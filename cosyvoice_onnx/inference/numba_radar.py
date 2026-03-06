import numpy as np
from numba import njit, prange

@njit
def build_prefix_index(hotword_ids_flat, hotword_offsets, vocab_size=32000):
    """
    构建 CSR 格式的 TokenID -> [WordID1, WordID2, ...] 的映射
    用于“帧找词”时快速定位哪些热词以此 Token 开头
    """
    num_hotwords = len(hotword_offsets) - 1
    counts = np.zeros(vocab_size, dtype=np.int32)
    
    # 1. 统计每个 Token 作为多少个热词的首字
    for i in range(num_hotwords):
        if hotword_offsets[i+1] > hotword_offsets[i]:
            first_tid = hotword_ids_flat[hotword_offsets[i]]
            if first_tid < vocab_size:
                counts[first_tid] += 1
                
    # 2. 计算偏移
    token_offsets = np.zeros(vocab_size + 1, dtype=np.int32)
    for i in range(vocab_size):
        token_offsets[i+1] = token_offsets[i] + counts[i]
        
    # 3. 填充 WordIDs
    word_ids = np.zeros(token_offsets[vocab_size], dtype=np.int32)
    current_pos = token_offsets.copy()
    for i in range(num_hotwords):
        if hotword_offsets[i+1] > hotword_offsets[i]:
            first_tid = hotword_ids_flat[hotword_offsets[i]]
            if first_tid < vocab_size:
                pos = current_pos[first_tid]
                word_ids[pos] = i
                current_pos[first_tid] += 1
                
    return token_offsets, word_ids

@njit()
def numba_parallel_scan(
    topk_ids,           # (T, K) int32
    topk_probs,         # (T, K) float32
    top1_indices,       # (T,) int32
    hotword_ids_flat,   # 所有热词展平后的 ID 序列
    hotword_offsets,    # 每个热词在 flat 中的起始偏移 (N+1)
    prefix_offsets,     # Token -> WordID 的 CSR 偏移
    prefix_word_ids,    # Token -> WordID 的 CSR 数据
    blank_id=0,
    max_lookahead=15,
    max_gap=1
):
    """
    [帧触发扫描算法]
    遍历每一帧，如果在 Top-K 中发现了热词的首字，则定向启动该热词的后续匹配。
    """
    T, K = topk_ids.shape
    num_hotwords = len(hotword_offsets) - 1
    
    results_t = np.zeros((T, 4), dtype=np.float32)
    results_word_id = np.full(T, -1, dtype=np.int32)
    token_frames_output_t = np.full((T, 32), -1, dtype=np.int32)

    for t in range(T):
        # 准入条件：必须是 Greedy 非空帧
        if top1_indices[t] == blank_id:
            continue
            
        # 检查当前帧的所有 Top-K 候选项
        best_match_word = -1
        best_match_prob = -1.0
        best_match_end = -1
        best_match_frames = np.full(32, -1, dtype=np.int32)

        for k in range(K):
            tid = topk_ids[t, k]
            if tid >= len(prefix_offsets) - 1: continue
            
            # 查找哪些热词以此 Token 开头
            p_start = prefix_offsets[tid]
            p_end = prefix_offsets[tid+1]
            
            for p_idx in range(p_start, p_end):
                word_idx = prefix_word_ids[p_idx]
                
                # 尝试完整匹配该热词
                start_off = hotword_offsets[word_idx]
                end_off = hotword_offsets[word_idx+1]
                word_len = end_off - start_off
                
                curr_token_idx = 0
                search_ptr = t
                match_end = -1
                match_prob_sum = 0.0
                temp_frames = np.full(32, -1, dtype=np.int32)
                
                while curr_token_idx < word_len and search_ptr < T:
                    target_tid = hotword_ids_flat[start_off + curr_token_idx]
                    found_t = -1
                    best_p = -1.0
                    
                    if curr_token_idx == 0:
                        # 首音就在当前帧确认为 tid
                        found_t = t
                        best_p = topk_probs[t, k]
                    else:
                        # 后续音跳跃查找
                        look_ahead_end = min(search_ptr + max_lookahead, T)
                        for tt in range(search_ptr, look_ahead_end):
                            # 间隙检查 (Gap)
                            last_t = match_end
                            gap_emissions = 0
                            for gap_k in range(last_t + 1, tt):
                                if top1_indices[gap_k] != blank_id:
                                    gap_emissions += 1
                            if gap_emissions > max_gap:
                                continue
                                
                            for kk in range(K):
                                if topk_ids[tt, kk] == target_tid:
                                    p = topk_probs[tt, kk]
                                    if p > best_p:
                                        best_p = p
                                        found_t = tt
                    
                    if found_t != -1:
                        match_end = found_t
                        match_prob_sum += best_p
                        temp_frames[curr_token_idx] = found_t
                        search_ptr = found_t + 1
                        curr_token_idx += 1
                    else:
                        break
                
                # 判定成功
                if curr_token_idx == word_len:
                    avg_p = match_prob_sum / word_len
                    # 在同一帧触发的多个热词中，取概率最高的那个
                    if avg_p > best_match_prob:
                        best_match_prob = avg_p
                        best_match_word = word_idx
                        best_match_end = match_end
                        best_match_frames[:] = temp_frames[:]
        
        # 记录本帧作为起点的最佳匹配
        if best_match_word != -1:
            results_t[t, 0] = 1.0
            results_t[t, 1] = float(t)
            results_t[t, 2] = float(best_match_end)
            results_t[t, 3] = best_match_prob
            results_word_id[t] = best_match_word
            token_frames_output_t[t, :] = best_match_frames[:]

    return results_t, results_word_id, token_frames_output_t

class FastHotwordRadar:
    def __init__(self, hotwords, tokenizer):
        import re
        self.tokenizer = tokenizer
        self.hotwords = hotwords
        self.search_hotwords = [re.sub(r'[^\w\s]+', ' ', w) for w in hotwords]
        self.hotword_tokens = [tokenizer.encode_as_pieces(sw) for sw in self.search_hotwords]
        
        all_ids = []
        offsets = [0]
        for sw in self.search_hotwords:
            ids = tokenizer.encode(sw)
            all_ids.extend(ids)
            offsets.append(len(all_ids))
            
        self.hotword_ids_flat = np.array(all_ids, dtype=np.int32)
        self.hotword_offsets = np.array(offsets, dtype=np.int32)
        
        # 核心：预构建前缀索引
        t_off, w_ids = build_prefix_index(self.hotword_ids_flat, self.hotword_offsets)
        self.prefix_offsets = t_off
        self.prefix_word_ids = w_ids

    def scan(self, topk_ids, topk_probs, top1_indices, blank_id=0):
        raw_res, raw_w_ids, raw_t_frames = numba_parallel_scan(
            topk_ids, topk_probs, top1_indices,
            self.hotword_ids_flat, self.hotword_offsets,
            self.prefix_offsets, self.prefix_word_ids,
            blank_id=blank_id
        )
        
        # 后处理：去重和并。因为不同帧触发的结果可能会重叠。
        # 简单策略：按照时间顺序遍历，如果当前命中点已被之前的命中覆盖，则跳过。
        final_detected = []
        last_covered_until = -1
        
        # 查找所有命中点
        hits = []
        for t in range(len(raw_res)):
            if raw_res[t, 0] > 0:
                hits.append({
                    "word_idx": raw_w_ids[t],
                    "start": raw_res[t, 1],
                    "end": raw_res[t, 2],
                    "prob": raw_res[t, 3],
                    "frame_indices": raw_t_frames[t, :]
                })
        
        # 按开始时间排序
        hits.sort(key=lambda x: x["start"])
        
        for h in hits:
            if h["start"] > last_covered_until:
                idx = h["word_idx"]
                tokens = self.hotword_tokens[idx]
                token_details = []
                for tk_pos in range(len(tokens)):
                    f_idx = h["frame_indices"][tk_pos]
                    if f_idx != -1:
                        token_details.append({"token": tokens[tk_pos], "time": f_idx * 0.060})
                
                final_detected.append({
                    "text": self.hotwords[idx],
                    "start": h["start"] * 0.060,
                    "end": h["end"] * 0.060,
                    "prob": h["prob"],
                    "tokens": token_details
                })
                last_covered_until = h["end"]
                
        return final_detected
