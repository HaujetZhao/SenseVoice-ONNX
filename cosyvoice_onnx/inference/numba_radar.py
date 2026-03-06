import numpy as np
from numba import njit, prange

@njit(parallel=True)
def numba_parallel_scan(
    topk_ids,           # (T, K) int32
    topk_probs,         # (T, K) float32
    top1_indices,       # (T,) int32
    hotword_ids_flat,   # 所有热词展平后的 ID 序列
    hotword_offsets,    # 每个热词在 flat 中的起始偏移 (N+1)
    blank_id=0,
    max_lookahead=15,
    max_gap=1
):
    """
    Numba 加速版并行热词雷达 (支持多次召回 & 逐词时间戳)
    返回: 
      - results (N, 8, 4): [found, start_frame, end_frame, avg_prob], 最多支持 8 次重复命中
      - token_frames_output (N, 8, 32): 每个命中的 Token 帧索引
    """
    num_hotwords = len(hotword_offsets) - 1
    T, K = topk_ids.shape
    results = np.zeros((num_hotwords, 8, 4), dtype=np.float32)
    token_frames_output = np.full((num_hotwords, 8, 32), -1, dtype=np.int32) 

    for i in prange(num_hotwords):
        start_off = hotword_offsets[i]
        end_off = hotword_offsets[i+1]
        word_len = end_off - start_off
        
        match_count = 0
        t_ptr = 0
        while t_ptr < T and match_count < 8:
            if top1_indices[t_ptr] == blank_id:
                t_ptr += 1
                continue
                
            current_token_idx = 0
            search_ptr = t_ptr
            match_start = -1
            match_end = -1
            match_prob_sum = 0.0
            temp_frames = np.full(32, -1, dtype=np.int32)
            
            while current_token_idx < word_len and search_ptr < T:
                target_token_id = hotword_ids_flat[start_off + current_token_idx]
                found_t = -1
                best_p = -1.0
                look_ahead_end = min(search_ptr + max_lookahead, T)
                
                for t in range(search_ptr, look_ahead_end):
                    if current_token_idx > 0:
                        last_t = match_end
                        gap_emissions = 0
                        for k in range(last_t + 1, t):
                            if top1_indices[k] != blank_id:
                                gap_emissions += 1
                        if gap_emissions > max_gap:
                            continue
                    
                    for k_idx in range(K):
                        if topk_ids[t, k_idx] == target_token_id:
                            prob = topk_probs[t, k_idx]
                            if prob > best_p:
                                best_p = prob
                                found_t = t
                
                if found_t != -1:
                    if current_token_idx == 0: match_start = found_t
                    match_end = found_t
                    match_prob_sum += best_p
                    temp_frames[current_token_idx] = found_t
                    search_ptr = found_t + 1
                    current_token_idx += 1
                else:
                    break
            
            if current_token_idx == word_len:
                # 记录当前这次命中（第 match_count 次）
                results[i, match_count, 0] = 1.0 
                results[i, match_count, 1] = float(match_start)
                results[i, match_count, 2] = float(match_end)
                results[i, match_count, 3] = match_prob_sum / word_len
                for token_pos in range(word_len):
                    token_frames_output[i, match_count, token_pos] = temp_frames[token_pos]
                
                match_count += 1
                # 关键：跳转到本次命中结束之后，确保“无重叠”
                t_ptr = match_end + 1
                continue
            
            t_ptr += 1
            
    return results, token_frames_output

class FastHotwordRadar:
    def __init__(self, hotwords, tokenizer):
        import re
        self.tokenizer = tokenizer
        self.hotwords = hotwords
        # 核心优化：搜索词 vs 显示词分离
        # 将所有标点符号（非单词、非空格字符）替换为空格，以匹配 ASR 的正常 Token 输出
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

    def scan(self, topk_ids, topk_probs, top1_indices, blank_id=0):
        raw_results, raw_token_frames = numba_parallel_scan(
            topk_ids,
            topk_probs,
            top1_indices,
            self.hotword_ids_flat,
            self.hotword_offsets,
            blank_id=blank_id
        )
        
        final_detected = []
        for i in range(len(self.hotwords)):
            # 遍历 8 个可能的重复命中槽位
            for match_idx in range(8):
                if raw_results[i, match_idx, 0] > 0:
                    tokens = self.hotword_tokens[i]
                    token_details = []
                    for t_idx in range(len(tokens)):
                        frame_idx = raw_token_frames[i, match_idx, t_idx]
                        if frame_idx != -1:
                            token_details.append({
                                "token": tokens[t_idx],
                                "time": frame_idx * 0.060
                            })
                    
                    final_detected.append({
                        "text": self.hotwords[i], 
                        "start": raw_results[i, match_idx, 1] * 0.060,
                        "end": raw_results[i, match_idx, 2] * 0.060,
                        "prob": raw_results[i, match_idx, 3],
                        "tokens": token_details
                    })
        return final_detected
