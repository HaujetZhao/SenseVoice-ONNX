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
    Numba 加速版并行热词雷达
    返回: match_results (N, 4) -> [found, start_frame, end_frame, avg_prob]
    """
    num_hotwords = len(hotword_offsets) - 1
    T, K = topk_ids.shape
    results = np.zeros((num_hotwords, 4), dtype=np.float32)

    # 并行遍历每一个热词
    for i in prange(num_hotwords):
        start_off = hotword_offsets[i]
        end_off = hotword_offsets[i+1]
        word_len = end_off - start_off
        
        # 扫描起始点
        t_ptr = 0
        while t_ptr < T:
            # 尝试在这个 t_ptr 为起点搜寻该热词
            current_token_idx = 0
            search_ptr = t_ptr
            
            # 用于记录路径的锚点信息
            match_start = -1
            match_end = -1
            match_prob_sum = 0.0
            has_real_emission = False
            
            while current_token_idx < word_len and search_ptr < T:
                target_token_id = hotword_ids_flat[start_off + current_token_idx]
                
                # 在窗口内寻找该 Token 的最佳锚点
                found_t = -1
                best_p = -1.0
                
                look_ahead_end = min(search_ptr + max_lookahead, T)
                
                for t in range(search_ptr, look_ahead_end):
                    # 间隙约束校验
                    if current_token_idx > 0:
                        last_t = match_end
                        # 统计 (last_t, t) 之间的非空 Greedy 帧数
                        gap_emissions = 0
                        for k in range(last_t + 1, t):
                            if top1_indices[k] != blank_id:
                                gap_emissions += 1
                        
                        if gap_emissions > max_gap:
                            continue
                    
                    # 搜寻 Top-K 候选项
                    for k_idx in range(K):
                        if topk_ids[t, k_idx] == target_token_id:
                            prob = topk_probs[t, k_idx]
                            if prob > best_p:
                                best_p = prob
                                found_t = t
                
                if found_t != -1:
                    # 锁定一个 Token 锚点
                    if current_token_idx == 0:
                        match_start = found_t
                    match_end = found_t
                    match_prob_sum += best_p
                    if top1_indices[found_t] != blank_id:
                        has_real_emission = True
                    
                    search_ptr = found_t + 1
                    current_token_idx += 1
                else:
                    # 路径断了
                    break
            
            # 最终路径校验
            if current_token_idx == word_len and has_real_emission:
                results[i, 0] = 1.0 # found
                results[i, 1] = float(match_start)
                results[i, 2] = float(match_end)
                results[i, 3] = match_prob_sum / word_len
                break # 该词已找到，停止搜索
            
            # 起点步进
            t_ptr += 1
            
    return results

class FastHotwordRadar:
    def __init__(self, hotwords, tokenizer):
        self.tokenizer = tokenizer
        self.hotwords = hotwords
        
        # 预编码热词为 ID 序列
        all_ids = []
        offsets = [0]
        for w in hotwords:
            ids = tokenizer.encode(w)
            all_ids.extend(ids)
            offsets.append(len(all_ids))
            
        self.hotword_ids_flat = np.array(all_ids, dtype=np.int32)
        self.hotword_offsets = np.array(offsets, dtype=np.int32)

    def scan(self, topk_ids, topk_probs, top1_indices, blank_id=0):
        """
        调用 Numba 并行加速扫描
        """
        raw_results = numba_parallel_scan(
            topk_ids,
            topk_probs,
            top1_indices,
            self.hotword_ids_flat,
            self.hotword_offsets,
            blank_id=blank_id
        )
        
        # 整理结果
        final_detected = []
        for i in range(len(self.hotwords)):
            if raw_results[i, 0] > 0:
                final_detected.append({
                    "text": self.hotwords[i],
                    "start": raw_results[i, 1] * 0.060,
                    "end": raw_results[i, 2] * 0.060,
                    "prob": raw_results[i, 3]
                })
        return final_detected
