import re
import numpy as np

class HotwordRadar:
    """
    热词召回组件 (基于 Top-K 矩阵搜索路径)
    """
    def __init__(self, frames_topk):
        """
        Args:
            frames_topk: List[List[Tuple[char, prob]]]，每帧的 Top-K 候选项
        """
        self.frames = frames_topk

    def clean_text(self, text):
        """清理掉空格、符号，并转为小写"""
        return re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5]', '', str(text)).lower()

    def scan(self, hotword: str, tokenizer, top1_indices: np.ndarray, blank_id=0):
        """
        [大师版] 带有“严格排放间隙”约束的路径搜索
        """
        tokens = tokenizer.encode_as_pieces(hotword)
        if not tokens: return {"found": False}
        
        T = len(self.frames)
        M = len(tokens)
        
        # 搜索状态
        t_ptr = 0
        while t_ptr < T:
            match_map = {}
            match_frames = []
            current_token_idx = 0
            search_ptr = t_ptr
            
            while current_token_idx < M and search_ptr < T:
                target_token = tokens[current_token_idx]
                target_cleaned = self.clean_text(target_token)
                
                # 在窗口内寻找该 Token 的最佳锚点
                found_t = -1
                best_p = -1.0
                
                # 寻找窗口：15 帧 (900ms) 足够覆盖一个 BPE Token 的长度
                look_ahead = min(search_ptr + 15, T)
                
                for t in range(search_ptr, look_ahead):
                    # --- 核心：间隙约束校验 ---
                    if current_token_idx > 0:
                        last_t = match_frames[-1]["frame"]
                        # 统计 (last_t, t) 之间的非空 Greedy 帧数
                        # 注意：不包含 last_t 和 t 本身
                        gap_emissions = np.sum(top1_indices[last_t + 1 : t] != blank_id)
                        if gap_emissions > 1:
                            # 间隙太大，这个 t 不予考虑
                            continue
                    
                    # 检查当前帧是否包含目标 Token
                    candidates = self.frames[t]
                    for char, prob in candidates:
                        if self.clean_text(char) == target_cleaned:
                            if prob > best_p:
                                best_p = prob
                                found_t = t
                    
                    # 性能优化：高置信度锁定
                    if found_t == t and best_p > 0.7: break

                if found_t != -1:
                    # 匹配到一个 Token
                    match_map[found_t] = target_token
                    match_frames.append({
                        "frame": found_t, "token": target_token, "prob": best_p,
                        "is_greedy": (top1_indices[found_t] != blank_id)
                    })
                    search_ptr = found_t + 1
                    current_token_idx += 1
                else:
                    # 当前 Token 找不到了，说明这条路径断了
                    break
            
            # --- 结果校验 ---
            if current_token_idx == M:
                # 至少要撞上一个 Greedy 的“实音”
                if any(m["is_greedy"] for m in match_frames):
                    avg_prob = sum(m["prob"] for m in match_frames) / M
                    return {
                        "found": True, 
                        "start": match_frames[0]["frame"] * 0.060,
                        "end": match_frames[-1]["frame"] * 0.060,
                        "prob": avg_prob, 
                        "match_map": match_map
                    }
            
            # 如果这条路没走通，起点往后挪一帧，继续大海捞针
            t_ptr += 1
            
        return {"found": False}
        
        return {"found": False}
