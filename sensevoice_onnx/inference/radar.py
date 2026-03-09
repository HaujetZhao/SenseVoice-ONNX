import re
import numpy as np

class HotwordRadar:
    """
    [直观版] 高性能热词召回组件 (纯 Python + 字典加速)
    核心算法：基于 Greedy 帧触发的前缀搜索
    """
    def __init__(self, hotwords, tokenizer):
        self.tokenizer = tokenizer
        self.hotwords = hotwords
        
        # 1. 预计算全量词表的小写映射 (加速 case-insensitive 匹配)
        # 注意: SentencePiece 可能包含特有的下划线 \u2581
        self.vocab_lower = []
        for i in range(tokenizer.get_piece_size()):
            piece = tokenizer.id_to_piece(i)
            # 我们将 piece 统一转小写，并去掉 SP 标记进行判定
            self.vocab_lower.append(piece.lower().replace('\u2581', '').strip())
        
        # 2. 预处理搜索词
        self.search_hotwords = [re.sub(r'[^\w\s]+', ' ', w) for w in hotwords]
        
        # 3. 构建前缀索引: {首字小写Piece: [(word_idx, [全量小写Piece序列]), ...]}
        self.prefix_index = {}
        self.hotword_lower_sequences = [] # 存储每个热词对应的小写 Piece 序列
        
        for idx, sw in enumerate(self.search_hotwords):
            token_ids = tokenizer.encode(sw)
            if not token_ids: 
                self.hotword_lower_sequences.append([])
                continue
            
            # 将该热词的所有 Token ID 预先转为小写 Piece 序列
            lower_pieces = [self.vocab_lower[tid] for tid in token_ids]
            self.hotword_lower_sequences.append(lower_pieces)
            
            first_p = lower_pieces[0]
            if not first_p: continue # 过滤空 piece
            
            if first_p not in self.prefix_index:
                self.prefix_index[first_p] = []
            self.prefix_index[first_p].append(idx)
            
        # 保存分段后的字符串形式用于回显
        self.hotword_pieces = [tokenizer.encode_as_pieces(sw) for sw in self.search_hotwords]

    def scan(self, topk_ids, topk_probs, top1_indices, blank_id=0, max_lookahead=15, max_gap=1):
        """
        [单次扫描] 获取所有非重叠命中结果
        """
        T, K = topk_ids.shape
        hits = []

        # 1. 帧驱动搜索
        for t in range(T):
            # 准入条件：必须是 Greedy 非空帧 (实音触发)
            if top1_indices[t] == blank_id:
                continue
            
            # 预提取该帧所有 Top-K 的小写 Piece
            frame_lower_pieces = [self.vocab_lower[tid] for tid in topk_ids[t]]
            
            # 检查当前帧 Top-K 中是否有能触发热词的首字 (小写匹配)
            best_match_in_frame = None
            
            # 记录已经检查过的小写 Piece，避免在一帧内重复扫描
            seen_pieces = set()
            
            for k in range(K):
                lp = frame_lower_pieces[k]
                if not lp or lp in seen_pieces: continue
                seen_pieces.add(lp)
                
                if lp not in self.prefix_index:
                    continue
                
                # 尝试匹配所有以此 Piece 开头的热词
                for word_idx in self.prefix_index[lp]:
                    lower_token_seq = self.hotword_lower_sequences[word_idx]
                    match_data = self._try_match(
                        t, k, lower_token_seq, topk_ids, topk_probs, top1_indices, 
                        blank_id, max_lookahead, max_gap
                    )
                    
                    if match_data:
                        if not best_match_in_frame or match_data["prob"] > best_match_in_frame["prob"]:
                            best_match_in_frame = {
                                "word_idx": word_idx,
                                "start_frame": t,
                                "end_frame": match_data["end_frame"],
                                "prob": match_data["prob"],
                                "frame_indices": match_data["frame_indices"]
                            }
            
            if best_match_in_frame:
                hits.append(best_match_in_frame)

        # 2. 后处理：非重叠合并
        return self._post_process(hits)

    def _try_match(self, t_start, k_start, lower_token_seq, topk_ids, topk_probs, top1_indices, blank_id, max_lookahead, max_gap):
        """内部尝试匹配一个特定词 (小写 Piece 匹配)"""
        T = topk_ids.shape[0]
        word_len = len(lower_token_seq)
        match_frames = []
        
        # 首字处理
        match_frames.append(t_start)
        prob_sum = topk_probs[t_start, k_start]
        last_t = t_start
        
        # 后续字跳跃匹配
        for i in range(1, word_len):
            target_lp = lower_token_seq[i]
            found_t = -1
            best_p = -1.0
            
            search_start = last_t + 1
            search_end = min(search_start + max_lookahead, T)
            
            for t in range(search_start, search_end):
                # 间隙约束
                gap_emissions = np.count_nonzero(top1_indices[last_t + 1 : t] != blank_id)
                if gap_emissions > max_gap:
                    continue
                
                # 在此帧的 Top-K 中找目标小写 Piece
                for k in range(topk_ids.shape[1]):
                    tid = topk_ids[t, k]
                    if self.vocab_lower[tid] == target_lp:
                        p = topk_probs[t, k]
                        if p > best_p:
                            best_p = p
                            found_t = t
            
            if found_t != -1:
                match_frames.append(found_t)
                prob_sum += best_p
                last_t = found_t
            else:
                return None # 链路中断
                
        return {
            "end_frame": last_t,
            "prob": prob_sum / word_len,
            "frame_indices": match_frames
        }

    def _post_process(self, hits):
        """去重合并，转换为用户友好的结构"""
        if not hits: return []
        
        # 按开始时间排序
        hits.sort(key=lambda x: x["start_frame"])
        
        final_detected = []
        last_covered_until = -1
        
        for h in hits:
            # 这里的判定条件确保取“第一个发现的名字”，且后续重叠部分不重复报
            if h["start_frame"] > last_covered_until:
                idx = h["word_idx"]
                pieces = self.hotword_pieces[idx]
                token_details = []
                
                for tk_pos, f_idx in enumerate(h["frame_indices"]):
                    token_details.append({
                        "token": pieces[tk_pos], 
                        "time": round(f_idx * 0.060, 3)
                    })
                
                final_detected.append({
                    "text": self.hotwords[idx],
                    "start": round(h["start_frame"] * 0.060, 3),
                    "end": round(h["end_frame"] * 0.060, 3),
                    "prob": round(h["prob"], 4),
                    "tokens": token_details
                })
                last_covered_until = h["end_frame"]
                
        return final_detected
