import re
import numpy as np

class HotwordRadar:
    """
    [字符级匹配版] 高性能热词召回组件
    核心算法：基于字符级前缀消耗的 DFS 回溯 + 记忆化搜索
    
    相比旧版（基于 Token ID 序列匹配），本版本解决了分词器切分与解码器
    Top-K 输出切分不一致导致匹配失败的问题。
    例如 "CapsWriter" 分词器拆为 ['Ca','ps','W','ri','ter']，
    但解码器输出 ['cap','s','writer']，旧版无法匹配，新版可以。
    """
    def __init__(self, hotwords, tokenizer):
        self.tokenizer = tokenizer
        self.hotwords = hotwords
        
        # 1. 预计算全量词表的小写映射 (去掉 SP 标记 ▁，用于字符级匹配)
        self.vocab_lower = []
        for i in range(tokenizer.get_piece_size()):
            piece = tokenizer.id_to_piece(i)
            self.vocab_lower.append(piece.lower().replace('\u2581', '').strip())
        
        # 2. 预处理搜索词：去除符号
        self.search_hotwords = [re.sub(r'[^\w\s]+', ' ', w) for w in hotwords]
        
        # 3. 构建热词的小写纯字符串（去符号、去空格、小写化）
        self.hotword_lower_strings = []
        for sw in self.search_hotwords:
            # 去除所有空白字符，然后小写化
            clean = re.sub(r'\s+', '', sw).lower()
            self.hotword_lower_strings.append(clean)
        
        # 4. 构建首字符前缀索引: {首字符: [word_idx, ...]}
        self.char_prefix_index = {}
        for idx, hw_lower in enumerate(self.hotword_lower_strings):
            if not hw_lower:
                continue
            first_char = hw_lower[0]
            if first_char not in self.char_prefix_index:
                self.char_prefix_index[first_char] = []
            self.char_prefix_index[first_char].append(idx)

    def scan(self, topk_ids, topk_probs, top1_indices, blank_id=0, max_lookahead=15, max_gap=0):
        """
        [单次扫描] 获取所有非重叠命中结果
        
        算法流程：
        1. 逐帧扫描，仅在 greedy 非空帧触发匹配
        2. 对该帧的 Top-K token，检查其字符是否为某热词的开头
        3. 若是，执行字符级 DFS 回溯匹配
        4. 后处理：非空帧过滤 + 长度优先去重
        """
        T, K = topk_ids.shape
        hits = []

        for t in range(T):
            # 准入条件：必须是 Greedy 非空帧
            if top1_indices[t] == blank_id:
                continue
            
            # 记录已尝试的 token 字符串，避免重复搜索
            seen_tokens = set()
            
            for k in range(K):
                tc = self.vocab_lower[topk_ids[t, k]]
                if not tc or tc in seen_tokens:
                    continue
                seen_tokens.add(tc)
                
                # 检查该 token 的首字符是否能触发某个热词
                first_char = tc[0]
                if first_char not in self.char_prefix_index:
                    continue
                
                for word_idx in self.char_prefix_index[first_char]:
                    hw_lower = self.hotword_lower_strings[word_idx]
                    
                    # 快速前缀检查：热词必须以该 token 的字符开头
                    if not hw_lower.startswith(tc):
                        continue
                    
                    match_data = self._try_match_chars(
                        t, k, hw_lower, topk_ids, topk_probs, top1_indices,
                        blank_id, max_lookahead
                    )
                    
                    if match_data:
                        hits.append({
                            "word_idx": word_idx,
                            "start_frame": t,
                            "end_frame": match_data["end_frame"],
                            "prob": match_data["prob"],
                            "frame_indices": match_data["frame_indices"],
                            "matched_tokens": match_data["matched_tokens"],
                            "non_blank_count": match_data["non_blank_count"],
                            "has_word_boundary": match_data.get("has_word_boundary", False)
                        })
                        name = self.hotwords[word_idx]
                        print(f"[Radar Scan] 发现匹配: '{name}' | 起始帧: {t} | "
                              f"Token路径: {match_data['frame_indices']} | "
                              f"匹配token: {match_data['matched_tokens']}")

        # 打印调试信息
        if hits:
            print(f"\n[Radar Debug] 原始命中 ({len(hits)}个):")
            for h in hits:
                name = self.hotwords[h['word_idx']]
                print(f"  - 候选: {name:<15} | 区间: {h['start_frame']:3d}-{h['end_frame']:3d} | "
                      f"概率: {h['prob']:.4f} | 非空帧: {h['non_blank_count']}")

        # 后处理：非重叠合并
        return self._post_process(hits, top1_indices, blank_id)

    def _try_match_chars(self, t_start, k_start, hotword_lower, topk_ids, topk_probs, top1_indices, blank_id, max_lookahead):
        """
        字符级前缀消耗 + DFS 回溯 + 概率最优路径选择
        
        核心思想：
        - 维护一个字符游标 cursor，指向热词中尚未匹配的位置
        - 每一步从当前帧的 Top-K 中选一个 token，其小写字符必须是
          remaining（hotword_lower[cursor:]）的前缀
        - 该 token "消耗" len(token_chars) 个字符，cursor 前进相应步数
        - DFS 探索**所有有效路径**，返回平均概率最高的那条
        - 仅对已证实无解的 (frame, cursor) 做记忆化剪枝
        
        Args:
            t_start: 触发帧
            k_start: 触发 token 在 Top-K 中的位置
            hotword_lower: 热词的小写纯字符串
            其余参数同 scan
        
        Returns:
            匹配成功返回 dict（含 frame_indices, matched_tokens 等），失败返回 None
        """
        T, K = topk_ids.shape
        
        # 首 token 检查（已在 scan 中做了前缀验证，这里直接消耗）
        first_tid = int(topk_ids[t_start, k_start])
        first_token = self.vocab_lower[first_tid]
        initial_cursor = len(first_token)
        initial_prob = float(topk_probs[t_start, k_start])
        
        # 检测首 token 原始 piece 是否带词边界标记 ▁（用于恢复空格）
        original_piece = self.tokenizer.id_to_piece(first_tid)
        has_word_boundary = original_piece.startswith('\u2581')
        
        # 首 token 就完成整个热词（极短热词）
        if initial_cursor >= len(hotword_lower):
            non_blank = 1 if top1_indices[t_start] != blank_id else 0
            return {
                "end_frame": t_start,
                "prob": initial_prob,
                "frame_indices": [t_start],
                "matched_tokens": [first_token],
                "non_blank_count": non_blank,
                "has_word_boundary": has_word_boundary
            }
        
        # DFS 回溯搜索后续 token
        # fail_set: 仅记录已证实无解的 (frame, cursor) 状态
        # 不缓存成功结果，以保证全局概率最优（不同前缀下最优后缀可能不同）
        fail_set = set()
        
        def dfs(search_from, cursor):
            """
            从 search_from 帧开始，尝试消耗 hotword_lower[cursor:] 的剩余字符。
            探索所有有效路径，返回**平均概率最高**的那条。
            
            回溯机制：
            - 每步尝试当前帧 Top-K 的所有 token
            - 若某 token 能消耗 remaining 的前缀，递归搜索后续帧
            - 不提前返回，而是收集所有成功路径，取概率最优
            - 已证实无解的 (frame, cursor) 通过 fail_set 剪枝
            
            Returns:
                最优路径: (frames_list, probs_list, tokens_list)
                无解: None
            """
            # 终止条件：所有字符已被消耗
            if cursor >= len(hotword_lower):
                return ([], [], [])
            
            key = (search_from, cursor)
            if key in fail_set:
                return None
            
            remaining = hotword_lower[cursor:]
            search_end = min(search_from + max_lookahead, T)
            
            best = None
            best_avg = -1.0
            
            for f in range(search_from, search_end):
                # 间隙约束：search_from 到 f 之间不能有 greedy 非空帧
                # （search_from - 1 是上一个匹配帧，f 是当前候选帧）
                if f > search_from:
                    gap_emissions = np.count_nonzero(
                        top1_indices[search_from:f] != blank_id
                    )
                    if gap_emissions > 0:
                        break  # 后续帧间隙只会更大，直接终止
                
                # 遍历该帧的 Top-K token
                for k in range(K):
                    tc = self.vocab_lower[topk_ids[f, k]]
                    if not tc:
                        continue
                    
                    # 核心匹配逻辑：token 字符必须是 remaining 的前缀
                    if remaining.startswith(tc):
                        new_cursor = cursor + len(tc)
                        sub_result = dfs(f + 1, new_cursor)
                        
                        if sub_result is not None:
                            frames, probs, tokens = sub_result
                            c_probs = [float(topk_probs[f, k])] + probs
                            avg = sum(c_probs) / len(c_probs)
                            if avg > best_avg:
                                best = (
                                    [f] + frames,
                                    c_probs,
                                    [tc] + tokens
                                )
                                best_avg = avg
            
            # 所有路径均失败 → 记入 fail_set 供后续剪枝
            if best is None:
                fail_set.add(key)
            return best
        
        # 从首 token 之后的帧开始 DFS
        sub_result = dfs(t_start + 1, initial_cursor)
        if sub_result is None:
            return None
        
        frames, probs, tokens = sub_result
        all_frames = [t_start] + frames
        all_probs = [initial_prob] + probs
        all_tokens = [first_token] + tokens
        
        # 统计匹配路径中覆盖的非空 Greedy 帧数量
        non_blank_count = sum(1 for f in all_frames if top1_indices[f] != blank_id)
        
        return {
            "end_frame": all_frames[-1],
            "prob": sum(all_probs) / len(all_probs),
            "frame_indices": all_frames,
            "matched_tokens": all_tokens,
            "non_blank_count": non_blank_count,
            "has_word_boundary": has_word_boundary
        }

    def _post_process(self, hits, top1_indices, blank_id):
        """
        去重合并：优先保留长度更长的热词
        约束：必须覆盖至少 2 个非空 Greedy 帧
        """
        if not hits: return []

        # 0. 过滤：只保留覆盖至少 2 个非空帧的热词
        filtered_hits = []
        for h in hits:
            if h['non_blank_count'] >= 2:
                filtered_hits.append(h)
            else:
                name = self.hotwords[h['word_idx']]
                print(f"[Radar Filter] ❌ 过滤掉 '{name}': 非空帧数={h['non_blank_count']} < 2")

        if not filtered_hits:
            print(f"[Radar Filter] 所有候选均被过滤（未覆盖足够非空帧）")
            return []

        # 1. 按开始时间排序
        filtered_hits.sort(key=lambda x: x["start_frame"])

        selected_hits = []
        i = 0
        while i < len(filtered_hits):
            curr = filtered_hits[i]
            best_h = curr
            best_len = len(self.hotwords[curr["word_idx"]])

            # 向后探测有冲突（区间重叠）的项
            j = i + 1
            while j < len(filtered_hits):
                nxt = filtered_hits[j]
                if nxt["start_frame"] <= best_h["end_frame"]:
                    nxt_len = len(self.hotwords[nxt["word_idx"]])
                    name_curr = self.hotwords[best_h['word_idx']]
                    name_nxt = self.hotwords[nxt['word_idx']]
                    print(f"[Radar Filter] 冲突发现: '{name_curr}'(len={best_len}) vs '{name_nxt}'(len={nxt_len})")
                    if nxt_len > best_len:
                        print(f"  >> 切换为更长者: '{name_nxt}'")
                        best_h = nxt
                        best_len = nxt_len
                    else:
                        print(f"  >> 保留原强者: '{name_curr}'")
                    j += 1
                else:
                    break

            selected_hits.append(best_h)
            i = j
            
        # 2. 转换为用户友好的结构
        final_detected = []
        for h in selected_hits:
            idx = h["word_idx"]
            token_details = []
            
            # 使用匹配路径中实际消耗的 token 字符串
            for i, f_idx in enumerate(h["frame_indices"]):
                token_details.append({
                    "token": h["matched_tokens"][i],
                    "time": round(f_idx * 0.060, 3)
                })
            
            # 若首 token 带有词边界标记 ▁，在热词前补空格以恢复词间距
            text = self.hotwords[idx]
            if h.get("has_word_boundary", False):
                text = " " + text
            
            final_detected.append({
                "text": text,
                "start": round(h["start_frame"] * 0.060, 3),
                "end": round(h["end_frame"] * 0.060, 3),
                "prob": round(h["prob"], 4),
                "tokens": token_details
            })
                
        return final_detected
