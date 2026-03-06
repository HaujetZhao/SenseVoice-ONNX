import numpy as np

class ResultIntegrator:
    @staticmethod
    def integrate(greedy_results, detected_hotwords):
        """
        [核心算法] 将 Greedy 识别流与热词匹配流进行无缝融合与替换
        Args:
            greedy_results: List[dict] -> [{'text': '...', 'start': ...}, ...]
            detected_hotwords: List[dict] -> [{'text': '...', 'start': ..., 'tokens': [...]}, ...]
        Returns:
            final_results: List[dict]
        """
        # 1. 按照时间顺序排列所有检测到的热词
        detected_hotwords.sort(key=lambda x: x["start"])
        
        final_results = []
        last_hotword_end = -1.0
        results_set = set() # 记录已处理的热词原始文本，防止同一热词在重叠区域被多次插入
        
        for g in greedy_results:
            replaced = False
            for hw in detected_hotwords:
                # 判定当前 Greedy 符号是否落在热词的“领土”内
                # 0.02s 的冗余是为了处理浮点数微小偏移和 CTC 的发散性
                if g["start"] >= hw["start"] - 0.02 and g["start"] <= hw["end"] + 0.02:
                    if hw["text"] not in results_set:
                        # 执行 Token 块级别插入
                        integrated_chunks = ResultIntegrator._merge_tokens_to_chunks(hw)
                        final_results.extend(integrated_chunks)
                        
                        results_set.add(hw["text"])
                        last_hotword_end = hw["end"]
                    replaced = True
                    break
            
            # 如果没被替换，且不在热词的封锁区内，则作为普通文本加入
            if not replaced and g["start"] > last_hotword_end:
                # 冗余压制逻辑：如果刚才刚结束一个热词，且现在的字跟热词里的字长得一样
                # 可能是由于对齐误差导致的重复，我们进行简单的静默跳过
                if last_hotword_end > 0 and (g["start"] - last_hotword_end < 0.5):
                    # 查找上一个落点的热词文本
                    last_hw = [h["text"] for h in detected_hotwords if abs(h["end"] - last_hotword_end) < 0.01]
                    if last_hw and g["text"] in last_hw[0]:
                        continue
                        
                final_results.append({
                    "text": g["text"],
                    "start": g["start"],
                    "is_hotword": False
                })
                
        return final_results

    @staticmethod
    def _merge_tokens_to_chunks(hw):
        """
        将热词内部的 Token 和原始文本进行“块对齐”切分
        """
        origin_text = hw["text"]
        search_base = origin_text.lower()
        chunks = []
        
        # 1. 寻找每个 Token 覆盖的字符起始位置
        anchors = [] # (idx_in_text, timestamp)
        curr_search_pos = 0
        for tk in hw["tokens"]:
            clean_tk = tk["token"].replace("\u2581", "").strip().lower()
            if not clean_tk: continue
            idx = search_base.find(clean_tk, curr_search_pos)
            if idx != -1:
                anchors.append((idx, tk["time"]))
                curr_search_pos = idx + len(clean_tk)
        
        if not anchors: 
            anchors.append((0, hw["start"]))
        elif anchors[0][0] != 0:
            anchors.insert(0, (0, hw["start"]))
            
        # 2. 根据锚点切割原始文本块
        for i in range(len(anchors)):
            start_idx, start_time = anchors[i]
            next_idx = anchors[i+1][0] if (i+1) < len(anchors) else len(origin_text)
            
            chunk_text = origin_text[start_idx:next_idx]
            chunks.append({
                "text": chunk_text,
                "start": start_time,
                "is_hotword": True
            })
        return chunks
