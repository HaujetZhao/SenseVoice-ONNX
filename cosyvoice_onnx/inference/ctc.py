import numpy as np

def greedy_search(log_probs, tokenizer, blank_id=0, prompt_len=4):
    """
    带时间戳的贪婪搜索。
    返回: List[dict] -> [{'text': '那', 'start': 0.72}, ...]
    """
    logits = log_probs[0, prompt_len:, :]
    indices = np.argmax(logits, axis=-1)
    
    # 连续去重
    collapsed = []
    if len(indices) > 0:
        current_id = indices[0]
        start_frame = 0
        for i in range(1, len(indices)):
            if indices[i] != current_id:
                collapsed.append((current_id, start_frame))
                current_id = indices[i]
                start_frame = i
        collapsed.append((current_id, start_frame))

    results = []
    for token_id, frame_idx in collapsed:
        if token_id == blank_id:
            continue
            
        # 强制转换为 Python int，防止 numpy.int64 导致 sentencepiece 报错
        token_id = int(token_id)
        
        # 解码文本
        char = tokenizer.id_to_piece(token_id) if hasattr(tokenizer, 'id_to_piece') else tokenizer.decode([token_id])
        if not char.strip():
            continue
            
        # 计算时间戳 (单位: 秒)
        t_start = frame_idx * 0.060 
        
        results.append({
            "text": char,
            "start": round(t_start, 3)
        })
        
    return results

def topk_search(log_probs, tokenizer, top_k=40, blank_id=0, prompt_len=4):
    """
    获取每一帧的 Top-K 候选项及其概率
    返回: List[List[Tuple[char, prob]]]
    """
    # 1. 裁剪掉前 prompt_len 帧
    # 注意: 输入 log_probs 形状通常是 (1, T_plus_4, Vocab)
    logits = log_probs[0, prompt_len:, :]
    T, V = logits.shape
    
    # 2. 计算 Softmax 概率
    # 为了数值稳定，我们在采样前转为 float32
    exp_logits = np.exp(logits.astype(np.float32) - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    # 3. 获取 Top-K 索引和概率
    # numpy 没有直接的 topk，我们用 partition 或 argsort
    # 为了获取前 K 个，我们用 argsort 取最后 K 个并翻转
    topk_data = []
    for t in range(T):
        p_frame = probs[t]
        # 获取最大值的索引
        indices = np.argsort(p_frame)[-top_k:][::-1]
        
        candidates = []
        for idx in indices:
            idx = int(idx)
            prob = float(p_frame[idx])
            
            if idx == blank_id:
                char = "[BLANK]"
            else:
                # 使用 sentencepiece 的 id_to_piece 或 decode
                char = tokenizer.id_to_piece(idx) if hasattr(tokenizer, 'id_to_piece') else tokenizer.decode([idx])
                # 处理一些特殊符号
                if not char.strip() and idx != blank_id:
                    char = f"<{idx}>"
            
            candidates.append((char, prob))
        topk_data.append(candidates)
        
    return topk_data
