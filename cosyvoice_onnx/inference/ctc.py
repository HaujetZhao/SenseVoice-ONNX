import numpy as np

def greedy_search(logits, blank_id=0, prompt_len=4):
    """
    极简 CTC 贪婪搜索。
    
    Args:
        logits: 模型输出 (1, T_plus_4, Vocab)
        blank_id: 空白占位符 ID
        prompt_len: 要跳过的 Prompt 帧数
    """
    # 1. 裁剪掉前 prompt_len 帧并做 ArgMax
    logits = logits[0, prompt_len:, :]
    indices = np.argmax(logits, axis=-1)
    
    # 2. 连续去重 & 移除空白
    decoded_ids = []
    last_id = -1
    for idx in indices:
        idx = int(idx)
        if idx != blank_id and idx != last_id:
            # 过滤特殊控制 Token (通常 > 25000 为时间戳或特殊符号)
            if idx < 25000:
                decoded_ids.append(idx)
        last_id = idx
        
    return decoded_ids
