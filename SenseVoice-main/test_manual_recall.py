import sys
import torch
import librosa
import numpy as np

# 引用 SenseVoice 目录
sys.path.append(r"d:\cosyvoice\SenseVoice-main")
from model import SenseVoiceSmall
from funasr import AutoModel

import re

def manual_greed_search(frames_top1):
    """最基本的贪心搜索：取每帧概率最大的字，去重，去BLANK"""
    tokens = []
    last_char = ""
    for char in frames_top1:
        if char != last_char:
            if char != "[BLANK]":
                tokens.append(char)
        last_char = char
    return "".join(tokens)

class HotwordRadar:
    def __init__(self, frames_top40):
        self.frames = frames_top40 # List of List[(char, prob)]

    def clean_text(self, text):
        """清理掉空格、符号，并转为小写"""
        return re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5]', '', text).lower()

    def scan(self, hotword):
        """
        在 Top-40 矩阵中搜索热词路径
        """
        cleaned_hotword = self.clean_text(hotword)
        if not cleaned_hotword:
            return {"found": False}
            
        chars = list(cleaned_hotword)
        current_char_idx = 0
        match_info = []

        for t, candidates in enumerate(self.frames):
            target = chars[current_char_idx]
            
            # 遍历当前帧的 Top 40
            for char, prob in candidates:
                # 同样清理候选字（处理 ASR 可能输出的空格或符号）
                cleaned_char = self.clean_text(char)
                if not cleaned_char:
                    continue
                
                # 如果是英文单词 Token（SenseVoice 可能会输出完整单词），只取开头看是否匹配
                # 这里我们假设简单的逐字符顺序匹配
                if cleaned_char.startswith(target) or target.startswith(cleaned_char):
                    match_info.append({"char": char, "frame": t, "prob": prob})
                    # 移动到热词中尚未匹配的部分
                    matched_len = min(len(cleaned_char), len(chars) - current_char_idx)
                    current_char_idx += matched_len
                    break 
            
            if current_char_idx >= len(chars):
                avg_prob = sum([m["prob"] for m in match_info]) / len(match_info)
                start_time = match_info[0]["frame"] * 60 / 1000
                end_time = match_info[-1]["frame"] * 60 / 1000
                return {"found": True, "start": start_time, "end": end_time, "prob": avg_prob}
        
        return {"found": False}

def main():
    # 更新热词和音频
    hotwords = ["Fun ASR Nano", "督工"]
    model_dir = "iic/SenseVoiceSmall"
    audio_path = r"d:\cosyvoice\test-fun.mp3"
    
    print(f"正在准备原生 SenseVoice 引擎...")
    loaded_model = AutoModel(model=model_dir, trust_remote_code=True, remote_code="./model.py", device="cuda:0")
    model, kwargs = loaded_model.model, loaded_model.kwargs
    tokenizer = kwargs["tokenizer"]
    model.eval()

    print(f"提取音频特征: {audio_path}")
    speech, _ = librosa.load(audio_path, sr=16000, mono=True)
    from funasr.utils.load_utils import extract_fbank
    audio_sample_list = [speech]
    speech_feat, feat_lengths = extract_fbank(audio_sample_list, data_type="sound", frontend=kwargs["frontend"])
    
    # 推理逻辑 (开启 withitn 以查看标点)
    with torch.no_grad():
        lid = model.embed(torch.LongTensor([[model.lid_dict["zh"]]]).to("cuda:0"))
        ee = model.embed(torch.LongTensor([[1, 2]]).to("cuda:0"))
        tn = model.embed(torch.LongTensor([[model.textnorm_dict["withitn"]]]).to("cuda:0"))
        
        full_speech = torch.cat((lid, ee, tn, speech_feat.to("cuda:0")), dim=1)
        encoder_out, _ = model.encoder(full_speech, feat_lengths.to("cuda:0") + 4)
        logits = model.ctc.log_softmax(encoder_out)[0, 4:]

    # --- 核心步骤：保存 Top-40 数据 ---
    frames_top40 = []
    frames_greedy = []
    
    for t in range(logits.shape[0]):
        probs = torch.softmax(logits[t], dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 40)
        
        candidates = []
        for k in range(40):
            tid = topk_indices[k].item()
            p = topk_probs[k].item()
            char = tokenizer.decode([tid]) if tid != 0 else "[BLANK]"
            if tid != 0 and not char.strip(): char = f"<{tid}>"
            candidates.append((char, p))
            if k == 0: frames_greedy.append(char)
        frames_top40.append(candidates)

    # --- 执行手动分析 ---
    print("\n" + "="*50)
    print("【1. 常规 Greedy 转录文本】")
    print(manual_greed_search(frames_greedy))
    
    print("\n【2. 热词雷达扫描报告】")
    radar = HotwordRadar(frames_top40)
    for word in hotwords:
        result = radar.scan(word)
        if result["found"]:
            print(f"✅ 召回成功: 【{word}】")
            print(f"   出现时间: {result['start']:.2f}s - {result['end']:.2f}s")
            print(f"   平均概率: {result['prob']:.4%}")
        else:
            print(f"❌ 未能召回: {word}")
    print("="*50)

if __name__ == "__main__":
    main()

    print("="*50)

if __name__ == "__main__":
    main()
