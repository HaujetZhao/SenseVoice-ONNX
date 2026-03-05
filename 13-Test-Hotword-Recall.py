import os
import librosa
import numpy as np
from cosyvoice_onnx.inference import SenseVoiceInference, HotwordRadar

def main():
    # 1. 初始化引擎
    model_dir = "./model"
    engine = SenseVoiceInference(model_dir, device="cpu")
    
    # 2. 准备音频
    audio_path = r"d:\cosyvoice\test-fun.mp3"
    if not os.path.exists(audio_path):
        print(f"❌ 找不到测试音频: {audio_path}")
        return
        
    print(f"\n正在处理音频: {os.path.basename(audio_path)} ...")
    audio, _ = librosa.load(audio_path, sr=16000)
    
    # 3. 运行推理（获取 Top-K 数据）
    # 我们故意设置 top_k=40 以便进行深入搜索
    topk_frames, greedy_text = engine.recognize_topk(audio, top_k=40, lid="zh")
    
    print("\n" + "="*50)
    print("【1. 常规识别（Greedy）】")
    print(f"文本: {greedy_text}")
    
    # 4. 初始化热词雷达
    print("\n【2. 热词召回雷达扫描】")
    radar = HotwordRadar(topk_frames)
    
    # 你之前想召回的词
    hotwords = ["Fun ASR Nano", "睡前消息", "事"]
    
    for word in hotwords:
        result = radar.scan(word)
        if result["found"]:
            print(f"✅ 召回成功: 【{word}】")
            print(f"   起始时间: {result['start']:.2f}s - {result['end']:.2f}s")
            print(f"   置信概率: {result['prob']:.4%}")
        else:
            print(f"❌ 召回失败: 【{word}】")
            
    print("="*50)

    # 特殊测试：显示第一个字是“事”的所有候选
    # 查找第一个有效非空白帧
    first_meaningful_frame = -1
    for t, cand in enumerate(topk_frames):
        if cand[0][0] != "[BLANK]":
            first_meaningful_frame = t
            break
            
    if first_meaningful_frame != -1:
        print(f"\n[调试] 观察第 {first_meaningful_frame} 帧（第一个发音帧）的 Top-10 候选:")
        for k in range(10):
            char, prob = topk_frames[first_meaningful_frame][k]
            print(f"   Rank {k+1}: {char} (prob: {prob:.4%})")

if __name__ == "__main__":
    main()
