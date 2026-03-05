import os
import librosa
import numpy as np
from cosyvoice_onnx.inference import SenseVoiceInference

def main():
    # 1. 初始化
    model_dir = "./model"
    engine = SenseVoiceInference(model_dir, device="cpu")
    
    # 2. 准备音频和热词
    # 特别加入测试热词，包括一个容易干扰的“虚假”热词看看效果
    audio_path = r"d:\cosyvoice\test-fun.mp3"
    hotwords = ["Fun-ASR-Nano", "事情"] 
    
    if not os.path.exists(audio_path):
        print(f"❌ 找不到测试音频: {audio_path}")
        return
        
    print(f"\n[SenseVoice] 正在处理音频: {os.path.basename(audio_path)}")
    print(f"[Hotwords] 注入热词列表: {hotwords}")
    
    audio, _ = librosa.load(audio_path, sr=16000)
    
    # 3. 运行增强推理
    # 内部会自动运行 Top-K 搜索 -> Radar 路径追踪 -> 严格间隙过滤 -> 文本替换
    results = engine.recognize_with_hotwords(audio, hotwords=hotwords, lid="zh")
    
    # 4. 展示结果
    print("\n" + "="*80)
    print(f"{'时间戳':<10} | {'字符':<10} | {'类型':<10}")
    print("-" * 80)
    
    full_text = ""
    for item in results:
        timestamp = f"{item['start']:6.2f}s"
        text = item['text']
        label = "🔥 HOTWORD" if item['is_hotword'] else "  Greedy"
        
        print(f"{timestamp} | {text:<10} | {label}")
        full_text += text

    print("-" * 80)
    print(f"【最终识别文本】\n{full_text}")
    print("="*80)

if __name__ == "__main__":
    main()
