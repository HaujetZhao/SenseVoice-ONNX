import os
import librosa
import numpy as np
from cosyvoice_onnx.inference import SenseVoiceInference

def main():
    # 1. 初始化引擎
    model_dir = "./model"
    engine = SenseVoiceInference(model_dir, device="cpu")
    
    # 2. 准备音频和热词
    audio_path = r"d:\cosyvoice\test-fun.mp3"
    hotwords = ["Fun ASR Nano", "事情", "但是"] # 这里的热词支持中文和英文
    
    if not os.path.exists(audio_path):
        print(f"❌ 找不到测试音频: {audio_path}")
        return
        
    audio, _ = librosa.load(audio_path, sr=16000)
    
    # 3. 执行“带热词召回”的推理
    # 我们不仅会得到文本，还会得到带有时间戳的列表
    print(f"正在识别并执行热词召回: {hotwords} ...")
    final_results = engine.recognize_with_hotwords(audio, hotwords=hotwords, lid="zh")
    
    # 4. 展示结果
    print("\n" + "="*70)
    print("【SenseVoice ONNX 自动化召回与替换报告】")
    print("-" * 70)
    
    full_text = []
    print(f"{'时间戳':<10} | {'字符':<5} | {'是否热词':<10}")
    print("-" * 35)
    
    for item in final_results:
        time_str = f"{item['start']:>6.2f}s"
        status = "✅ YES" if item['is_hotword'] else "  NO"
        char = item['text']
        
        print(f"{time_str:<10} | {char:<5} | {status}")
        full_text.append(char)
        
    print("-" * 70)
    print(f"最终完整文本: {''.join(full_text)}")
    print("="*70)

if __name__ == "__main__":
    main()
