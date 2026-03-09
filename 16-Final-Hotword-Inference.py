import os
import time
import numpy as np
from sensevoice_onnx.inference import SenseVoiceInference, ASREngineConfig, load_audio

def main():
    # 1. 初始化引擎 (使用 ASREngineConfig)
    config = ASREngineConfig(
        model_dir="./model",
        onnx_provider="dml",
        hotwords='hot.txt',
        pad_to=30, 
        precision='int4', 
        top_k=5
    )
    engine = SenseVoiceInference(config)
    
    # 2. 准备音频
    audio_path = r"test-fun.mp3"
    audio_path = r"dugong.mp3"
    
    if not os.path.exists(audio_path):
        print(f"❌ 找不到测试音频: {audio_path}")
        return
        
    print(f"\n[SenseVoice] 正在处理音频: {os.path.basename(audio_path)}")
    print(f"[Hotwords] 当前热词列表: {config.hotwords}")
    
    audio = load_audio(audio_path)
    
    # 3. 运行推理 (连续三遍测试速度)
    print(f"\n[Performance] 开始测速...")
    
    for i in range(1, 4):
        # 此时引擎内部已持有 radar，不会重复构建索引
        res_obj = engine.recognize(audio, lid="auto")
        tm = res_obj.timings
        
        print(f" >>> 第 {i} 轮耗时: {tm.total*1000:7.2f}ms | 识别文本: {res_obj.text}")
        print(f"     [细节] Frontend: {tm.frontend*1000:4.1f}ms | Encoder: {tm.encoder*1000:4.1f}ms | Decoder: {tm.decoder*1000:4.1f}ms | Radar: {tm.radar*1000:4.1f}ms")

    # 4. 展示最后一次的详细结果
    print("\n" + "="*50)
    print(f"{'时间戳':>10} | {'字符':<10} | {'类型'}")
    print("-" * 50)
    for r in res_obj.results:
        char_type = "🔥 HOTWORD" if r.is_hotword else "  Greedy"
        print(f"  {r.start:5.2f}s | {r.text:<10} | {char_type}")
    print("-" * 50)
    print(f"【检测到的热词】: {res_obj.hotwords}")
    print("【最终识别文本】")
    print(res_obj.text)
    print("="*50)

if __name__ == "__main__":
    main()
