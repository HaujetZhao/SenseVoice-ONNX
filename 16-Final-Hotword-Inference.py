import os
import time
import numpy as np
from sensevoice_onnx.inference import SenseVoiceInference, ASREngineConfig, load_audio

def main():
    # 1. 准备热词
    hotword_file = "hot.txt"
    with open(hotword_file, "r", encoding="utf-8") as f:
        hotwords = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

    # 2. 初始化引擎 (显式指定路径实现解耦)
    model_dir = "./model"
    precision = "int4"
    config = ASREngineConfig(
        encoder_path=f"{model_dir}/SenseVoice-Encoder.{precision}.onnx",
        decoder_path=f"{model_dir}/SenseVoice-CTC.{precision}.onnx",
        tokenizer_path=f"{model_dir}/tokenizer.bpe.model",
        onnx_provider="cpu",
        dml_pad_to=30, 
        top_k=5
    )
    engine = SenseVoiceInference(config)
    engine.update_hotwords(hotwords)
    
    # 2. 准备音频
    audio_path = r"test-fun.mp3"
    # audio_path = r"dugong.mp3"
    # audio_path = r"test-try.mp3"
    
    if not os.path.exists(audio_path):
        print(f"❌ 找不到测试音频: {audio_path}")
        return
        
    print(f"\n[SenseVoice] 正在处理音频: {os.path.basename(audio_path)}")
    print(f"[Hotwords] 当前热词列表: {hotwords}")
    
    audio = load_audio(audio_path)
    
    # 3. 运行推理 (测试速度)
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
