import os
import time
import numpy as np
from cosyvoice_onnx.inference import SenseVoiceInference, ASREngineConfig, load_audio
from cosyvoice_onnx.inference.exporters import export_to_srt, export_to_txt

def main():
    # 1. 初始化引擎
    # 长音频转录建议使用 CPU，或者 DML (不设置 pad_to 或设置较大的 pad_to)
    config = ASREngineConfig(
        model_dir="./model",
        device="dml", # 长音频文件转录，CPU 通常更稳定且显存压力小
        hotwords='hot.txt'
    )
    engine = SenseVoiceInference(config)
    
    # 2. 加载长音频 (前 300 秒)
    audio_path = r"d:\cosyvoice\睡前消息.m4a"
    if not os.path.exists(audio_path):
        print(f"❌ 找不到测试音频: {audio_path}")
        return
        
    print(f"\n[SenseVoice] 正在加载音频 (前 300 秒): {os.path.basename(audio_path)}")
    t0 = time.perf_counter()
    audio = load_audio(audio_path, duration=300)
    print(f"[SenseVoice] 音频加载完成，耗时: {time.perf_counter()-t0:.2f}s")
    
    # 3. 运行长音频识别
    # chunk_size=40s, overlap=2s (根据用户定义)
    print(f"\n[SenseVoice] 开始长音频分段转录 (Chunk=40s, Overlap=2s)...")
    t0 = time.perf_counter()
    result = engine.recognize(audio, chunk_size=40, overlap=2)
    t_total = time.perf_counter() - t0
    
    print(f"\n[SenseVoice] 转录完成！")
    print(f" - 总耗时: {t_total:.2f}s")
    print(f" - 音频时长: {len(audio)/16000:.2f}s")
    print(f" - 实时率 (RTF): {t_total / (len(audio)/16000):.4f}")
    
    # 4. 导出结果
    print(f"\n[SenseVoice] 正在导出结果...")
    export_to_srt("result.srt", result)
    export_to_txt("result.txt", result)
    
    print("\n[Preview] 前 100 个字符:")
    print("-" * 40)
    print(result.text[:100] + "...")
    print("-" * 40)

if __name__ == "__main__":
    main()
