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
        device="dml", 
        hotwords='hot.txt', 
        precision='int8'
    )
    engine = SenseVoiceInference(config)
    
    # 2. 运行长音频转录 (直接使用 transcribe 方法)
    audio_path = r"d:\cosyvoice\睡前消息.m4a"
    if not os.path.exists(audio_path):
        print(f"❌ 找不到测试音频: {audio_path}")
        return
        
    print(f"\n[SenseVoice] 开始转录音频文件: {os.path.basename(audio_path)} (Chunk=40s, Overlap=2s)...")
    
    t0 = time.perf_counter()
    result = engine.transcribe(audio_path, chunk_size=40, overlap=2, duration=0)
    t_total = time.perf_counter() - t0
    
    # 3. 结果汇总
    audio_duration = result.timings.total # 总耗时由 recognize 级联返回
    print(f"\n[SenseVoice] 转录完成！")
    print(f" - 总处理耗时: {t_total:.2f}s")
    print(f" - 最终文本长度: {len(result.text)} 字符")
    
    # 4. 导出结果
    print(f"\n[SenseVoice] 正在导出结果...")
    export_to_srt("睡前消息.srt", result)
    export_to_txt("睡前消息.txt", result)
    
    print("\n[Preview] 前 200 个字符:")
    print("-" * 40)
    print(result.text[:200] + "...")
    print("-" * 40)

if __name__ == "__main__":
    main()
