import os
import librosa
import numpy as np
from cosyvoice_onnx.inference import SenseVoiceInference

def main():
    # 1. 初始化引擎
    # model_dir 指向包含 onnx, npy, model 文件的目录
    model_dir = "./model"
    engine = SenseVoiceInference(model_dir, device="cpu")
    
    # 2. 准备音频
    # 这里可以使用 librosa 加载，或者纯 numpy 载入原始二进制
    audio_path = r"d:\cosyvoice\test-fun.mp3"
    if not os.path.exists(audio_path):
        print(f"❌ 找不到测试音频: {audio_path}")
        return
        
    print(f"\n正在识别: {os.path.basename(audio_path)}...")
    audio, _ = librosa.load(audio_path, sr=16000)
    
    import time
    
    # 3. 循环测试性能
    for i in range(5): # 执行 5 轮：1 轮预热 + 4 轮测试
        t_start = time.perf_counter()
        result = engine(audio, lid="zh", itn=True)
        t_end = time.perf_counter()
        
        duration = t_end - t_start
        if i == 0:
            print(f"【预热轮】识别结果: {result} | 用时: {duration:.4f}s")
        else:
            print(f"【第 {i} 轮】识别结果: {result} | 用时: {duration:.4f}s")
    
    print("\n" + "="*50)
    print("【SenseVoice (ONNX) 模块化推理结果】")
    print(f"识别文本: {result}")
    print("="*50)

if __name__ == "__main__":
    main()
