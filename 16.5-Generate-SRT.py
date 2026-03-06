import os
import librosa
import numpy as np
from cosyvoice_onnx.inference import SenseVoiceInference

def format_timestamp(seconds):
    """将秒数转换为 SRT 时间戳格式: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def result_to_srt(results, output_path):
    """将推理结果转换为 SRT 文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(len(results)):
            item = results[i]
            start_t = item["start"]
            
            # 计算结束时间：如果是最后一个 Token，持续 0.3s；否则持续到下一个 Token 开始
            if i + 1 < len(results):
                end_t = results[i+1]["start"]
                # 兼容处理：确保 end > start
                if end_t <= start_t:
                    end_t = start_t + 0.1
            else:
                end_t = start_t + 0.3
            
            # 写入 SRT 条目
            f.write(f"{i + 1}\n")
            f.write(f"{format_timestamp(start_t)} --> {format_timestamp(end_t)}\n")
            
            # 标记热词，方便在播放器中区分
            text = item["text"]
            if item.get("is_hotword"):
                text = f"🔥 {text}"
            
            f.write(f"{text}\n\n")

def main():
    # 1. 初始化引擎
    model_dir = "./model"
    engine = SenseVoiceInference(model_dir, device="cpu")
    
    # 2. 准备音频和热词
    audio_path = r"d:\cosyvoice\test-fun.mp3"
    hotwords = ["Fun-ASR-Nano", "事情"] 
    
    if not os.path.exists(audio_path):
        print(f"❌ 找不到测试音频: {audio_path}")
        return
        
    print(f"[SenseVoice] 正在处理音频: {os.path.basename(audio_path)}")
    audio, _ = librosa.load(audio_path, sr=16000)
    
    # 3. 运行推理
    results = engine.recognize_with_hotwords(audio, hotwords=hotwords, lid="zh")
    
    # 4. 生成 SRT
    srt_path = audio_path.replace(".mp3", ".srt").replace(".wav", ".srt")
    result_to_srt(results, srt_path)
    
    print("="*80)
    print(f"✅ SRT 字幕创建成功: {srt_path}")
    print("请使用播放器（如 VLC, PotPlayer）打开音频并开启该字幕进行校对。")
    print("="*80)

if __name__ == "__main__":
    main()
