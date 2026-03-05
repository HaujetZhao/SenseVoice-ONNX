
import os
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# 配置路径
# 使用官方模型名称，funasr 会自动在 ~/.cache/modelscope 中寻找
model_dir = "iic/SenseVoiceSmall"
audio_path = r"d:\cosyvoice\test-fun.mp3"

def main():
    if not os.path.exists(audio_path):
        print(f"错误: 找不到音频文件 {audio_path}")
        return

    print(f"正在加载 SenseVoice Small 模型...")
    # 显式指定使用本地的 model.py 以确保逻辑一致
    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        remote_code="./model.py",  
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device="cpu",
    )

    print(f"正在处理音频: {audio_path}")
    res = model.generate(
        input=audio_path,
        cache={},
        language="auto",  # 自动语种识别
        use_itn=True,     # 包含标点与逆文本正则化
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
    )

    if res and len(res) > 0:
        raw_text = res[0]["text"]
        # 使用富文本后处理，清理掉情感和事件标签
        clean_text = rich_transcription_postprocess(raw_text)
        
        print("\n" + "="*30)
        print("识别结果 (带标签):")
        print(raw_text)
        print("-" * 30)
        print("清理后的文本:")
        print(clean_text)
        print("="*30)
    else:
        print("转录失败，未返回结果。")

if __name__ == "__main__":
    main()
