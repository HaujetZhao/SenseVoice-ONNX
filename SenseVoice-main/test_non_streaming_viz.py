import os
import sys
import torch
import librosa
import numpy as np

# 使用原生 SenseVoice 目录
sys.path.append(r"d:\cosyvoice\SenseVoice-main")
from model import SenseVoiceSmall

def main():
    print(f"正在初始化非流式 SenseVoice 模型...")
    # 使用 AutoModel 方式加载，并获取内部模型
    from funasr import AutoModel
    model_dir = "iic/SenseVoiceSmall"
    
    # 构造复用之前的成功逻辑
    loaded_model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        remote_code="./model.py",
        device="cuda:0"
    )
    
    # 获取真正的 SenseVoiceSmall 实例和参数
    model = loaded_model.model
    kwargs = loaded_model.kwargs
    model.eval()

    audio_path = r"d:\cosyvoice\test-fun.mp3"
    print(f"加载音频: {audio_path}")
    speech, _ = librosa.load(audio_path, sr=16000, mono=True)
    
    # 转换为 Tensor，并增加 Batch 维度
    speech_tensor = torch.from_numpy(speech).float().unsqueeze(0).to("cuda:0")
    speech_lengths = torch.tensor([speech_tensor.shape[1]]).to("cuda:0")

    print("\n" + "="*50)
    print(" 非流式全局解码调试模式 ")
    print("="*50 + "\n")

    with torch.no_grad():
        # 模拟模型内部的编码流程
        # 1. 注入与推理代码一致的 Prompt (Language, Event, Emotion, Itn)
        # 这里我们模拟 model.inference 里的逻辑
        language = "zh"
        use_itn = True
        
        # 语言 Embed
        lid = model.lid_dict[language]
        lang_query = model.embed(torch.LongTensor([[lid]]).to("cuda:0"))
        
        # ITN Embed
        textnorm = "withitn" if use_itn else "woitn"
        tn_query = model.embed(torch.LongTensor([[model.textnorm_dict[textnorm]]]).to("cuda:0"))
        
        # Event/Emo Embed
        ee_query = model.embed(torch.LongTensor([[1, 2]]).to("cuda:0")) # 默认 1, 2
        
        # 获取 Fbank 特征 (使用 kwargs 里的前端配置)
        from funasr.utils.load_utils import extract_fbank
        # 注意：这里需要构造一个符合 frontend 要求的输入
        # 为了简单，我们直接调用 model.inference 的一部分逻辑
        # 但既然要看 Logits，我们手动拼接 Query
        
        # 重新提取 Fbank (这里直接用 funasr 的工具更准)
        # extract_fbank 期望的是一个包含 numpy 数组的列表
        audio_sample_list = [speech]
        speech_feat, feat_lengths = extract_fbank(
            audio_sample_list, data_type="sound", frontend=kwargs["frontend"]
        )
        speech_feat = speech_feat.to("cuda:0")
        feat_lengths = feat_lengths.to("cuda:0")

        # 拼接 Query
        full_speech = torch.cat((lang_query, ee_query, tn_query, speech_feat), dim=1)
        full_lengths = feat_lengths + 4

        # Encoder 前向传播
        encoder_out, encoder_out_lens = model.encoder(full_speech, full_lengths)
        # 获取 CTC Logits
        logits = model.ctc.log_softmax(encoder_out)[0] # (Time, Vocab)
        
        # 移除前 4 帧 Query 对应的输出
        logits = logits[4 : encoder_out_lens[0].item()]

    # 交互式显示
    for t in range(logits.shape[0]):
        probs = torch.softmax(logits[t], dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 20)
        
        print(f"\n>> 全局帧: {t} | 时间戳约: {t*60/1000:.2f}s")
        print(f"{'排名':<5} {'候选字':<10} {'概率':<10}")
        print("-" * 30)
        
        for k in range(20):
            token_id = topk_indices[k].item()
            prob_val = topk_probs[k].item()
            char = kwargs["tokenizer"].decode([token_id]) if token_id != 0 else "[BLANK]"
            if not char.strip() and token_id != 0: char = f"[ID:{token_id}]"
            print(f"#{k+1:<4} {char:<10} {prob_val:.4%}")
        
        res = input("\n[回车继续，q退出] > ")
        if res.lower() == 'q':
            break

if __name__ == "__main__":
    main()
