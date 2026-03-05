import sys
import torch
import librosa
import numpy as np
from asr_decoder import CTCDecoder

# 引用 SenseVoice 目录
sys.path.append(r"d:\cosyvoice\SenseVoice-main")
from model import SenseVoiceSmall
from funasr import AutoModel

def main():
    # 1. 定义热词
    hotwords = ["睡前消息", "督工"]
    model_dir = "iic/SenseVoiceSmall"
    audio_path = r"d:\cosyvoice\test.mp3"  # 使用你的原始测试音频
    
    print(f"正在加载模型并初始化热词解码器...")
    loaded_model = AutoModel(model=model_dir, trust_remote_code=True, remote_code="./model.py", device="cuda:0")
    model = loaded_model.model
    kwargs = loaded_model.kwargs
    tokenizer = kwargs["tokenizer"]
    model.eval()

    # 2. 准备符号表 (Symbol Table) 与 BPE 模型
    # CTCDecoder 需要词表来映射 ID 到字
    symbol_table = {}
    for i in range(tokenizer.get_vocab_size()):
        symbol_table[tokenizer.decode(i)] = i
    
    bpe_model = kwargs["tokenizer_conf"]["bpemodel"]
    
    # 初始化解码器，传入热词
    decoder = CTCDecoder(hotwords, symbol_table, bpe_model)

    # 3. 提取特征并推理获取 Logits
    print(f"处理音频: {audio_path}")
    speech, _ = librosa.load(audio_path, sr=16000, mono=True)
    audio_sample_list = [speech]
    
    from funasr.utils.load_utils import extract_fbank
    speech_feat, feat_lengths = extract_fbank(audio_sample_list, data_type="sound", frontend=kwargs["frontend"])
    speech_feat = speech_feat.to("cuda:0")
    feat_lengths = feat_lengths.to("cuda:0")

    # 模拟 Prompt 拼接
    lid = model.lid_dict["zh"]
    lang_query = model.embed(torch.LongTensor([[lid]]).to("cuda:0"))
    tn_query = model.embed(torch.LongTensor([[model.textnorm_dict["withitn"]]]).to("cuda:0"))
    ee_query = model.embed(torch.LongTensor([[1, 2]]).to("cuda:0"))
    
    full_speech = torch.cat((lang_query, ee_query, tn_query, speech_feat), dim=1)
    full_lengths = feat_lengths + 4

    with torch.no_grad():
        encoder_out, encoder_out_lens = model.encoder(full_speech, full_lengths)
        # 获取 Softmax 概率矩阵，这是 Beam Search 的输入
        probs = torch.softmax(model.ctc.log_softmax(encoder_out), dim=-1)[0]
        # 去掉前 4 帧 Prompt
        probs = probs[4 : encoder_out_lens[0].item()]

    # 4. 执行 Beam Search 热词召回
    print(f"开始执行 Beam Search (含有 {len(hotwords)} 个热词)...")
    # beam_size 越大召回能力越强，但也越慢
    res = decoder.ctc_prefix_beam_search(probs.cpu(), beam_size=10, is_last=True)
    
    # 5. 解码结果
    # res["tokens"][0] 是最优路径的 ID 序列
    best_tokens = res["tokens"][0]
    final_text = tokenizer.decode(best_tokens)
    
    print("\n" + "="*30)
    print("【热词召回结果】")
    print(final_text)
    print("="*30)

if __name__ == "__main__":
    main()
