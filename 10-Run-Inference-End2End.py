import os
import json
import torch
import librosa
import numpy as np
import onnxruntime as ort
from funasr.utils.load_utils import extract_fbank
from export_config import EXPORT_DIR

class SenseVoiceONNXRuntime:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        
        # 1. 加载配置和 Embedding
        with open(os.path.join(model_dir, "inference_config.json"), "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.prompt_embed = np.load(os.path.join(model_dir, "prompt_embed.npy"))
        
        # 2. 初始化 ONNX 会话
        enc_path = os.path.join(model_dir, "sensevoice_encoder.onnx")
        ctc_path = os.path.join(model_dir, "sensevoice_ctc.onnx")
        
        print(f"正在加载 Encoder: {enc_path}")
        self.enc_sess = ort.InferenceSession(enc_path, providers=['CPUExecutionProvider'])
        print(f"正在加载 CTC: {ctc_path}")
        self.ctc_sess = ort.InferenceSession(ctc_path, providers=['CPUExecutionProvider'])
        
        # 3. 加载 Tokenizer (这里临时借用 funasr 的 Tokenizer 处理，实际生产可替换为离线 tiktoken)
        # 为了演示端到端，我们这里直接从官方目录通过 funasr 加载，或者用户提供 tokens.json
        # 假设我们通过 model_dir 附近的官方目录加载
        from funasr.tokenizer.sentencepiece_tokenizer import SentencepiecesTokenizer
        # 注意：SenseVoice 实际使用的是 tiktoken 或其变体，这里根据 config.yaml 应该是 Sentencepieces
        # 但我们之前看到有 tokens.json。简单起见，我们先用官方 AutoModel 里的 tokenizer
        from funasr import AutoModel
        temp_model = AutoModel(model="iic/SenseVoiceSmall", trust_remote_code=True, device="cpu")
        self.tokenizer = temp_model.kwargs['tokenizer']
        # 初始化前端特征提取器
        self.frontend = temp_model.kwargs['frontend']

    def construct_prompt(self, lid="zh", itn=True):
        """
        构造 4 帧 Prompt Embedding
        逻辑参考 model.py 第 730-741 行:
        1. language_query (1帧)
        2. event_emo_query (2帧)
        3. style_query (1帧 - ITN)
        """
        lid_idx = self.config["lid_dict"].get(lid, 3) 
        itn_idx = self.config["textnorm_dict"].get("withitn" if itn else "woitn", 14)
        
        # 按照官方拼接顺序: language (1) -> event_emo (2) -> style (1)
        lid_embed = self.prompt_embed[lid_idx:lid_idx+1]
        event_emo_embed = self.prompt_embed[1:3]
        style_embed = self.prompt_embed[itn_idx:itn_idx+1]
        
        res = np.concatenate([lid_embed, event_emo_embed, style_embed], axis=0)
        return res[np.newaxis, ...].astype(np.float32) # (1, 4, 560)

    def run(self, audio_path, lid="zh", itn=True):
        # 1. 提取 Fbank
        print(f"提取音频特征: {audio_path}")
        speech, _ = librosa.load(audio_path, sr=16000)
        
        # 使用初始化好的 frontend
        speech_feat, _ = self.frontend(torch.from_numpy(speech).unsqueeze(0), torch.tensor([len(speech)]))
        speech_feat = speech_feat.numpy()
        
        # 2. 准备输入
        prompt_feat = self.construct_prompt(lid, itn).astype(np.float32)
        mask = np.ones((1, speech_feat.shape[1])).astype(np.float32)
        
        # 3. Encoder 推理
        print("运行 Encoder...")
        enc_out = self.enc_sess.run(None, {
            "speech_feat": speech_feat.astype(np.float32),
            "mask": mask,
            "prompt_feat": prompt_feat
        })[0]
        
        # 4. CTC 推理
        print("运行 CTC Decoder...")
        log_probs = self.ctc_sess.run(None, {"enc_out": enc_out})[0]
        
        # 5. Greedy 解码
        # 跳过前 4 帧 (Prompt 区域)
        log_probs = log_probs[:, 4:, :]
        indices = np.argmax(log_probs, axis=-1)[0]
        
        # 移除连续重复和 Blank (ID 0)
        decoded_ids = []
        last_id = -1
        for idx in indices:
            idx = int(idx) # 关键：转换为 Python int，numpy.int64 会导致 sentencepiece 崩溃
            if idx != 0 and idx != last_id:
                # 过滤掉时间戳等特殊 Token (如果是 25055 这种超大 ID 可能是特殊符号)
                if idx < 25000: # 快速过滤时间戳，具体的范围视 vocab 而定
                    decoded_ids.append(idx)
            last_id = idx
            
        print(f"解码 Token 序列: {decoded_ids}")
        text = self.tokenizer.decode(decoded_ids)
        return text

def main():
    # 使用之前导出的目录
    model_dir = EXPORT_DIR
    audio_path = r"d:\cosyvoice\test-fun.mp3"
    
    if not os.path.exists(audio_path):
        print(f"❌ 找不到测试音频: {audio_path}")
        return

    engine = SenseVoiceONNXRuntime(model_dir)
    result = engine.run(audio_path, lid="zh", itn=True)
    
    print("\n" + "="*50)
    print("【ONNX 端到端推理结果】")
    print(f"音频: {os.path.basename(audio_path)}")
    print(f"识别文本: {result}")
    print("="*50)

if __name__ == "__main__":
    main()
