import os
import json
import numpy as np
import onnxruntime as ort
import sentencepiece as spm
import librosa
from pathlib import Path

# ============================================================================
# 1. 纯 NumPy 特征提取器 (移植自 FunASRMelExtractor)
# ============================================================================

class NumPyMelExtractor:
    def __init__(self, sr=16000, n_fft=400, n_mels=80, f_min=20, f_max=8000):
        self.sr, self.n_fft, self.n_mels = sr, n_fft, n_mels
        
        # 1. 生成梅尔矩阵
        hz_to_mel = lambda f: 2595.0 * np.log10(1.0 + (f / 700.0))
        mel_to_hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
        all_freqs = np.linspace(0, sr // 2, n_fft // 2 + 1)
        m_pts = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
        f_pts = mel_to_hz(m_pts)
        f_diff = np.diff(f_pts)
        slopes = f_pts[np.newaxis, :] - all_freqs[:, np.newaxis]
        fb = np.maximum(0, np.minimum((-1.0 * slopes[:, :-2]) / f_diff[:-1], slopes[:, 2:] / f_diff[1:]))
        self.filters = fb.astype(np.float32)
        
        self.hop_length = 160
        self.window = (0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(self.n_fft) / self.n_fft)).astype(np.float32)
        self.pre_emphasis = 0.97

    def extract(self, audio: np.ndarray) -> np.ndarray:
        # 归一化
        audio = audio - np.mean(audio)
        # 预加重
        audio_pe = np.empty_like(audio)
        audio_pe[0] = audio[0]
        audio_pe[1:] = audio[1:] - self.pre_emphasis * audio[:-1]
        
        # STFT
        half_n_fft = self.n_fft // 2
        y = np.pad(audio_pe, (half_n_fft, half_n_fft), mode='constant')
        num_frames = 1 + (len(y) - self.n_fft) // self.hop_length
        frames = np.lib.stride_tricks.as_strided(y, shape=(num_frames, self.n_fft), strides=(y.strides[0] * self.hop_length, y.strides[0]))
        
        win_frames = frames * self.window
        stft_res = np.fft.rfft(win_frames, n=self.n_fft, axis=1)
        magnitudes = np.abs(stft_res)**2 
        
        mel_spec = np.dot(magnitudes, self.filters) 
        log_mel = np.log(mel_spec + 1e-7)
        
        # LFR Stack (7帧叠加, 6帧跳跃)
        T_mel = log_mel.shape[0]
        T_lfr = (T_mel + 5) // 6
        left_pad = np.repeat(log_mel[:1, :], 3, axis=0)
        right_pad_len = (T_lfr * 6 + 7) - T_mel
        right_pad = np.repeat(log_mel[-1:, :], right_pad_len, axis=0)
        padded = np.concatenate([left_pad, log_mel, right_pad], axis=0)
        
        lfr_feat = np.empty((T_lfr, 560), dtype=np.float32)
        for i in range(7):
            lfr_feat[:, i*80 : (i+1)*80] = padded[i : i + T_lfr * 6 : 6, :]
        return lfr_feat

# ============================================================================
# 2. 纯 ONNX + NumPy 推理引擎
# ============================================================================

class PureSenseVoiceInference:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        
        # 1. 加载配置和资源
        with open(os.path.join(model_dir, "inference_config.json"), "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.prompt_embed = np.load(os.path.join(model_dir, "prompt_embed.npy"))
        
        # 2. 加载 BPE 分词器
        print("正在加载 SentencePiece 分词器...")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(os.path.join(model_dir, "tokenizer.bpe.model"))
        
        # 3. 初始化 ONNX Runtime
        print("正在加载 ONNX 模型 (CPU)...")
        self.enc_sess = ort.InferenceSession(os.path.join(model_dir, "sensevoice_encoder.onnx"), providers=['CPUExecutionProvider'])
        self.ctc_sess = ort.InferenceSession(os.path.join(model_dir, "sensevoice_ctc.onnx"), providers=['CPUExecutionProvider'])
        
        self.frontend = NumPyMelExtractor()

    def construct_prompt(self, lid="zh", itn=True):
        lid_idx = self.config["lid_dict"].get(lid, 3) 
        itn_idx = self.config["textnorm_dict"].get("withitn" if itn else "woitn", 14)
        lid_embed, event_emo_embed, style_embed = self.prompt_embed[lid_idx:lid_idx+1], self.prompt_embed[1:3], self.prompt_embed[itn_idx:itn_idx+1]
        res = np.concatenate([lid_embed, event_emo_embed, style_embed], axis=0)
        return res[np.newaxis, ...].astype(np.float32)

    def run(self, audio_path, lid="zh", itn=True):
        print(f"提取音频特征: {audio_path}")
        audio, _ = librosa.load(audio_path, sr=16000)
        lfr_feat = self.frontend.extract(audio)
        
        prompt_feat = self.construct_prompt(lid, itn)
        mask = np.ones((1, lfr_feat.shape[0])).astype(np.float32)
        
        # 推理
        enc_out = self.enc_sess.run(None, {"speech_feat": lfr_feat[np.newaxis, ...], "mask": mask, "prompt_feat": prompt_feat})[0]
        log_probs = self.ctc_sess.run(None, {"enc_out": enc_out})[0]
        
        # CTC 解码
        logits = log_probs[0, 4:, :] # 跳过 Prompt
        indices = np.argmax(logits, axis=-1)
        
        decoded_ids, last_id = [], -1
        for idx in indices:
            idx = int(idx)
            if idx != 0 and idx != last_id:
                if idx < 25000: decoded_ids.append(idx)
            last_id = idx
            
        return self.sp.decode(decoded_ids)

def main():
    model_dir = Path("./model")
    audio_path = r"d:\cosyvoice\test-fun.mp3"
    
    if not os.path.exists(audio_path):
        print(f"❌ 找不到音频: {audio_path}")
        return

    print("\n--- [终极验证] 纯 NumPy + ONNX 推理 (无 Torch, 无 funasr) ---")
    engine = PureSenseVoiceInference(model_dir)
    result = engine.run(audio_path, lid="zh", itn=True)
    
    print("\n" + "="*50)
    print("【纯净版识别结果】")
    print(f"识别文本: {result}")
    print("="*50)

if __name__ == "__main__":
    main()
