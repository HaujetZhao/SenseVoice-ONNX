import numpy as np

class NumPyMelExtractor:
    """纯 NumPy 实现的特征提取器 (对齐 torchaudio & funasr)"""
    def __init__(self, sr=16000, n_fft=400, n_mels=80, f_min=20, f_max=8000):
        self.sr, self.n_fft, self.n_mels = sr, n_fft, n_mels
        
        # 1. 静态计算梅尔矩阵
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
        # 汉明窗
        self.window = (0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(self.n_fft) / self.n_fft)).astype(np.float32)
        self.pre_emphasis = 0.97

    def extract(self, audio: np.ndarray) -> np.ndarray:
        # 均值归一化
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
        
        # 2. LFR Stack (7帧拼接, 6帧跳跃)
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
