from pathlib import Path
import numpy as np
import onnxruntime as ort
from .ctc import greedy_search, topk_search

class SenseVoiceDecoder:
    def __init__(self, decoder_path: str, device="cpu", pad_to: int = 30):
        # 1. 资源路径
        decoder_path = Path(decoder_path)
        
        # 2. 初始化会话
        providers = ['CPUExecutionProvider']
        if device.lower() == "dml":
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        
        session_opts = ort.SessionOptions()
        session_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session_opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        print(f"[Decoder] 正在初始化 ONNX 会话 (EP: {providers[0]})...")
        self.session = ort.InferenceSession(str(decoder_path), providers=providers, sess_options=session_opts)

        # 3. 精度适配
        in_type = self.session.get_inputs()[0].type
        self.input_dtype = np.float16 if 'float16' in in_type else np.float32
        print(f"[Decoder] 自动检测输入精度: {self.input_dtype}")

        # 4. DML 预热
        self.use_dml = (device.lower() == "dml")
        self.fixed_len = int(pad_to * 17) + 4 # 1s ≈ 17帧 + 4帧 Prompt
        if self.use_dml:
            self.warmup()

    def warmup(self):
        """执行一次全量形状推理，触发 CTC Head 算子特化"""
        # CTC Decoder 的输入形状通常是 (1, T_plus_4, 512)
        dummy_enc = np.zeros((1, self.fixed_len, 512), dtype=self.input_dtype)
        print(f"[Decoder] DML 推理模式：正在使用形状为 {dummy_enc.shape} 的数据进行预热...")
        self.session.run(None, {"enc_out": dummy_enc})
        print("[Decoder] DML 预热完成。")

    def forward(self, enc_out):
        """执行 CTC Head 推理，获取 log_probs"""
        # 确保输入精度正确
        if enc_out.dtype != self.input_dtype:
            enc_out = enc_out.astype(self.input_dtype)
            
        print(f"[Decoder] 推理：输入形状 {enc_out.shape}")
        log_probs = self.session.run(None, {"enc_out": enc_out})[0]
        return log_probs

    def decode_greedy(self, enc_out, sp, blank_id=0, prompt_len=4, T_valid=None):
        """封装贪婪搜索"""
        log_probs = self.forward(enc_out)
        
        # 如果提供了有效长度，则切片以加速后续 CPU 处理
        if T_valid is not None:
            log_probs = log_probs[:, :T_valid + prompt_len, :]
            
        return greedy_search(log_probs, sp, blank_id=blank_id, prompt_len=prompt_len)

    def get_topk_space(self, enc_out, top_k=20, T_valid=None):
        """
        准备 Numba 雷达所需的搜索空间
        Args:
            enc_out: Encoder 输出 (B, T, D)
            top_k: Top-K 深度
            T_valid: 有效帧数 (LFR 步数)
        返回: topk_indices, topk_probs, top1_indices, log_probs
        """
        log_probs = self.forward(enc_out)
        
        # 核心优化：在 CPU 处理 (exp, argsort) 前先切回有效长度
        # 同时保留 log_probs 变量供后续 greedy_search 使用（如果需要）
        if T_valid is not None:
            valid_log_probs = log_probs[:, 4:T_valid + 4, :]
        else:
            valid_log_probs = log_probs[:, 4:, :]
            
        # 计算概率
        probs = np.exp(valid_log_probs[0, :, :])
        
        # 获取 Top-K 索引
        topk_indices = np.argsort(-probs, axis=-1)[:, :top_k].astype(np.int32)
        
        # 获取 Top-K 概率
        topk_probs = np.zeros((probs.shape[0], top_k), dtype=np.float32)
        for t in range(probs.shape[0]):
            topk_probs[t] = probs[t, topk_indices[t]]
            
        top1_indices = topk_indices[:, 0]
        
        return topk_indices, topk_probs, top1_indices, log_probs
