from pathlib import Path
import numpy as np
import onnxruntime as ort
from .ctc import greedy_search, topk_search

class SenseVoiceDecoder:
    def __init__(self, decoder_path: str, device="cpu"):
        # 1. 资源路径
        decoder_path = Path(decoder_path)
        
        # 2. 初始化会话
        providers = ['CPUExecutionProvider']
        if device.lower() == "dml":
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        
        session_opts = ort.SessionOptions()
        session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        print(f"[Decoder] 正在初始化 ONNX 会话 (EP: {providers[0]})...")
        self.session = ort.InferenceSession(str(decoder_path), providers=providers, sess_options=session_opts)

    def forward(self, enc_out):
        """执行 CTC Head 推理，获取 log_probs"""
        log_probs = self.session.run(None, {"enc_out": enc_out})[0]
        return log_probs

    def decode_greedy(self, enc_out, sp, blank_id=0, prompt_len=4):
        """封装贪婪搜索"""
        log_probs = self.forward(enc_out)
        return greedy_search(log_probs, sp, blank_id=blank_id, prompt_len=prompt_len)

    def get_topk_space(self, enc_out, top_k=20):
        """
        准备 Numba 雷达所需的搜索空间
        返回: topk_indices, topk_probs, top1_indices, log_probs
        """
        log_probs = self.forward(enc_out)
        
        # 计算概率 (跳过前 4 帧 Prompt)
        probs = np.exp(log_probs[0, 4:, :])
        
        # 获取 Top-K 索引
        topk_indices = np.argsort(-probs, axis=-1)[:, :top_k].astype(np.int32)
        
        # 获取 Top-K 概率
        topk_probs = np.zeros((probs.shape[0], top_k), dtype=np.float32)
        for t in range(probs.shape[0]):
            topk_probs[t] = probs[t, topk_indices[t]]
            
        top1_indices = topk_indices[:, 0]
        
        return topk_indices, topk_probs, top1_indices, log_probs
