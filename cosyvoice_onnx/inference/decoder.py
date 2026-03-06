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
        """
        执行 CTC Head 推理
        返回: 
            topk_log_probs: (1, T, 100)
            topk_indices: (1, T, 100)
        """
        # 确保输入精度正确
        if enc_out.dtype != self.input_dtype:
            enc_out = enc_out.astype(self.input_dtype)
            
        print(f"[Decoder] DML 推理：输入形状 {enc_out.shape}")
        # 现在模型有两个输出
        topk_log_probs, topk_indices = self.session.run(None, {"enc_out": enc_out})
        return topk_log_probs, topk_indices

    def decode_greedy(self, enc_out, sp, blank_id=0, prompt_len=4, T_valid=None):
        """
        封装贪婪搜索 (使用模型返回的 Top-1)
        """
        _, topk_indices = self.forward(enc_out)
        
        # 取 Top-1 索引进行 CTC 解码
        # topk_indices 形状是 (1, T, 100) -> 取 (1, T, 0)
        log_probs_placeholder = np.zeros((1, topk_indices.shape[1], 1), dtype=np.float32)
        
        # 为了兼容原有的 greedy_search 逻辑，我们构造一个只包含 Top-1 的假 log_probs
        # 或者直接修改输出逻辑。这里我们对 topk_indices 进行切片处理
        if T_valid is not None:
            active_indices = topk_indices[0, prompt_len : T_valid + prompt_len, 0]
        else:
            active_indices = topk_indices[0, prompt_len:, 0]
            
        # 连续去重
        collapsed = []
        if len(active_indices) > 0:
            current_id = active_indices[0]
            start_frame = 0
            for i in range(1, len(active_indices)):
                if active_indices[i] != current_id:
                    collapsed.append((current_id, start_frame))
                    current_id = active_indices[i]
                    start_frame = i
            collapsed.append((current_id, start_frame))

        results = []
        for token_id, frame_idx in collapsed:
            if token_id == blank_id: continue
            token_id = int(token_id)
            char = sp.id_to_piece(token_id)
            if not char.strip(): continue
            results.append({
                "text": char,
                "start": round(frame_idx * 0.060, 3)
            })
        return results

    def get_topk_space(self, enc_out, top_k=20, T_valid=None):
        """
        准备 Numba 雷达所需的搜索空间 (直接利用模型 TopK 输出)
        返回: topk_indices, topk_probs, top1_indices, log_probs_dummy
        """
        topk_log_probs, topk_indices = self.forward(enc_out)
        
        # 1. 切片到有效长度 (跳过 Prompt)
        # 注意：模型已经帮我们在 GPU 上排好序了，我们只需要取前 top_k 个即可
        start = 4
        end = (T_valid + 4) if T_valid is not None else topk_indices.shape[1]
        
        # 裁剪到请求的 top_k 深度 (模型默认输出 100)
        final_indices = topk_indices[0, start:end, :top_k].astype(np.int32)
        # 将对数概率转回概率 [0, 1] 供雷达使用
        final_probs = np.exp(topk_log_probs[0, start:end, :top_k].astype(np.float32))
        
        top1_indices = final_indices[:, 0]
        
        # 为了保持接口兼容性，返回一个空的 log_probs 占位符
        return final_indices, final_probs, top1_indices, None
