from pathlib import Path
import numpy as np
import onnxruntime as ort

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
        if self.use_dml and isinstance(pad_to, int) and pad_to > 0:
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
        执行 CTC Head 推理 (单次推理)
        """
        if enc_out.dtype != self.input_dtype:
            enc_out = enc_out.astype(self.input_dtype)
            
        # 模型一次性返回 Top-100 的概率和索引
        topk_log_probs, topk_indices = self.session.run(None, {"enc_out": enc_out})
        return topk_log_probs, topk_indices

    def decode_all(self, enc_out, sp, top_k=20, prompt_len=4, T_valid=None, blank_id=0):
        """
        [核心接口] 单次推理获取所有解码信息
        返回: (greedy_results, radar_indices, radar_probs, top1_indices)
        """
        # 1. 唯一的一次推理调用
        topk_log_probs, topk_indices = self.forward(enc_out)
        
        # 确定有效范围 (跳过 Prompt 区域)
        start = prompt_len
        end = (T_valid + prompt_len) if T_valid is not None else topk_indices.shape[1]
        
        # --- A. 提取雷达所需 Top-K 空间 ---
        # 裁剪模型输出的 100 到用户请求的 top_k (通常为20)
        radar_indices = topk_indices[0, start:end, :top_k].astype(np.int32)
        radar_probs = np.exp(topk_log_probs[0, start:end, :top_k].astype(np.float32))
        top1_indices = radar_indices[:, 0]
        
        # --- B. 构造 Greedy 结果 (基于 Top-1) ---
        greedy_ids = top1_indices
        collapsed = []
        if len(greedy_ids) > 0:
            curr_id = greedy_ids[0]
            start_frame = 0
            for i in range(1, len(greedy_ids)):
                if greedy_ids[i] != curr_id:
                    collapsed.append((curr_id, start_frame))
                    curr_id = greedy_ids[i]
                    start_frame = i
            collapsed.append((curr_id, start_frame))

        greedy_results = []
        for tid, fidx in collapsed:
            if tid == blank_id: continue
            char = sp.id_to_piece(int(tid))
            if not char.strip(): ignore_char = True
            else:
                greedy_results.append({
                    "text": char,
                    "start": round(fidx * 0.060, 3)
                })

        return greedy_results, radar_indices, radar_probs, top1_indices
