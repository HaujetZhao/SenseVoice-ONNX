from pathlib import Path
import json
import numpy as np
import onnxruntime as ort

class SenseVoiceEncoder:
    def __init__(self, encoder_path: str, inference_config_path: str, prompt_embed_path: str, device="cpu", pad_to: int = 30):
        # 1. 资源路径
        encoder_path = Path(encoder_path)
        inference_config_path = Path(inference_config_path)
        prompt_embed_path = Path(prompt_embed_path)

        # 2. 加载资源
        if not inference_config_path.exists():
            raise FileNotFoundError(f"找不到配置: {inference_config_path}")
            
        with open(inference_config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.prompt_embed = np.load(prompt_embed_path)
        
        # 3. 初始化会话
        providers = ['CPUExecutionProvider']
        if device.lower() == "dml":
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        
        session_opts = ort.SessionOptions()
        session_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session_opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        print(f"[Encoder] 正在初始化 ONNX 会话 (EP: {providers[0]})...")
        self.session = ort.InferenceSession(str(encoder_path), providers=providers, sess_options=session_opts)
        
        # 4. 精度适配 (检测模型是 FP32 还是 FP16)
        in_type = self.session.get_inputs()[0].type
        self.input_dtype = np.float16 if 'float16' in in_type else np.float32
        print(f"[Encoder] 自动检测输入精度: {self.input_dtype}")

        # 5. DML 策略设置 (仅在 DML 模式下生效)
        self.use_dml = (device.lower() == "dml")
        self.fixed_len = int(pad_to * 17) # 1s ≈ 17帧 LFR
        if self.use_dml and isinstance(pad_to, int) and pad_to > 0:
            self.warmup()

    def warmup(self):
        """执行一次全量形状推理，触发 DML 算子特化"""
        dummy_lfr = np.random.randn(1, self.fixed_len, 560).astype(self.input_dtype)
        dummy_mask = np.ones((1, self.fixed_len), dtype=self.input_dtype)
        dummy_prompt = np.zeros((1, 4, 560), dtype=self.input_dtype)
        print(f"[Encoder] DML 推理模式：正在使用形状为 {dummy_lfr.shape} 的 {self.fixed_len//17}s 随机数据进行预热...")
        self.session.run(None, {
            "speech_feat": dummy_lfr,
            "mask": dummy_mask,
            "prompt_feat": dummy_prompt
        })
        print("[Encoder] DML 预热完成。")

    def construct_prompt(self, lid="auto", itn=True):
        """构造 4 帧 Prompt Embedding"""
        lid_dict = self.config.get("lid_dict", {})
        itn_dict = self.config.get("textnorm_dict", {})
        
        lid_idx = lid_dict.get(lid, 3) 
        itn_str = "withitn" if itn else "woitn"
        itn_idx = itn_dict.get(itn_str, 14)
        
        # 核心逻辑镜像 engine.py: Language(1) -> Event_Emo(2) -> Style(1)
        lid_vec = self.prompt_embed[lid_idx:lid_idx+1]
        event_emo_vec = self.prompt_embed[1:3]
        style_vec = self.prompt_embed[itn_idx:itn_idx+1]
        
        prompt = np.concatenate([lid_vec, event_emo_vec, style_vec], axis=0)
        return prompt[np.newaxis, ...].astype(self.input_dtype)

    def forward(self, lfr_feat, lid="zh", itn=True):
        """
        执行 Encoder 推理
        返回: enc_out (1, T+4, 512)
        """
        # 1. 构造 Prompt
        prompt_feat = self.construct_prompt(lid=lid, itn=itn)
        
        T_valid = lfr_feat.shape[0]
        
        if self.use_dml and T_valid < self.fixed_len:
            # DML 填充策略：Uniform Padding + Replicate Padding
            T_target = self.fixed_len
            
            # 构造 Mask (1为有效, 0为填充)
            mask = np.zeros((1, T_target), dtype=self.input_dtype)
            mask[0, :T_valid] = 1.0
            
            # 构造填充后的特征 (使用最后一帧复读填充)
            full_feat = np.empty((1, T_target, 560), dtype=self.input_dtype)
            full_feat[0, :T_valid, :] = lfr_feat.astype(self.input_dtype)
            full_feat[0, T_valid:, :] = lfr_feat[-1, :].astype(self.input_dtype) # Replicate
            
            # 3. 推理
            print(f"[Encoder] DML 推理：输入形状 {full_feat.shape}, 有效步数 {T_valid}")
            enc_out = self.session.run(None, {
                "speech_feat": full_feat,
                "mask": mask,
                "prompt_feat": prompt_feat
            })[0]
            
            # 4. 直接返回全量结果 (包括填充部分)
            # 填充区的输出已被内部掩码清零，保留它们可使 Decoder 形状同样保持稳定
            return enc_out
        else:
            # 动态轴推理 (非 DML 模式或长度已超过固定值)
            print(f"[Encoder] 推理：输入形状 {(1, T_valid, 560)}")
            mask = np.ones((1, T_valid), dtype=self.input_dtype)
            enc_out = self.session.run(None, {
                "speech_feat": lfr_feat[np.newaxis, ...].astype(self.input_dtype),
                "mask": mask,
                "prompt_feat": prompt_feat
            })[0]
            return enc_out
