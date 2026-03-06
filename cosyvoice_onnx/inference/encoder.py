from pathlib import Path
import json
import numpy as np
import onnxruntime as ort

class SenseVoiceEncoder:
    def __init__(self, encoder_path: str, inference_config_path: str, prompt_embed_path: str, device="cpu"):
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
        
        # 4. 其它设置
        self.use_dml = (device.lower() == "dml")

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
        return prompt[np.newaxis, ...].astype(np.float32)

    def forward(self, lfr_feat, lid="zh", itn=True):
        """
        执行 Encoder 推理
        返回: enc_out (1, T+4, 512)
        """
        # 1. 构造 Prompt
        prompt_feat = self.construct_prompt(lid=lid, itn=itn)
        
        T_valid = lfr_feat.shape[0]
        
        # 动态轴推理
        mask = np.ones((1, T_valid), dtype=np.float32)
        enc_out = self.session.run(None, {
            "speech_feat": lfr_feat[np.newaxis, ...],
            "mask": mask,
            "prompt_feat": prompt_feat
        })[0]
        return enc_out
