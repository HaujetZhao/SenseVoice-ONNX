import os
import json
import numpy as np
import onnxruntime as ort
import sentencepiece as spm
from .audio import NumPyMelExtractor
from .ctc import greedy_search

class SenseVoiceInference:
    """
    SenseVoice ONNX 推理引擎 (纯净版)
    - 零 PyTorch 依赖
    - 动态 Prompt 构造
    - ONNX CPU/DML 支持
    """
    def __init__(self, model_dir, device="cpu"):
        self.model_dir = model_dir
        
        # 1. 资源路径
        inference_config_path = os.path.join(model_dir, "inference_config.json")
        prompt_embed_path = os.path.join(model_dir, "prompt_embed.npy")
        tokenizer_path = os.path.join(model_dir, "tokenizer.bpe.model")
        enc_onnx = os.path.join(model_dir, "sensevoice_encoder.onnx")
        ctc_onnx = os.path.join(model_dir, "sensevoice_ctc.onnx")

        # 2. 检查并加载资源
        if not os.path.exists(inference_config_path):
            raise FileNotFoundError(f"找不到配置: {inference_config_path}")
            
        with open(inference_config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.prompt_embed = np.load(prompt_embed_path)
        
        # 3. 初始化分词器
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)
        
        # 4. 初始化会话
        providers = ['CPUExecutionProvider']
        if device.lower() == "dml":
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        
        session_opts = ort.SessionOptions()
        session_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session_opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        print(f"[SenseVoice] 正在初始化 ONNX 会话 (EP: {providers[0]})...")
        self.enc_sess = ort.InferenceSession(enc_onnx, providers=providers, sess_options=session_opts)
        self.ctc_sess = ort.InferenceSession(ctc_onnx, providers=providers, sess_options=session_opts)
        
        # 5. 特征提取组件
        self.frontend = NumPyMelExtractor()

    def construct_prompt(self, lid="auto", itn=True):
        """构造 4 帧 Prompt Embedding"""
        lid_dict = self.config.get("lid_dict", {})
        itn_dict = self.config.get("textnorm_dict", {})
        
        lid_idx = lid_dict.get(lid, 3) 
        itn_str = "withitn" if itn else "woitn"
        itn_idx = itn_dict.get(itn_str, 14)
        
        # 拼接: Language(1) -> Event_Emo(2) -> Style(1)
        lid_vec = self.prompt_embed[lid_idx:lid_idx+1]
        event_emo_vec = self.prompt_embed[1:3]
        style_vec = self.prompt_embed[itn_idx:itn_idx+1]
        
        prompt = np.concatenate([lid_vec, event_emo_vec, style_vec], axis=0)
        return prompt[np.newaxis, ...].astype(np.float32)

    def __call__(self, audio_data: np.ndarray, lid="zh", itn=True):
        """
        执行推理流程
        Args:
            audio_data: (T,) NumPy 数组 (16k 采样率)
        """
        # 1. 提取 LFR 特征 (NumPy)
        lfr_feat = self.frontend.extract(audio_data)
        
        # 2. 准备 Prompt & Mask
        prompt_feat = self.construct_prompt(lid=lid, itn=itn)
        mask = np.ones((1, lfr_feat.shape[0])).astype(np.float32)
        
        # 3. Encoder 推理
        enc_out = self.enc_sess.run(None, {
            "speech_feat": lfr_feat[np.newaxis, ...], 
            "mask": mask, 
            "prompt_feat": prompt_feat
        })[0]
        
        # 4. CTC 推理 (Decoder Head)
        log_probs = self.ctc_sess.run(None, {"enc_out": enc_out})[0]
        
        # 5. 解码
        decoded_ids = greedy_search(log_probs, blank_id=0, prompt_len=4)
        text = self.sp.decode(decoded_ids)
        
        return text
