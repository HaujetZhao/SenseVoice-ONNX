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
        # greedy_search 现在返回 List[dict]
        greedy_res = greedy_search(log_probs, self.sp, blank_id=0, prompt_len=4)
        text = "".join([item['text'] for item in greedy_res])
        
        return text

    def recognize_topk(self, audio_data: np.ndarray, top_k=40, lid="zh", itn=True):
        """
        推理并获取每帧的 Top-K 候选词极其概率 (用于热词召回分析)
        返回: topk_data, greedy_text
        """
        # 1. 提取 LFR 特征
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
        
        # 4. CTC 推理
        log_probs = self.ctc_sess.run(None, {"enc_out": enc_out})[0]
        
        # 5. 多路径解析
        from .ctc import greedy_search, topk_search
        
        # 路径 A: 快速贪婪
        greedy_res = greedy_search(log_probs, self.sp, blank_id=0, prompt_len=4)
        greedy_text = "".join([item['text'] for item in greedy_res])
        
        # 路径 B: 保留每帧 Top-K (用于后续 Radar 处理)
        topk_data = topk_search(log_probs, self.sp, top_k=top_k, blank_id=0, prompt_len=4)
        
        return topk_data, greedy_text

    def recognize_with_hotwords(self, audio_data: np.ndarray, hotwords: list, lid="zh", itn=True):
        """
        [核心] 带有热词替换功能的推理
        返回: List[dict] -> [{'text': '...', 'start': ...}, ...]
        """
        # 1. 获取基础数据 (带有时间戳的 greedy 序列)
        lfr_feat = self.frontend.extract(audio_data)
        prompt_feat = self.construct_prompt(lid=lid, itn=itn)
        mask = np.ones((1, lfr_feat.shape[0])).astype(np.float32)
        
        enc_out = self.enc_sess.run(None, {"speech_feat": lfr_feat[np.newaxis, ...], "mask": mask, "prompt_feat": prompt_feat})[0]
        log_probs = self.ctc_sess.run(None, {"enc_out": enc_out})[0]
        
        from .ctc import greedy_search, topk_search
        from .radar import HotwordRadar
        
        # 获取 Top-1 索引用于幻觉校验
        top1_indices = np.argmax(log_probs[0, 4:, :], axis=-1)
        
        greedy_results = greedy_search(log_probs, self.sp, prompt_len=4)
        topk_frames = topk_search(log_probs, self.sp, top_k=40, prompt_len=4)
        
        # 2. 运行热词雷达扫描
        radar = HotwordRadar(topk_frames)
        detected_hotwords = []
        for word in hotwords:
            # 传入 self.sp (分词器) 和 top1_indices
            res = radar.scan(word, tokenizer=self.sp, top1_indices=top1_indices)
            if res["found"]:
                detected_hotwords.append({
                    "text": word,
                    "start": res["start"],
                    "end": res["end"],
                    "prob": res["prob"]
                })
        
        # 3. 按时间顺序进行替换
        # 排序检测到的热词（按出现时间）
        detected_hotwords.sort(key=lambda x: x["start"])
        
        final_results = []
        last_hotword_end = -1.0
        
        # 对每一个 Greedy 结果进行检查，看它是否落在某个热词的时间范围内
        for g in greedy_results:
            # 查找该 greedy 字是否落入任何已识别的热词区间
            replaced = False
            for h in detected_hotwords:
                if g["start"] >= h["start"] and g["start"] <= h["end"]:
                    # 如果该热词尚未被添加（它是该区间第一个字），则添加整个热词
                    if h["start"] > last_hotword_end:
                        # 计算热词内每个字的估算时间（线性分布）
                        h_chars = list(h["text"])
                        n_chars = len(h_chars)
                        duration = h["end"] - h["start"]
                        step = (duration / n_chars) if n_chars > 1 else 0
                        
                        for i, char in enumerate(h_chars):
                            final_results.append({
                                "text": char,
                                "start": round(h["start"] + i * step, 3),
                                "is_hotword": True
                            })
                        last_hotword_end = h["end"]
                    replaced = True
                    break
            
            if not replaced and g["start"] > last_hotword_end:
                # [新增：抑制逻辑] 检查这个字是否是刚才热词的冗余重复
                # 如果当前字出现在最近一个热词结束后的 1.0 秒内
                if last_hotword_end > 0 and (g["start"] - last_hotword_end < 1.0):
                    # 简单检查：如果这个字包含在刚才识别出的热词里，跳过它
                    # 这是一个粗略的重复压制
                    last_hw_text = [h["text"] for h in detected_hotwords if h["end"] == last_hotword_end]
                    if last_hw_text and g["text"] in last_hw_text[0]:
                        continue
                        
                final_results.append({
                    "text": g["text"],
                    "start": g["start"],
                    "is_hotword": False
                })
                
        return final_results
