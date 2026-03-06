import os
import json
import numpy as np
import onnxruntime as ort
import sentencepiece as spm
from .audio import NumPyMelExtractor
from .encoder import SenseVoiceEncoder
from .decoder import SenseVoiceDecoder
from .integrator import ResultIntegrator
from .ctc import greedy_search, topk_search

class SenseVoiceInference:
    """
    SenseVoice ONNX 推理引擎 (纯净版)
    - 零 PyTorch 依赖
    - 动态 Prompt 构造
    - ONNX CPU/DML 支持
    """
    def __init__(self, model_dir, device="cpu"):
        self.model_dir = model_dir
        
        # 1. 编码器、解码器与前端
        self.encoder = SenseVoiceEncoder(model_dir, device=device)
        self.decoder = SenseVoiceDecoder(model_dir, device=device)
        self.frontend = NumPyMelExtractor()
        
        # 2. 资源路径
        tokenizer_path = os.path.join(model_dir, "tokenizer.bpe.model")
        
        # 3. 初始化分词器
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)

    def __call__(self, audio_data: np.ndarray, lid="zh", itn=True):
        # 1. 前端与编码
        lfr_feat = self.frontend.extract(audio_data)
        enc_out = self.encoder.forward(lfr_feat, lid=lid, itn=itn)
        
        # 2. 调用解码器生成贪婪序列
        greedy_res = self.decoder.decode_greedy(enc_out, self.sp, blank_id=0, prompt_len=4)
        return "".join([item['text'] for item in greedy_res])

    def recognize_topk(self, audio_data: np.ndarray, top_k=40, lid="zh", itn=True):
        # 1. 前端与编码
        lfr_feat = self.frontend.extract(audio_data)
        enc_out = self.encoder.forward(lfr_feat, lid=lid, itn=itn)
        
        # 2. 调用解码器
        log_probs = self.decoder.forward(enc_out)
        greedy_res = greedy_search(log_probs, self.sp, blank_id=0, prompt_len=4)
        greedy_text = "".join([item['text'] for item in greedy_res])
        topk_data = topk_search(log_probs, self.sp, top_k=top_k, blank_id=0, prompt_len=4)
        return topk_data, greedy_text


    def recognize_with_hotwords(self, audio_data: np.ndarray, hotwords: list, lid="zh", itn=True, top_k: int = 20):
        """
        [核心] 带有热词替换功能的推理
        返回: List[dict] -> [{'text': '...', 'start': ...}, ...]
        """
        # 1. 前端与编码
        lfr_feat = self.frontend.extract(audio_data)
        enc_out = self.encoder.forward(lfr_feat, lid=lid, itn=itn)
        
        # 2. 获取并行雷达所需的搜索空间
        topk_indices, topk_probs, top1_indices, log_probs = self.decoder.get_topk_space(enc_out, top_k=top_k)
        
        from .numba_radar import FastHotwordRadar
        
        # 2. 运行并行扫描 (Numba)
        from .numba_radar import FastHotwordRadar
        radar = FastHotwordRadar(hotwords, self.sp)
        detected_hotwords = radar.scan(topk_indices, topk_probs, top1_indices)
        
        # 3. 准备基础 Greedy 序列
        greedy_results = greedy_search(log_probs, self.sp, prompt_len=4)
        
        # 4. 调用整合器进行碰撞检测与 Token 块合成
        return ResultIntegrator.integrate(greedy_results, detected_hotwords)
