import time
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import sentencepiece as spm
from .audio import NumPyMelExtractor
from .encoder import SenseVoiceEncoder
from .decoder import SenseVoiceDecoder
from .integrator import ResultIntegrator
from .ctc import greedy_search, topk_search
from .numba_radar import FastHotwordRadar
from .schema import ASREngineConfig, TranscriptionResult, Timings, RecognitionResult

class SenseVoiceInference:
    """
    SenseVoice ONNX 推理引擎 (纯净版)
    - 零 PyTorch 依赖
    - 动态 Prompt 构造
    - ONNX CPU/DML 支持
    """
    def __init__(self, model_dir_or_config, device="cpu", hotwords: list = None):
        if isinstance(model_dir_or_config, ASREngineConfig):
            self.config = model_dir_or_config
            self.model_dir = Path(self.config.model_dir)
            self.device = self.config.device
            init_hotwords = self.config.hotwords
        else:
            self.model_dir = Path(model_dir_or_config)
            self.device = device
            init_hotwords = hotwords

        # 1. 编码器、解码器与前端
        self.encoder = SenseVoiceEncoder(self.model_dir, device=self.device)
        self.decoder = SenseVoiceDecoder(self.model_dir, device=self.device)
        self.frontend = NumPyMelExtractor()
        
        # 2. 资源路径
        tokenizer_path = self.model_dir / "tokenizer.bpe.model"
        
        # 3. 初始化分词器
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(tokenizer_path))
        
        # 4. 初始化热词雷达 (默认为空或使用传入值)
        self.radar = None
        self.set_hotwords(init_hotwords or [])

    def set_hotwords(self, hotwords):
        """
        设置或更新热词列表
        支持格式:
        - List[str]: ['word1', 'word2']
        - str (path): 'd:/path/to/hotwords.txt'
        - str (text): 'word1\nword2'
        """
        final_list = []
        if isinstance(hotwords, list):
            final_list = hotwords
        elif isinstance(hotwords, (str, Path)):
            hotword_path = Path(hotwords)
            # 1. 检查是否为有效路径
            if hotword_path.exists() and hotword_path.is_file():
                with open(hotword_path, "r", encoding="utf-8") as f:
                    final_list = [line.strip() for line in f if line.strip()]
            else:
                # 2. 否则视为以换行符分隔的多行文本 (仅针对字符串)
                if isinstance(hotwords, str):
                    final_list = [line.strip() for line in hotwords.splitlines() if line.strip()]
        
        from .numba_radar import FastHotwordRadar
        self.radar = FastHotwordRadar(final_list, self.sp)

    def __call__(self, audio_data: np.ndarray, lid="zh", itn=True):
        """[默认识别接口] 使用热词雷达进行解码"""
        return self.recognize(audio_data, lid=lid, itn=itn)

    def recognize(self, audio_data: np.ndarray, lid="zh", itn=True, top_k: int = 20):
        """
        全功能识别接口，使用引擎当前持有的热词列表。
        """
        return self.recognize_with_hotwords(audio_data, lid=lid, itn=itn, top_k=top_k)

    def recognize_greedy(self, audio_data: np.ndarray, lid="zh", itn=True):
        """[调试用] 纯贪婪搜索解码"""
        lfr_feat = self.frontend.extract(audio_data)
        enc_out = self.encoder.forward(lfr_feat, lid=lid, itn=itn)
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


    def recognize_with_hotwords(self, audio_data: np.ndarray, lid="zh", itn=True, top_k: int = 20):
        """
        [核心] 带有热词替换功能的推理。使用引擎当前持有的 radar 实例。
        """
        t_start = time.perf_counter()
        
        # 1. 前端特提取
        t0 = time.perf_counter()
        lfr_feat = self.frontend.extract(audio_data)
        t_frontend = time.perf_counter() - t0
        
        # 2. 编码器推理
        t0 = time.perf_counter()
        enc_out = self.encoder.forward(lfr_feat, lid=lid, itn=itn)
        t_encoder = time.perf_counter() - t0
        
        # 3. 解码器获取搜索空间
        t0 = time.perf_counter()
        topk_indices, topk_probs, top1_indices, log_probs = self.decoder.get_topk_space(enc_out, top_k=top_k)
        t_decoder = time.perf_counter() - t0
        
        # 4. 运行并行扫描 (Numba)
        t0 = time.perf_counter()
        detected_hotwords = self.radar.scan(topk_indices, topk_probs, top1_indices)
        t_radar = time.perf_counter() - t0
        
        # 5. 准备基础 Greedy 序列
        greedy_results = greedy_search(log_probs, self.sp, prompt_len=4)
        
        # 6. 调用整合器进行碰撞检测与 Token 块合成
        t0 = time.perf_counter()
        integrated_list = ResultIntegrator.integrate(greedy_results, detected_hotwords)
        t_integrate = time.perf_counter() - t0
        
        t_total = time.perf_counter() - t_start
        
        # 7. 构造结构化结果
        recognition_results = [
            RecognitionResult(text=item["text"], start=item["start"], is_hotword=item["is_hotword"])
            for item in integrated_list
        ]
        
        timings = Timings(
            frontend=t_frontend,
            encoder=t_encoder,
            decoder=t_decoder,
            radar=t_radar,
            integrate=t_integrate,
            total=t_total
        )
        
        return TranscriptionResult(
            text="".join([r.text for r in recognition_results]),
            results=recognition_results,
            timings=timings
        )
