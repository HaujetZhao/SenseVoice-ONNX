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
from .radar import HotwordRadar
from .schema import ASREngineConfig, TranscriptionResult, Timings, RecognitionResult

class SenseVoiceInference:
    """
    SenseVoice ONNX 推理引擎 (纯净版)
    - 零 PyTorch 依赖
    - 动态 Prompt 构造
    - ONNX CPU/DML 支持
    """
    def __init__(self, config: ASREngineConfig):
        """
        初始化 SenseVoice 推理引擎
        只接受 ASREngineConfig 实例，所有参数均封装在其中。
        """
        self.config = config
        self.device = config.device
        model_dir = Path(config.model_dir)
            
        # 1. 内部路径解析 (统一映射逻辑)
        encoder_path = model_dir / "SenseVoice-Encoder.fp16.onnx"
        decoder_path = model_dir / "SenseVoice-CTC.fp16.onnx"
        tokenizer_path = model_dir / "tokenizer.bpe.model"
        inference_config_path = model_dir / "inference_config.json"
        prompt_embed_path = model_dir / "prompt_embed.npy"

        # 2. 构造编码器、解码器与前端
        self.encoder = SenseVoiceEncoder(
            encoder_path=encoder_path, 
            inference_config_path=inference_config_path, 
            prompt_embed_path=prompt_embed_path, 
            device=self.device,
            pad_to=self.config.pad_to
        )
        self.decoder = SenseVoiceDecoder(
            decoder_path=decoder_path, 
            device=self.device,
            pad_to=self.config.pad_to
        )
        self.frontend = NumPyMelExtractor()
        
        # 3. 初始化分词器
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(tokenizer_path))
        
        # 4. 初始化热词雷达 (从配置中获取)
        self.radar = None
        if self.config.hotwords:
            self.set_hotwords(self.config.hotwords)
            
        # 5. 结果整合器
        self.integrator = ResultIntegrator()

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
        
        from .radar import HotwordRadar
        self.radar = HotwordRadar(final_list, self.sp)

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
        
        # 3. 解码器处理 (单次推理产出所有数据)
        t0 = time.perf_counter()
        T_valid = lfr_feat.shape[0]
        greedy_results, topk_indices, topk_probs, top1_indices = self.decoder.decode_all(
            enc_out, self.sp, top_k=top_k, T_valid=T_valid
        )
        t_decoder = time.perf_counter() - t0
        
        # 4. 运行并行扫描 (Numba)
        t0 = time.perf_counter()
        detected_hotwords = self.radar.scan(topk_indices, topk_probs, top1_indices)
        t_radar = time.perf_counter() - t0
        
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
