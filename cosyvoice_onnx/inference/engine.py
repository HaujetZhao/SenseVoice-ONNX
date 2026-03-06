import os
import json
import numpy as np
import onnxruntime as ort
import sentencepiece as spm
from .audio import NumPyMelExtractor
from .encoder import SenseVoiceEncoder
from .decoder import SenseVoiceDecoder
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
        
        # 2. 运行 Numba 并行雷达
        radar = FastHotwordRadar(hotwords, self.sp)
        detected_hotwords = radar.scan(topk_indices, topk_probs, top1_indices)
        
        greedy_results = greedy_search(log_probs, self.sp, prompt_len=4)
        
        # 3. 按时间顺序进行替换
        # 排序检测到的热词（按出现时间）
        detected_hotwords.sort(key=lambda x: x["start"])
        
        final_results = []
        last_hotword_end = -1.0
        results_set = set() # 用于记录已插入的热词，避免重复
        
        # 对每一个 Greedy 结果进行检查，看它是否落在某个热词的时间范围内
        for g in greedy_results:
            # 检查当前 Greedy 字符是否落在任何已命中热词的领土内 [start, end]
            replaced = False
            for hw in detected_hotwords:
                if g["start"] >= hw["start"] - 0.02 and g["start"] <= hw["end"] + 0.02:
                    # 被热词封锁，如果还没插入则插入
                    if hw["text"] not in results_set:
                        # 核心升级：Token 块模式输出
                        origin_text = hw["text"]
                        search_base = origin_text.lower()
                        
                        # 1. 寻找每个 Token 覆盖的字符起始位置
                        anchors = [] # 存储 (idx_in_text, timestamp)
                        curr_search_pos = 0
                        for tk in hw["tokens"]:
                            clean_tk = tk["token"].replace("\u2581", "").strip().lower()
                            if not clean_tk: continue
                            idx = search_base.find(clean_tk, curr_search_pos)
                            if idx != -1:
                                anchors.append((idx, tk["time"]))
                                curr_search_pos = idx + len(clean_tk)
                        
                        # 确保至少有一个起始锚点
                        if not anchors: 
                            anchors.append((0, hw["start"]))
                        elif anchors[0][0] != 0:
                            # 如果第一个 Token 不是从 0 开始（比如开头是标点），强制补全 0
                            anchors.insert(0, (0, hw["start"]))
                            
                        # 2. 根据锚点进行分块切割
                        # 将字符串切成：[start_idx, next_start_idx)
                        for i in range(len(anchors)):
                            start_idx, start_time = anchors[i]
                            end_idx = anchors[i+1][0] if (i+1) < len(anchors) else len(origin_text)
                            
                            chunk_text = origin_text[start_idx:end_idx]
                            final_results.append({
                                "text": chunk_text,
                                "start": start_time,
                                "is_hotword": True
                            })
                            
                        results_set.add(hw["text"])
                        last_hotword_end = hw["end"]
                    replaced = True
                    break
            
            if not replaced and g["start"] > last_hotword_end:
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
