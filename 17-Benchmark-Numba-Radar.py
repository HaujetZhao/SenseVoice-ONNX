import time
import numpy as np
import librosa
from cosyvoice_onnx.inference import SenseVoiceInference
from cosyvoice_onnx.inference.numba_radar import FastHotwordRadar

def main():
    # 1. 初始化
    engine = SenseVoiceInference("./model", device="cpu")
    audio, _ = librosa.load(r"d:\cosyvoice\test-fun.mp3", sr=16000)
    
    # 获取推理后的 Top-K 数据
    # 我们扩展一下 ctc.py，让 topk_search 能直接返回原始 NumPy 数组
    from cosyvoice_onnx.inference.ctc import topk_search
    lfr_feat = engine.frontend.extract(audio)
    prompt_feat = engine.construct_prompt(lid="zh")
    mask = np.ones((1, lfr_feat.shape[0])).astype(np.float32)
    enc_out = engine.enc_sess.run(None, {"speech_feat": lfr_feat[np.newaxis, ...], "mask": mask, "prompt_feat": prompt_feat})[0]
    log_probs = engine.ctc_sess.run(None, {"enc_out": enc_out})[0]
    
    # 准备 Numba 需要的输入
    T_total = log_probs.shape[1] - 4
    K = 20
    probs = np.exp(log_probs[0, 4:, :])
    topk_indices = np.argsort(-probs, axis=-1)[:, :K].astype(np.int32)
    topk_probs = np.zeros((T_total, K), dtype=np.float32)
    for t in range(T_total):
        topk_probs[t] = probs[t, topk_indices[t]]
    
    top1_indices = topk_indices[:, 0]
    
    # 2. 构造 5000 个热词的巨大列表
    # 包含目标词
    test_hotwords = ["Fun ASR Nano", "事情", "测试一下", "那我来", "代码加速"]
    # 注入 5000 个随机字符组合来模拟巨大词表
    for i in range(5000):
        test_hotwords.append(f"随机词汇{i}")
        
    print(f"🚀 开始压力测试，热词数量: {len(test_hotwords)}")
    
    # 3. 初始化 FastRadar (预编码热词)
    t0 = time.time()
    radar = FastHotwordRadar(test_hotwords, engine.sp)
    print(f"   [初始化/预编码耗时]: {(time.time()-t0)*1000:.2f}ms")
    
    # 4. 执行多核并行扫描 (预热)
    radar.scan(topk_indices, topk_probs, top1_indices)
    
    # 5. 正式测速
    t1 = time.time()
    hits = radar.scan(topk_indices, topk_probs, top1_indices)
    t2 = time.time()
    
    print(f"🎯 扫描成功，命中数量: {len(hits)}")
    print(f"🔥 [万词并行扫描耗时]: {(t2-t1)*1000:.2f}ms")
    
    # 打印部分结果校验
    print("\n命中示例:")
    for h in hits[:5]:
        print(f" - {h['text']} ({h['start']:.2f}s ~ {h['end']:.2f}s, prob: {h['prob']:.4f})")

if __name__ == "__main__":
    main()
