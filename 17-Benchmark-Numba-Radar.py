import time
import numpy as np
import librosa
from cosyvoice_onnx.inference import SenseVoiceInference, ASREngineConfig
from cosyvoice_onnx.inference.radar import HotwordRadar

def main():
    # 1. 初始化引擎
    config = ASREngineConfig(model_dir="./model", device="cpu")
    engine = SenseVoiceInference(config)
    audio, _ = librosa.load(r"test-fun.mp3", sr=16000)
    
    # 2. 准备搜索空间 (Top-K 概率和索引)
    lfr_feat = engine.frontend.extract(audio)
    enc_out = engine.encoder.forward(lfr_feat, lid="zh")
    
    # 直接调用重构后的解码器方法获取搜索空间
    # decode_all 返回: (greedy_results, radar_indices, radar_probs, top1_indices)
    _, topk_indices, topk_probs, top1_indices = engine.decoder.decode_all(
        enc_out, engine.sp, top_k=20, T_valid=len(lfr_feat)
    )
    
    # 2. 构造 5000 个热词的巨大列表
    # 包含目标词
    test_hotwords = ["Fun-ASR-Nano", "事情", "测试一下", "那我来", "代码加速"]
    # 注入 5000 个随机字符组合来模拟巨大词表
    for i in range(5000):
        test_hotwords.append(f"随机词汇{i}")
        
    print(f"🚀 开始压力测试，热词数量: {len(test_hotwords)}")
    
    # 3. 初始化 Radar (预编码热词，此时使用高效字典索引)
    t0 = time.time()
    radar = HotwordRadar(test_hotwords, engine.sp)
    print(f"   [初始化/预编码耗时]: {(time.time()-t0)*1000:.2f}ms")
    
    # 4. 执行多核并行扫描 (预热)
    radar.scan(topk_indices, topk_probs, top1_indices)
    
    # 5. 正式测速 (运行 10 次)
    print(f"\n🔥 开始性能实测 (10 次迭代):")
    durations = []
    for i in range(1, 11):
        t1 = time.perf_counter()
        hits = radar.scan(topk_indices, topk_probs, top1_indices)
        t2 = time.perf_counter()
        
        d = (t2 - t1) * 1000
        durations.append(d)
        print(f"  >>> 第 {i:2d} 次耗时: {d:6.3f}ms")
    
    avg_d = sum(durations) / len(durations)
    print(f"\n� 平均扫描耗时: {avg_d:6.3f}ms | 命中数量: {len(hits)}")
    
    # 打印部分结果校验
    print("\n命中示例 (详细时间轴):")
    for h in hits[:5]:
        print(f" - {h['text']} ({h['start']:.2f}s ~ {h['end']:.2f}s, prob: {h['prob']:.4f})")
        for t in h['tokens']:
            print(f"    └─ {t['token']:<6} : {t['time']:.2f}s")

if __name__ == "__main__":
    main()
