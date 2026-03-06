import os
import librosa
import numpy as np
from cosyvoice_onnx.inference import SenseVoiceInference
from cosyvoice_onnx.inference.numba_radar import FastHotwordRadar

def main():
    # 1. 初始化引擎
    model_dir = "./model"
    engine = SenseVoiceInference(model_dir, device="cpu")
    
    # 2. 准备音频和热词
    audio_path = r"d:\cosyvoice\test-fun.mp3"
    hotwords = ["Fun ASR Nano", "事情", "但是"]
    
    if not os.path.exists(audio_path):
        print(f"❌ 找不到测试音频: {audio_path}")
        return
        
    audio, _ = librosa.load(audio_path, sr=16000)
    
    # 3. 运行推理并获取调试数据
    # 我们不仅获取 Top-K，还要获取整体的 Greedy 索引用于 Radar 校验
    # 注意：engine.recognize_topk 内部跳过了前 4 帧
    topk_frames, _ = engine.recognize_topk(audio, top_k=20, lid="zh")
    
    # 重新推理一遍获取 top1 索引（为了对齐，我们需要 4 帧偏移后的结果）
    # 这里直接从 topk_frames 提取第一个索引即可，因为 ctc.py.topk_search 已经处理好了
    # 但由于 topk_search 已经转成了 char，我们需要一个能跟 radar 对接的版本
    # 实际上，engine 内部在调用 recognize_with_hotwords 时已经做了这些
    
    # 推理一次获取原始 log_probs
    lfr_feat = engine.frontend.extract(audio)
    enc_out = engine.encoder.forward(lfr_feat, lid="zh")
    log_probs = engine.ctc_sess.run(None, {"enc_out": enc_out})[0]
    
    # 准备 Numba 格式的 Top-K 数组
    probs = np.exp(log_probs[0, 4:, :])
    top_k = 20
    topk_indices = np.argsort(-probs, axis=-1)[:, :top_k].astype(np.int32)
    topk_probs = np.zeros((probs.shape[0], top_k), dtype=np.float32)
    for t in range(probs.shape[0]):
        topk_probs[t] = probs[t, topk_indices[t]]
    top1_indices = topk_indices[:, 0]
    
    # 运行扫描
    radar = FastHotwordRadar(hotwords, engine.sp)
    detected = radar.scan(topk_indices, topk_probs, top1_indices)
    
    # 合并搜集所有热词的 match_map 用于表格显示
    global_match_map = {}
    for item in detected:
        for tk in item["tokens"]:
            # 计算帧索引
            t_idx = int(round(tk["time"] / 0.060))
            global_match_map[t_idx] = tk["token"].replace("\u2581", "")

    # 4. 打印四列调试表格
    print("\n" + "="*160)
    print(f"{'帧时间':<10} | {'Greedy (Top-1)':<15} | {'热词 Token 锚点':<15} | {'Top-25 搜索空间 (由高到低排列)':<80}")
    print("-" * 160)
    
    for t in range(len(topk_frames)):
        timestamp = t * 0.060
        candidates = topk_frames[t]
        
        # Column 2: Greedy
        greedy_char = candidates[0][0]
        greedy_display = f"【{greedy_char}】" if greedy_char != "[BLANK]" else "   ."
        
        # Column 3: Hotword Token Anchor
        hw_display = global_match_map.get(t, "")
        
        # Column 4: Top-25 Space
        top25_chars = [c[0] for c in candidates]
        search_space = " ".join(top25_chars).replace("[BLANK]", "·")
        
        print(f"{timestamp:>6.2f}s    | {greedy_display:<15} | {hw_display:<15} | {search_space}")

    print("="*150)
    print("表格说明：'热词召回结果' 列显示的是雷达在 Top-25 空间中捕捉到的匹配路径。")

if __name__ == "__main__":
    main()
