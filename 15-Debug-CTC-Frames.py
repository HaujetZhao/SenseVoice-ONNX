import os
import librosa
import numpy as np
from cosyvoice_onnx.inference import SenseVoiceInference, HotwordRadar

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
    
    # 为了演示，我们手动模拟一次 radar 的扫描过程
    from cosyvoice_onnx.inference.ctc import greedy_search
    # 推理一次获取原始 log_probs
    lfr_feat = engine.frontend.extract(audio)
    prompt_feat = engine.construct_prompt(lid="zh")
    mask = np.ones((1, lfr_feat.shape[0])).astype(np.float32)
    enc_out = engine.enc_sess.run(None, {"speech_feat": lfr_feat[np.newaxis, ...], "mask": mask, "prompt_feat": prompt_feat})[0]
    log_probs = engine.ctc_sess.run(None, {"enc_out": enc_out})[0]
    
    top1_indices = np.argmax(log_probs[0, 4:, :], axis=-1)
    radar = HotwordRadar(topk_frames)
    
    # 合并搜集所有热词的 match_map
    global_match_map = {}
    for word in hotwords:
        # 使用高级版 scan (传入 engine.sp)
        res = radar.scan(word, tokenizer=engine.sp, top1_indices=top1_indices)
        if res["found"]:
            # 记录哪些帧被识别为什么热词 Token (锚点)
            for t_idx, token_txt in res["match_map"].items():
                global_match_map[t_idx] = f"{token_txt}"

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
