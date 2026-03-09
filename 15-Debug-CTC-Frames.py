import os
import numpy as np
from sensevoice_onnx.inference import SenseVoiceInference, ASREngineConfig, load_audio
from sensevoice_onnx.inference.radar import HotwordRadar

def main():
    # 1. 初始化引擎
    model_dir = "./model"
    config = ASREngineConfig(model_dir=model_dir, onnx_provider="cpu", precision="int4")
    engine = SenseVoiceInference(config)
    
    # 2. 准备音频
    audio_path = r"test-fun.mp3"
    audio_path = r"睡前消息.m4a"
    audio_path = r"dugong.mp3"

    # 准备热词
    hotword_file = "hot.txt"
    with open('hot.txt', "r", encoding="utf-8") as f:
        hotwords = [line.strip() for line in f if line.strip()]
    
    if not os.path.exists(audio_path):
        print(f"❌ 找不到测试音频: {audio_path}")
        return
        
    audio = load_audio(audio_path, duration = 30)
    
    # 3. 运行推理并获取调试数据
    # 提取特征
    lfr_feat = engine.frontend.extract(audio)
    T_valid = lfr_feat.shape[0]
    
    # 配置显示参数
    display_top_k = 20 # <--- 变量定义：设置想要显示的搜索深度以及雷达扫描深度
    
    # Encoder
    enc_out = engine.encoder.forward(lfr_feat, lid="zh")
    
    # Decoder (获取由模型输出的 Top-K)
    topk_log_probs, topk_indices = engine.decoder.forward(enc_out)
    
    # 转换为概率并切片到 display_top_k，使雷达扫描范围和显示范围完全一致
    topk_probs = np.exp(topk_log_probs[0, 4:, :display_top_k])
    topk_ids = topk_indices[0, 4:, :display_top_k]
    top1_indices = topk_ids[:, 0]
    
    # 构造 topk_frames 用于显示 (char, prob)
    # 这里直接利用已有的 topk 结果
    topk_frames = []
    for t in range(topk_ids.shape[0]):
        frame_candidates = []
        for k in range(min(topk_ids.shape[1], display_top_k)): 
            tid = int(topk_ids[t, k])
            prob = float(topk_probs[t, k])
            if tid == 0: # Blank
                char = "[BLANK]"
            else:
                char = engine.sp.id_to_piece(tid)
                if not char.strip():
                    char = f"<{tid}>"
            frame_candidates.append((char, prob))
        topk_frames.append(frame_candidates)

    # 4. 运行雷达扫描
    radar = HotwordRadar(hotwords, engine.sp)
    detected = radar.scan(topk_ids, topk_probs, top1_indices)
    
    # 合并搜集所有热词的 match_map 用于表格显示
    global_match_map = {}
    for item in detected:
        for tk in item["tokens"]:
            # 计算帧索引
            t_idx = int(round(tk["time"] / 0.060))
            global_match_map[t_idx] = tk["token"].replace("\u2581", "")

    # 5. 打印四列调试表格
    print("\n" + "="*50)
    print(f"{'帧时间':<5} | {'Greedy':<5} | {'热词锚点':<5} | {f'搜索空间':<80}")
    print("-" * 50)
    
    for t in range(len(topk_frames)):
        timestamp = t * 0.060
        candidates = topk_frames[t]
        
        # Column 2: Greedy
        greedy_char = candidates[0][0]
        greedy_display = f"{greedy_char}" if greedy_char != "[BLANK]" else "   ."
        
        # Column 3: Hotword Token Anchor
        hw_display = global_match_map.get(t, "")
        
        # Column 4: 搜索空间
        top_k_chars = [c[0] for c in candidates]
        search_space = " ".join(top_k_chars).replace("[BLANK]", "·")
        
        print(f"{timestamp:>6.2f}s    | {greedy_display:<5} | {hw_display:<5} | {search_space}")

    print("="*50)
    print(f"表格说明：'热词召回结果' 列显示的是雷达在 Top-{display_top_k} 空间中捕捉到的匹配路径。")

if __name__ == "__main__":
    main()
