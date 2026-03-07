import os
import sys
import torch
import numpy as np
import onnxruntime as ort
from funasr import AutoModel
from export_config import MODEL_DIR, EXPORT_DIR

def main():
    print("\n[Step 03] 数值比对: Torch (Official) vs ONNX (Dynamo)")
    
    # 1. 路径设置
    onnx_path = os.path.join(EXPORT_DIR, "SenseVoice-Encoder.fp32.onnx")
    
    # 2. 加载官方 Torch 模型 (使用最原始的定义进行基准测试)
    print("正在加载官方 Torch 模型作为基准...")
    # 显式指定加载官方 SenseVoice-main 里的 model.py，确保 ilens 逻辑是原始的
    official_model_wrapper = AutoModel(
        model=str(MODEL_DIR),
        trust_remote_code=True,
        remote_code=r"d:\cosyvoice\SenseVoice-main\model.py",
        device="cpu"
    )
    torch_model = official_model_wrapper.model.encoder
    torch_model.eval()

    # 3. 加载 ONNX 模型
    if not os.path.exists(onnx_path):
        print(f"❌ 错误: 找不到模型文件 {onnx_path}")
        return
        
    print(f"正在加载 ONNX 模型: {onnx_path}")
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # 4. 准备测试数据 (测试填充一致性)
    # 模拟场景：10秒有效信号，放在 30秒的容器里进行 DML 推理
    batch_size = 1
    T_valid = 170 # 10s
    T_total = 510 # 30s
    D = 560 # 特征维度
    
    print(f"准备数据: 有效长度={T_valid}, 容器长度={T_total}")
    
    # 基础特征 (有效部分)
    speech_feat_valid = torch.randn(batch_size, T_valid, D)
    prompt_feat = torch.randn(batch_size, 4, D)
    
    # --- Torch 基准推理准备 (纯净 10s 推理) ---
    full_input_torch = torch.cat([prompt_feat, speech_feat_valid], dim=1)
    full_lengths_torch = torch.tensor([T_valid + 4], dtype=torch.int32)

    # --- ONNX 填充推理准备 (30s 容器 + Replicate Padding) ---
    # Replicate Padding 最后一帧
    speech_feat_padding = speech_feat_valid[:, -1:, :].repeat(1, T_total - T_valid, 1)
    speech_feat_full = torch.cat([speech_feat_valid, speech_feat_padding], dim=1)
    
    # Mask: 10s 有效，20s 无效
    mask_full = torch.zeros(batch_size, T_total)
    mask_full[:, :T_valid] = 1.0

    onnx_inputs = {
        "speech_feat": speech_feat_full.numpy(),
        "mask": mask_full.numpy(),
        "prompt_feat": prompt_feat.numpy()
    }

    # 5. 执行推理
    print("运行 Torch 基准推理 (10s 隔离推理)...")
    with torch.no_grad():
        torch_out, _ = torch_model(full_input_torch, full_lengths_torch)

    print(f"运行 ONNX 填充推理 ({T_total}帧容器)...")
    onnx_out_full = session.run(None, onnx_inputs)[0]
    
    # 对比截断：只对比有效部分 (T_valid + 4)
    onnx_out = onnx_out_full[:, :T_valid + 4, :]
    torch_out_np = torch_out.numpy()

    # 6. 数值比对分析
    torch_out_np = torch_out.numpy()
    
    # 计算余弦相似度 (Cosine Similarity)
    t_flat = torch_out_np.flatten()
    o_flat = onnx_out.flatten()
    cos_sim = np.dot(t_flat, o_flat) / (np.linalg.norm(t_flat) * np.linalg.norm(o_flat))
    
    # 计算绝对误差
    abs_diff = np.abs(torch_out_np - onnx_out)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    print("\n" + "="*50)
    print("【数值一致性报告】")
    print(f"测试维度: T_valid={T_valid}, T_total={T_total}, Output_Dim={torch_out_np.shape[-1]}")
    print("-" * 30)
    print(f"余弦相似度: {cos_sim:.10f}")
    print(f"最大绝对误差: {max_diff:.3e}")
    print(f"平均绝对误差: {mean_diff:.3e}")
    print("-" * 30)

    if cos_sim > 0.99999:
        print("✅ 对比结果：极高度一致 (Excellent)")
    elif cos_sim > 0.9999:
        print("✅ 对比结果：高度一致 (Good)")
    else:
        print("⚠️ 警告：数值存在偏差，请确认算子改动对精度影响。")
    print("="*50)

if __name__ == "__main__":
    main()
