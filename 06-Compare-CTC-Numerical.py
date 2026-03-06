import os
import torch
import numpy as np
import onnxruntime as ort
from funasr import AutoModel
from export_config import MODEL_DIR, EXPORT_DIR

def main():
    print("\n[Step 06] CTC 数值比对: Torch (Official) vs ONNX (Dynamo)")
    
    # 1. 路径设置
    onnx_path = os.path.join(EXPORT_DIR, "SenseVoice-CTC.fp32.onnx")
    
    # 2. 加载官方 Torch 模型
    print("正在加载官方 Torch 模型 CTC 部分...")
    official_model_wrapper = AutoModel(
        model=str(MODEL_DIR),
        trust_remote_code=True,
        device="cpu"
    )
    torch_ctc = official_model_wrapper.model.ctc
    torch_ctc.eval()

    # 3. 加载 ONNX 模型
    if not os.path.exists(onnx_path):
        print(f"❌ 错误: 找不到模型文件 {onnx_path}")
        return
        
    print(f"正在加载 CTC ONNX 模型: {onnx_path}")
    session = ort.InferenceSession(onnx_path, providers=['DmlExecutionProvider'])

    # 4. 准备随机输入数据
    batch_size = 1
    T_plus_4 = 30
    D = 512
    
    enc_out = torch.randn(batch_size, T_plus_4, D)

    # 5. 执行推理
    print("运行 Torch CTC 基准推理...")
    with torch.no_grad():
        # 官方 CTC.log_softmax 接受 enc_out
        torch_out = torch_ctc.log_softmax(enc_out)

    print("运行 ONNX CTC 导出模型推理...")
    onnx_inputs = {"enc_out": enc_out.numpy()}
    onnx_out = session.run(None, onnx_inputs)[0]

    # 6. 数值比对分析
    torch_out_np = torch_out.numpy()
    
    # 计算余弦相似度
    # 由于词表维度较大，分切片计算或展平计算
    t_flat = torch_out_np.flatten()
    o_flat = onnx_out.flatten()
    
    # 避免数值溢出，使用 float64 计算相似度
    dot_prod = np.dot(t_flat.astype(np.float64), o_flat.astype(np.float64))
    norm_t = np.linalg.norm(t_flat.astype(np.float64))
    norm_o = np.linalg.norm(o_flat.astype(np.float64))
    cos_sim = dot_prod / (norm_t * norm_o)
    
    # 计算绝对误差
    abs_diff = np.abs(torch_out_np - onnx_out)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    print("\n" + "="*50)
    print("【CTC 数值一致性报告】")
    print(f"测试维度: T_plus_4={T_plus_4}, Vocab_Dim={torch_out_np.shape[-1]}")
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
        print("⚠️ 警告：数值存在偏差。")
    print("="*50)

if __name__ == "__main__":
    main()
