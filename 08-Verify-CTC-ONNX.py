import os
import numpy as np
import onnxruntime as ort
from export_config import EXPORT_DIR

def main():
    print("\n[Step 05] 验证 CTC ONNX 模型推理...")
    
    onnx_path = os.path.join(EXPORT_DIR, "SenseVoice-CTC.fp32.onnx")
    if not os.path.exists(onnx_path):
        print(f"❌ 错误: 找不到模型文件 {onnx_path}")
        return

    # 1. 创建 ONNX Runtime 会话 (CPU)
    print(f"正在加载 CTC ONNX 模型: {onnx_path}")
    session = ort.InferenceSession(onnx_path, providers=['DmlExecutionProvider'])

    # 2. 准备随机输入
    # 输入维度是 (Batch, T, 512)
    batch_size = 1
    T_plus_4 = 25 
    
    print(f"准备测试输入 (Batch={batch_size}, T_plus_4={T_plus_4})...")
    enc_out = np.random.randn(batch_size, T_plus_4, 512).astype(np.float32)

    # 3. 执行推理
    print("开始 CTC ONNX 推理...")
    inputs = {"enc_out": enc_out}
    outputs = session.run(None, inputs)
    log_probs = outputs[0]

    # 4. 输出结果信息
    print("\n" + "="*30)
    print("【CTC 推理成功】")
    print(f"输出结果形状: {log_probs.shape}")
    # Vocab size 通常是 60515 左右
    print("-" * 30)
    print("前 5 个输出数值示例 (Log Probs):")
    print(log_probs[0, 0, :5])
    print("="*30)

if __name__ == "__main__":
    main()
