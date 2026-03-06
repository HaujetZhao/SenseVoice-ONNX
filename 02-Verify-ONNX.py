import os
import numpy as np
import onnxruntime as ort
from export_config import EXPORT_DIR

def main():
    print("\n[Step 02] 验证 ONNX 模型推理...")
    
    onnx_path = os.path.join(EXPORT_DIR, "SenseVoice-Encoder.fp32.onnx")
    if not os.path.exists(onnx_path):
        print(f"❌ 错误: 找不到模型文件 {onnx_path}")
        return

    # 1. 创建 ONNX Runtime 会话 (强制使用 CPU)
    print(f"正在加载 ONNX 模型: {onnx_path}")
    sess_options = ort.SessionOptions()
    # 强制使用 CPU
    providers = ['DmlExecutionProvider']
    session = ort.InferenceSession(onnx_path, sess_options, providers=providers)

    # 2. 准备随机输入
    batch_size = 1
    T = 20 # 随机长度进行动态轴测试
    
    print(f"准备测试输入 (Batch={batch_size}, T={T})...")
    # 注意: ONNX Runtime 期望的是 numpy 数组
    speech_feat = np.random.randn(batch_size, T, 560).astype(np.float32)
    mask = np.ones((batch_size, T)).astype(np.float32)
    prompt_feat = np.random.randn(batch_size, 4, 560).astype(np.float32)

    # 3. 执行推理
    print("开始 ONNX 推理...")
    inputs = {
        "speech_feat": speech_feat,
        "mask": mask,
        "prompt_feat": prompt_feat
    }
    
    outputs = session.run(None, inputs)
    encoder_out = outputs[0]

    # 4. 输出结果信息
    print("\n" + "="*30)
    print("【推理成功】")
    print(f"输出结果形状: {encoder_out.shape}")
    print(f"预期形状应为: (1, {4 + T}, 512)") # 4帧Prompt + T帧音频, Encoder输出维度通常是512
    print("-" * 30)
    print("前 5 个输出数值示例:")
    print(encoder_out[0, 0, :5])
    print("="*30)

    # 5. 测试动态轴 (不同的 T)
    print("\n测试动态轴 (T=50)...")
    T2 = 50
    speech_feat2 = np.random.randn(batch_size, T2, 560).astype(np.float32)
    mask2 = np.ones((batch_size, T2)).astype(np.float32)
    inputs2 = {
        "speech_feat": speech_feat2,
        "mask": mask2,
        "prompt_feat": prompt_feat
    }
    outputs2 = session.run(None, inputs2)
    print(f"动态轴 T=50 输出形状: {outputs2[0].shape}")

if __name__ == "__main__":
    main()
