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

    # 2. 准备输入：30秒容器，10秒有效
    batch_size = 1
    T_total = 510 # 约 30s (1s = 17 帧 LFR)
    T_valid = 170 # 约 10s
    
    print(f"准备测试输入: 总长度={T_total}, 有效长度={T_valid}...")
    
    # 构造语音特征：前 T_valid 帧为随机值，后面部分填充（例如复读最后一帧）
    speech_feat_valid = np.random.randn(batch_size, T_valid, 560).astype(np.float32)
    # 按照 DML 优化逻辑，填充区通常使用 Replicate Padding
    speech_feat_padding = np.repeat(speech_feat_valid[:, -1:, :], T_total - T_valid, axis=1)
    speech_feat = np.concatenate([speech_feat_valid, speech_feat_padding], axis=1)
    
    # 构造 Mask：1 表示有效，0 表示无效
    mask = np.zeros((batch_size, T_total), dtype=np.float32)
    mask[:, :T_valid] = 1.0
    
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
    print(f"ONNX 输出形状: {encoder_out.shape}")
    print(f"有效部分形状: (1, {T_valid + 4}, 512)")
    
    # 我们可以观察有效部分之后的值是否被掩码清零了（或者保持一致）
    # 按照我们在 model.py 里的 Fire-wall Sweeping，有效部分后面应该全是 0
    after_valid = encoder_out[0, T_valid+4:, :]
    max_after_valid = np.max(np.abs(after_valid))
    print(f"有效长度之后的最大数值: {max_after_valid:.2e} (预期接近 0)")
    
    print("-" * 30)
    print("前 5 个输出数值示例 (Prompt + Speech):")
    print(encoder_out[0, 0, :5])
    print("="*30)

if __name__ == "__main__":
    main()
