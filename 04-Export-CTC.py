import os
import torch
import warnings
from pathlib import Path

# 引入配置
from export_config import MODEL_DIR, EXPORT_DIR

# 引入 Wrapper
from cosyvoice_onnx.export.wrappers import CTCExportWrapper
from funasr import AutoModel

def main():
    print("\n[Step 04] 开始导出 SenseVoice CTC Decoder ONNX (FP32)...")
    
    # 1. 准备目录
    os.makedirs(EXPORT_DIR, exist_ok=True)
    onnx_path = EXPORT_DIR / "SenseVoice-CTC.fp32.onnx"
    
    # 2. 加载官方模型获取 CTC 部分
    print(f"正在从 {MODEL_DIR} 提取 CTC 权重...")
    official_model_wrapper = AutoModel(
        model=str(MODEL_DIR),
        trust_remote_code=True,
        device="cpu"
    )
    official_model = official_model_wrapper.model
    
    # 3. 初始化 CTC 包装器
    ctc_wrapper = CTCExportWrapper(official_model.ctc)
    ctc_wrapper.eval()

    # 4. 准备虚拟输入数据 (Dummy Inputs)
    # 输入是 Encoder 的输出: (Batch, T_plus_4, 512)
    batch_size = 1
    T_plus_4 = 21 # 17帧音频 + 4帧Prompt
    dummy_enc_out = torch.randn(batch_size, T_plus_4, 512)

    # 5. 执行 ONNX 导出
    print(f"正在导出 CTC ONNX 到: {onnx_path}")
    
    warnings.filterwarnings("ignore")
    
    torch.onnx.export(
        ctc_wrapper, 
        (dummy_enc_out,), 
        str(onnx_path),
        input_names=['enc_out'],
        output_names=['topk_log_probs', 'topk_indices'],
        dynamic_axes={
            'enc_out': {1: 'T_plus_4'},
            'topk_log_probs': {1: 'T_plus_4'},
            'topk_indices': {1: 'T_plus_4'}
        },
        opset_version=18,
        dynamo=True # 同样使用 Dynamo 模式
    )

    print(f"\n✅ CTC 导出完成！")
    print(f"模型文件: {onnx_path}")

if __name__ == "__main__":
    main()
