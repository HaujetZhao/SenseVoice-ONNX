import os
import torch
import warnings
from pathlib import Path

# 引入配置
from export_config import MODEL_DIR, EXPORT_DIR

# 引入改造后的模型定义和 Wrapper
# 由于 model_definition 需要加载官方权重，我们先用 funasr 加载官方权重，再将其 state_dict 注入到我们的导出版本
from cosyvoice_onnx.export.model import SenseVoiceSmall
from cosyvoice_onnx.export.wrappers import EncoderExportWrapper
from funasr import AutoModel

def main():
    print("\n[Step 01] 开始导出 SenseVoice ONNX (FP32)...")
    
    # 1. 准备目录
    os.makedirs(EXPORT_DIR, exist_ok=True)
    onnx_path = EXPORT_DIR / "SenseVoice-Encoder.fp32.onnx"
    
    # 2. 加载官方原始模型以获取权重
    print(f"正在从 {MODEL_DIR} 加载官方模型权重...")
    # 使用 AutoModel 主要是为了方便直接获取配置好的模型实例
    official_model_wrapper = AutoModel(
        model=str(MODEL_DIR),
        trust_remote_code=True,
        device="cpu"
    )
    official_model = official_model_wrapper.model
    
    # 3. 初始化我们改造过的导出专用模型
    # 我们需要根据官方模型的配置来初始化我们的 SenseVoiceSmall
    # 注意：这里我们直接从 official_model 拷贝参数或重用其内部的 Encoder 部分
    # 为了保证算子的一致性（使用我们修改过的 model.py），我们将权重加载到我们的类中
    print("正在初始化导出版本编码器...")
    export_encoder = official_model.encoder
    # 将 export_encoder 包装到我们的导出包装器中
    wrapper = EncoderExportWrapper(export_encoder)
    wrapper.eval()

    # 4. 准备虚拟输入数据 (Dummy Inputs)
    # speech_feat: (Batch, T, 560) -> 假设 1 秒音频，约 17 帧 LFR
    # mask: (Batch, T)
    # prompt_feat: (Batch, 4, 560)
    batch_size = 1
    T = 17 
    dummy_speech = torch.randn(batch_size, T, 560)
    dummy_mask = torch.ones(batch_size, T)
    dummy_prompt = torch.randn(batch_size, 4, 560)

    # 5. 执行 ONNX 导出
    print(f"正在导出 ONNX 到: {onnx_path}")
    
    # 忽略一些导出时的警告
    warnings.filterwarnings("ignore")
    
    torch.onnx.export(
        wrapper, 
        (dummy_speech, dummy_mask, dummy_prompt), 
        str(onnx_path),
        input_names=['speech_feat', 'mask', 'prompt_feat'],
        output_names=['encoder_out'],
        dynamic_axes={
            'speech_feat': {0: 'batch', 1: 'T'},
            'mask': {0: 'batch', 1: 'T'},
            'prompt_feat': {0: 'batch'},
            'encoder_out': {0: 'batch', 1: 'T_plus_4'}
        },
        opset_version=18, 
        dynamo=True # 使用 Dynamo 导出模式
    )

    print(f"\n✅ 导出完成！")
    print(f"模型文件: {onnx_path}")

if __name__ == "__main__":
    main()
