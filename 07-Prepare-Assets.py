import os
import shutil
import torch
import numpy as np
from funasr import AutoModel
from export_config import MODEL_DIR, EXPORT_DIR

def main():
    print("\n[Step 07] 准备推理资源 (Embedding & Tokenizer)...")
    
    # 1. 确保目标目录存在
    os.makedirs(EXPORT_DIR, exist_ok=True)

    # 2. 拷贝词表文件
    tiktoken_src = os.path.join(MODEL_DIR, "multilingual.tiktoken")
    tiktoken_dst = os.path.join(EXPORT_DIR, "multilingual.tiktoken")
    
    if os.path.exists(tiktoken_src):
        print(f"正在拷贝词表: {tiktoken_src} -> {tiktoken_dst}")
        shutil.copy2(tiktoken_src, tiktoken_dst)
    else:
        print(f"⚠️ 警告: 未在官方目录找到 multilingual.tiktoken")

    # 3. 提取 Prompt Embedding 权重
    print(f"正在从 {MODEL_DIR} 提取 Embedding 权重...")
    official_model_wrapper = AutoModel(
        model=str(MODEL_DIR),
        trust_remote_code=True,
        device="cpu"
    )
    # 获取 Embedding 层权重 (torch.Tensor)
    embed_weight = official_model_wrapper.model.embed.weight.detach().cpu().numpy()
    
    embed_path = os.path.join(EXPORT_DIR, "prompt_embed.npy")
    print(f"正在保存 Embedding 矩阵 (形状: {embed_weight.shape}) 到: {embed_path}")
    np.save(embed_path, embed_weight)

    # 4. 保存一份简单的映射关系，方便推理时查表
    # 这部分可以硬编码在推理代码里，也可以写成 json
    import json
    meta = {
        "lid_dict": official_model_wrapper.model.lid_dict,
        "textnorm_dict": official_model_wrapper.model.textnorm_dict,
        "emo_dict": official_model_wrapper.model.emo_dict,
        "input_size": 560,
        "output_size": 512
    }
    meta_path = os.path.join(EXPORT_DIR, "inference_config.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    print(f"基础配置信息已保存到: {meta_path}")

    print(f"\n✅ 推理资源准备就绪！")

if __name__ == "__main__":
    main()
