import os
import shutil
import torch
import json
import numpy as np
from funasr import AutoModel
from export_config import MODEL_DIR, EXPORT_DIR

def main():
    print("\n[Step 07] 准备推理资源 (Embedding & Tokenizer)...")
    
    # 1. 确保目标目录存在
    os.makedirs(EXPORT_DIR, exist_ok=True)

    # 2. 处理词表 (Tokenizer)
    tokens_src = os.path.join(MODEL_DIR, "tokens.json")
    tokens_dst = os.path.join(EXPORT_DIR, "tokens.json")
    if os.path.exists(tokens_src):
        print(f"正在准备词表: {tokens_dst}")
        shutil.copy2(tokens_src, tokens_dst)
    
    bpe_src = os.path.join(MODEL_DIR, "chn_jpn_yue_eng_ko_spectok.bpe.model")
    bpe_dst = os.path.join(EXPORT_DIR, "Tokenizer.bpe.model")
    if os.path.exists(bpe_src):
        print(f"正在拷贝 BPE 模型: {bpe_dst}")
        shutil.copy2(bpe_src, bpe_dst)

    # 3. 基础配置提取
    print(f"正在从 {MODEL_DIR} 提取基础配置...")
    official_model_wrapper = AutoModel(
        model=str(MODEL_DIR),
        trust_remote_code=True,
        device="cpu"
    )

    print(f"\n✅ 推理资源准备就绪！")

if __name__ == "__main__":
    main()
