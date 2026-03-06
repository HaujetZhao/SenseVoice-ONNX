import torch
import torch.nn as nn
from .model import SenseVoiceEncoderSmall

class EncoderExportWrapper(nn.Module):
    """
    SenseVoice Encoder 导出包装器
    支持输入:
        - speech_feat: (Batch, T, 560) -> 原始 Fbank 特征
        - mask: (Batch, T) -> 特征有效掩码 (1代表有效, 0代表填充)
        - prompt_feat: (Batch, 4, 560) -> 预先查好表的 4 帧 Prompt Embedding
    """
    def __init__(self, encoder: SenseVoiceEncoderSmall):
        super().__init__()
        self.encoder = encoder

    def forward(self, speech_feat: torch.Tensor, mask: torch.Tensor, prompt_feat: torch.Tensor):
        # 1. 拼接 Prompt 与语音特征 -> (Batch, 4 + T, 560)
        x = torch.cat([prompt_feat, speech_feat], dim=1)

        # 2. 构造完整的 Mask
        # Prompt 区域(前4帧)始终为有效状态(1)
        batch_size = mask.size(0)
        prompt_mask = torch.ones((batch_size, 4), device=mask.device, dtype=mask.dtype)
        full_mask = torch.cat([prompt_mask, mask], dim=1)

        # 3. 调用编码器
        # 我们之前已经修改了 model.py，使其 encoder.forward 接受 (x, mask)
        encoder_out = self.encoder(x, full_mask)

        return encoder_out

class CTCExportWrapper(nn.Module):
    """
    SenseVoice CTC 解码头导出包装器 (TopK 优化版)
    输入:
        - enc_out: (Batch, T_plus_4, 512)
    输出:
        - topk_log_probs: (Batch, T_plus_4, 100) -> 前 100 个最大对数概率
        - topk_indices: (Batch, T_plus_4, 100) -> 对应的字符 ID
    """
    def __init__(self, ctc, k=100):
        super().__init__()
        self.ctc = ctc
        self.k = k

    def forward(self, enc_out: torch.Tensor):
        # 1. 计算全量 Log Softmax
        log_probs = self.ctc.log_softmax(enc_out)
        
        # 2. 在 GPU 侧直接提取 TopK
        # 这将极大减少 DML 的回传数据量 (1.2M -> 5KB)
        topk_log_probs, topk_indices = torch.topk(log_probs, self.k, dim=-1)
        
        return topk_log_probs, topk_indices.to(torch.int32)
