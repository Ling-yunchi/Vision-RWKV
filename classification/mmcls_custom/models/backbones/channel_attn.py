import math

import torch
from torch import nn
import torch.nn.functional as F


class SeparableAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dim, attn_dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.qkv_proj = nn.Conv2d(embed_dim, 1 + (hidden_dim * 2), 1)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.out_proj = nn.Conv2d(hidden_dim, embed_dim, 1)

    def forward(self, x):
        """
        x: B H W C
        """
        qkv = self.qkv_proj(x)
        query, key, value = torch.split(qkv, [1, self.hidden_dim, self.hidden_dim], dim=1)

        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_drop(context_scores)  # B H W 1

        context_vector = key * context_scores  # B H W D
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)  # B H W 1

        output = F.relu(value) * context_vector  # B H W D
        output = self.out_proj(output)
        return output


class ECA(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=None):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if k_size is None:
            t = int(abs((math.log(channel, 2) + 1) / 2))
            k_size = k_size if t % 2 else t + 1
        self.k_size = k_size
        self.conv = nn.Conv1d(1, 1, kernel_size=self.k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
