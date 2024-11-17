import torch
import torch.nn.functional as F
from torch import nn


def create_distance_matrix(kernel_size, w, u):
    # 计算中心位置的索引
    center_idx = kernel_size // 2

    # 创建坐标网格
    x_coords = torch.arange(kernel_size)
    y_coords = torch.arange(kernel_size)
    y_grid, x_grid = torch.meshgrid(
        y_coords, x_coords, indexing="ij"
    )  # `indexing='ij'` 确保y在第0维，x在第1维

    # 计算距离矩阵：绝对距离之和
    distance_matrix = torch.abs(x_grid - center_idx) + torch.abs(y_grid - center_idx)

    # 使用距离矩阵进行权重计算
    weight_matrix = -distance_matrix.unsqueeze(0) * w.unsqueeze(1).unsqueeze(
        2
    )  # 广播到每个通道
    weight_matrix[:, center_idx, center_idx] += u  # 中心元素加上每个通道的u

    return weight_matrix  # 不再展开为1D，保持为2D


class WKV2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, B, H, W, C, w, u, k, v):
        device = w.device
        kernel_size = 2 * (max(H, W) - 1) + 1
        pad = kernel_size // 2

        # 枃建距离矩阵
        distance_weights = create_distance_matrix(
            kernel_size, w, u
        ).to(device)  # Shape: (C, kernel_size, kernel_size)

        # 处理k的展开
        k = k.permute(0, 3, 1, 2)  # 转换为 B x C x H x W
        unfold_k = F.unfold(k, kernel_size=(kernel_size, kernel_size), padding=pad)
        unfolded_k = unfold_k.view(
            B, C, kernel_size * kernel_size, H * W
        )  # B x C x (K*K) x (H*W)

        # 应用距离权重并求和得到调整后的k
        distance_weights = distance_weights.view(
            C, kernel_size * kernel_size, 1
        )  # C x (K*K) x 1
        weight_adjusted_k = unfolded_k + distance_weights  # B x C x (K*K) x (H*W)
        weight_adjusted_k = F.relu(weight_adjusted_k, inplace=True)
        k_adjusted = weight_adjusted_k.sum(dim=2)  # B x C x (H * W)

        # 计算exp(k)
        exp_k = torch.exp(k_adjusted)  # B x C x (H * W)

        # 处理v并与exp(k)相乘
        v = v.permute(0, 3, 1, 2).reshape(B, C, -1)  # B x C x (H * W)
        result = exp_k * v  # 逐元素相乘 B x C x (H * W)

        # 将结果reshape为 B x C x H x W
        output = result.view(B, C, H, W)
        output = output.permute(0, 2, 3, 1)  # 转换回 B x H x W x C

        return output
