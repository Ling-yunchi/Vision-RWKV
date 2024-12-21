import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F


class UWShift(nn.Module):
    def __init__(self, n_features, kernel_size):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        self.w = nn.Parameter(torch.ones(n_features) / 2)
        self.u = nn.Parameter(torch.ones(n_features) / 2)
        self.alpha = nn.Parameter(torch.ones(n_features) / 2)
        self.kernel_size = kernel_size
        self.padding = tuple(k // 2 for k in [self.kernel_size, self.kernel_size])
        with torch.no_grad():
            self.register_buffer("distance_matrix", self.create_distance_matrix(kernel_size))

    @staticmethod
    def create_distance_matrix(kernel_size):
        center_idx = kernel_size // 2

        x_coords = torch.arange(kernel_size)
        y_coords = torch.arange(kernel_size)
        y_grid, x_grid = torch.meshgrid(
            y_coords, x_coords, indexing="ij"
        )  # `indexing='ij'` 确保y在第0维，x在第1维

        # 计算距离矩阵：绝对距离之和
        distance_matrix = torch.abs(x_grid - center_idx) + torch.abs(y_grid - center_idx)
        return distance_matrix

    def calc_decay_matrix(self):
        center_idx = self.kernel_size // 2
        dis_mat = self.distance_matrix
        decay_matrix = -dis_mat.unsqueeze(0) * self.w.unsqueeze(1).unsqueeze(2)
        decay_matrix[:, center_idx, center_idx] += self.u
        return decay_matrix.exp()  # 确保衰减为正数

    def forward(self, x, patch_resolution):
        """
        x: B,T,C
        """
        B, T, C = x.shape
        H, W = patch_resolution
        xx = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        decay_mat = self.calc_decay_matrix()
        decay_mat = rearrange(decay_mat, 'c k1 k2 -> c () k1 k2')

        xx = F.conv2d(xx, decay_mat, padding=self.padding, groups=C)

        xx = rearrange(xx, 'b c h w -> b (h w) c')
        output = x * self.alpha + xx * (1 - self.alpha)
        return output


if __name__ == "__main__":
    x = torch.ones(2, 9, 1)
    shift = UWShift(1, 3)
    print(shift(x, (3, 3)))
