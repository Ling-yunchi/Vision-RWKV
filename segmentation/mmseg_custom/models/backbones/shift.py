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


class OmniShift(nn.Module):
    def __init__(self, dim):
        super(OmniShift, self).__init__()
        # Define the layers for training
        self.conv1x1 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1, groups=dim, bias=False
        )
        self.conv3x3 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            groups=dim,
            bias=False,
        )
        self.conv5x5 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=5,
            padding=2,
            groups=dim,
            bias=False,
        )
        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True)

        # Register buffers for testing
        self.register_buffer('combined_weight', None)
        self.repram_flag = True

    def forward_train(self, x):
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)

        out = (
                self.alpha[0] * x
                + self.alpha[1] * out1x1
                + self.alpha[2] * out3x3
                + self.alpha[3] * out5x5
        )
        return out

    def reparam_5x5(self):
        # Combine the parameters of conv1x1, conv3x3, and conv5x5 to form a single 5x5 depth-wise convolution

        padded_weight_1x1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2))
        padded_weight_3x3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1))

        identity_weight = F.pad(torch.ones_like(self.conv1x1.weight), (2, 2, 2, 2))

        combined_weight = (
                self.alpha[0] * identity_weight
                + self.alpha[1] * padded_weight_1x1
                + self.alpha[2] * padded_weight_3x3
                + self.alpha[3] * self.conv5x5.weight
        )

        device = self.conv5x5.weight.device
        combined_weight = combined_weight.to(device)

        # Store the combined weight in the buffer
        self.combined_weight = combined_weight

    def forward(self, x):
        if self.training:
            self.repram_flag = True
            out = self.forward_train(x)
        else:
            if self.repram_flag:
                self.reparam_5x5()
                self.repram_flag = False

            # Use F.conv2d with the stored combined weights for testing
            out = F.conv2d(x, self.combined_weight, padding=2, groups=x.size(1))
        return out


if __name__ == "__main__":
    x = torch.ones(2, 9, 1)
    shift = UWShift(1, 3)
    print(shift(x, (3, 3)))
