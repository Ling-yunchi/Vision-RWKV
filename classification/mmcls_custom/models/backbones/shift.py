import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class UWShift(nn.Module):
    def __init__(self, n_features, kernel_size=7):
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


class MLDCShift(nn.Module):
    def __init__(self, dim):
        super(MLDCShift, self).__init__()
        self.conv1x1_in = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1, bias=False
        )
        self.conv3x3_1 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=dim,
            bias=False,
        )
        self.conv3x3_2 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=3,
            dilation=3,
            groups=dim,
            bias=False,
        )
        self.conv1x1_out = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1, bias=False
        )

    def forward(self, x):
        xx = self.conv1x1_in(x)
        x1 = self.conv3x3_1(xx)
        x2 = self.conv3x3_2(xx)
        xx = self.conv1x1_out(x1 + x2)
        out = x + xx
        return out


class MVCShift(nn.Module):
    def __init__(self, dim):
        super(MVCShift, self).__init__()
        self.conv1x1 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1, bias=False
        )
        self.conv3x3d1 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            groups=dim,
            bias=False,
        )
        self.conv3x3d2 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=dim,
            bias=False,
        )
        self.conv3x3d3 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=3,
            dilation=3,
            groups=dim,
            bias=False,
        )
        self.conv1x1_1 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1, bias=False
        )
        self.conv1x1_2 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1, bias=False
        )
        self.conv1x1_3 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1, bias=False
        )

    def forward(self, x):
        xx = self.conv1x1(x)
        x1 = self.conv3x3d1(xx)
        x2 = self.conv3x3d2(xx)
        x3 = self.conv3x3d3(xx)
        x1o = self.conv1x1_1(x1)
        x2o = self.conv1x1_2(x2)
        x3o = self.conv1x1_3(x3)
        out = x + x1o + x2o + x3o
        return out


class DeformShift(nn.Module):
    def __init__(self, dim, kernel_size=3, offset_ks=3):
        super(DeformShift, self).__init__()
        o_p = offset_ks // 2
        padding = kernel_size // 2
        self.offset_conv = nn.Conv2d(
            dim,
            2 * kernel_size * kernel_size,
            kernel_size=offset_ks,
            stride=1,
            padding=o_p,
            bias=True
        )
        self.deform_conv = DeformConv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=dim,  # 使用depth-wise convolution
            bias=False
        )

    def forward(self, x):
        offset = self.offset_conv(x)
        out = self.deform_conv(x, offset)
        return out


if __name__ == "__main__":
    dim = 192
    x = torch.ones(2, dim, 3, 3)
    omni_shift = OmniShift(dim)
    uw_shift = UWShift(dim)
    mldc_shift = MLDCShift(dim)
    mvc_shift = MVCShift(dim)
    deform_shift = DeformShift(dim)


    # print all shift total params
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(f'OmniShift: {count_parameters(omni_shift)}')
    print(f'UWShift: {count_parameters(uw_shift)}')
    print(f'MLDCShift: {count_parameters(mldc_shift)}')
    print(f'MVCShift: {count_parameters(mvc_shift)}')
    print(f'DeformShift: {count_parameters(deform_shift)}')
