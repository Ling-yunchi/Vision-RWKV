import math

import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
from torch.fft import fftshift, ifftshift, fft2, ifft2


class Down_conv(nn.Module):
    def __init__(self, n_dim, down_factor=2):
        super(Down_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_dim, n_dim // down_factor, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x


class Down_fft(nn.Module):
    def __init__(self, in_channels, out_channels, down_factor=2, kernel_size=3, padding=1):
        super(Down_fft, self).__init__()
        self.down_factor = down_factor
        self.conv_downsample = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                         stride=self.down_factor, padding=padding)
        self.register_buffer('gaussian_kernel', None, persistent=True)

    def _compute_gaussian_kernel(self, H, W):
        """ 计算并返回高斯核 """
        crow, ccol = H // 2, W // 2
        y_range = torch.arange(0, H, dtype=torch.float32) - crow
        x_range = torch.arange(0, W, dtype=torch.float32) - ccol
        Y, X = torch.meshgrid(y_range, x_range, indexing='ij')
        gaussian_kernel = torch.exp(-(X ** 2 + Y ** 2) / (2 * (min(H, W) / 6) ** 2))
        return gaussian_kernel.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        N, C, H, W = x.shape

        if self.gaussian_kernel is None or self.gaussian_kernel.shape[-2:] != (H, W):
            self.register_buffer('gaussian_kernel', self._compute_gaussian_kernel(H, W), persistent=True)

        f_x = fftshift(fft2(x, dim=(-2, -1)), dim=(-2, -1))
        weighted_f_x = f_x * self.gaussian_kernel.expand(N, C, -1, -1)
        weighted_x = torch.abs(ifft2(ifftshift(weighted_f_x, dim=(-2, -1)), dim=(-2, -1)))
        output = self.conv_downsample(weighted_x)
        return output


class DCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(DCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        result = x * self.weight
        return result

    @staticmethod
    def build_filter(pos, freq, t_pos):
        result = math.cos(math.pi * freq * (pos + 0.5) / t_pos) / math.sqrt(t_pos)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)
        c_part = channel // len(mapper_x)
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = (
                            self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y))

        return dct_filter


class Down_dct(nn.Module):
    def __init__(self, dct_h, dct_w, channel, down_factor=2, freq_sel_method='top16'):
        super(Down_dct, self).__init__()
        self.dct_h = dct_h
        self.dct_w = dct_w
        self.down_factor = down_factor

        mapper_x, mapper_y = self.get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]

        self.dct_layer = DCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.down_conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=self.down_factor, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def get_freq_indices(method):

        assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                          'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                          'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
        num_freq = int(method[3:])
        if 'top' in method:
            all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2,
                                 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2, 6, 1]
            all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2,
                                 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0, 5, 3]
            mapper_x = all_top_indices_x[:num_freq]
            mapper_y = all_top_indices_y[:num_freq]
        elif 'low' in method:
            all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0,
                                 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
            all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5,
                                 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3]
            mapper_x = all_low_indices_x[:num_freq]
            mapper_y = all_low_indices_y[:num_freq]
        elif 'bot' in method:
            all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5,
                                 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5, 3, 6]
            all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1,
                                 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3, 3, 3]
            mapper_x = all_bot_indices_x[:num_freq]
            mapper_y = all_bot_indices_y[:num_freq]
        else:
            raise NotImplementedError
        return mapper_x, mapper_y

    def forward(self, x):
        x_dct = self.dct_layer(x)
        output = self.down_conv(x_dct)
        return output


if __name__ == '__main__':
    x = torch.randn(1, 64, 224, 224)
    conv = Down_conv(64)
    wt = Down_wt(64, 64)
    fft = Down_fft(64, 64)
    dct = Down_dct(224, 224, 64)
    x_conv = conv(x)
    x_wt = wt(x)
    x_fft = fft(x)
    x_dct = dct(x)
    print(f"input shape {x.shape}")
    print(f"conv {x_conv.shape}")
    print(f"wt {x_wt.shape}")
    print(f"fft {x_fft.shape}")
    print(f"dct {x_dct.shape}")
