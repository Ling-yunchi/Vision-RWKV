import os
import time

import torch
from torch.utils.cpp_extension import load

T_MAX = 8192  # increase this if your ctx_len is long [NOTE: TAKES LOTS OF VRAM!]
# it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice

file_path = os.path.dirname(os.path.realpath(__file__))
is_windows = os.name == "nt"
build_dir = f"{file_path}/build_2d"
if not os.path.exists(build_dir):
    os.makedirs(build_dir)

wkv_cuda = load(
    name="wkv",
    sources=[f"{file_path}/cuda/wkv_2d_op.cpp", f"{file_path}/cuda/wkv_2d_cuda.cu"],
    verbose=True,
    build_directory=build_dir,
    extra_cuda_cflags=[
        "-res-usage",
        f"--maxrregcount{'=' if is_windows else ' '}60",
        "--use_fast_math",
        "-O3",
        "-Xptxas",
        "-O3",
        f"-DTmax={T_MAX}",
    ],
)


class WKV2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, W, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        ctx.H = H
        ctx.W = W
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = w.dtype == torch.half
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()

        ctx.save_for_backward(w, u, k, v)
        y = torch.empty((B, T, C), device="cuda", memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, H, W, w, u, k, v, y)
        if half_mode:
            y = y.half()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        H = ctx.H
        W = ctx.W
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device="cuda").contiguous()
        gu = torch.zeros((B, C), device="cuda").contiguous()
        gk = torch.zeros((B, T, C), device="cuda").contiguous()
        gv = torch.zeros((B, T, C), device="cuda").contiguous()
        half_mode = gy.dtype == torch.half
        wkv_cuda.backward(B, T, C, H, W, w, u, k, v, gy.contiguous(), gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (
                None,
                None,
                None,
                None,
                None,
                gw.half(),
                gu.half(),
                gk.half(),
                gv.half(),
            )
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, None, None, gw, gu, gk, gv)


def RUN_CUDA_2d(B, T, C, H, W, w, u, k, v):
    return WKV2d.apply(B, T, C, H, W, w.cuda(), u.cuda(), k.cuda(), v.cuda())


if __name__ == "__main__":
    H, W = 2, 2
    B, T, C = 2, H * W, 1
    w = torch.randn((B, C), device="cuda").requires_grad_()
    u = torch.randn((B, C), device="cuda").requires_grad_()
    k = torch.randn((B, T, C), device="cuda").requires_grad_()
    v = torch.randn((B, T, C), device="cuda").requires_grad_()
    st = time.time()
    y_2d = RUN_CUDA_2d(B, T, C, H, W, w, u, k, v)
    print(f"{time.time() - st:.20f}s")
    print(y_2d)
