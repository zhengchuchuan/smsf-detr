"""
Lightweight P2 detail bridge fusion modules for HybridEncoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation


class ConvNormAct(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, kernel_size=3, stride=1, groups: int = 1, act: str = "silu"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            padding = tuple((int(k) - 1) // 2 for k in kernel_size)
        else:
            padding = (int(kernel_size) - 1) // 2
        self.conv = nn.Conv2d(
            int(ch_in),
            int(ch_out),
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=int(groups),
            bias=False,
        )
        self.norm = nn.BatchNorm2d(int(ch_out))
        self.act = get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class SPDConv2x(nn.Module):
    def __init__(self, channels: int, factor: int = 2, act: str = "silu"):
        super().__init__()
        factor = int(factor)
        if factor <= 0:
            raise ValueError(f"factor must be > 0, got {factor}")
        self.factor = factor
        ch_mid = int(channels) * factor * factor
        self.reduce = ConvNormAct(ch_mid, int(channels), kernel_size=1, stride=1, act=act)
        self.refine = ConvNormAct(int(channels), int(channels), kernel_size=3, stride=1, groups=int(channels), act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.factor > 1:
            x = F.pixel_unshuffle(x, self.factor)
        x = self.reduce(x)
        x = self.refine(x)
        return x


class MKB(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 7,
        use_global_branch: bool = True,
        expand_ratio: float = 0.5,
        act: str = "silu",
    ) -> None:
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1

        hidden = max(16, int(out_channels * float(expand_ratio)))
        self.pre = ConvNormAct(in_channels, hidden, kernel_size=1, stride=1, act=act)

        self.local = nn.Sequential(
            ConvNormAct(hidden, hidden, kernel_size=3, stride=1, groups=hidden, act=act),
            ConvNormAct(hidden, hidden, kernel_size=1, stride=1, act=act),
        )
        self.large = nn.Sequential(
            ConvNormAct(hidden, hidden, kernel_size=(1, kernel_size), stride=1, groups=hidden, act=act),
            ConvNormAct(hidden, hidden, kernel_size=(kernel_size, 1), stride=1, groups=hidden, act=act),
            ConvNormAct(hidden, hidden, kernel_size=1, stride=1, act=act),
        )

        self.use_global_branch = bool(use_global_branch)
        if self.use_global_branch:
            self.global_fc = nn.Conv2d(hidden, hidden, kernel_size=1, bias=True)

        self.post = ConvNormAct(hidden, out_channels, kernel_size=1, stride=1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        y = self.local(x) + self.large(x)
        if self.use_global_branch:
            g = torch.sigmoid(self.global_fc(F.adaptive_avg_pool2d(x, output_size=1)))
            y = y + x * g
        return self.post(y)


class P2DBFBridge(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        spd_factor: int = 2,
        use_mkb: bool = True,
        mkb_kernel_size: int = 7,
        mkb_use_global_branch: bool = True,
        mkb_expand_ratio: float = 0.5,
        res_scale_init: float = 0.1,
        act: str = "silu",
    ) -> None:
        super().__init__()
        channels = int(channels)
        self.p2_mapper = SPDConv2x(channels, factor=spd_factor, act=act)
        self.use_mkb = bool(use_mkb)
        if self.use_mkb:
            self.fuse = MKB(
                in_channels=2 * channels,
                out_channels=channels,
                kernel_size=mkb_kernel_size,
                use_global_branch=mkb_use_global_branch,
                expand_ratio=mkb_expand_ratio,
                act=act,
            )
        else:
            self.fuse = ConvNormAct(2 * channels, channels, kernel_size=1, stride=1, act=act)

        self.delta = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.res_scale = nn.Parameter(torch.tensor(float(res_scale_init), dtype=torch.float32))
        self._init_identity()

    def _init_identity(self) -> None:
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, x_p2: torch.Tensor, x_p3: torch.Tensor) -> torch.Tensor:
        if x_p2.ndim != 4 or x_p3.ndim != 4:
            raise ValueError(f"P2DBFBridge expects BCHW tensors, got p2={x_p2.shape}, p3={x_p3.shape}")
        b2, c2, _, _ = x_p2.shape
        b3, c3, h3, w3 = x_p3.shape
        if b2 != b3 or c2 != c3:
            raise ValueError(f"P2DBFBridge expects matching batch/channels, got p2={x_p2.shape}, p3={x_p3.shape}")

        p2_to_p3 = self.p2_mapper(x_p2)
        if p2_to_p3.shape[-2:] != (h3, w3):
            p2_to_p3 = F.interpolate(p2_to_p3, size=(h3, w3), mode="bilinear", align_corners=False)

        fused = self.fuse(torch.cat([p2_to_p3, x_p3], dim=1))
        delta = self.delta(fused)
        return x_p3 + self.res_scale * delta
