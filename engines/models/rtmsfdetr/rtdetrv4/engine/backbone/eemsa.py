from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_act(name: str) -> nn.Module:
    act = str(name).strip().lower()
    if act in {"relu"}:
        return nn.ReLU(inplace=True)
    if act in {"silu", "swish"}:
        return nn.SiLU(inplace=True)
    if act in {"gelu"}:
        return nn.GELU()
    if act in {"identity", "none", ""}:
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


def _make_norm(kind: str, channels: int) -> nn.Module:
    k = str(kind).strip().lower()
    if k in {"bn", "batchnorm", "batch_norm"}:
        return nn.BatchNorm2d(channels)
    if k in {"gn", "groupnorm", "group_norm"}:
        groups = min(32, channels)
        while groups > 1 and channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if k in {"identity", "none", ""}:
        return nn.Identity()
    raise ValueError(f"Unsupported norm: {kind}")


class ECALayer(nn.Module):
    """Efficient Channel Attention (ECA) for 2D feature maps."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        k = int(kernel_size)
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, 1, c)
        y = self.conv1d(y)
        y = self.act(y).view(b, c, 1, 1)
        return x * y


def _make_divisible(v: int, divisor: int = 16) -> int:
    if divisor <= 0:
        return int(v)
    return max(divisor, int(math.ceil(float(v) / float(divisor)) * divisor))


def _resolve_ema_groups(channels: int, groups: int | None) -> int:
    g = int(groups) if groups is not None else 8
    if g <= 0:
        g = 1
    g = min(g, int(channels))
    while g > 1 and (channels % g != 0):
        g -= 1
    return max(1, g)


class _EMASpatialAttention(nn.Module):
    """EMA-style spatial attention with grouped cross-spatial interaction."""

    def __init__(
        self,
        channels: int,
        *,
        groups: int = 8,
        conv_kernel_size: int = 3,
        use_group_norm: bool = True,
    ) -> None:
        super().__init__()
        c = int(channels)
        if c <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")

        k = int(conv_kernel_size)
        if k <= 0 or k % 2 == 0:
            raise ValueError(f"conv_kernel_size must be odd and > 0, got {conv_kernel_size}")

        self.groups = _resolve_ema_groups(c, groups)
        cg = c // self.groups
        if cg <= 0:
            raise ValueError(f"Invalid EMA grouped channels: channels={c}, groups={self.groups}")

        self.conv1x1 = nn.Conv2d(cg, cg, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv3x3 = nn.Conv2d(cg, cg, kernel_size=k, stride=1, padding=k // 2, bias=True)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.gn = nn.GroupNorm(1, cg) if bool(use_group_norm) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        g = self.groups
        cg = c // g

        xg = x.reshape(b * g, cg, h, w)
        x_h = xg.mean(dim=3, keepdim=True)
        x_w = xg.mean(dim=2, keepdim=True).transpose(2, 3)

        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x_w = x_w.transpose(2, 3)

        x1 = self.gn(xg * torch.sigmoid(x_h) * torch.sigmoid(x_w))
        x2 = self.conv3x3(xg)

        x11 = F.softmax(self.agp(x1).reshape(b * g, cg, 1).transpose(1, 2), dim=-1)
        x12 = x2.reshape(b * g, cg, h * w)
        x21 = F.softmax(self.agp(x2).reshape(b * g, cg, 1).transpose(1, 2), dim=-1)
        x22 = x1.reshape(b * g, cg, h * w)

        weights = (torch.bmm(x11, x12) + torch.bmm(x21, x22)).reshape(b * g, 1, h, w)
        out = (xg * torch.sigmoid(weights)).reshape(b, c, h, w)
        return out


class EEMSA(nn.Module):
    """Edge-Enhanced Multi-Scale Attention: reduce -> (Edge + EMA) -> fusion -> (ECA) -> expand -> residual."""

    def __init__(
        self,
        channels: int,
        *,
        ratio: float = 0.25,
        min_channels: int = 32,
        edge_dw_kernel_size: int = 3,
        edge_use_pointwise: bool = True,
        ema_groups: int = 8,
        ema_conv_kernel_size: int = 3,
        ema_use_group_norm: bool = True,
        fusion: str = "weighted_sum",
        norm: str = "bn",
        act: str = "silu",
        use_eca: bool = True,
        eca_kernel_size: int = 3,
        alpha_init: float = 0.1,
    ) -> None:
        super().__init__()
        c = int(channels)
        if c <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")

        r = float(ratio)
        if r <= 0:
            raise ValueError(f"ratio must be > 0, got {ratio}")

        c_mid = _make_divisible(max(int(min_channels), int(round(c * r))), 16)
        c_mid = min(max(16, c_mid), c)

        k = int(edge_dw_kernel_size)
        if k <= 0 or k % 2 == 0:
            raise ValueError(f"edge_dw_kernel_size must be odd and > 0, got {edge_dw_kernel_size}")

        self.reduce = nn.Conv2d(c, c_mid, kernel_size=1, stride=1, padding=0, bias=False)

        self.edge_dw = nn.Conv2d(c_mid, c_mid, kernel_size=k, stride=1, padding=k // 2, groups=c_mid, bias=False)
        self.edge_dw_norm = _make_norm(norm, c_mid)
        self.edge_dw_act = _make_act(act)

        self.edge_use_pointwise = bool(edge_use_pointwise)
        if self.edge_use_pointwise:
            self.edge_pw = nn.Conv2d(c_mid, c_mid, kernel_size=1, stride=1, padding=0, bias=False)
            self.edge_pw_norm = _make_norm(norm, c_mid)
            self.edge_pw_act = _make_act(act)
        else:
            self.edge_pw = nn.Identity()
            self.edge_pw_norm = nn.Identity()
            self.edge_pw_act = nn.Identity()

        self.ema = _EMASpatialAttention(
            c_mid,
            groups=int(ema_groups),
            conv_kernel_size=int(ema_conv_kernel_size),
            use_group_norm=bool(ema_use_group_norm),
        )

        fmode = str(fusion).strip().lower()
        if fmode in {"weighted", "weighted_sum", "wsum", "sum", "gated_sum"}:
            self.fusion_mode = "weighted_sum"
            self.fusion_logits = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float32))
            self.fuse_proj = None
        elif fmode in {"concat1x1", "concat", "cat1x1"}:
            self.fusion_mode = "concat1x1"
            self.fusion_logits = None
            self.fuse_proj = nn.Conv2d(2 * c_mid, c_mid, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            raise ValueError(f"Unsupported fusion mode: {fusion}")

        self.fuse_norm = _make_norm(norm, c_mid)
        self.fuse_act = _make_act(act)
        self.eca = ECALayer(c_mid, kernel_size=int(eca_kernel_size)) if bool(use_eca) else nn.Identity()
        self.expand = nn.Conv2d(c_mid, c, kernel_size=1, stride=1, padding=0, bias=False)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.reduce(x)

        edge = self.edge_dw_act(self.edge_dw_norm(self.edge_dw(feat)))
        edge = self.edge_pw_act(self.edge_pw_norm(self.edge_pw(edge)))

        ema = self.ema(feat)

        if self.fusion_mode == "weighted_sum":
            assert self.fusion_logits is not None
            w = F.softmax(self.fusion_logits, dim=0).to(dtype=feat.dtype, device=feat.device)
            fused = w[0] * edge + w[1] * ema
        else:
            assert self.fuse_proj is not None
            fused = self.fuse_proj(torch.cat([edge, ema], dim=1))

        fused = self.fuse_act(self.fuse_norm(fused))
        fused = self.eca(fused)
        fused = self.expand(fused)

        alpha = self.alpha.to(dtype=x.dtype, device=x.device)
        return x + alpha * fused


__all__ = ["EEMSA"]
