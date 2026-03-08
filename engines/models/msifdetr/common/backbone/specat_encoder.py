# ------------------------------------------------------------------------
# SPECAT Encoder (dependency-free, encoder-only)
# ------------------------------------------------------------------------
# 说明：
# - 该实现参考 third_party/SPECAT 的核心注意力结构（CAB / SSM_AB / FeedForward / MA），
#   但移除了对 einops/timm/scipy 的依赖，便于在本仓库训练环境中直接使用。
# - 仅实现“encoder-only”的特征提取，用于检测融合（不包含解码器/重建头）。
#
# 注意：
# - attention_type 支持 "base" 与 "full"（对应 third_party 的开关）；
# - "base" 模式不依赖 HSA/Swin 等额外模块，更适合作为检测侧的第一版落地。

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class GELU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        # x: [B,H,W,C]
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        hidden = int(dim * mult)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False, groups=hidden),
            GELU(),
            nn.Conv2d(hidden, dim, 1, 1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B,H,W,C] -> [B,C,H,W] -> [B,H,W,C]
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class MA(nn.Module):
    """Mask Attention（对应 third_party/SPECAT 的 optical filter-based 版本）。"""

    def __init__(self, n_feat: int):
        super().__init__()
        self.depth_conv = nn.Conv2d(
            n_feat, n_feat, kernel_size=5, padding=2, bias=True, groups=n_feat
        )

    def forward(self, mask_3d: Tensor) -> Tensor:
        attn_map = torch.sigmoid(self.depth_conv(mask_3d))
        res = mask_3d * attn_map
        return res + mask_3d


def _bnhd(x: Tensor, heads: int) -> Tensor:
    """
    将 [B, N, H*D] reshape 为 [B, H, N, D]，替代 einops.rearrange。
    """
    b, n, hd = x.shape
    if hd % heads != 0:
        raise ValueError(f"heads={heads} 不能整除最后维度 hd={hd}")
    d = hd // heads
    return x.view(b, n, heads, d).permute(0, 2, 1, 3).contiguous()


class SSMAB(nn.Module):
    """
    Spatial-Spectral-Mask Attention Block（对应 third_party/SPECAT 的 SSM_AB）。

    该注意力是“通道/光谱维度”的相关性建模：attn 的维度是 dim_head×dim_head。
    """

    def __init__(
        self,
        dim: int,
        *,
        dim_head: int = 64,
        heads: int = 8,
        attention_type: Literal["base", "full"] = "base",
    ):
        super().__init__()
        self.dim = int(dim)
        self.dim_head = int(dim_head)
        self.num_heads = int(heads)
        self.attention_type = str(attention_type)

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)

        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.ma = MA(dim)

        if self.attention_type not in {"base", "full"}:
            raise ValueError(f"不支持 attention_type={self.attention_type}")

        # full 模式额外引入空间注意力分支（HSA），用于增强跨空间的可用提示信号。
        if self.attention_type == "full":
            self.sa = HSA(dim=dim, num_heads=self.num_heads, window_size=8)
            self.sa_conv = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim, bias=False)

    def forward(self, x_in: Tensor, mask: Tensor) -> Tensor:
        """
        x_in: [B,H,W,C]
        mask: [B,H,W,C]（注意：这里的 mask 是 SPECAT 内部的 mask 特征，不是 padding mask）
        """
        b, h, w, c = x_in.shape
        if self.attention_type == "base":
            x = x_in.reshape(b, h * w, c)
        else:
            # full：加入空间注意力（对残余平移更鲁棒的“提示”来源之一）
            x_mid = x_in.permute(0, 3, 1, 2)
            x_mid_out = self.sa(x_mid)
            spa_g = mask.permute(0, 3, 1, 2)
            x_sa_emb = self.sa_conv(spa_g) + spa_g
            if x_sa_emb.shape[-2:] != x_mid_out.shape[-2:]:
                x_sa_emb = F.interpolate(
                    x_sa_emb, size=x_mid_out.shape[-2:], mode="bilinear", align_corners=False
                )
            x = (x_mid_out * x_sa_emb).permute(0, 2, 3, 1).reshape(b, h * w, c)

        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)

        mask_attn = self.ma(mask.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(
            b, h * w, c
        )

        q = _bnhd(q_inp, self.num_heads)
        k = _bnhd(k_inp, self.num_heads)
        v = _bnhd(v_inp, self.num_heads)
        mask_attn_h = _bnhd(mask_attn, self.num_heads)

        if self.attention_type == "full":
            v = v * mask_attn_h

        # q/k/v: [B, heads, N, D] -> 转置为 [B, heads, D, N]，对“空间维度”做归一化
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)

        # A = K^T * Q，得到 [B, heads, D, D] 的通道相关性注意力
        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)

        # [B, heads, D, N]
        x_attn = attn @ v
        x_attn = x_attn.permute(0, 3, 1, 2).reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x_attn).view(b, h, w, c)

        # 位置/局部增强分支（depthwise conv）
        out_p = self.pos_emb(v_inp.view(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return out_c + out_p


class DropPath(nn.Module):
    """DropPath（Stochastic Depth），避免引入 timm 依赖。"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def window_partition(x: Tensor, window_size: int) -> Tensor:
    # x: [B,H,W,C] -> [B*nW, Ws*Ws, C]
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, c)
    return windows


def window_reverse(windows: Tensor, window_size: int, h: int, w: int) -> Tensor:
    # windows: [B*nW, Ws*Ws, C] -> [B,H,W,C]
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention with relative position bias."""

    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.num_heads = int(num_heads)
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        ws = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * ws - 1) * (2 * ws - 1), num_heads)
        )

        coords_h = torch.arange(ws)
        coords_w = torch.arange(ws)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Ws, Ws
        coords_flatten = torch.flatten(coords, 1)  # 2, Ws*Ws
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, N, N
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N, N, 2
        relative_coords[:, :, 0] += ws - 1
        relative_coords[:, :, 1] += ws - 1
        relative_coords[:, :, 0] *= 2 * ws - 1
        relative_position_index = relative_coords.sum(-1)  # N, N
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        # x: [B*nW, N, C]
        b_, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b_, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B_, heads, N, dim]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [B_, heads, N, N]

        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            n, n, -1
        )
        bias = bias.permute(2, 0, 1).contiguous()  # [heads, N, N]
        attn = attn + bias.unsqueeze(0)

        if attn_mask is not None:
            # attn_mask: [nW, N, N]
            nw = attn_mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n)
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj(x)


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.mlp_ratio = float(mlp_ratio)

        if self.shift_size >= self.window_size:
            raise ValueError("shift_size must be < window_size")

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=self.num_heads)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), GELU(), nn.Linear(hidden, dim))

    @staticmethod
    def _calc_attn_mask(h: int, w: int, window_size: int, shift_size: int, device) -> Optional[Tensor]:
        if shift_size == 0:
            return None
        img_mask = torch.zeros((1, h, w, 1), device=device)
        cnt = 0
        h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, window_size)  # [nW, N, 1]
        mask_windows = mask_windows.view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x: Tensor) -> Tensor:
        # x: [B,H,W,C]
        b, h, w, c = x.shape
        ws = self.window_size
        ss = min(self.shift_size, ws - 1)

        pad_b = (ws - h % ws) % ws
        pad_r = (ws - w % ws) % ws
        if pad_b or pad_r:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        hp, wp = x.shape[1], x.shape[2]

        shortcut = x
        x = self.norm1(x)

        if ss > 0:
            shifted = torch.roll(x, shifts=(-ss, -ss), dims=(1, 2))
            attn_mask = self._calc_attn_mask(hp, wp, ws, ss, x.device)
        else:
            shifted = x
            attn_mask = None

        x_windows = window_partition(shifted, ws)  # [B*nW, N, C]
        attn_windows = self.attn(x_windows, attn_mask=attn_mask)
        shifted_back = window_reverse(attn_windows, ws, hp, wp)

        if ss > 0:
            x = torch.roll(shifted_back, shifts=(ss, ss), dims=(1, 2))
        else:
            x = shifted_back

        x = self.drop_path(x) + shortcut
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if pad_b or pad_r:
            x = x[:, :h, :w, :].contiguous()
        return x


class HSA(nn.Module):
    """
    Hierarchical Spatial Attention（对应 third_party/SPECAT 的 HSA）。
    - 使用两个 Swin block（W-MSA + SW-MSA）+ FFN。
    """

    def __init__(self, dim: int, *, num_heads: int, window_size: int = 8):
        super().__init__()
        self.wa = SwinTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
        )
        self.swa = SwinTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2,
        )
        self.pn = PreNorm(dim, FeedForward(dim=dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: [B,C,H,W] -> [B,H,W,C]
        x_hw = x.permute(0, 2, 3, 1)
        x_hw = self.wa(x_hw) + x_hw
        x_hw = self.swa(x_hw) + x_hw
        x_hw = self.pn(x_hw) + x_hw
        return x_hw.permute(0, 3, 1, 2).contiguous()


class CAB(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        dim_head: int = 64,
        heads: int = 8,
        num_blocks: int = 1,
        attention_type: Literal["base", "full"] = "base",
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        SSMAB(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            attention_type=attention_type,
                        ),
                        PreNorm(dim, FeedForward(dim=dim)),
                    ]
                )
                for _ in range(int(num_blocks))
            ]
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        # x/mask: [B,C,H,W]
        x_hw = x.permute(0, 2, 3, 1)
        mask_hw = mask.permute(0, 2, 3, 1)
        for attn, ff in self.blocks:
            x_hw = attn(x_hw, mask_hw) + x_hw
            x_hw = ff(x_hw) + x_hw
        return x_hw.permute(0, 3, 1, 2).contiguous()


@dataclass(frozen=True)
class SpecatEncoderConfig:
    in_channels: int = 7
    base_dim: int = 64
    stage: int = 3
    num_blocks: Sequence[int] = (2, 2, 2, 2)
    dim_head: Optional[int] = None
    attention_type: Literal["base", "full"] = "base"
    out_channels: int = 256


class SpecatEncoderBackbone(nn.Module):
    """
    SPECAT encoder-only backbone.

    输出：
    - feat: Tensor[B, out_channels, H/(2^stage), W/(2^stage)]
    """

    def __init__(self, cfg: SpecatEncoderConfig):
        super().__init__()
        self.cfg = cfg

        dim0 = int(cfg.base_dim)
        in_ch = int(cfg.in_channels)
        self.embedding = nn.Conv2d(in_ch, dim0, 3, 1, 1, bias=False)
        self.embedding2 = nn.Conv2d(in_ch, dim0, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        stage = int(cfg.stage)
        num_blocks = list(cfg.num_blocks)
        if len(num_blocks) < stage + 1:
            raise ValueError(
                f"SpecatEncoderConfig.num_blocks 长度需 >= stage+1（stage={stage}），但得到 {len(num_blocks)}"
            )

        dim_head = int(cfg.dim_head or cfg.base_dim)
        attention_type: Literal["base", "full"] = cfg.attention_type

        self.encoder_layers = nn.ModuleList()
        dim_stage = dim0
        for i in range(stage):
            heads = max(1, dim_stage // dim_head)
            self.encoder_layers.append(
                nn.ModuleList(
                    [
                        CAB(
                            dim=dim_stage,
                            num_blocks=int(num_blocks[i]),
                            dim_head=dim_head,
                            heads=heads,
                            attention_type=attention_type,
                        ),
                        nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
                        nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
                    ]
                )
            )
            dim_stage *= 2

        heads = max(1, dim_stage // dim_head)
        self.bottleneck = CAB(
            dim=dim_stage,
            dim_head=dim_head,
            heads=heads,
            num_blocks=int(num_blocks[stage]),
            attention_type=attention_type,
        )

        self.proj = nn.Sequential(
            nn.Conv2d(dim_stage, int(cfg.out_channels), 1, 1, bias=False),
            nn.GroupNorm(8 if int(cfg.out_channels) % 8 == 0 else 1, int(cfg.out_channels)),
        )

    def forward(self, x: Tensor) -> Tensor:
        # mask 特征（不是 padding mask）
        mask = self.lrelu(self.embedding2(x))
        fea = self.lrelu(self.embedding(x))

        for cab, fea_down, mask_down in self.encoder_layers:
            fea = cab(fea, mask)
            fea = fea_down(fea)
            mask = mask_down(mask)

        fea = self.bottleneck(fea, mask)
        return self.proj(fea)
