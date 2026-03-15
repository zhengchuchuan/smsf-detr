from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DeformableAlign2D"]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=3,
            stride=int(stride),
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(int(out_channels))
        self.conv2 = nn.Conv2d(int(out_channels), int(out_channels), kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(out_channels))

        self.shortcut = nn.Identity()
        if int(stride) != 1 or int(in_channels) != int(out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, stride=int(stride), bias=False),
                nn.BatchNorm2d(int(out_channels)),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        out_channels = int(out_channels)
        mid_channels = max(1, out_channels // int(self.expansion))

        self.conv1 = nn.Conv2d(int(in_channels), mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=int(stride), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()
        if int(stride) != 1 or int(in_channels) != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(int(in_channels), out_channels, kernel_size=1, stride=int(stride), bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class DeformableAlign2D(nn.Module):
    """
    Deformable alignment via dense offset prediction + grid_sample warping.

    Adapted from MRT-DETR's `DeformableModule` (offset_predict / presnet backbone).
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_keypoints: int = 5,
        offset_scale: float = 6.0,
        offset_enabled: bool = True,
        per_channel_offset: bool = False,
        attention_norm: str = "sigmoid",  # sigmoid|softmax
        padding_mode: str = "border",  # zeros|border|reflection
        align_corners: bool = True,
        loss_type: str = "cosine",  # cosine|infonce
        loss_downsample: float | None = None,
        nce_num_patches: int = 64,
        nce_patch_size: int = 5,
        nce_tau: float = 0.2,
        infonce_weight: float = 1.0,
        lncc_weight: float = 0.2,
        lncc_window_size: int = 5,
        lncc_eps: float = 1e-6,
        affine_enabled: bool = False,
        affine_scale: float = 0.1,
        affine_init_identity: bool = True,
        affine_per_channel: bool = False,
        affine_type: str = "affine",
    ) -> None:
        super().__init__()
        in_channels = int(in_channels)
        num_keypoints = int(num_keypoints)
        offset_scale = float(offset_scale)
        offset_enabled = bool(offset_enabled)
        affine_per_channel = bool(affine_per_channel)
        per_channel_offset = bool(per_channel_offset) or affine_per_channel
        attention_norm = str(attention_norm).strip().lower()
        padding_mode = str(padding_mode).strip().lower()
        align_corners = bool(align_corners)
        loss_type_norm = str(loss_type).strip().lower()
        if loss_type_norm in {"cos", "cosine", "cosine_similarity", "cos_sim"}:
            loss_type_norm = "cosine"
        elif loss_type_norm in {"infonce", "info_nce", "info-nce", "nce", "patch_nce", "patch-nce"}:
            loss_type_norm = "infonce"
        elif loss_type_norm in {"lncc", "local_ncc", "local-ncc", "ncc_local"}:
            loss_type_norm = "lncc"
        elif loss_type_norm in {
            "infonce_lncc",
            "info_nce_lncc",
            "info-nce-lncc",
            "infonce+lncc",
            "combo_infonce_lncc",
        }:
            loss_type_norm = "infonce_lncc"
        else:
            raise ValueError(f"Unsupported loss_type={loss_type} (supported: cosine|infonce|lncc|infonce_lncc)")
        loss_downsample_value = None if loss_downsample is None else float(loss_downsample)
        if loss_downsample_value is not None:
            if loss_downsample_value <= 0 or loss_downsample_value > 1.0:
                raise ValueError(f"loss_downsample must be in (0,1], got {loss_downsample}")
        nce_num_patches = int(nce_num_patches)
        if nce_num_patches <= 0:
            raise ValueError(f"nce_num_patches must be > 0, got {nce_num_patches}")
        nce_patch_size = int(nce_patch_size)
        if nce_patch_size <= 0:
            raise ValueError(f"nce_patch_size must be > 0, got {nce_patch_size}")
        if nce_patch_size % 2 == 0:
            nce_patch_size += 1
        nce_tau = float(nce_tau)
        if nce_tau <= 0:
            raise ValueError(f"nce_tau must be > 0, got {nce_tau}")
        infonce_weight = float(infonce_weight)
        lncc_weight = float(lncc_weight)
        if infonce_weight < 0:
            raise ValueError(f"infonce_weight must be >= 0, got {infonce_weight}")
        if lncc_weight < 0:
            raise ValueError(f"lncc_weight must be >= 0, got {lncc_weight}")
        lncc_window_size = int(lncc_window_size)
        if lncc_window_size <= 0:
            raise ValueError(f"lncc_window_size must be > 0, got {lncc_window_size}")
        if lncc_window_size % 2 == 0:
            lncc_window_size += 1
        lncc_eps = float(lncc_eps)
        if lncc_eps <= 0:
            raise ValueError(f"lncc_eps must be > 0, got {lncc_eps}")
        if loss_type_norm == "infonce_lncc" and infonce_weight <= 0 and lncc_weight <= 0:
            raise ValueError("infonce_lncc requires infonce_weight > 0 or lncc_weight > 0")
        affine_enabled = bool(affine_enabled)
        affine_scale = float(affine_scale)
        if affine_scale <= 0:
            raise ValueError(f"affine_scale must be > 0, got {affine_scale}")
        affine_init_identity = bool(affine_init_identity)
        affine_type_norm = str(affine_type).strip().lower()
        if affine_type_norm in {"affine", "full", "all"}:
            affine_type_norm = "affine"
        elif affine_type_norm in {"translation", "translate", "shift", "trans", "move"}:
            affine_type_norm = "translation"
        else:
            raise ValueError(f"Unsupported affine_type={affine_type} (supported: affine|translation)")

        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}")
        if num_keypoints <= 0:
            raise ValueError(f"num_keypoints must be > 0, got {num_keypoints}")
        if offset_enabled:
            if offset_scale <= 0:
                raise ValueError(f"offset_scale must be > 0, got {offset_scale}")
        else:
            if offset_scale < 0:
                raise ValueError(f"offset_scale must be >= 0 when offset_enabled is False, got {offset_scale}")
        if attention_norm not in {"sigmoid", "softmax"}:
            raise ValueError(f"Unsupported attention_norm={attention_norm} (supported: sigmoid|softmax)")
        if padding_mode not in {"zeros", "border", "reflection"}:
            raise ValueError(f"Unsupported padding_mode={padding_mode} (supported: zeros|border|reflection)")

        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.offset_scale = offset_scale
        self.offset_enabled = offset_enabled
        self.per_channel_offset = per_channel_offset
        self.attention_norm = attention_norm
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.loss_type = loss_type_norm
        self.loss_downsample = loss_downsample_value
        self.nce_num_patches = nce_num_patches
        self.nce_patch_size = nce_patch_size
        self.nce_tau = nce_tau
        self.infonce_weight = infonce_weight
        self.lncc_weight = lncc_weight
        self.lncc_window_size = lncc_window_size
        self.lncc_eps = lncc_eps
        self.affine_enabled = affine_enabled
        self.affine_scale = affine_scale
        self.affine_init_identity = affine_init_identity
        self.affine_per_channel = affine_per_channel
        self.affine_type = affine_type_norm

        self.offset_predict = nn.Sequential(
            # Keep bias=True to stay close to MRT-DETR's implementation.
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.res_block1 = ResidualBlock(in_channels, in_channels)
        self.bottle_block1 = BottleneckBlock(in_channels, in_channels)
        self.res_block2 = ResidualBlock(in_channels, in_channels)

        offset_channels = 2 * num_keypoints * in_channels if self.per_channel_offset else 2 * num_keypoints
        self.offset_head = nn.Conv2d(in_channels, offset_channels, kernel_size=3, padding=1, bias=True)
        self.attention_head = nn.Conv2d(in_channels, num_keypoints, kernel_size=3, padding=1, bias=True)
        if self.affine_enabled:
            self.affine_pool = nn.AdaptiveAvgPool2d(1)
            if self.affine_type == "translation":
                affine_out = 2 * (in_channels if self.affine_per_channel else 1)
            else:
                affine_out = 6 * (in_channels if self.affine_per_channel else 1)
            self.affine_head = nn.Linear(in_channels, int(affine_out), bias=True)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        # Start from identity warp: offset=0.
        nn.init.constant_(self.offset_head.weight, 0.0)
        if self.offset_head.bias is not None:
            nn.init.constant_(self.offset_head.bias, 0.0)

        # Start from near-uniform attention.
        nn.init.normal_(self.attention_head.weight, mean=0.0, std=0.01)
        if self.attention_head.bias is not None:
            if self.attention_norm == "sigmoid":
                nn.init.constant_(self.attention_head.bias, 1.0 / float(self.num_keypoints))
            else:
                nn.init.constant_(self.attention_head.bias, 0.0)
        if self.affine_enabled:
            nn.init.constant_(self.affine_head.weight, 0.0)
            if self.affine_head.bias is not None:
                nn.init.constant_(self.affine_head.bias, 0.0)

    @staticmethod
    def _create_base_grid(h: int, w: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid grid size: h={h} w={w}")
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, dtype=torch.float32, device=device),
            torch.linspace(-1, 1, w, dtype=torch.float32, device=device),
            indexing="ij",
        )
        grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0)  # (1, H, W, 2)
        return grid.to(dtype=dtype)

    def _predict_affine_theta(self, feat: torch.Tensor) -> torch.Tensor:
        pooled = self.affine_pool(feat).flatten(1)
        delta = self.affine_head(pooled)
        delta = torch.tanh(delta) * self.affine_scale
        b = delta.shape[0]
        if self.affine_per_channel:
            if self.affine_type == "translation":
                delta = delta.view(b, self.in_channels, 2)
                theta = torch.zeros(
                    (b, self.in_channels, 2, 3),
                    device=delta.device,
                    dtype=delta.dtype,
                )
                theta[..., 0, 0] = 1.0
                theta[..., 1, 1] = 1.0
                theta[..., 0, 2] = delta[..., 0]
                theta[..., 1, 2] = delta[..., 1]
            else:
                theta = delta.view(b, self.in_channels, 2, 3)
                if self.affine_init_identity:
                    identity = torch.tensor(
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        device=delta.device,
                        dtype=delta.dtype,
                    ).unsqueeze(0).unsqueeze(0)
                    theta = theta + identity
        else:
            if self.affine_type == "translation":
                delta = delta.view(b, 2)
                theta = torch.zeros(
                    (b, 2, 3),
                    device=delta.device,
                    dtype=delta.dtype,
                )
                theta[:, 0, 0] = 1.0
                theta[:, 1, 1] = 1.0
                theta[:, 0, 2] = delta[:, 0]
                theta[:, 1, 2] = delta[:, 1]
            else:
                theta = delta.view(b, 2, 3)
                if self.affine_init_identity:
                    identity = torch.tensor(
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        device=delta.device,
                        dtype=delta.dtype,
                    ).unsqueeze(0)
                    theta = theta + identity
        return theta

    def predict(
        self, x_ref: torch.Tensor, x_src: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x_ref.shape != x_src.shape:
            raise ValueError(f"DeformableAlign2D expects same shapes, got ref={x_ref.shape} src={x_src.shape}")
        if x_ref.ndim != 4:
            raise ValueError(f"DeformableAlign2D expects BCHW tensors, got {x_ref.shape}")
        if x_ref.shape[1] != self.in_channels:
            raise ValueError(
                f"Channel mismatch: expected {self.in_channels}, got ref={x_ref.shape[1]} src={x_src.shape[1]}"
            )

        feat = None
        if self.offset_enabled or self.affine_enabled:
            offset_input = torch.cat([x_ref, x_src], dim=1)  # (B, 2C, H, W)
            feat = self.offset_predict(offset_input)
            feat = self.res_block1(feat)
            feat = self.bottle_block1(feat)
            feat = self.res_block2(feat)

        b, _, h, w = x_ref.shape
        if self.offset_enabled:
            assert feat is not None
            # (B, 2K, H, W) in "pixel" units, then normalize to [-1,1] grid scale.
            offset = self.offset_head(feat)
            offset = torch.tanh(offset) * float(self.offset_scale)

            denom_x = max(int(w) - 1, 1) / 2.0
            denom_y = max(int(h) - 1, 1) / 2.0
            if self.per_channel_offset:
                offset = offset.view(b, self.in_channels, self.num_keypoints, 2, h, w)
                offset_x = offset[:, :, :, 0]
                offset_y = offset[:, :, :, 1]
                offset_x = offset_x / float(denom_x)
                offset_y = offset_y / float(denom_y)
            else:
                offset = offset.view(b, self.num_keypoints, 2, h, w)
                offset_x = offset[:, :, 0]
                offset_y = offset[:, :, 1]
                offset_x = offset_x / float(denom_x)
                offset_y = offset_y / float(denom_y)

            attn = self.attention_head(feat)
            if self.attention_norm == "sigmoid":
                attn = torch.sigmoid(attn)
            else:
                attn = torch.softmax(attn, dim=1)
        else:
            if self.per_channel_offset:
                offset_x = x_ref.new_zeros((b, self.in_channels, self.num_keypoints, h, w))
                offset_y = x_ref.new_zeros((b, self.in_channels, self.num_keypoints, h, w))
            else:
                offset_x = x_ref.new_zeros((b, self.num_keypoints, h, w))
                offset_y = x_ref.new_zeros((b, self.num_keypoints, h, w))
            attn = x_ref.new_full((b, self.num_keypoints, h, w), 1.0 / float(self.num_keypoints))

        if self.affine_enabled:
            if feat is None:
                offset_input = torch.cat([x_ref, x_src], dim=1)
                feat = self.offset_predict(offset_input)
                feat = self.res_block1(feat)
                feat = self.bottle_block1(feat)
                feat = self.res_block2(feat)
            affine_theta = self._predict_affine_theta(feat)
            return offset_x, offset_y, attn, affine_theta
        return offset_x, offset_y, attn

    def warp(
        self,
        x: torch.Tensor,
        *,
        offset_x: torch.Tensor,
        offset_y: torch.Tensor,
        attention: torch.Tensor,
        affine_theta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"DeformableAlign2D.warp expects BCHW tensor, got {x.shape}")
        b, c, h, w = x.shape
        if self.per_channel_offset:
            k = int(offset_x.shape[2])
            if offset_x.shape != (b, c, k, h, w) or offset_y.shape != (b, c, k, h, w):
                raise ValueError(
                    "offset_x/offset_y shape mismatch: "
                    f"x={x.shape} offset_x={offset_x.shape} offset_y={offset_y.shape}"
                )
        else:
            k = int(offset_x.shape[1])
            if offset_x.shape != (b, k, h, w) or offset_y.shape != (b, k, h, w):
                raise ValueError(
                    "offset_x/offset_y shape mismatch: "
                    f"x={x.shape} offset_x={offset_x.shape} offset_y={offset_y.shape}"
                )
        if attention.shape not in {(b, k, h, w), (b, c, k, h, w)}:
            raise ValueError(f"attention shape mismatch: x={x.shape} attention={attention.shape}")

        fused, _, _ = self.deform_with_attention(
            x,
            offset_x=offset_x,
            offset_y=offset_y,
            attention_weights=attention,
            affine_theta=affine_theta,
        )
        return fused

    def deform_with_attention(
        self,
        feature_map: torch.Tensor,
        *,
        offset_x: torch.Tensor,
        offset_y: torch.Tensor,
        attention_weights: torch.Tensor,
        affine_theta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MRT-DETR style API: warp + (optional) return sampled features + expanded attention weights.

        Returns:
        - fused_features: (B, C, H, W)
        - sampled_features: (B, K, C, H, W)
        - attention_weights: (B, K, 1, H, W)
        """
        if feature_map.ndim != 4:
            raise ValueError(f"DeformableAlign2D.deform_with_attention expects BCHW tensor, got {feature_map.shape}")
        b, c, h, w = feature_map.shape
        if self.per_channel_offset:
            k = int(offset_x.shape[2])
            if offset_x.shape != (b, c, k, h, w) or offset_y.shape != (b, c, k, h, w):
                raise ValueError(
                    "offset_x/offset_y shape mismatch: "
                    f"feature_map={feature_map.shape} offset_x={offset_x.shape} offset_y={offset_y.shape}"
                )
            if attention_weights.shape not in {(b, k, h, w), (b, c, k, h, w)}:
                raise ValueError(
                    f"attention_weights shape mismatch: feature_map={feature_map.shape} attn={attention_weights.shape}"
                )
        else:
            k = int(offset_x.shape[1])
            if offset_x.shape != (b, k, h, w) or offset_y.shape != (b, k, h, w):
                raise ValueError(
                    "offset_x/offset_y shape mismatch: "
                    f"feature_map={feature_map.shape} offset_x={offset_x.shape} offset_y={offset_y.shape}"
                )
            if attention_weights.shape != (b, k, h, w):
                raise ValueError(
                    f"attention_weights shape mismatch: feature_map={feature_map.shape} attn={attention_weights.shape}"
                )

        if affine_theta is not None:
            if affine_theta.ndim == 4:
                if affine_theta.shape[0] != b or affine_theta.shape[1] != c:
                    raise ValueError(
                        "affine_theta shape mismatch for per-channel affine: "
                        f"expected (B,C,2,3)=({b},{c},2,3), got {affine_theta.shape}"
                    )
                theta = affine_theta.to(dtype=feature_map.dtype).reshape(b * c, 2, 3)
                base_grid = F.affine_grid(
                    theta,
                    size=(b * c, 1, h, w),
                    align_corners=self.align_corners,
                ).view(b, c, h, w, 2)
            else:
                base_grid = F.affine_grid(
                    affine_theta.to(dtype=feature_map.dtype),
                    size=(b, 1, h, w),
                    align_corners=self.align_corners,
                )
        else:
            base_grid = self._create_base_grid(h, w, device=feature_map.device, dtype=feature_map.dtype)
            if base_grid.shape[0] != b:
                base_grid = base_grid.expand(b, -1, -1, -1)
        if self.per_channel_offset:
            if base_grid.ndim == 5:
                grid_x = base_grid[..., 0].unsqueeze(2)
                grid_y = base_grid[..., 1].unsqueeze(2)
            else:
                grid_x = base_grid[..., 0].reshape(b, 1, 1, h, w)
                grid_y = base_grid[..., 1].reshape(b, 1, 1, h, w)
            grid_x = grid_x + offset_x.to(dtype=feature_map.dtype)
            grid_y = grid_y + offset_y.to(dtype=feature_map.dtype)
            sampling_grid = torch.stack((grid_x, grid_y), dim=-1).clamp(-1, 1)  # (B,C,K,H,W,2)

            feature_map_expanded = feature_map.unsqueeze(2).expand(b, c, k, h, w).reshape(b * c * k, 1, h, w)
            sampling_grid_reshaped = sampling_grid.reshape(b * c * k, h, w, 2)
            sampled_features = F.grid_sample(
                feature_map_expanded,
                sampling_grid_reshaped,
                mode="bilinear",
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            ).reshape(b, c, k, h, w).permute(0, 2, 1, 3, 4)
            if attention_weights.shape == (b, c, k, h, w):
                attn = attention_weights.to(dtype=feature_map.dtype).permute(0, 2, 1, 3, 4)
            else:
                attn = attention_weights.to(dtype=feature_map.dtype).unsqueeze(2)  # (B,K,1,H,W)
        else:
            if base_grid.ndim == 5:
                raise ValueError(
                    "Per-channel affine requires per_channel_offset=True to match sampling grid shape."
                )
            grid_x = base_grid[..., 0].unsqueeze(1).expand(b, k, h, w)
            grid_y = base_grid[..., 1].unsqueeze(1).expand(b, k, h, w)

            grid_x = grid_x + offset_x.to(dtype=feature_map.dtype)
            grid_y = grid_y + offset_y.to(dtype=feature_map.dtype)
            sampling_grid = torch.stack((grid_x, grid_y), dim=-1).clamp(-1, 1)  # (B,K,H,W,2)

            feature_map_expanded = feature_map.unsqueeze(1).expand(b, k, c, h, w).reshape(b * k, c, h, w)
            sampling_grid_reshaped = sampling_grid.reshape(b * k, h, w, 2)
            sampled_features = F.grid_sample(
                feature_map_expanded,
                sampling_grid_reshaped,
                mode="bilinear",
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            ).reshape(b, k, c, h, w)
            attn = attention_weights.to(dtype=feature_map.dtype).unsqueeze(2)  # (B,K,1,H,W)

        fused_features = (sampled_features * attn).sum(dim=1)  # (B,C,H,W)
        return fused_features, sampled_features, attn

    def _maybe_downsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.loss_downsample is None or self.loss_downsample == 1.0:
            return x
        scale = float(self.loss_downsample)
        h = max(1, int(round(x.shape[2] * scale)))
        w = max(1, int(round(x.shape[3] * scale)))
        if h == x.shape[2] and w == x.shape[3]:
            return x
        return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=self.align_corners)

    @staticmethod
    def _gather_by_index(feat: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        b, c, _ = feat.shape
        idx = indices.unsqueeze(1).expand(b, c, indices.shape[1])
        return torch.gather(feat, 2, idx)

    def _info_nce_loss(self, x_ref: torch.Tensor, x_aligned: torch.Tensor) -> torch.Tensor:
        x_ref = self._maybe_downsample(x_ref)
        x_aligned = self._maybe_downsample(x_aligned)
        patch = min(self.nce_patch_size, x_ref.shape[2], x_ref.shape[3])
        if patch % 2 == 0:
            patch = max(1, patch - 1)
        if patch > 1:
            x_ref = F.avg_pool2d(x_ref, kernel_size=patch, stride=1, padding=patch // 2)
            x_aligned = F.avg_pool2d(x_aligned, kernel_size=patch, stride=1, padding=patch // 2)
        b, c, h, w = x_ref.shape
        total = h * w
        if total <= 0:
            return x_ref.new_tensor(0.0)
        num_samples = min(self.nce_num_patches, total)
        indices = torch.randint(0, total, (b, num_samples), device=x_ref.device)
        ref_flat = x_ref.view(b, c, total)
        aligned_flat = x_aligned.view(b, c, total)
        ref = self._gather_by_index(ref_flat, indices).permute(0, 2, 1)
        aligned = self._gather_by_index(aligned_flat, indices).permute(0, 2, 1)
        ref = F.normalize(ref, dim=-1)
        aligned = F.normalize(aligned, dim=-1)
        ref = ref.reshape(-1, c)
        aligned = aligned.reshape(-1, c)
        logits = (ref @ aligned.t()).float() / self.nce_tau
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        return loss

    def _local_ncc_loss(self, x_ref: torch.Tensor, x_aligned: torch.Tensor) -> torch.Tensor:
        x_ref = self._maybe_downsample(x_ref)
        x_aligned = self._maybe_downsample(x_aligned)
        if x_ref.shape != x_aligned.shape:
            raise ValueError(f"LNCC expects same shapes, got ref={x_ref.shape} aligned={x_aligned.shape}")

        b, c, h, w = x_ref.shape
        win = min(int(self.lncc_window_size), int(h), int(w))
        if win <= 1:
            cosine_sim = F.cosine_similarity(x_aligned, x_ref, dim=1)
            return 1.0 - cosine_sim.mean()
        if win % 2 == 0:
            win = max(1, win - 1)

        pad = win // 2
        filt = torch.ones((c, 1, win, win), device=x_ref.device, dtype=x_ref.dtype)
        win_size = float(win * win)

        sum_x = F.conv2d(x_ref, filt, padding=pad, groups=c)
        sum_y = F.conv2d(x_aligned, filt, padding=pad, groups=c)
        sum_x2 = F.conv2d(x_ref * x_ref, filt, padding=pad, groups=c)
        sum_y2 = F.conv2d(x_aligned * x_aligned, filt, padding=pad, groups=c)
        sum_xy = F.conv2d(x_ref * x_aligned, filt, padding=pad, groups=c)

        mean_x = sum_x / win_size
        mean_y = sum_y / win_size

        cross = sum_xy - mean_y * sum_x - mean_x * sum_y + mean_x * mean_y * win_size
        var_x = sum_x2 - 2.0 * mean_x * sum_x + mean_x * mean_x * win_size
        var_y = sum_y2 - 2.0 * mean_y * sum_y + mean_y * mean_y * win_size

        denom = torch.sqrt(var_x.clamp_min(self.lncc_eps) * var_y.clamp_min(self.lncc_eps) + self.lncc_eps)
        cc = (cross / denom).clamp(-1.0, 1.0)
        return 1.0 - cc.mean()

    def loss_calculate(
        self,
        x_ref: torch.Tensor,
        offset_x: torch.Tensor,
        offset_y: torch.Tensor,
        x_aligned: torch.Tensor,
        attention_weights: torch.Tensor,
        affine_theta: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        MRT-DETR style alignment loss.

        Mirrors `third_party/MRT-DETR/.../presnet.py:loss_calculate`:
        - use cosine similarity or patch InfoNCE between aligned feature and reference feature.

        Note: offset_x/offset_y/attention_weights are accepted for API compatibility; current loss only uses x_ref/x_aligned.
        """
        _ = offset_x, offset_y, attention_weights, affine_theta
        aux_losses: dict[str, torch.Tensor] = {}
        if self.loss_type == "cosine":
            x_ref = self._maybe_downsample(x_ref)
            x_aligned = self._maybe_downsample(x_aligned)
            cosine_sim = F.cosine_similarity(x_aligned, x_ref, dim=1)  # (B,H,W)
            loss = 1.0 - cosine_sim.mean()
        elif self.loss_type == "infonce":
            loss = self._info_nce_loss(x_ref, x_aligned)
        elif self.loss_type == "lncc":
            loss = self._local_ncc_loss(x_ref, x_aligned)
        elif self.loss_type == "infonce_lncc":
            terms = []
            if self.infonce_weight > 0:
                terms.append(self.infonce_weight * self._info_nce_loss(x_ref, x_aligned))
            if self.lncc_weight > 0:
                terms.append(self.lncc_weight * self._local_ncc_loss(x_ref, x_aligned))
            if not terms:
                raise RuntimeError("infonce_lncc produced no active loss terms")
            loss = torch.stack(terms).sum()
        else:
            raise ValueError(f"Unsupported loss_type={self.loss_type}")
        aux_losses["loss_deform_align"] = loss
        return aux_losses

    def forward(self, x_ref: torch.Tensor, x_src: torch.Tensor) -> torch.Tensor:
        pred = self.predict(x_ref, x_src)
        if self.affine_enabled:
            offset_x, offset_y, attn, affine_theta = pred
            return self.warp(x_src, offset_x=offset_x, offset_y=offset_y, attention=attn, affine_theta=affine_theta)
        offset_x, offset_y, attn = pred
        return self.warp(x_src, offset_x=offset_x, offset_y=offset_y, attention=attn)
