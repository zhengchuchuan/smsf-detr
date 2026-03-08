from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from engines.models.base import BaseDetector
from utils.misc import NestedTensor


class RTDETRv4Detector(BaseDetector):
    """
    将 RT-DETRv4 的 RTv4 模型封装为本工程 BaseTrainer 兼容的接口。

    关键适配点：
    - 本工程数据管线可能输出两类 RGB 张量：
      1) ImageNet mean/std 归一化后的张量（范围通常在 [-2, 2] 左右）；
      2) 已在 [0, 1] 的线性/最值归一化张量（如 per_channel_minmax）。
      RT-DETRv4 原始训练期望输入为 [0, 1] float，因此在 forward 内支持“反归一化”到 [0, 1]；
      同时为避免与非 ImageNet 归一化重复处理，默认会根据输入数值范围自动跳过不必要的反归一化。
    - third_party 的 RTv4Criterion 在 eval 模式仍要求 outputs 包含 aux_outputs / enc_aux_outputs；
      因此在推理输出里补齐空列表，保证验证阶段可计算 loss。
    """

    supported_modalities = ("rgb", "ms", "rgb_ms")

    def __init__(
        self,
        model: nn.Module,
        teacher_model: nn.Module | None = None,
        *,
        expected_input_channels: int | None = None,
        rgb_channels: int = 3,
        input_denormalize: bool = True,
        rgb_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        rgb_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        clamp_after_denormalize: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.expected_input_channels = expected_input_channels
        self.rgb_channels = int(rgb_channels)
        self.input_denormalize = bool(input_denormalize)
        self.clamp_after_denormalize = bool(clamp_after_denormalize)

        # 注意：teacher_model 是“运行时依赖”，不应被写入 state_dict/checkpoint。
        # 使用 __dict__ 直接赋值避免被 nn.Module 注册为子模块。
        self.__dict__["_teacher_model"] = teacher_model

        mean = torch.tensor(list(rgb_mean), dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(list(rgb_std), dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_rgb_mean", mean, persistent=False)
        self.register_buffer("_rgb_std", std, persistent=False)

    @staticmethod
    def _needs_denormalize(images: torch.Tensor, *, rgb_channels: int = 3) -> bool:
        if not torch.is_floating_point(images):
            return False
        if rgb_channels <= 0:
            return False
        if images.ndim not in (3, 4):
            return True
        try:
            if images.ndim == 4 and images.shape[1] >= rgb_channels:
                probe = images[:, :rgb_channels]
            elif images.ndim == 3 and images.shape[0] >= rgb_channels:
                probe = images[:rgb_channels]
            else:
                probe = images
            img_min = float(probe.amin())
            img_max = float(probe.amax())
        except Exception:
            return True
        # ImageNet normalize 后通常会越界到 [0,1] 之外；而线性/最值归一化会落在 [0,1]。
        return img_min < -0.05 or img_max > 1.05

    @staticmethod
    def _extract_tensor(sample: Any) -> torch.Tensor:
        if isinstance(sample, NestedTensor) or hasattr(sample, "tensors"):
            return sample.tensors  # type: ignore[return-value]
        if isinstance(sample, torch.Tensor):
            return sample
        raise TypeError(f"RTDETRv4Detector 期望输入 Tensor/NestedTensor，实际为: {type(sample)}")

    def _prepare_images(self, samples: Any) -> torch.Tensor:
        if isinstance(samples, dict):
            # 参考 msifdetr 的多模态输入约定：优先按 rgb->ms 拼接，其它 key 作为附加通道追加。
            ordered_keys = []
            for key in ("rgb", "ms"):
                if key in samples:
                    ordered_keys.append(key)
            ordered_keys.extend([k for k in samples.keys() if k not in set(ordered_keys)])
            tensors = [self._extract_tensor(samples[k]) for k in ordered_keys]
            if not tensors:
                raise ValueError("samples 为空 dict，无法构造输入。")
            cat_dim = 1 if tensors[0].ndim == 4 else 0
            images = torch.cat(tensors, dim=cat_dim)
        else:
            images = self._extract_tensor(samples)

        if self.expected_input_channels is not None:
            if images.ndim == 4:
                actual = int(images.shape[1])
            elif images.ndim == 3:
                actual = int(images.shape[0])
            else:
                actual = -1
            if actual != int(self.expected_input_channels):
                raise ValueError(
                    "RTDETRv4Detector 输入通道数不匹配："
                    f"expected={self.expected_input_channels}, got={actual}. "
                    "请检查 data.channel_splits/dual_stream_output 与 HGNetv2.in_chs 配置是否一致。"
                )

        if self.rgb_channels <= 0:
            return images

        if self.input_denormalize and self._needs_denormalize(images, rgb_channels=self.rgb_channels):
            if images.ndim == 4:
                if images.shape[1] < self.rgb_channels:
                    raise ValueError(
                        f"期望至少 {self.rgb_channels} 个 RGB 通道以执行反归一化，但输入为 {images.shape}。"
                    )
                images = images.clone()
                rgb = images[:, : self.rgb_channels]
                rgb = rgb * self._rgb_std + self._rgb_mean
                if self.clamp_after_denormalize:
                    rgb = rgb.clamp(0.0, 1.0)
                images[:, : self.rgb_channels] = rgb
            else:
                # (C,H,W)
                if images.shape[0] < self.rgb_channels:
                    raise ValueError(
                        f"期望至少 {self.rgb_channels} 个 RGB 通道以执行反归一化，但输入为 {images.shape}。"
                    )
                images = images.clone()
                rgb_mean = self._rgb_mean.view(3, 1, 1)
                rgb_std = self._rgb_std.view(3, 1, 1)
                rgb = images[: self.rgb_channels]
                rgb = rgb * rgb_std + rgb_mean
                if self.clamp_after_denormalize:
                    rgb = rgb.clamp(0.0, 1.0)
                images[: self.rgb_channels] = rgb
        return images

    def forward(self, samples: Any, targets=None):
        images = self._prepare_images(samples)
        teacher_model = self.__dict__.get("_teacher_model")
        if self.training and teacher_model is not None:
            if self.rgb_channels <= 0:
                raise ValueError("已启用 teacher 蒸馏，但 rgb_channels<=0，无法为 teacher 提供 RGB 输入。")
            teacher_images = images[:, : self.rgb_channels] if images.ndim == 4 else images[: self.rgb_channels]
            with torch.no_grad():
                teacher_encoder_output = teacher_model(teacher_images).detach()
            outputs = self.model(
                images,
                targets=targets,
                teacher_encoder_output=teacher_encoder_output,
            )
        else:
            outputs = self.model(images, targets=targets)
        if not isinstance(outputs, dict):
            raise TypeError(f"RT-DETRv4 输出应为 dict，实际为: {type(outputs)}")

        # RTv4Criterion 在 forward 内假定 aux_outputs/enc_aux_outputs 存在；推理模式下补齐空列表。
        outputs.setdefault("aux_outputs", [])
        if "enc_aux_outputs" not in outputs:
            outputs["enc_aux_outputs"] = []
        if "enc_meta" not in outputs:
            outputs["enc_meta"] = {"class_agnostic": False}
        return outputs


__all__ = ["RTDETRv4Detector"]
