from __future__ import annotations

"""
Grad-CAM 可视化脚本（RTMSF-DETR / RT-DETRv4 wrapper）。

目标：
- 读取 outputs/**/config.yaml + checkpoint_best.pth；
- 从 val/test 数据集中抽样，前向推理后对某个 detection logit 做梯度回传；
- 在指定层（默认 model.backbone 的最后一级特征）上计算 Grad-CAM，并叠加到 RGB 图像保存。

示例（按你给的 run_dir）：
  python tools/gradcam_rtmsfdetr.py \\
    --config outputs/oil_rgb_msi_20260115_3cls/rtmsfdetr/.../config.yaml \\
    --checkpoint outputs/oil_rgb_msi_20260115_3cls/rtmsfdetr/.../checkpoint_best.pth \\
    --device cuda \\
    --split val \\
    --sample-idx 0 \\
    --num-samples 8 \\
    --target-layer model.backbone \\
    --feature-index -1 \\
    --alpha 0.45

如需挑 layer：
  python tools/gradcam_rtmsfdetr.py --config ... --list-layers --layer-filter backbone
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError("需要安装 opencv-python 才能进行热力图上色/叠加。") from exc


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _pick_device(device: str) -> torch.device:
    d = str(device).strip().lower()
    if d.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("请求使用 CUDA，但当前不可用，自动回退到 CPU。")
        return torch.device("cpu")
    return torch.device(d)


def _load_config_any(
    config: str | Path,
    *,
    config_dir: str | Path,
    overrides: list[str] | None,
) -> Any:
    """
    支持两种输入：
    1) `configs/` 目录内的 Hydra 配置（走 load_config + defaults 组合）
    2) `outputs/**/config.yaml` 这类“已保存的完整配置”（直接 OmegaConf.load）
    """
    config_path = _as_path(config).expanduser()
    if config_path.is_file():
        cfg = OmegaConf.load(str(config_path))
        OmegaConf.set_struct(cfg, False)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
            OmegaConf.set_struct(cfg, False)
        return cfg
    from engines.core.parse_config import load_config

    return load_config(config, config_dir=_as_path(config_dir), overrides=overrides)


def _load_checkpoint(path: Path, *, weights_only: bool) -> Any:
    return torch.load(path, map_location="cpu", weights_only=weights_only)


def _extract_state_dict(checkpoint: Any, *, prefer_ema: bool) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        if prefer_ema and checkpoint.get("ema") is not None:
            ema_state = checkpoint.get("ema")
            if hasattr(ema_state, "state_dict"):
                ema_state = ema_state.state_dict()
            if isinstance(ema_state, Mapping):
                return ema_state  # type: ignore[return-value]
        model_state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    else:
        model_state = checkpoint
    if hasattr(model_state, "state_dict"):
        model_state = model_state.state_dict()
    if not isinstance(model_state, Mapping):
        raise TypeError(f"无法从 checkpoint 解析 state_dict，得到类型={type(model_state)}")
    return model_state  # type: ignore[return-value]


def _filter_compatible_state_dict(
    model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor]
) -> Mapping[str, torch.Tensor]:
    """仅保留 shape 与当前模型一致的参数，避免 strict=False 仍因 shape mismatch 抛错。"""
    model_sd = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k not in model_sd:
            continue
        if model_sd[k].shape != v.shape:
            continue
        filtered[k] = v
    return filtered


def _resolve_named_module(root: torch.nn.Module, name: str) -> torch.nn.Module:
    name = str(name or "").strip()
    if name == "" or name == ".":
        return root

    modules = dict(root.named_modules())
    if name in modules:
        return modules[name]

    # 允许用户只写末尾片段（如 "backbone"），但必须唯一。
    candidates = [k for k in modules.keys() if k.endswith(name)]
    if len(candidates) == 1:
        return modules[candidates[0]]
    if not candidates:
        raise KeyError(f"未找到 layer: {name}")
    cand_preview = "\n  - " + "\n  - ".join(sorted(candidates)[:50])
    raise KeyError(f"layer={name} 不唯一，候选有 {len(candidates)} 个，请指定更精确的名字。候选示例：{cand_preview}")


@dataclass(frozen=True)
class DetTarget:
    query_idx: int
    class_idx: int
    score: float
    box_xyxy: tuple[float, float, float, float]


class GradCAMExtractor:
    def __init__(self, model: torch.nn.Module, *, target_layer: torch.nn.Module, feature_index: int = -1) -> None:
        self.model = model
        self.target_layer = target_layer
        self.feature_index = int(feature_index)
        self._handle: Any | None = None
        self.activations: torch.Tensor | None = None  # [B,C,H,W]

    def _forward_hook(self, _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        act: Any = output
        if isinstance(act, (list, tuple)):
            if not act:
                raise ValueError("target_layer 输出为空 list/tuple，无法计算 Grad-CAM。")
            act = act[self.feature_index]
        if isinstance(act, dict):
            raise TypeError(
                "target_layer 输出为 dict，当前脚本未实现从 dict 里选 activation。"
                "请把 --target-layer 指向某个返回 Tensor 或 list[Tensor] 的模块。"
            )
        if not torch.is_tensor(act):
            raise TypeError(f"target_layer 输出类型不支持: {type(act)}")
        if act.ndim != 4:
            raise ValueError(f"target_layer 输出维度需要为 [B,C,H,W]，当前 shape={tuple(act.shape)}")
        self.activations = act

    def __enter__(self) -> "GradCAMExtractor":
        self.activations = None
        self._handle = self.target_layer.register_forward_hook(self._forward_hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            try:
                self._handle.remove()
            except Exception:
                pass
        self._handle = None

    def compute_cam(
        self,
        *,
        target_score: torch.Tensor,
        input_size_hw: tuple[int, int],
        method: str = "gradcam",
    ) -> np.ndarray:
        if self.activations is None:
            raise RuntimeError("未捕获到 activations：请确认 target_layer 是否在前向中被调用。")

        # 只对 activations 求梯度，避免把所有参数的 grad 都存下来。
        grads = torch.autograd.grad(
            target_score,
            self.activations,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]

        acts = self.activations

        mode = str(method or "gradcam").strip().lower()
        if mode == "layercam":
            # LayerCAM: use positive gradients as spatial weights (more localized than global-average weights).
            cam = (F.relu(grads) * acts).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
        else:
            # Classic Grad-CAM.
            weights = grads.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
            cam = (weights * acts).sum(dim=1, keepdim=True)  # [B,1,H,W]
            cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_size_hw, mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)
        return cam.detach().float().cpu().numpy()


def _denormalize_rgb_for_vis(rgb_tensor: torch.Tensor, *, cfg: Any) -> np.ndarray:
    """
    输入：
      rgb_tensor: [3,H,W] float，通常是 ImageNet normalize 之后的张量
    输出：
      uint8 RGB, shape [H,W,3]
    """
    if rgb_tensor.ndim != 3 or rgb_tensor.shape[0] < 3:
        raise ValueError(f"rgb_tensor shape 需要为 [3,H,W]，当前={tuple(rgb_tensor.shape)}")

    mode = str(getattr(getattr(cfg, "data", None), "rgb_normalize_mode", "imagenet")).lower()
    rgb = rgb_tensor[:3].detach().float().cpu()

    if mode == "imagenet":
        mean = torch.tensor(getattr(getattr(cfg, "data", None), "rgb_mean", (0.485, 0.456, 0.406)), dtype=torch.float32)
        std = torch.tensor(getattr(getattr(cfg, "data", None), "rgb_std", (0.229, 0.224, 0.225)), dtype=torch.float32)
        rgb = rgb * std.view(3, 1, 1) + mean.view(3, 1, 1)
    # 其它模式（linear/image_max/per_channel_minmax）基本都已在 [0,1]，直接 clamp 即可。
    rgb = rgb.clamp(0.0, 1.0)

    rgb_np = (rgb.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return rgb_np


def _select_det_target(
    outputs: Mapping[str, torch.Tensor],
    *,
    num_classes: int,
    input_size_hw: tuple[int, int],
    target_class: int | None,
) -> DetTarget:
    logits = outputs["pred_logits"]  # [B,Q,C] or [B,Q,C+1]
    boxes = outputs["pred_boxes"]  # [B,Q,4] cxcywh normalized
    if logits.ndim != 3 or boxes.ndim != 3:
        raise ValueError(f"unexpected pred shapes: logits={tuple(logits.shape)} boxes={tuple(boxes.shape)}")

    b, q, c = logits.shape
    if b != 1:
        raise ValueError("当前脚本仅支持 batch_size=1（便于逐图可视化）。")

    # 推断 focal vs softmax-background
    if c == int(num_classes) + 1:
        # softmax, last class is background
        probs = logits.softmax(dim=-1)[..., :num_classes]
    else:
        probs = logits.sigmoid()
        if probs.shape[-1] != int(num_classes):
            # 兜底：如果 cfg/train.num_classes 与模型不一致，仍尽量走“按实际通道数”。
            num_classes = int(probs.shape[-1])
            probs = probs[..., :num_classes]

    probs0 = probs[0]  # [Q,C]
    if target_class is not None:
        cls = int(target_class)
        if cls < 0 or cls >= probs0.shape[1]:
            raise ValueError(f"--target-class 越界：{cls} not in [0, {probs0.shape[1]-1}]")
        score_per_q = probs0[:, cls]
        qidx = int(score_per_q.argmax().item())
        cidx = cls
        score = float(score_per_q[qidx].item())
    else:
        flat = probs0.flatten(0, 1)  # [Q*C]
        k = int(flat.argmax().item())
        qidx = int(k // probs0.shape[1])
        cidx = int(k % probs0.shape[1])
        score = float(flat[k].item())

    # box: cxcywh normalized -> xyxy pixel (on model input)
    box_cxcywh = boxes[0, qidx]
    cx, cy, w, h = [float(x) for x in box_cxcywh.detach().cpu().tolist()]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    H, W = input_size_hw
    x1 *= W
    x2 *= W
    y1 *= H
    y2 *= H
    x1 = max(0.0, min(float(W - 1), x1))
    x2 = max(0.0, min(float(W - 1), x2))
    y1 = max(0.0, min(float(H - 1), y1))
    y2 = max(0.0, min(float(H - 1), y2))

    return DetTarget(query_idx=qidx, class_idx=cidx, score=score, box_xyxy=(x1, y1, x2, y2))


def _cxcywh_to_xyxy_norm(boxes: torch.Tensor) -> torch.Tensor:
    """cxcywh (normalized) -> xyxy (normalized). boxes: [...,4]."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    IoU for xyxy boxes in normalized coords.
    boxes1: [N,4], boxes2: [M,4] -> [N,M]
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((int(boxes1.shape[0]), int(boxes2.shape[0])), device=boxes1.device, dtype=torch.float32)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))[:, None]
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))[None, :]
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def _box_cxcywh_to_xyxy_pixel(box_cxcywh: torch.Tensor, *, out_hw: tuple[int, int]) -> tuple[float, float, float, float]:
    """cxcywh normalized -> xyxy pixel."""
    cx, cy, w, h = [float(x) for x in box_cxcywh.detach().cpu().tolist()]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    H, W = out_hw
    x1 *= W
    x2 *= W
    y1 *= H
    y2 *= H
    x1 = max(0.0, min(float(W - 1), x1))
    x2 = max(0.0, min(float(W - 1), x2))
    y1 = max(0.0, min(float(H - 1), y1))
    y2 = max(0.0, min(float(H - 1), y2))
    return (x1, y1, x2, y2)


def _extract_gt_boxes_labels(target: Any) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not isinstance(target, Mapping):
        return None
    boxes = target.get("boxes", None)
    labels = target.get("labels", None)
    if not torch.is_tensor(boxes) or not torch.is_tensor(labels):
        return None
    if boxes.ndim != 2 or boxes.shape[-1] != 4 or labels.ndim != 1:
        return None
    return boxes, labels


def _select_queries_for_class_from_logits(
    pred_logits_qc: torch.Tensor,
    *,
    num_classes: int,
    target_class: int,
    topk: int,
    score_thr: float,
    allow_fallback: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Like _select_queries_for_class but operates on [Q,C] and supports skipping when below thr."""
    if pred_logits_qc.ndim != 2:
        raise ValueError(f"pred_logits expected [Q,C], got {tuple(pred_logits_qc.shape)}")
    q, c = pred_logits_qc.shape
    cls = int(target_class)
    if cls < 0:
        raise ValueError("target_class must be >=0")

    if c == int(num_classes) + 1:
        probs = pred_logits_qc.softmax(dim=-1)[:, :num_classes]
        if cls >= int(num_classes):
            raise ValueError(f"target_class out of range: {cls} (num_classes={num_classes})")
        prob_q = probs[:, cls]
    else:
        probs = pred_logits_qc.sigmoid()
        if cls >= probs.shape[-1]:
            raise ValueError(f"target_class out of range: {cls} (C={probs.shape[-1]})")
        prob_q = probs[:, cls]

    keep = prob_q >= float(score_thr) if float(score_thr) > 0 else torch.ones_like(prob_q, dtype=torch.bool)
    idx_all = torch.nonzero(keep, as_tuple=False).squeeze(1)
    if idx_all.numel() == 0:
        if not bool(allow_fallback):
            return None
        idx_all = torch.tensor([int(prob_q.argmax().item())], device=prob_q.device, dtype=torch.long)

    if int(topk) > 0 and idx_all.numel() > int(topk):
        sel = idx_all[torch.argsort(prob_q[idx_all])[-int(topk) :]]
    else:
        sel = idx_all
    sel = sel.sort().values
    return sel, prob_q[sel]


def _select_queries_by_gt_iou(
    pred_logits_qc: torch.Tensor,
    pred_boxes_q4: torch.Tensor,
    gt_boxes_cxcywh: torch.Tensor,
    *,
    num_classes: int,
    target_class: int,
    topk: int,
    score_thr: float,
    iou_thr: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """
    Select queries that match GT boxes of target_class by IoU, then take top-k by class prob.
    Returns (sel_q, sel_prob, sel_iou).
    """
    if pred_boxes_q4.ndim != 2 or pred_boxes_q4.shape[-1] != 4:
        raise ValueError(f"pred_boxes expected [Q,4], got {tuple(pred_boxes_q4.shape)}")
    if gt_boxes_cxcywh.ndim != 2 or gt_boxes_cxcywh.shape[-1] != 4:
        raise ValueError(f"gt_boxes expected [N,4], got {tuple(gt_boxes_cxcywh.shape)}")
    if gt_boxes_cxcywh.numel() == 0:
        return None

    picked = _select_queries_for_class_from_logits(
        pred_logits_qc,
        num_classes=num_classes,
        target_class=int(target_class),
        topk=0,  # do topk after IoU filtering
        score_thr=float(score_thr),
        allow_fallback=True,  # ensure prob_q is computable for fallback later
    )
    if picked is None:
        return None
    _, prob_q_all = picked
    # Above helper returns only selected entries; we need full prob_q. Recompute cheaply:
    if pred_logits_qc.shape[1] == int(num_classes) + 1:
        prob_q = pred_logits_qc.softmax(dim=-1)[:, :num_classes][:, int(target_class)]
    else:
        prob_q = pred_logits_qc.sigmoid()[:, int(target_class)]

    pred_xyxy = _cxcywh_to_xyxy_norm(pred_boxes_q4).clamp(0.0, 1.0)
    gt_xyxy = _cxcywh_to_xyxy_norm(gt_boxes_cxcywh).clamp(0.0, 1.0)
    iou = _box_iou_xyxy(pred_xyxy, gt_xyxy)  # [Q,Ng]
    best_iou_per_q = iou.max(dim=1).values if iou.numel() else torch.zeros((pred_boxes_q4.shape[0],), device=pred_boxes_q4.device)

    keep = best_iou_per_q >= float(iou_thr) if float(iou_thr) > 0 else torch.ones_like(best_iou_per_q, dtype=torch.bool)
    if float(score_thr) > 0:
        keep = keep & (prob_q >= float(score_thr))
    idx_all = torch.nonzero(keep, as_tuple=False).squeeze(1)
    if idx_all.numel() == 0:
        return None

    if int(topk) > 0 and idx_all.numel() > int(topk):
        sel = idx_all[torch.argsort(prob_q[idx_all])[-int(topk) :]]
    else:
        sel = idx_all
    sel = sel.sort().values
    return sel, prob_q[sel], best_iou_per_q[sel]


def _select_queries_for_class(
    outputs: Mapping[str, torch.Tensor],
    *,
    num_classes: int,
    target_class: int,
    topk: int,
    score_thr: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      (query_indices [K], probs [K]) for the given class.
    Selection is based on sigmoid/softmax probabilities (not logits).
    """
    logits = outputs["pred_logits"]  # [B,Q,C] or [B,Q,C+1]
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(f"pred_logits shape expected [1,Q,C], got {tuple(logits.shape)}")

    _, q, c = logits.shape
    cls = int(target_class)
    if cls < 0:
        raise ValueError("target_class must be >=0")

    if c == int(num_classes) + 1:
        probs = logits.softmax(dim=-1)[..., :num_classes]
        if cls >= int(num_classes):
            raise ValueError(f"target_class out of range: {cls} (num_classes={num_classes})")
        prob_q = probs[0, :, cls]
    else:
        probs = logits.sigmoid()
        if cls >= probs.shape[-1]:
            raise ValueError(f"target_class out of range: {cls} (C={probs.shape[-1]})")
        prob_q = probs[0, :, cls]

    keep = prob_q >= float(score_thr) if float(score_thr) > 0 else torch.ones_like(prob_q, dtype=torch.bool)
    idx_all = torch.nonzero(keep, as_tuple=False).squeeze(1)
    if idx_all.numel() == 0:
        # fallback to best query
        idx_all = torch.tensor([int(prob_q.argmax().item())], device=prob_q.device, dtype=torch.long)

    if int(topk) > 0 and idx_all.numel() > int(topk):
        sel = idx_all[torch.argsort(prob_q[idx_all])[-int(topk) :]]
    else:
        sel = idx_all

    sel = sel.sort().values  # stable order
    return sel, prob_q[sel]


def _overlay_cam(rgb_uint8: np.ndarray, cam01: np.ndarray, *, alpha: float) -> np.ndarray:
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError(f"rgb_uint8 must be HWC RGB, got shape={rgb_uint8.shape}")
    h, w = rgb_uint8.shape[:2]
    if cam01.shape != (h, w):
        cam01 = cv2.resize(cam01.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    cam_u8 = np.clip(cam01 * 255.0, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    alpha = float(alpha)
    alpha = 0.0 if alpha < 0.0 else 1.0 if alpha > 1.0 else alpha
    overlay = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0.0)
    return overlay


def _draw_box(
    img_bgr: np.ndarray,
    *,
    box_xyxy: tuple[float, float, float, float],
    label: str,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    x1 = max(0, min(img_bgr.shape[1] - 1, x1))
    x2 = max(0, min(img_bgr.shape[1] - 1, x2))
    y1 = max(0, min(img_bgr.shape[0] - 1, y1))
    y2 = max(0, min(img_bgr.shape[0] - 1, y2))
    out = img_bgr.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=2)
    if label:
        # 背景条
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        th_total = th + baseline + 6
        cv2.rectangle(out, (x1, max(0, y1 - th_total)), (x1 + tw + 6, y1), color, thickness=-1)
        cv2.putText(out, label, (x1 + 3, y1 - 4), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grad-CAM 可视化（RTMSF-DETR）")
    parser.add_argument("--config", required=True, help="训练时使用的 config（支持 outputs/**/config.yaml 或 configs/**.yaml）。")
    parser.add_argument("--config-dir", default="configs", help="Hydra 搜索配置根目录（默认 configs）。")
    parser.add_argument("--opts", nargs="*", default=None, help="可选覆盖项：KEY=VALUE（按顺序生效）。")
    parser.add_argument("--checkpoint", default=None, help="checkpoint 路径（默认与 config 同目录下 best/last）。")
    parser.add_argument("--device", default="cuda", help="cpu/cuda/cuda:0 等（cuda 不可用会自动回退 CPU）。")
    parser.add_argument("--weights-only", action="store_true", help="以 weights_only=True 方式加载 ckpt（更安全）。")
    parser.add_argument("--no-ema", action="store_true", help="不使用 EMA 权重（即使 checkpoint 中包含）。")
    parser.add_argument(
        "--keep-distill",
        action="store_true",
        help="保持训练时的 distill 配置并构建 teacher（更慢且占显存；Grad-CAM 推理一般不需要）。",
    )
    parser.add_argument(
        "--load-pretrain",
        action="store_true",
        help="构建模型时仍加载 pretrain_weights（通常没必要，因为随后会加载 checkpoint）。",
    )

    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="从哪个 split 抽样。")
    parser.add_argument("--sample-idx", type=int, default=0, help="从数据集第几个样本开始。")
    parser.add_argument("--num-samples", type=int, default=1, help="连续可视化多少个样本。")

    parser.add_argument("--target-layer", default="model.backbone", help="用于 Grad-CAM 的层名（model.named_modules() 中的 key）。")
    parser.add_argument("--feature-index", type=int, default=-1, help="当 target-layer 输出为 list/tuple 时选择第几个特征（默认最后一个）。")
    parser.add_argument("--target-class", type=int, default=None, help="只对指定类别做 Grad-CAM（默认取全局最高分）。")
    parser.add_argument(
        "--target-topk",
        type=int,
        default=1,
        help="当 --target-class 指定时：用该类别 top-k queries 的分数求和做反传（1=仅取最高分 query）。",
    )
    parser.add_argument(
        "--target-score-thr",
        type=float,
        default=0.0,
        help="当 --target-class 指定时：只使用该类别分数 >= thr 的 queries（0=不筛）。",
    )
    parser.add_argument(
        "--target-score-mode",
        type=str,
        default="logit",
        choices=["logit", "sigmoid"],
        help="反传目标分数来源：logit=原始 logits（默认更稳定）；sigmoid=概率。",
    )
    parser.add_argument(
        "--cam-method",
        type=str,
        default="gradcam",
        choices=["gradcam", "layercam"],
        help="CAM 生成方式：gradcam(默认) / layercam(更局部、更锐利)。",
    )
    parser.add_argument(
        "--target-from",
        type=str,
        default="pred",
        choices=["pred", "gt"],
        help="选择用于反传的 queries：pred=按预测分数选；gt=按与 GT IoU 匹配（仅当数据集提供 labels/boxes）。",
    )
    parser.add_argument(
        "--gt-iou-thr",
        type=float,
        default=0.3,
        help="target-from=gt 时，匹配 GT 的 IoU 阈值（0=不筛）。",
    )
    parser.add_argument(
        "--require-gt",
        action="store_true",
        help="target-from=gt 时，如果当前图像没有 target-class 的 GT，则跳过该样本（不输出）。",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="当 target-class 指定且 score_thr/gt_iou_thr 筛选后为空时，不回退到 best query；直接跳过该样本。",
    )
    parser.add_argument(
        "--draw-gt",
        action="store_true",
        help="在输出图中额外绘制 target-class 的 GT 框（红色）。",
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="热力图叠加透明度（0~1）。")

    parser.add_argument("--output-dir", default="outputs/gradcam_rtmsfdetr", help="输出目录。")

    parser.add_argument("--list-layers", action="store_true", help="打印可用 layer 列表并退出。")
    parser.add_argument("--layer-filter", default="", help="配合 --list-layers：仅打印名字包含该子串的模块。")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config_root = _as_path(args.config_dir).resolve()
    cfg = _load_config_any(args.config, config_dir=config_root, overrides=list(args.opts or []))

    # 先决定实际 device（cuda 不可用时回退 cpu），再写回 cfg，避免 trainer.build_model() 仍尝试 .to('cuda')。
    device = _pick_device(str(args.device))

    with open_dict(cfg):
        cfg.mode = "test"
        if "runtime" not in cfg:
            cfg.runtime = {}
        cfg.runtime.device = str(device)
        if "model" not in cfg:
            cfg.model = {}
        # Grad-CAM 主要用来解释推理行为；teacher 不参与推理也不应占用显存。
        # 因此默认禁用 distill（避免构建 DINOv3 teacher），需要时用 --keep-distill 打开。
        if not bool(args.keep_distill):
            cfg.model.disable_distill = True
        # 随后会加载 checkpoint，所以默认跳过 pretrain_weights，避免重复 I/O。
        if not bool(args.load_pretrain):
            cfg.model.force_no_pretrain = True

    # 构建 trainer/model（复用项目内 build_model 逻辑，保证与训练一致）
    from hydra.utils import instantiate
    from datasets import build_dataset as build_any_dataset

    trainer_cfg = getattr(cfg, "trainer", None)
    if trainer_cfg is None:
        raise KeyError("配置中缺少 trainer 节点，无法实例化训练器。")
    trainer = instantiate(trainer_cfg, cfg)
    model = trainer.build_model()
    model.eval().to(device)

    if bool(args.list_layers):
        filt = str(args.layer_filter or "").strip()
        names = []
        for name, module in model.named_modules():
            if not name:
                continue
            if filt and filt not in name:
                continue
            # 仅打印常见“可做 CAM”的层，避免输出过多；用户仍可用 --layer-filter 缩小范围。
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Sequential, torch.nn.ModuleList)) or name.endswith(
                ("backbone", "encoder", "decoder")
            ):
                names.append(name)
        for n in sorted(set(names)):
            print(n)
        return 0

    ckpt_path: Path | None = None
    if args.checkpoint:
        ckpt_path = _as_path(args.checkpoint).expanduser()
    else:
        config_path = _as_path(args.config).expanduser()
        if config_path.is_file():
            cand_dir = config_path.parent
            for name in ("checkpoint_best.pth", "checkpoint.pth"):
                p = cand_dir / name
                if p.is_file():
                    ckpt_path = p
                    break
    if ckpt_path is None or (not ckpt_path.is_file()):
        raise FileNotFoundError("未找到可用 checkpoint，请显式传 --checkpoint。")

    checkpoint = _load_checkpoint(ckpt_path, weights_only=bool(args.weights_only))
    state_dict = _extract_state_dict(checkpoint, prefer_ema=not bool(args.no_ema))
    state_dict = _filter_compatible_state_dict(model, state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logging.warning("加载 checkpoint：missing=%d unexpected=%d", len(missing), len(unexpected))
    logging.info("加载权重完成: %s", ckpt_path)

    # 构建 dataset，并按 idx 抽样
    data_args = trainer._ensure_data_args()  # noqa: SLF001 - 工具脚本中复用 trainer 内部逻辑

    if args.split == "train":
        split_name = "train"
    elif args.split == "val":
        split_name = str(getattr(data_args, "val_split", None) or "val")
    else:
        split_name = str(getattr(data_args, "test_split", None) or "test")

    dataset = build_any_dataset(split_name, data_args, int(getattr(data_args, "img_size", 640)))
    if len(dataset) <= 0:
        raise RuntimeError(f"dataset(split={split_name}) 为空。")

    # 解析类别名
    class_names = getattr(getattr(cfg, "data", None), "class_names", None) or []
    if hasattr(class_names, "__iter__") and not isinstance(class_names, (str, bytes)):
        class_names = [str(x) for x in list(class_names)]
    else:
        class_names = []

    num_classes = int(getattr(getattr(cfg, "train", None), "num_classes", 0) or 0)
    if num_classes <= 0:
        # 兜底：用 class_names 长度
        num_classes = len(class_names)

    target_layer = _resolve_named_module(model, str(args.target_layer))
    output_dir = _as_path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    start = max(0, int(args.sample_idx))
    count = max(1, int(args.num_samples))
    end = min(len(dataset), start + count)

    logging.info("split=%s, idx=[%d, %d), target_layer=%s, feature_index=%d", split_name, start, end, args.target_layer, args.feature_index)
    logging.info("输出目录: %s", output_dir)

    for idx in range(start, end):
        sample, target = dataset[idx]

        # 取 RGB 用于可视化（默认优先 sample['rgb']，否则尝试 sample[:3]）
        if isinstance(sample, dict):
            rgb_tensor = sample.get("rgb", None)
            if rgb_tensor is None:
                raise KeyError("dataset 返回 dict sample，但缺少 key='rgb'（无法叠加热力图）。")
            model_input = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) and v.ndim == 3 else v.to(device)) for k, v in sample.items() if torch.is_tensor(v)}
        else:
            if not torch.is_tensor(sample):
                raise TypeError(f"dataset 返回 sample 类型不支持: {type(sample)}")
            rgb_tensor = sample[:3]
            model_input = sample.unsqueeze(0).to(device) if sample.ndim == 3 else sample.to(device)

        rgb_vis = _denormalize_rgb_for_vis(rgb_tensor, cfg=cfg)
        H, W = rgb_vis.shape[:2]

        # Grad-CAM
        with GradCAMExtractor(model, target_layer=target_layer, feature_index=int(args.feature_index)) as cam_extractor:
            outputs = model(model_input, targets=None)
            if not isinstance(outputs, Mapping) or "pred_logits" not in outputs or "pred_boxes" not in outputs:
                raise TypeError(f"模型输出不符合预期：{type(outputs)} keys={getattr(outputs, 'keys', lambda: [])()}")

            # Backprop target:
            # - no target-class: best (q,cls) logit
            # - target-class: sum over selected queries for that class
            #   - target-from=pred: select by predicted class score
            #   - target-from=gt: select by IoU matched to GT boxes of that class (when available)
            det: DetTarget
            if args.target_class is not None:
                pred_logits_qc = outputs["pred_logits"][0]
                pred_boxes_q4 = outputs["pred_boxes"][0]

                sel: tuple[torch.Tensor, torch.Tensor] | None = None
                if str(args.target_from).lower() == "gt":
                    gt = _extract_gt_boxes_labels(target)
                    if gt is None:
                        if bool(args.require_gt):
                            logging.info("idx=%d skip (no gt labels/boxes provided by dataset)", idx)
                            continue
                    else:
                        gt_boxes, gt_labels = gt
                        keep_gt = gt_labels == int(args.target_class)
                        gt_boxes_cls = gt_boxes[keep_gt]
                        if gt_boxes_cls.numel() == 0:
                            if bool(args.require_gt):
                                logging.info("idx=%d skip (no GT of class=%d)", idx, int(args.target_class))
                                continue
                        else:
                            picked = _select_queries_by_gt_iou(
                                pred_logits_qc,
                                pred_boxes_q4,
                                gt_boxes_cls,
                                num_classes=num_classes,
                                target_class=int(args.target_class),
                                topk=int(args.target_topk),
                                score_thr=float(args.target_score_thr),
                                iou_thr=float(args.gt_iou_thr),
                            )
                            if picked is not None:
                                sel_q, sel_prob, sel_iou = picked
                                sel = (sel_q, sel_prob)
                                logging.info(
                                    "idx=%d target-from=gt matched_queries=%d best_iou=%.3f",
                                    idx,
                                    int(sel_q.numel()),
                                    float(sel_iou.max().item()) if sel_iou.numel() else float("nan"),
                                )

                if sel is None:
                    sel = _select_queries_for_class_from_logits(
                        outputs["pred_logits"][0],
                        num_classes=num_classes,
                        target_class=int(args.target_class),
                        topk=int(args.target_topk),
                        score_thr=float(args.target_score_thr),
                        allow_fallback=not bool(args.no_fallback),
                    )
                    if sel is None:
                        logging.info("idx=%d skip (no query meets score_thr=%.3f)", idx, float(args.target_score_thr))
                        continue
                sel_q, sel_prob = sel
                mode = str(args.target_score_mode or "logit").lower()
                if mode == "sigmoid":
                    score = torch.sigmoid(outputs["pred_logits"][0, sel_q, int(args.target_class)]).sum()
                else:
                    score = outputs["pred_logits"][0, sel_q, int(args.target_class)].sum()
                target_score = score
                logging.info(
                    "target-class=%d selected_queries=%d top_prob=%.4f",
                    int(args.target_class),
                    int(sel_q.numel()),
                    float(sel_prob.max().item()) if sel_prob.numel() else float("nan"),
                )

                # Choose a representative query for drawing bbox/label (highest prob among selected).
                q_draw = int(sel_q[torch.argmax(sel_prob)].item())
                box_xyxy = _box_cxcywh_to_xyxy_pixel(outputs["pred_boxes"][0, q_draw], out_hw=(H, W))
                det = DetTarget(query_idx=q_draw, class_idx=int(args.target_class), score=float(sel_prob.max().item()), box_xyxy=box_xyxy)
            else:
                det = _select_det_target(
                    outputs,
                    num_classes=num_classes,
                    input_size_hw=(H, W),
                    target_class=None,
                )
                target_score = outputs["pred_logits"][0, det.query_idx, det.class_idx]

            cam01 = cam_extractor.compute_cam(
                target_score=target_score,
                input_size_hw=(H, W),
                method=str(args.cam_method),
            )

        # 叠加 + 画框
        overlay_bgr = _overlay_cam(rgb_vis, cam01, alpha=float(args.alpha))
        cls_name = class_names[det.class_idx] if 0 <= det.class_idx < len(class_names) else f"class_{det.class_idx}"
        label = f"{cls_name} {det.score:.3f} (q={det.query_idx})"
        overlay_bgr = _draw_box(overlay_bgr, box_xyxy=det.box_xyxy, label=label)

        if args.target_class is not None and bool(args.draw_gt):
            gt = _extract_gt_boxes_labels(target)
            if gt is not None:
                gt_boxes, gt_labels = gt
                keep_gt = gt_labels == int(args.target_class)
                gt_boxes_cls = gt_boxes[keep_gt]
                if gt_boxes_cls.numel() > 0:
                    for b in gt_boxes_cls:
                        overlay_bgr = _draw_box(
                            overlay_bgr,
                            box_xyxy=_box_cxcywh_to_xyxy_pixel(b, out_hw=(H, W)),
                            label="GT",
                            color=(0, 0, 255),
                        )

        # 文件名尽量可追溯到 dataset 文件
        stem = f"{idx:06d}"
        try:
            img_id = getattr(dataset, "ids", [None])[idx]
            coco = getattr(dataset, "coco", None)
            if coco is not None and img_id is not None:
                info = coco.loadImgs(int(img_id))[0]
                file_name = str(info.get("file_name", "")).strip()
                if file_name:
                    stem = f"{idx:06d}_{Path(file_name).stem}"
        except Exception:
            pass

        out_path = output_dir / f"{stem}_gradcam.png"
        cv2.imwrite(str(out_path), overlay_bgr)
        logging.info("saved: %s", out_path)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
