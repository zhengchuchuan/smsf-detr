from __future__ import annotations

"""
EigenCAM 可视化（参考 docs/grad_cam.ipynb 的做法，基于 pytorch-grad-cam）。

要点：
- 采用 pytorch-grad-cam 的 EigenCAM（无梯度），对指定层的特征做 PCA 投影得到 CAM；
- 支持 backbone/encoder 这类输出为 list/tuple/dict 的层，通过 reshape_transform 将其变成单个 Tensor；
- 可选在图上绘制某个类别（如 oil=0）的预测框，便于核对“热力图是否与检测对象一致”。

注意：
EigenCAM 是“类无关”的显著性图（targets 对 EigenCAM 不起作用），它更像“哪里特征激活强”，
不是严格的“oil 类贡献热力图”。如果你需要强类别相关，可优先用：
- vis/featuremap_heatmap_rtmsfdetr.py 的 heat_source=attn_cam / det_heatmap
- tools/gradcam_rtmsfdetr.py 的 target-from=gt（GT 匹配反传）
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError("需要安装 opencv-python 才能进行热力图上色/叠加。") from exc

try:
    from pytorch_grad_cam import EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except Exception as exc:  # pragma: no cover
    raise ImportError("需要安装 pytorch-grad-cam 才能运行本脚本。") from exc


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
    candidates = [k for k in modules.keys() if k.endswith(name)]
    if len(candidates) == 1:
        return modules[candidates[0]]
    if not candidates:
        raise KeyError(f"未找到 layer: {name}")
    cand_preview = "\n  - " + "\n  - ".join(sorted(candidates)[:50])
    raise KeyError(f"layer={name} 不唯一，候选有 {len(candidates)} 个，请指定更精确的名字。候选示例：{cand_preview}")


def _denormalize_rgb(rgb_tensor: torch.Tensor, *, cfg: Any) -> np.ndarray:
    """[3,H,W] -> float32 RGB [H,W,3] in [0,1]."""
    if rgb_tensor.ndim != 3 or rgb_tensor.shape[0] < 3:
        raise ValueError(f"rgb_tensor shape 需要为 [3,H,W]，当前={tuple(rgb_tensor.shape)}")
    mode = str(getattr(getattr(cfg, "data", None), "rgb_normalize_mode", "imagenet")).lower()
    rgb = rgb_tensor[:3].detach().float().cpu()
    if mode == "imagenet":
        mean = torch.tensor(getattr(getattr(cfg, "data", None), "rgb_mean", (0.485, 0.456, 0.406)), dtype=torch.float32)
        std = torch.tensor(getattr(getattr(cfg, "data", None), "rgb_std", (0.229, 0.224, 0.225)), dtype=torch.float32)
        rgb = rgb * std.view(3, 1, 1) + mean.view(3, 1, 1)
    rgb = rgb.clamp(0.0, 1.0)
    return rgb.permute(1, 2, 0).numpy().astype(np.float32)


def _cxcywh_to_xyxy_pixel(boxes: torch.Tensor, *, out_hw: tuple[int, int]) -> np.ndarray:
    """cxcywh normalized -> xyxy pixel, boxes: [N,4]."""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise TypeError(f"boxes 期望 [N,4]，实际为: shape={tuple(boxes.shape)}")
    h, w = int(out_hw[0]), int(out_hw[1])
    b = boxes.detach().float().cpu().numpy()
    cx, cy, bw, bh = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    x1 = (cx - 0.5 * bw) * w
    y1 = (cy - 0.5 * bh) * h
    x2 = (cx + 0.5 * bw) * w
    y2 = (cy + 0.5 * bh) * h
    x1 = np.clip(x1, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)
    x2 = np.clip(x2, 0, w - 1)
    y2 = np.clip(y2, 0, h - 1)
    return np.stack([x1, y1, x2, y2], axis=1)


def _draw_boxes_bgr(
    img_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    *,
    labels: Sequence[str],
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    out = img_bgr.copy()
    for box, label in zip(boxes_xyxy, labels):
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=2)
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), baseline = cv2.getTextSize(label, font, 0.6, 1)
            th_total = th + baseline + 6
            cv2.rectangle(out, (x1, max(0, y1 - th_total)), (x1 + tw + 6, y1), color, thickness=-1)
            cv2.putText(out, label, (x1 + 3, y1 - 4), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EigenCAM 可视化（RTMSF-DETR, pytorch-grad-cam）")
    parser.add_argument("--config", required=True, help="训练时使用的 config（支持 outputs/**/config.yaml 或 configs/**.yaml）。")
    parser.add_argument("--config-dir", default="configs", help="Hydra 搜索配置根目录（默认 configs）。")
    parser.add_argument("--opts", nargs="*", default=None, help="可选覆盖项：KEY=VALUE（按顺序生效）。")
    parser.add_argument("--checkpoint", default=None, help="checkpoint 路径（默认与 config 同目录下 best/last）。")
    parser.add_argument("--device", default="cuda", help="cpu/cuda/cuda:0 等（cuda 不可用会自动回退 CPU）。")
    parser.add_argument("--weights-only", action="store_true", help="以 weights_only=True 方式加载 ckpt（更安全）。")
    parser.add_argument("--no-ema", action="store_true", help="不使用 EMA 权重（即使 checkpoint 中包含）。")
    parser.add_argument("--keep-distill", action="store_true", help="保持训练时 distill 配置并构建 teacher（更慢）。")
    parser.add_argument("--load-pretrain", action="store_true", help="构建模型时仍加载 pretrain_weights（通常没必要）。")

    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="从哪个 split 抽样。")
    parser.add_argument("--sample-idx", type=int, default=0, help="从数据集第几个样本开始。")
    parser.add_argument("--num-samples", type=int, default=1, help="连续可视化多少个样本。")

    parser.add_argument("--target-layer", default="model.backbone", help="用于 CAM 的层名（model.named_modules() 中的 key）。")
    parser.add_argument(
        "--feature-map-indice",
        default="mean",
        help="当 target-layer 输出为 list/tuple/dict 时，如何选特征：0/1/2/... 或 mean(聚合).",
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="热力图叠加透明度（0~1）。")

    parser.add_argument("--draw-pred", action="store_true", help="绘制预测框（绿色）。")
    parser.add_argument("--target-class", type=int, default=None, help="仅绘制/筛选该类别的预测框（0-based）。")
    parser.add_argument("--pred-score-thr", type=float, default=0.3, help="绘制预测框的最低分数阈值（sigmoid prob）。")
    parser.add_argument("--pred-topk", type=int, default=50, help="绘制预测框的 top-k（按分数）。")

    parser.add_argument("--output-dir", default="outputs/eigencam_rtmsfdetr", help="输出目录。")
    parser.add_argument("--list-layers", action="store_true", help="打印可用 layer 列表并退出。")
    parser.add_argument("--layer-filter", default="", help="配合 --list-layers：仅打印名字包含该子串的模块。")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config_root = _as_path(args.config_dir).resolve()
    cfg = _load_config_any(args.config, config_dir=config_root, overrides=list(args.opts or []))
    device = _pick_device(str(args.device))

    with open_dict(cfg):
        cfg.mode = "test"
        if "runtime" not in cfg:
            cfg.runtime = {}
        cfg.runtime.device = str(device)
        if "model" not in cfg:
            cfg.model = {}
        if not bool(args.keep_distill):
            cfg.model.disable_distill = True
        if not bool(args.load_pretrain):
            cfg.model.force_no_pretrain = True

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
        for name, _module in model.named_modules():
            if not name:
                continue
            if filt and filt not in name:
                continue
            print(name)
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

    data_args = trainer._ensure_data_args()  # noqa: SLF001
    if args.split == "train":
        split_name = "train"
    elif args.split == "val":
        split_name = str(getattr(data_args, "val_split", None) or "val")
    else:
        split_name = str(getattr(data_args, "test_split", None) or "test")

    dataset = build_any_dataset(split_name, data_args, int(getattr(data_args, "img_size", 640)))
    if len(dataset) <= 0:
        raise RuntimeError(f"dataset(split={split_name}) 为空。")

    class_names = getattr(getattr(cfg, "data", None), "class_names", None) or []
    if hasattr(class_names, "__iter__") and not isinstance(class_names, (str, bytes)):
        class_names = [str(x) for x in list(class_names)]
    else:
        class_names = []

    target_layer = _resolve_named_module(model, str(args.target_layer))

    feature_map_indice = str(args.feature_map_indice).strip().lower()

    def reshape_transform(x: Any) -> torch.Tensor:
        # list/tuple/dict -> Tensor[B,C,H,W]
        if torch.is_tensor(x):
            if x.ndim != 4:
                raise ValueError(f"target-layer 输出 Tensor 需为 [B,C,H,W]，got {tuple(x.shape)}")
            return x
        if isinstance(x, dict):
            vals = list(x.values())
        elif isinstance(x, (list, tuple)):
            vals = list(x)
        else:
            raise TypeError(f"target-layer 输出类型不支持: {type(x)} (需要 Tensor/list/tuple/dict)")
        vals = [v for v in vals if torch.is_tensor(v)]
        if not vals:
            raise ValueError("target-layer 输出为空或不包含 Tensor。")
        # 统一到最后一层的空间分辨率（与 docs/grad_cam.ipynb 保持一致）
        target_size = vals[-1].shape[-2:]
        if feature_map_indice == "mean":
            acts = [F.interpolate(torch.abs(v), size=target_size, mode="bilinear", align_corners=False) for v in vals]
            return torch.cat(acts, dim=1)
        idx = int(feature_map_indice)
        if idx < 0 or idx >= len(vals):
            raise IndexError(f"feature-map-indice 越界：{idx} not in [0, {len(vals)-1}]")
        return vals[idx]

    out_dir = _as_path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    start = max(0, int(args.sample_idx))
    count = max(1, int(args.num_samples))
    end = min(len(dataset), start + count)
    logging.info("split=%s, idx=[%d, %d), target_layer=%s, feature_map_indice=%s", split_name, start, end, args.target_layer, args.feature_map_indice)
    logging.info("输出目录: %s", out_dir)

    cam = EigenCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)

    for idx in range(start, end):
        sample, _target = dataset[idx]

        if isinstance(sample, dict):
            ordered = []
            for k in ("rgb", "ms"):
                if k in sample:
                    ordered.append(k)
            ordered.extend([k for k in sample.keys() if k not in set(ordered)])
            tensors = [sample[k] for k in ordered if torch.is_tensor(sample[k])]
            if not tensors:
                raise ValueError("dataset sample(dict) 中没有 Tensor。")
            x = torch.cat(tensors, dim=0)
            rgb_tensor = sample.get("rgb", tensors[0])  # best effort
        else:
            if not torch.is_tensor(sample):
                raise TypeError(f"dataset 返回 sample 类型不支持: {type(sample)}")
            x = sample
            rgb_tensor = sample[:3]

        # EigenCAM expects input_tensor: [B,C,H,W]
        input_tensor = x.unsqueeze(0).to(device)

        rgb01 = _denormalize_rgb(rgb_tensor, cfg=cfg)  # [H,W,3] float32 0..1
        h, w = rgb01.shape[:2]

        # targets for EigenCAM are ignored, but must not be None (otherwise BaseCAM tries argmax on model outputs).
        dummy_targets = [0]
        grayscale_cam = cam(input_tensor, targets=dummy_targets)[0]  # [H,W] in [0,1] (scaled)
        cam_rgb = show_cam_on_image(rgb01, grayscale_cam, use_rgb=True)  # uint8 RGB
        out_bgr = cv2.cvtColor(cam_rgb, cv2.COLOR_RGB2BGR)

        if bool(args.draw_pred):
            with torch.inference_mode():
                out = model(input_tensor, targets=None)
            if isinstance(out, Mapping) and "pred_logits" in out and "pred_boxes" in out:
                logits = out["pred_logits"][0]  # [Q,C] or [Q,C+1]
                boxes = out["pred_boxes"][0]  # [Q,4] cxcywh
                if logits.ndim != 2:
                    raise TypeError(f"pred_logits expected [Q,C], got {tuple(logits.shape)}")
                prob = torch.sigmoid(logits)
                if args.target_class is not None:
                    cls = int(args.target_class)
                    if cls < 0 or cls >= prob.shape[1]:
                        raise ValueError(f"--target-class 越界：{cls} not in [0, {prob.shape[1]-1}]")
                    scores = prob[:, cls]
                    labels = torch.full_like(scores, fill_value=cls, dtype=torch.long)
                else:
                    scores, labels = prob.max(dim=-1)
                keep = scores >= float(args.pred_score_thr) if float(args.pred_score_thr) > 0 else torch.ones_like(scores, dtype=torch.bool)
                idx_all = torch.nonzero(keep, as_tuple=False).squeeze(1)
                if idx_all.numel() > 0 and int(args.pred_topk) > 0 and idx_all.numel() > int(args.pred_topk):
                    sel = idx_all[torch.argsort(scores[idx_all])[-int(args.pred_topk) :]]
                else:
                    sel = idx_all
                sel = sel.sort().values
                if sel.numel() > 0:
                    boxes_sel = boxes[sel]
                    labels_sel = labels[sel].detach().cpu().numpy().tolist()
                    scores_sel = scores[sel].detach().cpu().numpy().tolist()
                    boxes_xyxy = _cxcywh_to_xyxy_pixel(boxes_sel, out_hw=(h, w))
                    label_str = []
                    for lab, sc in zip(labels_sel, scores_sel):
                        name = class_names[lab] if 0 <= lab < len(class_names) else f"class_{lab}"
                        label_str.append(f"{name} {sc:.2f}")
                    out_bgr = _draw_boxes_bgr(out_bgr, boxes_xyxy, labels=label_str, color=(0, 255, 0))

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

        out_path = out_dir / f"{stem}_eigencam.png"
        cv2.imwrite(str(out_path), out_bgr)
        logging.info("saved: %s", out_path)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

