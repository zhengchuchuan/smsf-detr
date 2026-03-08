from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from PIL import Image
from torchvision.transforms import functional as tvf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.misc import NestedTensor


"""  
- 1) 导出特征（每张图一个 npz）：用 tools/repr_export_rtmsfdetr.py

    python tools/repr_export_rtmsfdetr.py \
      --resolved-config outputs/oil_rgb_all_data/rtmsfdetr/rtv4_hgnetv2_l_distill/260104-160214-rtmsfdetr_oil_rgb_det_rtv4_hgnetv2_l_distill_all_data/config.yaml \
      --device cuda --amp \
      --input-dir data/oil_20260101/rgb/val \
      --feat-source fpn --feat-level 0 \
      --output-dir outputs/repr/rtmsfdetr \
      --run-name oil_rgb_val_k32x16_20260104
    产物在 outputs/repr/rtmsfdetr/oil_rgb_val/features/*.npz（含 feat[C,H,W] + meta）。
    产物在 outputs/repr/rtmsfdetr/oil_rgb_val/features/*.npz（含 feat[C,H,W] + meta）。
"""


@dataclass(frozen=True)
class FeatureMeta:
    image_path: str
    orig_hw: tuple[int, int]
    input_hw: tuple[int, int]
    feat_hw: tuple[int, int]
    feat_channels: int
    stride_hw: tuple[int, int]
    feat_source: str
    feat_level: int
    dtype: str


def get_config(node: Any, key: str, default: Any | None = None) -> Any:
    if node is None:
        return default
    if hasattr(node, "get"):
        return node.get(key, default)
    return getattr(node, key, default)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="导出已训练 RTMSFDETR（RGB-only）dense 表征特征图（用于聚类/伪彩色可视化）。"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str, help="Hydra 配置路径（在 configs/ 下或任意 yaml 路径）。")
    group.add_argument("--resolved-config", type=str, help="outputs/**/config.yaml 这类落盘完整配置。")
    parser.add_argument("--config-dir", type=str, default="configs", help="Hydra 配置根目录（默认 configs）。")
    parser.add_argument("--opts", nargs="*", default=None, help="可选 dotlist overrides。")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="checkpoint 路径（默认：若 resolved-config 来自 outputs/**/config.yaml，则自动找同目录 checkpoint_best.pth / checkpoint.pth）。",
    )
    parser.add_argument("--use-ema", action="store_true", help="若 ckpt 包含 ema 权重则优先使用。")
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="以 weights_only=True 方式加载 ckpt（更安全，但 BaseTrainer 默认保存了 config 对象，通常需要关闭）。",
    )

    parser.add_argument("--device", default="cuda", help="cpu/cuda/cuda:0 等（cuda 不可用会自动回退 CPU）。")
    parser.add_argument("--amp", action="store_true", help="导出特征时启用 AMP（仅 CUDA 有效）。")

    parser.add_argument(
        "--input-dir",
        type=str,
        default="",
        help="待导出图片目录（png/jpg/tif...）；不传则尝试从 cfg.data.dataset_dir 推导。",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="当未显式传 --input-dir 时，使用 dataset_dir/rgb/<split>（默认 val）。",
    )
    parser.add_argument("--recursive", action="store_true", help="递归扫描 input-dir。")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少张（0 表示不限制）。")
    parser.add_argument("--batch-size", type=int, default=1, help="导出 batch size（默认 1）。")
    parser.add_argument("--img-size", type=int, default=0, help="输入 resize 到方形大小（0=自动按 cfg.img_size 或 640）。")

    parser.add_argument(
        "--feat-source",
        type=str,
        default="fpn",
        choices=["fpn", "backbone"],
        help="导出特征来源：fpn=HybridEncoder 输出；backbone=HGNetv2 stage 输出。",
    )
    parser.add_argument(
        "--feat-level",
        type=int,
        default=0,
        help="特征层级索引：fpn/backbone 通常为 0/1/2 对应 stride 8/16/32。",
    )
    parser.add_argument(
        "--save-dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="保存到 npz 的特征精度（默认 float16）。",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/repr/rtmsfdetr",
        help="输出根目录（默认 outputs/repr/rtmsfdetr）。",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help='输出子目录名（默认自动："<input_dir>-YYYYMMDD-HHMM"）。',
    )
    return parser.parse_args()


def _pick_device(device: str) -> torch.device:
    device = str(device).strip().lower()
    if device.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("请求使用 CUDA，但当前不可用，自动回退到 CPU。")
        return torch.device("cpu")
    return torch.device(device)


def _sanitize_dirname(name: str) -> str:
    name = str(name).strip()
    name = name.replace("/", "_").replace("\\", "_")
    name = "_".join([p for p in name.split() if p])
    return name or "run"


def _iter_images(input_dir: Path, *, recursive: bool) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if recursive:
        paths = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    else:
        paths = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(paths)


def _resolve_input_dir(cfg: Any, *, input_dir: str, split: str) -> Path:
    if str(input_dir).strip():
        return Path(input_dir).expanduser()
    dataset_dir = get_config(getattr(cfg, "data", None), "dataset_dir", None)
    if not dataset_dir:
        raise ValueError("未提供 --input-dir，且 cfg.data.dataset_dir 为空，无法推导输入目录。")
    return Path(str(dataset_dir)).expanduser() / "rgb" / str(split)


def _load_cfg(args: argparse.Namespace):
    config = args.resolved_config or args.config
    if args.resolved_config:
        cfg = OmegaConf.load(str(Path(config).expanduser()))
        OmegaConf.set_struct(cfg, False)
        if args.opts:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(args.opts)))
            OmegaConf.set_struct(cfg, False)
    else:
        try:
            from engines.core.parse_config import load_config  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "当前环境缺少 hydra-core，无法通过 --config 解析 defaults/config group；"
                "请改用 --resolved-config（outputs/**/config.yaml），或安装 hydra-core。"
            ) from exc
        cfg = load_config(config, config_dir=args.config_dir, overrides=list(args.opts or []))
    with open_dict(cfg):
        if "runtime" not in cfg:
            cfg.runtime = {}
    return cfg


def _resolve_ckpt(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        p = Path(args.checkpoint).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"--checkpoint 不存在: {p}")
        return p

    if args.resolved_config:
        cfg_path = Path(args.resolved_config).expanduser()
        if cfg_path.is_file():
            cand_dir = cfg_path.parent
            for name in ("checkpoint_best.pth", "checkpoint.pth"):
                p = cand_dir / name
                if p.is_file():
                    return p
    raise FileNotFoundError("未找到可用 checkpoint：请显式传 --checkpoint，或传 --resolved-config 指向 outputs/**/config.yaml。")


def _extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping) and checkpoint.get("model") is not None:
        state = checkpoint["model"]
    else:
        state = checkpoint
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    if not isinstance(state, Mapping):
        raise TypeError(f"无法从 checkpoint 解析 state_dict，得到类型={type(state)}")
    return state


def _filter_compatible_state_dict(
    model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor]
) -> Mapping[str, torch.Tensor]:
    model_sd = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k not in model_sd:
            continue
        if model_sd[k].shape != v.shape:
            continue
        filtered[k] = v
    return filtered


def _build_image_batch(
    paths: list[Path],
    *,
    img_size: int,
    normalize: bool,
    rgb_mean: tuple[float, float, float],
    rgb_std: tuple[float, float, float],
) -> tuple[torch.Tensor, list[tuple[int, int]], list[tuple[int, int]]]:
    tensors: list[torch.Tensor] = []
    orig_hws: list[tuple[int, int]] = []
    input_hws: list[tuple[int, int]] = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        orig_w, orig_h = img.size
        orig_hws.append((int(orig_h), int(orig_w)))
        if img_size > 0:
            img = img.resize((int(img_size), int(img_size)), Image.BILINEAR)
        in_w, in_h = img.size
        input_hws.append((int(in_h), int(in_w)))
        x = tvf.to_tensor(img)
        if normalize:
            x = tvf.normalize(x, mean=list(rgb_mean), std=list(rgb_std))
        tensors.append(x)
    batch = torch.stack(tensors, dim=0)
    return batch, orig_hws, input_hws


def _compute_stride(input_hw: tuple[int, int], feat_hw: tuple[int, int]) -> tuple[int, int]:
    in_h, in_w = input_hw
    fh, fw = feat_hw
    if fh <= 0 or fw <= 0:
        return (0, 0)
    return (max(1, int(round(in_h / fh))), max(1, int(round(in_w / fw))))


def main() -> int:
    args = _parse_args()
    cfg = _load_cfg(args)

    device = _pick_device(args.device)
    amp_enabled = bool(args.amp) and device.type == "cuda"

    input_dir = _resolve_input_dir(cfg, input_dir=str(args.input_dir), split=str(args.split))
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input-dir 不存在或不是目录: {input_dir}")
    img_paths = _iter_images(input_dir, recursive=bool(args.recursive))
    if args.limit and int(args.limit) > 0:
        img_paths = img_paths[: int(args.limit)]
    if not img_paths:
        raise FileNotFoundError(f"input-dir 未找到图片: {input_dir}")

    ckpt_path = _resolve_ckpt(args)

    with open_dict(cfg):
        cfg.runtime.device = str(device)
        cfg.runtime.device_ids = []
        cfg.runtime.world_size = 1
        cfg.mode = "test"

    trainer_cfg = getattr(cfg, "trainer", None)
    if trainer_cfg is None:
        logging.warning("cfg.trainer 未配置：将直接按 cfg.model/cfg.train/cfg.data 构造 rtmsfdetr 模型。")

    data_cfg = getattr(cfg, "data", None)
    rgb_mean = tuple(get_config(data_cfg, "rgb_mean", (0.485, 0.456, 0.406)))
    rgb_std = tuple(get_config(data_cfg, "rgb_std", (0.229, 0.224, 0.225)))
    normalize = bool(get_config(getattr(cfg, "model", None), "input_denormalize", True))

    if int(args.img_size) > 0:
        img_size = int(args.img_size)
    else:
        img_size = int(
            get_config(getattr(cfg, "data", None), "img_size", None)
            or get_config(getattr(cfg, "train", None), "img_size", None)
            or 640
        )

    from argparse import Namespace
    from engines.models.rtmsfdetr.builder import build_model_and_processors

    model_cfg = getattr(cfg, "model", None) or {}
    train_cfg = getattr(cfg, "train", None) or {}

    rgb_ch = int(get_config(getattr(cfg, "data", None), "rgb_input_channels", 3) or 3)
    ms_ch = int(get_config(getattr(cfg, "data", None), "ms_input_channels", 0) or 0)
    if bool(get_config(getattr(cfg, "data", None), "use_msi_input", False)) or ms_ch > 0:
        raise RuntimeError("当前脚本仅支持 RGB-only 导特征：请确保 data.use_msi_input=false 且 ms_input_channels=0。")

    build_args = Namespace(
        device=str(device),
        num_classes=int(get_config(train_cfg, "num_classes", get_config(model_cfg, "num_classes", 1))),
        img_size=int(img_size),
        # RT-DETRv4 YAML config
        rtdetrv4_config=get_config(model_cfg, "rtdetrv4_config", None),
        disable_distill=bool(get_config(model_cfg, "disable_distill", True)),
        teacher_repo_path=get_config(model_cfg, "teacher_repo_path", None),
        teacher_weights_path=get_config(model_cfg, "teacher_weights_path", None),
        # HGNetv2 预训练与冻结
        hgnet_pretrained=bool(get_config(model_cfg, "hgnet_pretrained", False)),
        hgnet_local_model_dir=get_config(model_cfg, "hgnet_local_model_dir", None),
        hgnet_freeze_at=int(get_config(model_cfg, "hgnet_freeze_at", -1)),
        hgnet_freeze_norm=bool(get_config(model_cfg, "hgnet_freeze_norm", False)),
        # 输入反归一化
        input_denormalize=bool(get_config(model_cfg, "input_denormalize", True)),
        clamp_after_denormalize=bool(get_config(model_cfg, "clamp_after_denormalize", True)),
        rgb_mean=tuple(rgb_mean),
        rgb_std=tuple(rgb_std),
        # 输入通道（RGB-only）
        rgb_input_channels=int(rgb_ch),
        ms_input_channels=int(ms_ch),
        input_channels=int(rgb_ch + ms_ch),
        # 兜底：保持单流
        dual_stream_backbone=bool(get_config(model_cfg, "dual_stream_backbone", False)),
        backbone_output_merge=str(get_config(model_cfg, "backbone_output_merge", "avg") or "avg"),
        backbone_fusion=get_config(model_cfg, "backbone_fusion", None),
    )

    model, _, _ = build_model_and_processors(build_args)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=bool(args.weights_only))
    model_state = checkpoint
    if args.use_ema and isinstance(checkpoint, Mapping) and checkpoint.get("ema") is not None:
        model_state = checkpoint["ema"]
    state_dict = _extract_state_dict(model_state)
    compatible = _filter_compatible_state_dict(model, state_dict)
    skipped = len(state_dict) - len(compatible)
    if skipped:
        logging.warning("checkpoint 有 %d 个参数 shape 不匹配或不存在，已跳过加载。", skipped)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if missing or unexpected:
        logging.info("加载 state_dict：missing=%d unexpected=%d", len(missing), len(unexpected))

    model.eval()
    model.to(device)

    out_root = Path(args.output_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    run_name = str(args.run_name).strip()
    if not run_name:
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        run_name = f"{_sanitize_dirname(input_dir.name)}-{stamp}"
    run_dir = out_root / run_name
    feat_dir = run_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    logging.info("配置: %s", args.resolved_config or args.config)
    logging.info("权重: %s", ckpt_path)
    logging.info("输入: %s (N=%d)", input_dir, len(img_paths))
    logging.info("输出: %s", run_dir)
    logging.info(
        "device=%s amp=%s img_size=%d feat_source=%s feat_level=%d save_dtype=%s",
        device,
        amp_enabled,
        img_size,
        str(args.feat_source),
        int(args.feat_level),
        str(args.save_dtype),
    )

    batch_size = max(1, int(args.batch_size))
    save_dtype = np.float16 if str(args.save_dtype) == "float16" else np.float32

    index: list[dict[str, Any]] = []
    for idx in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[idx : idx + batch_size]
        batch_tensor, orig_hws, input_hws = _build_image_batch(
            batch_paths,
            img_size=img_size,
            normalize=normalize,
            rgb_mean=rgb_mean,
            rgb_std=rgb_std,
        )

        mask = torch.zeros((batch_tensor.shape[0], batch_tensor.shape[2], batch_tensor.shape[3]), dtype=torch.bool)
        samples = NestedTensor(batch_tensor.to(device), mask.to(device))

        detector = model.module if hasattr(model, "module") else model
        if not hasattr(detector, "_prepare_images") or not hasattr(detector, "model"):
            raise TypeError(f"当前模型不是预期的 RTDETRv4Detector 封装，类型={type(detector)}")

        with torch.inference_mode():
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    images = detector._prepare_images(samples)  # type: ignore[attr-defined]
                    raw = detector.model  # type: ignore[attr-defined]
                    feats_backbone = raw.backbone(images)
                    if str(args.feat_source) == "backbone":
                        feat_maps = feats_backbone
                    else:
                        feat_maps = raw.encoder(feats_backbone)
            else:
                images = detector._prepare_images(samples)  # type: ignore[attr-defined]
                raw = detector.model  # type: ignore[attr-defined]
                feats_backbone = raw.backbone(images)
                if str(args.feat_source) == "backbone":
                    feat_maps = feats_backbone
                else:
                    feat_maps = raw.encoder(feats_backbone)

        if isinstance(feat_maps, tuple) and len(feat_maps) >= 1:
            feat_maps = feat_maps[0]
        if not isinstance(feat_maps, (list, tuple)):
            raise TypeError(f"特征输出期望 list/tuple，实际为: {type(feat_maps)}")

        level = int(args.feat_level)
        if level < 0 or level >= len(feat_maps):
            raise IndexError(f"feat_level 越界：level={level} but len(feat_maps)={len(feat_maps)}")

        selected = feat_maps[level]
        if not torch.is_tensor(selected) or selected.ndim != 4:
            raise TypeError(f"选中的特征不是 4D Tensor，实际为: {type(selected)} shape={getattr(selected, 'shape', None)}")

        # Save per-image npz: feat shape [C, H, W]
        selected_cpu = selected.detach().float().cpu()
        for p, orig_hw, in_hw, feat_one in zip(batch_paths, orig_hws, input_hws, selected_cpu):
            c, fh, fw = feat_one.shape
            stride_hw = _compute_stride(in_hw, (int(fh), int(fw)))
            meta = FeatureMeta(
                image_path=str(p),
                orig_hw=tuple(orig_hw),
                input_hw=tuple(in_hw),
                feat_hw=(int(fh), int(fw)),
                feat_channels=int(c),
                stride_hw=stride_hw,
                feat_source=str(args.feat_source),
                feat_level=int(args.feat_level),
                dtype=str(args.save_dtype),
            )
            out_npz = feat_dir / f"{p.stem}.npz"
            np.savez_compressed(
                out_npz,
                feat=feat_one.numpy().astype(save_dtype, copy=False),
                meta=json.dumps(asdict(meta), ensure_ascii=False),
            )
            index.append({"npz": str(out_npz), "image": str(p)})

    meta_out = run_dir / "meta.json"
    meta_out.write_text(
        json.dumps(
            {
                "config": str(args.resolved_config or args.config),
                "checkpoint": str(ckpt_path),
                "input_dir": str(input_dir),
                "split": str(args.split),
                "img_size": int(img_size),
                "normalize": bool(normalize),
                "rgb_mean": list(rgb_mean),
                "rgb_std": list(rgb_std),
                "feat_source": str(args.feat_source),
                "feat_level": int(args.feat_level),
                "save_dtype": str(args.save_dtype),
                "count": len(index),
                "files": index,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logging.info("完成：%d 张，已写入 %s", len(index), meta_out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
