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
import torch.nn.functional as F_torch
from omegaconf import OmegaConf, open_dict
from PIL import Image
from torchvision.transforms import functional as tvf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.misc import NestedTensor


"""
导出 RTMSFDETR（支持 RGB+MSI dual-stream）dense 特征图（每张图一个 npz）用于聚类/油类可视化。

典型用法（配合 tools/cluster_analysis 其它脚本）：
  python tools/cluster_analysis/repr_export_rtmsfdetr_rgb_msi.py \
    --resolved-config outputs/.../config.yaml \
    --device cuda --amp \
    --split val \
    --feat-source fpn --feat-level 0 \
    --output-dir outputs/repr/rtmsfdetr \
    --run-name oil_rgb_msi_val

之后可运行：
  python tools/cluster_analysis/repr_cluster_rtmsfdetr.py --run-dir outputs/repr/rtmsfdetr/<run_name> --k1 32 --k2 16 --l2norm --save-vis
  python tools/cluster_analysis/repr_auto_select_oil_clusters.py --run-dir ... --ann data/.../val.json
  python tools/cluster_analysis/repr_oil_heatmap.py --run-dir ...
  python tools/cluster_analysis/repr_oil_heatmap_panel.py --run-dir ... --ann ...
"""


@dataclass(frozen=True)
class FeatureMeta:
    # Keep compatibility with repr_cluster_rtmsfdetr.py (expects `image_path`).
    image_path: str
    msi_path: str
    orig_hw: tuple[int, int]
    input_hw: tuple[int, int]
    feat_hw: tuple[int, int]
    feat_channels: int
    stride_hw: tuple[int, int]
    feat_source: str
    feat_level: int
    dtype: str


@dataclass(frozen=True)
class PairItem:
    rgb: Path
    msi: Path


def get_config(node: Any, key: str, default: Any | None = None) -> Any:
    if node is None:
        return default
    if hasattr(node, "get"):
        return node.get(key, default)
    return getattr(node, key, default)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="导出已训练 RTMSFDETR（RGB+MSI）dense 表征特征图（用于聚类/伪彩色/油相关热力图）。"
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
        "--rgb-dir",
        type=str,
        default="",
        help="RGB 图片目录（默认按 cfg.data.dataset_dir 推导 dataset_dir/rgb/<split>）。",
    )
    parser.add_argument(
        "--msi-dir",
        type=str,
        default="",
        help="MSI 图片目录（默认按 cfg.data.dataset_dir 推导 dataset_dir/msi/<split>）。",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="当未显式传 --rgb-dir/--msi-dir 时，用 dataset_dir/{rgb,msi}/<split>（默认 val）。",
    )
    parser.add_argument("--recursive", action="store_true", help="递归扫描 rgb-dir。")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少张（0 表示不限制）。")
    parser.add_argument("--batch-size", type=int, default=1, help="导出 batch size（默认 1）。")
    parser.add_argument("--img-size", type=int, default=0, help="输入 resize 到方形大小（0=自动按 cfg 或 640）。")

    parser.add_argument(
        "--feat-source",
        type=str,
        default="fpn",
        choices=["fpn", "backbone"],
        help="导出特征来源：fpn=HybridEncoder 输出；backbone=HGNetv2 stage 输出（已融合）。",
    )
    parser.add_argument(
        "--feat-level",
        type=int,
        default=0,
        help="特征层级索引：fpn/backbone 通常为 0/1/2 对应 stride 8/16/32（取决于 RTv4 YAML 的 return_idx）。",
    )
    parser.add_argument(
        "--save-dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="保存到 npz 的特征精度（默认 float16）。",
    )
    parser.add_argument(
        "--strict-pairs",
        action="store_true",
        help="若开启，则遇到缺失 msi 配对文件直接报错；否则跳过并 warning。",
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
        help='输出子目录名（默认自动："<split>-rgb_msi-YYYYMMDD-HHMM"）。',
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


def _load_cfg(args: argparse.Namespace):
    config = args.resolved_config or args.config
    if args.resolved_config:
        cfg = OmegaConf.load(str(Path(config).expanduser()))
        OmegaConf.set_struct(cfg, False)
        if args.opts:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(args.opts)))
            OmegaConf.set_struct(cfg, False)
    else:
        from engines.core.parse_config import load_config  # type: ignore

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


def _filter_compatible_state_dict(model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    model_sd = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k not in model_sd:
            continue
        if model_sd[k].shape != v.shape:
            continue
        filtered[k] = v
    return filtered


def _resolve_dirs(cfg: Any, *, rgb_dir: str, msi_dir: str, split: str) -> tuple[Path, Path]:
    data_cfg = getattr(cfg, "data", None)
    dataset_dir = str(get_config(data_cfg, "dataset_dir", "")).strip()
    if not dataset_dir and (not str(rgb_dir).strip() or not str(msi_dir).strip()):
        raise ValueError("未提供 --rgb-dir/--msi-dir，且 cfg.data.dataset_dir 为空，无法推导输入目录。")
    root = Path(dataset_dir).expanduser() if dataset_dir else None

    if str(rgb_dir).strip():
        rgb = Path(rgb_dir).expanduser()
    else:
        assert root is not None
        rgb = root / "rgb" / str(split)

    if str(msi_dir).strip():
        msi = Path(msi_dir).expanduser()
    else:
        assert root is not None
        msi = root / "msi" / str(split)

    return rgb, msi


def _pair_rgb_msi(
    rgb_paths: list[Path],
    *,
    msi_dir: Path,
    ms_suffix: str,
    strict: bool,
) -> list[PairItem]:
    out: list[PairItem] = []
    suffix = str(ms_suffix or ".tif")
    if not suffix.startswith("."):
        suffix = "." + suffix
    for p in rgb_paths:
        stem = p.stem
        m = (msi_dir / f"{stem}{suffix}").expanduser()
        if m.is_file():
            out.append(PairItem(rgb=p, msi=m))
            continue
        if strict:
            raise FileNotFoundError(f"缺失 MSI 配对文件: rgb={p} expected_msi={m}")
        logging.warning("缺失 MSI 配对文件，跳过: rgb=%s expected_msi=%s", p, m)
    return out


def _rgb_normalize(
    x: torch.Tensor,
    *,
    mode: str,
    rgb_mean: tuple[float, float, float],
    rgb_std: tuple[float, float, float],
) -> torch.Tensor:
    mode = str(mode or "imagenet").lower()
    if mode == "imagenet":
        return tvf.normalize(x, mean=list(rgb_mean), std=list(rgb_std))
    if mode == "linear":
        return x
    if mode == "image_max":
        denom = float(x.amax().clamp_min(1e-6))
        return x / denom
    if mode == "per_channel_minmax":
        mins = x.amin(dim=(1, 2), keepdim=True)
        maxs = x.amax(dim=(1, 2), keepdim=True)
        denom = (maxs - mins).clamp_min(1e-6)
        return (x - mins) / denom
    raise ValueError(f"未知的 rgb_normalize_mode: {mode}")


def _load_msi_as_tensor(msi_path: Path, *, expected_channels: int) -> torch.Tensor:
    if not msi_path.is_file():
        raise FileNotFoundError(f"未找到多光谱文件：{msi_path}")

    try:
        import tifffile  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("需要 tifffile 才能读取多光谱 TIF。") from exc

    suffix = msi_path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        array = tifffile.imread(str(msi_path))
    else:
        # 允许“伪多光谱/红外图”以 jpg/png 存储
        with Image.open(msi_path) as img:
            img.load()
            array = np.array(img)

    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError(f"多光谱图像形状异常（需要 3 维）：{msi_path}, shape={array.shape}")

    dim0, dim1, dim2 = array.shape
    is_chw = dim0 <= 32 and dim0 < dim1 and dim0 < dim2
    is_hwc = dim2 <= 32 and dim2 < dim0 and dim2 < dim1
    if is_chw and is_hwc:
        exp = int(expected_channels)
        if dim0 == exp and dim2 != exp:
            is_hwc = False
        elif dim2 == exp and dim0 != exp:
            is_chw = False

    if is_chw and not is_hwc:
        array_hwc = np.transpose(array, (1, 2, 0))
    else:
        array_hwc = array

    c = int(array_hwc.shape[2])
    exp = int(expected_channels)
    if c == exp:
        pass
    elif exp == 1:
        if c > 1:
            array_hwc = array_hwc.mean(axis=2, keepdims=True)
    elif c == 1 and exp > 1:
        array_hwc = np.repeat(array_hwc, repeats=exp, axis=2)
    elif c > exp:
        array_hwc = array_hwc[..., :exp]
    else:
        raise ValueError(f"多光谱通道数不足：{msi_path}, channels={c}, expected={exp}")

    tensor = torch.from_numpy(np.transpose(array_hwc, (2, 0, 1)).astype(np.float32))
    return tensor


def _normalize_ms_tensor(
    ms: torch.Tensor,
    *,
    mode: str,
    scale_value: float | None,
) -> torch.Tensor:
    mode = str(mode or "per_channel_minmax").lower()
    if mode == "none":
        return ms
    if mode == "linear":
        if scale_value is None:
            raise ValueError("ms_normalize_mode=linear 需要 scale_value（data.ms_fixed_scale）。")
        return ms / float(scale_value)
    if mode == "per_channel_minmax":
        mins = ms.amin(dim=(1, 2), keepdim=True)
        maxs = ms.amax(dim=(1, 2), keepdim=True)
        denom = (maxs - mins).clamp_min(1e-6)
        return (ms - mins) / denom
    if mode == "tensor_minmax":
        lo = ms.amin()
        hi = ms.amax()
        denom = (hi - lo).clamp_min(1e-6)
        return (ms - lo) / denom
    if mode == "image_max":
        denom = ms.amax().clamp_min(1e-6)
        return ms / denom
    if mode == "fixed_scale":
        if scale_value is None:
            raise ValueError("ms_normalize_mode=fixed_scale 需要 scale_value（data.ms_fixed_scale）。")
        return torch.clamp(ms / float(scale_value), 0.0, 1.0)
    raise ValueError(f"未知的 ms_normalize_mode: {mode}")


def _resize_ms_tensor(ms: torch.Tensor, *, size_hw: tuple[int, int]) -> torch.Tensor:
    ms = ms.unsqueeze(0)
    out = F_torch.interpolate(ms, size=size_hw, mode="bilinear", align_corners=False)
    return out.squeeze(0)


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

    ckpt_path = _resolve_ckpt(args)

    with open_dict(cfg):
        cfg.runtime.device = str(device)
        cfg.runtime.device_ids = []
        cfg.runtime.world_size = 1
        cfg.mode = "test"

    data_cfg = getattr(cfg, "data", None)
    model_cfg = getattr(cfg, "model", None) or {}
    train_cfg = getattr(cfg, "train", None) or {}

    rgb_dir, msi_dir = _resolve_dirs(cfg, rgb_dir=str(args.rgb_dir), msi_dir=str(args.msi_dir), split=str(args.split))
    if not rgb_dir.is_dir():
        raise NotADirectoryError(f"rgb-dir 不存在或不是目录: {rgb_dir}")
    if not msi_dir.is_dir():
        raise NotADirectoryError(f"msi-dir 不存在或不是目录: {msi_dir}")

    rgb_paths = _iter_images(rgb_dir, recursive=bool(args.recursive))
    if args.limit and int(args.limit) > 0:
        rgb_paths = rgb_paths[: int(args.limit)]
    if not rgb_paths:
        raise FileNotFoundError(f"rgb-dir 未找到图片: {rgb_dir}")

    rgb_ch = int(get_config(data_cfg, "rgb_input_channels", 3) or 3)
    ms_ch = int(get_config(data_cfg, "ms_input_channels", 0) or 0)
    if ms_ch <= 0:
        raise ValueError("ms_input_channels<=0：该脚本用于 RGB+MSI 导特征，请检查配置 data.ms_input_channels。")
    if not bool(get_config(data_cfg, "use_msi_input", True)):
        raise ValueError("data.use_msi_input=false：该脚本用于 RGB+MSI 导特征，请检查配置。")

    rgb_mean = tuple(get_config(data_cfg, "rgb_mean", (0.485, 0.456, 0.406)))
    rgb_std = tuple(get_config(data_cfg, "rgb_std", (0.229, 0.224, 0.225)))
    rgb_mode = str(get_config(data_cfg, "rgb_normalize_mode", "imagenet") or "imagenet")
    ms_mode = str(get_config(data_cfg, "ms_normalize_mode", "per_channel_minmax") or "per_channel_minmax")
    ms_fixed_scale = get_config(data_cfg, "ms_fixed_scale", None)
    ms_center_to_rgb_range = bool(get_config(data_cfg, "ms_center_to_rgb_range", False))

    if int(args.img_size) > 0:
        img_size = int(args.img_size)
    else:
        img_size = int(get_config(data_cfg, "img_size", None) or get_config(train_cfg, "img_size", None) or 640)

    ms_suffix = str(get_config(data_cfg, "ms_suffix", ".tif") or ".tif")
    pairs = _pair_rgb_msi(rgb_paths, msi_dir=msi_dir, ms_suffix=ms_suffix, strict=bool(args.strict_pairs))
    if not pairs:
        raise RuntimeError("没有任何可用的 RGB/MSI 配对样本（可能全缺失 msi，或 --limit 太小）。")

    from argparse import Namespace
    from engines.models.rtmsfdetr.builder import build_model_and_processors

    build_args = Namespace(
        device=str(device),
        num_classes=int(get_config(train_cfg, "num_classes", get_config(model_cfg, "num_classes", 1))),
        img_size=int(img_size),
        rtdetrv4_config=get_config(model_cfg, "rtdetrv4_config", None),
        disable_distill=bool(get_config(model_cfg, "disable_distill", True)),
        teacher_repo_path=get_config(model_cfg, "teacher_repo_path", None),
        teacher_weights_path=get_config(model_cfg, "teacher_weights_path", None),
        hgnet_pretrained=bool(get_config(model_cfg, "hgnet_pretrained", False)),
        hgnet_local_model_dir=get_config(model_cfg, "hgnet_local_model_dir", None),
        hgnet_freeze_at=int(get_config(model_cfg, "hgnet_freeze_at", -1)),
        hgnet_freeze_norm=bool(get_config(model_cfg, "hgnet_freeze_norm", False)),
        input_denormalize=bool(get_config(model_cfg, "input_denormalize", True)),
        clamp_after_denormalize=bool(get_config(model_cfg, "clamp_after_denormalize", True)),
        rgb_mean=tuple(rgb_mean),
        rgb_std=tuple(rgb_std),
        rgb_input_channels=int(rgb_ch),
        ms_input_channels=int(ms_ch),
        input_channels=int(rgb_ch + ms_ch),
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
        run_name = f"{_sanitize_dirname(str(args.split))}-rgb_msi-{stamp}"
    run_dir = out_root / run_name
    feat_dir = run_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    batch_size = max(1, int(args.batch_size))
    save_dtype = np.float16 if str(args.save_dtype) == "float16" else np.float32

    logging.info("config=%s", args.resolved_config or args.config)
    logging.info("ckpt=%s", ckpt_path)
    logging.info("rgb_dir=%s msi_dir=%s (pairs=%d)", rgb_dir, msi_dir, len(pairs))
    logging.info(
        "device=%s amp=%s img_size=%d batch=%d feat_source=%s feat_level=%d save_dtype=%s",
        device,
        amp_enabled,
        int(img_size),
        batch_size,
        str(args.feat_source),
        int(args.feat_level),
        str(args.save_dtype),
    )
    logging.info("out=%s", run_dir)

    index: list[dict[str, Any]] = []

    for i0 in range(0, len(pairs), batch_size):
        batch = pairs[i0 : i0 + batch_size]
        tensors: list[torch.Tensor] = []
        orig_hws: list[tuple[int, int]] = []
        input_hws: list[tuple[int, int]] = []
        rgb_paths_batch: list[Path] = []
        msi_paths_batch: list[Path] = []

        for it in batch:
            rgb_img = Image.open(it.rgb).convert("RGB")
            orig_w, orig_h = rgb_img.size
            orig_hws.append((int(orig_h), int(orig_w)))

            if img_size > 0:
                rgb_resized = rgb_img.resize((int(img_size), int(img_size)), Image.BILINEAR)
            else:
                rgb_resized = rgb_img
            in_w, in_h = rgb_resized.size
            input_hws.append((int(in_h), int(in_w)))

            rgb_tensor = tvf.to_tensor(rgb_resized)
            rgb_tensor = _rgb_normalize(rgb_tensor, mode=rgb_mode, rgb_mean=rgb_mean, rgb_std=rgb_std)
            if rgb_tensor.shape[0] != rgb_ch:
                raise ValueError(f"RGB 通道数异常：expect {rgb_ch} got {rgb_tensor.shape[0]} file={it.rgb}")

            ms_tensor = _load_msi_as_tensor(it.msi, expected_channels=int(ms_ch))
            # resize ms to match rgb before normalization (训练管线要求同尺寸)
            if ms_tensor.shape[1:] != (orig_h, orig_w):
                # 兜底：允许 ms 与 rgb 原图不一致时，先对齐到 rgb 原图尺寸
                ms_tensor = _resize_ms_tensor(ms_tensor, size_hw=(int(orig_h), int(orig_w)))
            if img_size > 0:
                ms_tensor = _resize_ms_tensor(ms_tensor, size_hw=(int(img_size), int(img_size)))
            else:
                # align to rgb_resized size (may differ if img_size=0 but rgb was already resized in caller)
                ms_tensor = _resize_ms_tensor(ms_tensor, size_hw=(int(in_h), int(in_w)))

            ms_tensor = _normalize_ms_tensor(
                ms_tensor,
                mode=ms_mode,
                scale_value=float(ms_fixed_scale) if ms_fixed_scale is not None else None,
            )
            if ms_center_to_rgb_range and ms_mode.lower() in {
                "per_channel_minmax",
                "tensor_minmax",
                "image_max",
                "fixed_scale",
            }:
                ms_tensor = (ms_tensor - 0.5) / 0.5
            if ms_tensor.shape[0] != ms_ch:
                raise ValueError(f"MSI 通道数异常：expect {ms_ch} got {ms_tensor.shape[0]} file={it.msi}")

            tensors.append(torch.cat([rgb_tensor, ms_tensor], dim=0))
            rgb_paths_batch.append(it.rgb)
            msi_paths_batch.append(it.msi)

        batch_tensor = torch.stack(tensors, dim=0)
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
                    feat_maps = feats_backbone if str(args.feat_source) == "backbone" else raw.encoder(feats_backbone)
            else:
                images = detector._prepare_images(samples)  # type: ignore[attr-defined]
                raw = detector.model  # type: ignore[attr-defined]
                feats_backbone = raw.backbone(images)
                feat_maps = feats_backbone if str(args.feat_source) == "backbone" else raw.encoder(feats_backbone)

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
        for rgb_p, msi_p, orig_hw, in_hw, feat_one in zip(rgb_paths_batch, msi_paths_batch, orig_hws, input_hws, selected_cpu):
            c, fh, fw = feat_one.shape
            stride_hw = _compute_stride(in_hw, (int(fh), int(fw)))
            meta = FeatureMeta(
                image_path=str(rgb_p),
                msi_path=str(msi_p),
                orig_hw=tuple(orig_hw),
                input_hw=tuple(in_hw),
                feat_hw=(int(fh), int(fw)),
                feat_channels=int(c),
                stride_hw=stride_hw,
                feat_source=str(args.feat_source),
                feat_level=int(args.feat_level),
                dtype=str(args.save_dtype),
            )
            out_npz = feat_dir / f"{rgb_p.stem}.npz"
            np.savez_compressed(
                out_npz,
                feat=feat_one.numpy().astype(save_dtype, copy=False),
                meta=json.dumps(asdict(meta), ensure_ascii=False),
            )
            index.append({"npz": str(out_npz), "image": str(rgb_p), "msi": str(msi_p)})

    meta_out = run_dir / "meta.json"
    meta_out.write_text(
        json.dumps(
            {
                "config": str(args.resolved_config or args.config),
                "checkpoint": str(ckpt_path),
                "rgb_dir": str(rgb_dir),
                "msi_dir": str(msi_dir),
                "split": str(args.split),
                "img_size": int(img_size),
                "rgb_normalize_mode": str(rgb_mode),
                "ms_normalize_mode": str(ms_mode),
                "ms_fixed_scale": float(ms_fixed_scale) if ms_fixed_scale is not None else None,
                "ms_center_to_rgb_range": bool(ms_center_to_rgb_range),
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
