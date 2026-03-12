# ------------------------------------------------------------------------
# 多模态 COCO 数据集（RGB 与多光谱 TIF）
# 目的：支持按照自定义目录结构（见 README 中示例）读取 RGB 及 7 通道多光谱图像，
#       并提供便捷的 Dataset / DataLoader 构建函数，便于在主训练脚本中切换。
# ------------------------------------------------------------------------
# dataset/
# ├── annotations/
# │   ├── train.json
# │   └── val.json
# ├── rgb/
# │   ├── train/
# │   │   ├── 0001.jpg
# │   │   ├── 0002.jpg
# │   │   └── ...
# │   └── val/
# │       ├── 0005.jpg
# │       └── ...
# ├── msi/
# │   ├── train/
# │   │   ├── 0001.tif
# │   │   ├── 0002.tif
# │   │   └── ...
# │   └── val/
# │       ├── 0005.tif
# │       └── ...



from __future__ import annotations

import random
from dataclasses import dataclass
import math
import logging
from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Sequence, Tuple, Union, List

import numpy as np
import torch
import torch.nn.functional as F_torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from PIL import Image
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import tifffile
except ImportError as exc:  # pragma: no cover - tifffile 为核心依赖，缺失需立即提示
    raise ImportError(
        "需要 `tifffile` 才能读取多光谱 TIF 图像，请先 `pip install tifffile`。"
    ) from exc

from . import transforms as det_transforms
from .coco import (
    CocoDetection,
    compute_multi_scale_scales,
    make_coco_transforms,
    make_coco_transforms_square_div_64,
)
from utils.box_ops import box_xyxy_to_cxcywh
from utils.misc import collate_fn as default_collate_fn

PathLike = Union[str, Path]
logger = logging.getLogger(__name__)


def select_annotation_file(ann_dir: Path, image_set: str, prefer_bbox: bool) -> Path:
    """
    根据任务类型选择合适的标注文件，支持以下结构：

    annotations/
      ├── train.json
      ├── train_bbox.json
      ├── detection/train.json (旧结构，可选)
      └── segmentation/train.json (旧结构，可选)
    """

    def candidate_pair(root: Path) -> List[Path]:
        base = root / f"{image_set}.json"
        bbox = root / f"{image_set}_bbox.json"
        return [bbox, base] if prefer_bbox else [base, bbox]

    # 新推荐结构：优先使用 annotations/{split}.json（或 *_bbox.json），便于检测/分割共用同一份 COCO 标注。
    # 旧结构（annotations/detection, annotations/segmentation）仅作为回退以兼容历史数据组织方式。
    if prefer_bbox:
        candidate_roots: List[Path] = [ann_dir, ann_dir / "detection", ann_dir / "segmentation"]
    else:
        candidate_roots = [ann_dir, ann_dir / "segmentation", ann_dir / "detection"]

    candidates: List[Path] = []
    seen: set = set()
    for root in candidate_roots:
        # ann_dir 可能本身就是文件存放处，至少保证加入一次
        paths = candidate_pair(root if root != ann_dir else ann_dir)
        for path in paths:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(path)

    tried = []
    for path in candidates:
        tried.append(path)
        if path.is_file():
            if prefer_bbox and "segmentation" in path.parts:
                logger.warning(
                    "Detection 任务优先使用 detection 标注，但当前回退到 %s",
                    path,
                )
            if not prefer_bbox and "detection" in path.parts:
                logger.warning(
                    "Segmentation 任务优先使用 segmentation 标注，但当前回退到 %s",
                    path,
                )
            return path

    tried_str = ", ".join(str(p) for p in tried) if tried else "无可用候选"
    raise FileNotFoundError(f"未找到 {image_set} 对应的标注文件，尝试过：{tried_str}")


def _ensure_dataset_paths(
    root: Path,
    image_set: str,
    *,
    prefer_bbox: bool,
    require_rgb: bool = True,
    require_msi: bool = True,
) -> Tuple[Optional[Path], Optional[Path], Path]:
    """根据约定的目录结构返回 (rgb_dir, msi_dir, ann_path)。"""
    rgb_dir = root / "rgb" / image_set
    msi_dir = root / "msi" / image_set
    ann_dir = root / "annotations"
    ann_path = select_annotation_file(ann_dir, image_set, prefer_bbox=prefer_bbox)
    if require_rgb and not rgb_dir.is_dir():
        raise FileNotFoundError(f"未找到 RGB 图片目录：{rgb_dir}")
    if require_msi and not msi_dir.is_dir():
        raise FileNotFoundError(f"未找到多光谱目录：{msi_dir}")
    return rgb_dir, msi_dir, ann_path


def _default_rgb_transforms(
    image_set: str,
    img_size: int,
    *,
    multi_scale: bool = False,
    expanded_scales: bool = False,
    skip_random_resize: bool = False,
    patch_size: int = 16,
    num_windows: int = 4,
    square_resize_div_64: bool = False,
) -> Callable:
    """复用既有 COCO 变换配置，便于与主干训练逻辑对齐。"""
    if square_resize_div_64:
        divisor = patch_size * num_windows
        aligned_img_size = int(math.ceil(img_size / divisor) * divisor)
        return make_coco_transforms_square_div_64(
            image_set=image_set,
            img_size=aligned_img_size,
            multi_scale=multi_scale,
            expanded_scales=expanded_scales,
            skip_random_resize=skip_random_resize,
            patch_size=patch_size,
            num_windows=num_windows,
        )

    return make_coco_transforms(
        image_set=image_set,
        img_size=img_size,
        multi_scale=multi_scale,
        expanded_scales=expanded_scales,
        skip_random_resize=skip_random_resize,
        patch_size=patch_size,
        num_windows=num_windows,
    )


def build_rgb_only_dataset(
    dataset_root: PathLike,
    image_set: str,
    img_size: int,
    *,
    include_masks: bool = False,
    filter_annotations_without_masks: bool = True,
    drop_images_without_masks: bool = True,
    transform_builder: Optional[Callable[..., Callable]] = None,
    multi_scale: bool = False,
    expanded_scales: bool = False,
    skip_random_resize: bool = False,
    patch_size: int = 16,
    num_windows: int = 4,
    square_resize_div_64: bool = False,
    remap_mscoco_category: bool = False,
    category_names: Optional[Sequence[str]] = None,
) -> CocoDetection:
    """
    构建仅包含 RGB 模态的 COCO 数据集，目录结构需满足：

        dataset/
        ├── annotations/{train,val}.json
        └── rgb/{train,val}/*.jpg

    返回值可直接搭配 `utils.misc.collate_fn` 与现有训练管线。
    """
    root = Path(dataset_root)
    rgb_dir, _, ann_path = _ensure_dataset_paths(
        root,
        image_set,
        prefer_bbox=not include_masks,
        require_msi=False,
    )
    transforms = (
        transform_builder(image_set=image_set, img_size=img_size)
        if transform_builder
        else _default_rgb_transforms(
            image_set=image_set,
            img_size=img_size,
            multi_scale=multi_scale,
            expanded_scales=expanded_scales,
            skip_random_resize=skip_random_resize,
            patch_size=patch_size,
            num_windows=num_windows,
            square_resize_div_64=square_resize_div_64,
        )
    )
    return CocoDetection(
        str(rgb_dir),
        str(ann_path),
        transforms=transforms,
        include_masks=include_masks,
        filter_annotations_without_masks=filter_annotations_without_masks,
        drop_images_without_masks=drop_images_without_masks,
        remap_mscoco_category=remap_mscoco_category,
        category_names=category_names if remap_mscoco_category else None,
    )


def build_rgb_only_dataloader(
    dataset_root: PathLike,
    image_set: str,
    img_size: int,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    pin_memory: bool = True,
    include_masks: bool = False,
    filter_annotations_without_masks: bool = True,
    drop_images_without_masks: bool = True,
    drop_last: bool = False,
    multi_scale: bool = False,
    expanded_scales: bool = False,
    skip_random_resize: bool = False,
    patch_size: int = 16,
    num_windows: int = 4,
    square_resize_div_64: bool = False,
    remap_mscoco_category: bool = False,
    category_names: Optional[Sequence[str]] = None,
) -> DataLoader:
    """RGB 专用 DataLoader，封装常用参数。"""
    dataset = build_rgb_only_dataset(
        dataset_root,
        image_set,
        img_size,
        include_masks=include_masks,
        filter_annotations_without_masks=filter_annotations_without_masks,
        drop_images_without_masks=drop_images_without_masks,
        multi_scale=multi_scale,
        expanded_scales=expanded_scales,
        skip_random_resize=skip_random_resize,
        patch_size=patch_size,
        num_windows=num_windows,
        square_resize_div_64=square_resize_div_64,
        remap_mscoco_category=remap_mscoco_category,
        category_names=category_names,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=default_collate_fn,
    )


def build_multispectral_dataset(
    image_set: str,
    args,
    img_size: int,
    *,
    use_rgb_input: bool = True,
    use_msi_input: bool = True,
):
    """供 datasets.build_dataset 统一调用的多光谱 COCO 构建函数。"""
    subset = image_set.split("_")[0]
    dataset_root = getattr(args, "ms_dataset_dir", None) or getattr(args, "dataset_dir", None)
    if dataset_root is None:
        raise ValueError("请通过 --ms_dataset_dir 或 --dataset_dir 指定多光谱数据根目录。")

    root = Path(dataset_root)
    assert root.exists(), f"provided multispectral dataset path {root} does not exist"

    config = MultispectralDatasetConfig(
        img_size=img_size,
        include_masks=getattr(args, "segmentation_head", False),
        expected_ms_channels=getattr(args, "ms_expected_channels", 7),
        ms_suffix=getattr(args, "ms_suffix", ".tif"),
        ms_npy_layout=getattr(args, "ms_npy_layout", "auto"),
        random_horizontal_flip=not getattr(args, "ms_disable_random_flip", False),
        flip_prob=getattr(args, "ms_flip_prob", 0.5),
        rgb_normalize_mode=getattr(args, "rgb_normalize_mode", "imagenet"),
        rgb_mean=getattr(args, "rgb_mean", (0.485, 0.456, 0.406)),
        rgb_std=getattr(args, "rgb_std", (0.229, 0.224, 0.225)),
        ms_normalize_mode=getattr(args, "ms_normalize_mode", "per_channel_minmax"),
        # 默认按 16-bit TIFF 处理；若数据是 8-bit，请在 config 中显式设为 255.0
        ms_fixed_scale=getattr(args, "ms_fixed_scale", 65535.0),
        ms_center_to_rgb_range=getattr(args, "ms_center_to_rgb_range", False),
        ms_color_jitter_strength=getattr(args, "ms_color_jitter_strength", 0.0),
        dual_stream_output=getattr(args, "dual_stream_output", False),
        use_rgb_input=use_rgb_input,
        use_msi_input=use_msi_input,
        shared_transform=getattr(args, "shared_transform", "simple"),
        letterbox_fill_value=int(getattr(args, "letterbox_fill_value", 114)),
        letterbox_ms_fill_value=float(getattr(args, "letterbox_ms_fill_value", 0.0)),
        hsv_h=float(getattr(args, "hsv_h", 0.0)),
        hsv_s=float(getattr(args, "hsv_s", 0.0)),
        hsv_v=float(getattr(args, "hsv_v", 0.0)),
        degrees=float(getattr(args, "degrees", 0.0)),
        translate=float(getattr(args, "translate", 0.0)),
        scale=float(getattr(args, "scale", 0.0)),
        shear=float(getattr(args, "shear", 0.0)),
        perspective=float(getattr(args, "perspective", 0.0)),
        flipud=float(getattr(args, "flipud", 0.0)),
        fliplr=float(getattr(args, "fliplr", 0.5)),
        mosaic=float(getattr(args, "mosaic", 0.0)),
        square_resize_div_64=getattr(args, "square_resize_div_64", False),
        patch_size=getattr(args, "patch_size", 16),
        num_windows=getattr(args, "num_windows", 4),
        multi_scale=getattr(args, "multi_scale", False),
        expanded_scales=getattr(args, "expanded_scales", False),
        do_random_resize_via_padding=getattr(args, "do_random_resize_via_padding", False),
        remap_mscoco_category=getattr(args, "remap_mscoco_category", False),
        category_names=getattr(args, "class_names", None)
        if getattr(args, "remap_mscoco_category", False)
        else None,
        filter_annotations_without_masks=getattr(args, "filter_annotations_without_masks", True),
        drop_images_without_masks=getattr(args, "drop_images_without_masks", True),
    )
    if getattr(args, "dual_stream_backbone", False):
        config.dual_stream_output = True

    return CocoRgbMultispectralDataset(
        dataset_root=root,
        image_set=subset,
        config=config,
    )

def _load_msi_as_tensor(
    msi_path: Path,
    *,
    expected_channels: Optional[int] = None,
    npy_layout: Literal["auto", "cwh", "chw", "hwc"] = "auto",
) -> Tensor:
    """读取多光谱文件 -> float32 Tensor，自动调整为 [C, H, W]。

    兼容：
    - 多通道 TIFF（tifffile）
    - 常见图像格式（jpg/png/webp/bmp），用于“伪多光谱/红外图”这类单/三通道文件
    """
    if not msi_path.is_file():
        raise FileNotFoundError(f"未找到多光谱文件：{msi_path}")

    suffix = msi_path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        array = tifffile.imread(str(msi_path))
    elif suffix in {".npy", ".npz"}:
        if suffix == ".npy":
            array = np.load(msi_path, mmap_mode="r")
        else:
            with np.load(msi_path, mmap_mode="r") as bundle:
                keys = sorted(bundle.files)
                if not keys:
                    raise ValueError(f"空的 npz 文件：{msi_path}")
                array = bundle[keys[0]]
    elif suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        # LLVIP 等数据集的“红外/热成像”经常以 8-bit jpg/png 存储
        with Image.open(msi_path) as img:
            img.load()
            array = np.array(img)
    else:
        # 未知后缀：尽量按 TIFF 读，失败则提示用户配置 ms_suffix 或转换数据
        try:
            array = tifffile.imread(str(msi_path))
        except Exception as exc:
            raise ValueError(
                f"不支持的多光谱文件格式：{msi_path}（suffix={suffix}）。"
                "请检查 data.ms_suffix / 数据实际后缀，或将 MSI 转为 .tif/.tiff。"
            ) from exc

    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError(f"多光谱图像形状异常（需要 3 维）：{msi_path}, shape={array.shape}")

    # 统一为 HWC
    # 对于 TIFF/常规图像，保持历史行为：只在 CHW 明确成立时做 CHW->HWC。
    # 对于 npy/npz，再额外支持 MODA 常见的 CWH（如 [8, 1200, 900]）。
    dim0, dim1, dim2 = array.shape
    if npy_layout == "hwc":
        array_hwc = array
    elif npy_layout == "chw":
        array_hwc = np.transpose(array, (1, 2, 0))
    elif npy_layout == "cwh":
        array_hwc = np.transpose(array, (2, 1, 0))
    else:
        is_ch_first = dim0 <= 32 and dim0 < dim1 and dim0 < dim2
        is_ch_last = dim2 <= 32 and dim2 < dim0 and dim2 < dim1
        if is_ch_first and is_ch_last and expected_channels is not None:
            exp = int(expected_channels)
            if dim0 == exp and dim2 != exp:
                is_ch_last = False
            elif dim2 == exp and dim0 != exp:
                is_ch_first = False

        if suffix in {".npy", ".npz"} and is_ch_first and not is_ch_last:
            # 仅对 numpy 存档保留 CWH 自动判定；TIFF 的 [C,H,W] 在 oil/MSI 数据中是常态。
            if expected_channels is not None and int(expected_channels) == int(dim0) and dim1 >= dim2:
                array_hwc = np.transpose(array, (2, 1, 0))
            else:
                array_hwc = np.transpose(array, (1, 2, 0))
        elif is_ch_first and not is_ch_last:
            array_hwc = np.transpose(array, (1, 2, 0))
        else:
            array_hwc = array

    if array_hwc.ndim != 3:
        raise ValueError(f"多光谱图像 shape 解析失败：{msi_path}, shape={array.shape}")

    c = int(array_hwc.shape[2])
    if expected_channels is not None:
        exp = int(expected_channels)
        if exp <= 0:
            raise ValueError(f"expected_channels 必须为正整数，当前={expected_channels}")
        if c == exp:
            pass
        elif exp == 1:
            # 常见：红外 jpg 其实是 3 通道，但语义上是单通道
            if c > 1:
                array_hwc = array_hwc.mean(axis=2, keepdims=True)
            else:
                array_hwc = array_hwc[..., None] if array_hwc.ndim == 2 else array_hwc
        elif c == 1 and exp > 1:
            array_hwc = np.repeat(array_hwc, repeats=exp, axis=2)
        elif c > exp:
            array_hwc = array_hwc[..., :exp]
        else:
            raise ValueError(
                f"多光谱通道数不足：{msi_path}, channels={c}, expected={exp}"
            )

    tensor = torch.from_numpy(np.transpose(array_hwc, (2, 0, 1)).astype(np.float32))
    return tensor


def _resize_ms_tensor(tensor: Tensor, size_hw: Tuple[int, int]) -> Tensor:
    """借助 bilinear interpolate 统一多光谱尺寸。"""
    tensor = tensor.unsqueeze(0)
    resized = F_torch.interpolate(
        tensor,
        size=size_hw,
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0)


def _letterbox_ms_tensor(
    tensor: Tensor,
    *,
    out_size: int,
    scale: float,
    pad_left: int,
    pad_top: int,
    pad_right: int,
    pad_bottom: int,
    pad_value: float = 0.0,
) -> Tensor:
    """等比缩放 + padding 到方形，输出 [C, out_size, out_size]。"""
    if tensor.ndim != 3:
        raise ValueError(f"ms_tensor must have shape [C,H,W], got {tuple(tensor.shape)}")
    if out_size <= 0:
        raise ValueError(f"out_size must be positive, got {out_size}")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")

    c, h0, w0 = [int(x) for x in tensor.shape]
    new_h = int(round(h0 * float(scale)))
    new_w = int(round(w0 * float(scale)))
    new_h = max(1, min(new_h, out_size))
    new_w = max(1, min(new_w, out_size))

    resized = _resize_ms_tensor(tensor, (new_h, new_w))
    padded = F_torch.pad(
        resized,
        (int(pad_left), int(pad_right), int(pad_top), int(pad_bottom)),
        mode="constant",
        value=float(pad_value),
    )
    if padded.shape[1:] != (out_size, out_size):
        raise ValueError(
            "letterbox ms_tensor size mismatch: "
            f"expected={(out_size, out_size)}, got={tuple(padded.shape[1:])} "
            f"(pads={(pad_left, pad_right, pad_top, pad_bottom)}, resized={(new_h, new_w)})"
        )
    return padded


def _augment_hsv_rgb(img_rgb: np.ndarray, *, hgain: float, sgain: float, vgain: float) -> None:
    """In-place HSV augmentation for uint8 RGB image (HWC)."""
    if hgain <= 0 and sgain <= 0 and vgain <= 0:
        return
    if cv2 is None:
        raise ImportError("需要安装 opencv-python 才能使用 shared_transform=s2adet 的 HSV 增强。")
    if img_rgb.dtype != np.uint8:
        img_rgb[:] = np.clip(img_rgb, 0, 255).astype(np.uint8)
    r = np.random.uniform(-1, 1, 3) * np.array([hgain, sgain, vgain], dtype=np.float32) + 1.0
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(hsv)
    dtype = img_rgb.dtype
    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
    hsv_aug = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    img_rgb[:] = cv2.cvtColor(hsv_aug, cv2.COLOR_HSV2RGB)


def _box_candidates(
    *,
    box1: np.ndarray,
    box2: np.ndarray,
    wh_thr: float = 2.0,
    ar_thr: float = 20.0,
    area_thr: float = 0.10,
    eps: float = 1e-16,
) -> np.ndarray:
    """
    Filter candidate boxes after geometric transform.

    Args:
        box1: shape (4, N), boxes before transform (xyxy).
        box2: shape (4, N), boxes after transform (xyxy).
    """
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)


def _random_perspective_pair(
    img_rgb: np.ndarray,
    img_ms: np.ndarray,
    targets: np.ndarray,
    *,
    degrees: float,
    translate: float,
    scale: float,
    shear: float,
    perspective: float,
    border: tuple[int, int],
    border_value_rgb: tuple[int, int, int] = (114, 114, 114),
    border_value_ms: float = 114.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the same random perspective/affine to both modalities (RGB + MSI)."""
    if cv2 is None:
        raise ImportError("需要安装 opencv-python 才能使用 shared_transform=s2adet 的 random_perspective。")
    if img_rgb.shape[:2] != img_ms.shape[:2]:
        raise ValueError(f"RGB/MS shape mismatch in random_perspective: {img_rgb.shape} vs {img_ms.shape}")
    if targets.ndim != 2 or targets.shape[1] != 5:
        raise ValueError(f"targets must have shape [N,5] as (cls,x1,y1,x2,y2), got {targets.shape}")

    height = int(img_rgb.shape[0] + border[0] * 2)
    width = int(img_rgb.shape[1] + border[1] * 2)

    # Center
    C = np.eye(3, dtype=np.float32)
    C[0, 2] = -img_rgb.shape[1] / 2.0
    C[1, 2] = -img_rgb.shape[0] / 2.0

    # Perspective
    P = np.eye(3, dtype=np.float32)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)

    # Rotation and Scale
    R = np.eye(3, dtype=np.float32)
    angle = random.uniform(-degrees, degrees)
    scale_factor = random.uniform(1.0 - scale, 1.0 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale_factor)

    # Shear
    S = np.eye(3, dtype=np.float32)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180.0)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180.0)

    # Translation
    T = np.eye(3, dtype=np.float32)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

    # Combined rotation matrix
    M = T @ S @ R @ P @ C

    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        if perspective:
            img_rgb = cv2.warpPerspective(img_rgb, M, dsize=(width, height), borderValue=border_value_rgb)
            img_ms = cv2.warpPerspective(img_ms, M, dsize=(width, height), borderValue=border_value_ms)
        else:
            img_rgb = cv2.warpAffine(img_rgb, M[:2], dsize=(width, height), borderValue=border_value_rgb)
            img_ms = cv2.warpAffine(img_ms, M[:2], dsize=(width, height), borderValue=border_value_ms)

    n = int(targets.shape[0])
    if n:
        xy = np.ones((n * 4, 3), dtype=np.float32)
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        xy = xy @ M.T
        if perspective:
            xy = xy[:, :2] / xy[:, 2:3]
        else:
            xy = xy[:, :2]
        xy = xy.reshape(n, 8)

        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        valid = _box_candidates(box1=targets[:, 1:5].T * float(scale_factor), box2=new.T, area_thr=0.10)
        targets = targets[valid]
        targets[:, 1:5] = new[valid]

    return img_rgb, img_ms, targets


def _normalize_ms_tensor(
    tensor: Tensor,
    mode: Literal[
        "none",
        "linear",
        "per_channel_minmax",
        "tensor_minmax",
        "image_max",
        "fixed_scale",
        "dataset_standardize",
    ],
    *,
    scale_value: Optional[float],
    dataset_stats: Optional[Tuple[Tensor, Tensor]] = None,
) -> Tensor:
    if mode == "none":
        return tensor
    if mode == "linear":
        if not scale_value:
            raise ValueError("使用 linear 归一化必须提供 scale_value。")
        return tensor / float(scale_value)
    if mode == "per_channel_minmax":
        mins = tensor.amin(dim=(1, 2), keepdim=True)
        maxs = tensor.amax(dim=(1, 2), keepdim=True)
        denom = (maxs - mins).clamp_min(1e-6)
        return (tensor - mins) / denom
    if mode == "tensor_minmax":
        min_val = tensor.amin()
        max_val = tensor.amax()
        denom = (max_val - min_val).clamp_min(1e-6)
        return (tensor - min_val) / denom
    if mode == "image_max":
        max_val = tensor.amax().clamp_min(1e-6)
        return tensor / max_val
    if mode == "fixed_scale":
        if not scale_value:
            raise ValueError("使用 fixed_scale 归一化必须提供 scale_value。")
        return torch.clamp(tensor / float(scale_value), 0.0, 1.0)
    raise ValueError(f"未知的归一化模式：{mode}")


def _apply_ms_color_jitter(tensor: Tensor, *, strength: float) -> Tensor:
    """对多光谱张量执行轻量随机亮度/对比度扰动，仿照 RGB HSV 抖动。"""
    if strength <= 0:
        return tensor
    brightness_scale = 1.0 + (random.random() * 2.0 - 1.0) * strength
    jittered = tensor * brightness_scale
    contrast_scale = 1.0 + (random.random() * 2.0 - 1.0) * (strength * 0.5)
    channel_mean = jittered.mean(dim=(1, 2), keepdim=True)
    jittered = (jittered - channel_mean) * contrast_scale + channel_mean
    return torch.clamp(jittered, min=-5.0, max=5.0)

def _has_valid_segmentation(segmentation) -> bool:
    """
    判断 COCO annotation 的 segmentation 是否有效。

    - polygon: list[list[float]]，允许多段多边形
    - rle: dict，包含 {counts, size}
    """
    if segmentation is None:
        return False
    if isinstance(segmentation, dict):
        return "counts" in segmentation and "size" in segmentation
    if isinstance(segmentation, (list, tuple)):
        if len(segmentation) == 0:
            return False
        # 允许 polygon list 或混合结构；只要存在一个像样的多边形就视为有效
        for poly in segmentation:
            if isinstance(poly, (list, tuple)) and len(poly) >= 6:
                return True
        # 也可能是 RLE dict 的 list
        for poly in segmentation:
            if isinstance(poly, dict) and "counts" in poly and "size" in poly:
                return True
        return False
    return False


@dataclass
class MultispectralDatasetConfig:
    img_size: int
    include_masks: bool = False
    expected_ms_channels: int = 7
    ms_suffix: str = ".tif"
    ms_npy_layout: Literal["auto", "cwh", "chw", "hwc"] = "auto"
    random_horizontal_flip: bool = True
    flip_prob: float = 0.5
    rgb_normalize_mode: Literal["imagenet", "linear", "image_max", "per_channel_minmax"] = "imagenet"
    rgb_mean: Sequence[float] = (0.485, 0.456, 0.406)
    rgb_std: Sequence[float] = (0.229, 0.224, 0.225)
    ms_normalize_mode: Literal[
        "none",
        "linear",
        "per_channel_minmax",
        "tensor_minmax",
        "image_max",
        "fixed_scale",
    ] = "per_channel_minmax"
    # 默认按 16-bit TIFF 处理；若数据是 8-bit，请在 config 中显式设为 255.0
    ms_fixed_scale: Optional[float] = 65535.0
    ms_center_to_rgb_range: bool = False
    ms_color_jitter_strength: float = 0.0
    dual_stream_output: bool = False
    use_rgb_input: bool = True
    use_msi_input: bool = True
    # shared geometric transforms
    shared_transform: Literal["simple", "coco", "letterbox", "s2adet"] = "simple"
    # letterbox
    letterbox_fill_value: int = 114
    letterbox_ms_fill_value: float = 0.0
    # YOLOv5/S2ADet-style augmentation (train only, when shared_transform="s2adet")
    hsv_h: float = 0.0
    hsv_s: float = 0.0
    hsv_v: float = 0.0
    degrees: float = 0.0
    translate: float = 0.0
    scale: float = 0.0
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 0.0
    square_resize_div_64: bool = False
    patch_size: int = 16
    num_windows: int = 4
    multi_scale: bool = False
    expanded_scales: bool = False
    do_random_resize_via_padding: bool = False
    remap_mscoco_category: bool = False
    category_names: Optional[Sequence[str]] = None
    # segmentation-only: 过滤掉缺失 segmentation 的实例/图片，避免 mask/box 不一致导致 loss 崩溃
    filter_annotations_without_masks: bool = True
    drop_images_without_masks: bool = True


class CocoRgbMultispectralDataset(Dataset):
    """返回 RGB、多光谱或二者拼接后的 Tensor，兼容现有 DETR 训练接口。"""

    def __init__(
        self,
        dataset_root: PathLike,
        image_set: str,
        *,
        config: MultispectralDatasetConfig,
    ):
        root = Path(dataset_root)
        if not (config.use_rgb_input or config.use_msi_input):
            raise ValueError("至少需要启用 RGB 或 MSI 其中一种输入模态。")

        require_rgb = bool(config.use_rgb_input)
        require_msi = bool(config.use_msi_input)
        rgb_dir, msi_dir, ann_path = _ensure_dataset_paths(
            root,
            image_set,
            prefer_bbox=not config.include_masks,
            require_rgb=require_rgb,
            require_msi=require_msi,
        )

        self.image_set = image_set
        self.cfg = config
        self.dataset_root = root
        self.rgb_dir: Optional[Path] = rgb_dir if require_rgb else None
        self.msi_dir: Optional[Path] = msi_dir if require_msi else None

        coco_img_root = str(self.rgb_dir or self.msi_dir or root)
        coco_helper = CocoDetection(
            coco_img_root,
            str(ann_path),
            transforms=None,
            include_masks=config.include_masks,
            remap_mscoco_category=config.remap_mscoco_category,
            category_names=config.category_names,
        )
        self.coco = coco_helper.coco
        self.prepare = coco_helper.prepare
        self.ids = list(coco_helper.ids)

        self._filter_ids_by_modality_files()
        self._filter_ids_by_segmentation_annotations()

        if isinstance(config.img_size, int):
            self.output_hw = (config.img_size, config.img_size)
        else:
            self.output_hw = tuple(config.img_size)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return len(self.ids)

    def _load_prepared_item(self, idx: int):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        file_name = img_info.get("file_name", "")

        ms_tensor: Optional[Tensor] = None
        if self.cfg.use_msi_input:
            msi_path = self._resolve_msi_path(file_name)
            ms_tensor = _load_msi_as_tensor(
                msi_path,
                expected_channels=self.cfg.expected_ms_channels,
                npy_layout=self.cfg.ms_npy_layout,
            )

        if self.cfg.use_rgb_input:
            rgb_path = self._resolve_rgb_path(file_name)
            rgb_img = Image.open(rgb_path).convert("RGB")
        else:
            if ms_tensor is None:
                raise RuntimeError("use_rgb_input=False 时必须启用 use_msi_input。")
            h, w = ms_tensor.shape[1:]
            rgb_img = Image.new("RGB", (w, h))

        if self.cfg.use_rgb_input and ms_tensor is not None:
            if ms_tensor.shape[1:] != rgb_img.size[::-1]:
                raise ValueError(
                    f"RGB/MSI 尺寸不一致：rgb={rgb_img.size[::-1]} msi={ms_tensor.shape[1:]} "
                    f"(file_name={file_name})"
                )

        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        if self.cfg.include_masks and self.cfg.filter_annotations_without_masks:
            annotations = [
                ann
                for ann in annotations
                if ann.get("iscrowd", 0) == 0 and _has_valid_segmentation(ann.get("segmentation"))
            ]
        target = {"image_id": img_id, "annotations": annotations}
        rgb_img, target = self.prepare(rgb_img, target)
        return rgb_img, ms_tensor, target

    def __getitem__(self, idx: int):
        rgb_img, ms_tensor, target = self._load_prepared_item(idx)

        shared_mode = str(getattr(self.cfg, "shared_transform", "simple")).lower()
        if shared_mode == "s2adet":
            rgb_img, ms_tensor, target = self._apply_shared_transforms_s2adet(idx, rgb_img, ms_tensor, target)
        else:
            rgb_img, ms_tensor, target = self._apply_shared_transforms(rgb_img, ms_tensor, target)

        modalities: List[Tensor] = []
        rgb_tensor: Optional[Tensor] = None

        if self.cfg.use_rgb_input:
            rgb_tensor = F.to_tensor(rgb_img)
            if self.cfg.rgb_normalize_mode == "imagenet":
                rgb_tensor = F.normalize(rgb_tensor, mean=self.cfg.rgb_mean, std=self.cfg.rgb_std)
            elif self.cfg.rgb_normalize_mode == "linear":
                # F.to_tensor 已将 uint8 像素线性缩放到 [0,1]
                pass
            elif self.cfg.rgb_normalize_mode == "image_max":
                rgb_tensor = rgb_tensor / rgb_tensor.amax().clamp_min(1e-6)
            elif self.cfg.rgb_normalize_mode == "per_channel_minmax":
                mins = rgb_tensor.amin(dim=(1, 2), keepdim=True)
                maxs = rgb_tensor.amax(dim=(1, 2), keepdim=True)
                rgb_tensor = (rgb_tensor - mins) / (maxs - mins).clamp_min(1e-6)
            else:
                raise ValueError(f"未知的 RGB 归一化模式：{self.cfg.rgb_normalize_mode}")
            modalities.append(rgb_tensor)

        if self.cfg.use_msi_input:
            if ms_tensor is None:
                raise RuntimeError("ms_tensor should not be None when use_msi_input=True")
            # 归一化模式
            ms_tensor = _normalize_ms_tensor(
                ms_tensor,
                mode=self.cfg.ms_normalize_mode,
                scale_value=self.cfg.ms_fixed_scale,
            )
            if self.cfg.ms_center_to_rgb_range and self.cfg.ms_normalize_mode in {
                "per_channel_minmax",
                "tensor_minmax",
                "image_max",
                "fixed_scale",
            }:
                ms_tensor = (ms_tensor - 0.5) / 0.5
            if self.cfg.ms_color_jitter_strength > 0:
                ms_tensor = _apply_ms_color_jitter(ms_tensor, strength=self.cfg.ms_color_jitter_strength)
            target_hw = rgb_tensor.shape[1:] if rgb_tensor is not None else ms_tensor.shape[1:]
            if ms_tensor.shape[1:] != target_hw:
                ms_tensor = _resize_ms_tensor(ms_tensor, target_hw)
            modalities.append(ms_tensor)

        if self.cfg.dual_stream_output:
            sample: Dict[str, Tensor] = {}
            if rgb_tensor is not None:
                sample["rgb"] = rgb_tensor
            if ms_tensor is not None:
                sample["ms"] = ms_tensor
            return sample, target

        return torch.cat(modalities, dim=0), target

    def _resolve_msi_path(self, file_name: str) -> Path:
        if self.msi_dir is None:
            raise FileNotFoundError("未配置多光谱目录（use_msi_input=False 或目录缺失）。")
        source = Path(file_name)
        stem = source.stem
        candidate = self.msi_dir / f"{stem}{self.cfg.ms_suffix}"
        if candidate.is_file():
            return candidate
        fallback = self.msi_dir / source.name
        if fallback.is_file():
            return fallback
        globbed = next(iter(self.msi_dir.glob(f"{stem}.*")), None)
        if globbed is not None and globbed.is_file():
            return globbed
        raise FileNotFoundError(
            f"无法找到 {file_name} 对应的多光谱文件，尝试过 {candidate} 与 {fallback}"
        )

    def _resolve_rgb_path(self, file_name: str) -> Path:
        if self.rgb_dir is None:
            raise FileNotFoundError("未配置 RGB 目录（use_rgb_input=False 或目录缺失）。")
        source = Path(file_name)
        if source.is_absolute() and source.is_file():
            return source
        candidate = self.rgb_dir / source
        if candidate.is_file():
            return candidate
        fallback = self.rgb_dir / source.name
        if fallback.is_file():
            return fallback
        globbed = next(iter(self.rgb_dir.glob(f"{source.stem}.*")), None)
        if globbed is not None and globbed.is_file():
            return globbed
        return candidate

    @staticmethod
    def _scan_dir_stems(directory: Optional[Path]) -> set[str]:
        if directory is None or not directory.is_dir():
            return set()
        return {p.stem for p in directory.iterdir() if p.is_file() and not p.name.startswith(".")}

    def _filter_ids_by_modality_files(self) -> None:
        original = len(self.ids)

        if self.cfg.use_rgb_input and self.cfg.use_msi_input:
            rgb_stems = self._scan_dir_stems(self.rgb_dir)
            msi_stems = self._scan_dir_stems(self.msi_dir)
            only_rgb = rgb_stems - msi_stems
            only_msi = msi_stems - rgb_stems
            logger.info(
                "[%s] RGB/MSI 文件配对统计：只有rgb:%d种, 只有msi:%d种",
                self.image_set,
                len(only_rgb),
                len(only_msi),
            )
            paired = rgb_stems & msi_stems

            kept: List[int] = []
            for img_id in self.ids:
                file_name = self.coco.loadImgs(img_id)[0].get("file_name", "")
                if Path(file_name).stem not in paired:
                    continue
                if not self._resolve_rgb_path(file_name).is_file():
                    continue
                if not self._resolve_msi_path(file_name).is_file():
                    continue
                kept.append(img_id)
            self.ids = kept
            logger.info(
                "[%s] COCO 样本过滤：原始=%d, 保留=%d, 跳过=%d",
                self.image_set,
                original,
                len(self.ids),
                original - len(self.ids),
            )
            return

    def _filter_ids_by_segmentation_annotations(self) -> None:
        """
        在 segmentation 模式下，剔除“只有 bbox 没有 segmentation”的标注（以及因此为空的图片）。

        目的：
        - 保证 target["boxes"] 与 target["masks"] 一一对应
        - 避免包含空 target 导致 SciPy 匈牙利匹配报错（num_targets==0）
        """
        if not self.cfg.include_masks:
            return
        if not (self.cfg.filter_annotations_without_masks or self.cfg.drop_images_without_masks):
            return

        original = len(self.ids)
        kept: List[int] = []
        skipped = 0

        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            # 与 ConvertCoco 对齐：忽略 iscrowd
            anns = [a for a in anns if a.get("iscrowd", 0) == 0]
            if self.cfg.filter_annotations_without_masks:
                anns = [a for a in anns if _has_valid_segmentation(a.get("segmentation"))]

            if self.cfg.drop_images_without_masks and len(anns) == 0:
                skipped += 1
                continue
            kept.append(img_id)

        self.ids = kept
        if skipped:
            logger.warning(
                "[%s] segmentation 样本过滤：原始=%d, 保留=%d, 剔除无有效mask图片=%d",
                self.image_set,
                original,
                len(self.ids),
                skipped,
            )

        if self.cfg.use_msi_input and not self.cfg.use_rgb_input:
            kept: List[int] = []
            skipped = 0
            for img_id in self.ids:
                file_name = self.coco.loadImgs(img_id)[0].get("file_name", "")
                try:
                    msi_path = self._resolve_msi_path(file_name)
                except FileNotFoundError:
                    skipped += 1
                    continue
                if not msi_path.is_file():
                    skipped += 1
                    continue
                kept.append(img_id)
            self.ids = kept
            if skipped:
                logger.warning("[%s] MSI-only 样本过滤：由于文件缺失跳过=%d", self.image_set, skipped)
            return

        if self.cfg.use_rgb_input and not self.cfg.use_msi_input:
            kept: List[int] = []
            skipped = 0
            for img_id in self.ids:
                file_name = self.coco.loadImgs(img_id)[0].get("file_name", "")
                if not self._resolve_rgb_path(file_name).is_file():
                    skipped += 1
                    continue
                kept.append(img_id)
            self.ids = kept
            if skipped:
                logger.warning("[%s] RGB-only 样本过滤：由于文件缺失跳过=%d", self.image_set, skipped)
            return

    def _apply_shared_transforms(self, rgb_img, ms_tensor, target):
        mode = str(getattr(self.cfg, "shared_transform", "simple")).lower()
        if mode == "coco":
            return self._apply_shared_transforms_coco(rgb_img, ms_tensor, target)
        if mode == "letterbox":
            return self._apply_shared_transforms_letterbox(rgb_img, ms_tensor, target)
        # default: simple
        return self._apply_shared_transforms_simple(rgb_img, ms_tensor, target)

    def _apply_shared_transforms_simple(self, rgb_img, ms_tensor, target):
        # 随机水平翻转：仅在训练集启用，验证/测试保持确定性，便于指标对照。
        if self.image_set == "train" and self.cfg.random_horizontal_flip and random.random() < self.cfg.flip_prob:
            rgb_img, target = det_transforms.hflip(rgb_img, target)
            if ms_tensor is not None:
                ms_tensor = torch.flip(ms_tensor, dims=[2])

        # 统一缩放到目标尺寸（固定方形）
        rgb_img, target = det_transforms.resize(rgb_img, target, size=self.output_hw, max_size=None)
        if ms_tensor is not None:
            ms_tensor = _resize_ms_tensor(ms_tensor, rgb_img.size[::-1])
        target = self._normalize_target_boxes(target, rgb_img.size[::-1])
        return rgb_img, ms_tensor, target

    def _letterbox_pair_abs(
        self,
        rgb_img,
        ms_tensor,
        target,
        *,
        out_size: int,
        rgb_fill: int,
        ms_fill: float,
    ):
        """
        对 RGB + MSI 执行等比缩放 + padding 到方形（letterbox），并同步更新 target（xyxy 像素坐标）。

        注意：该函数不会把 boxes 归一化到 cxcywh；调用方需要自行调用 `_normalize_target_boxes`。
        """
        if out_size <= 0:
            raise ValueError(f"invalid img_size={out_size}")

        w0, h0 = rgb_img.size
        if w0 <= 0 or h0 <= 0:
            raise ValueError(f"invalid image size: {(w0, h0)}")

        # letterbox params
        scale = min(out_size / float(w0), out_size / float(h0))
        new_w = int(round(w0 * scale))
        new_h = int(round(h0 * scale))
        new_w = max(1, min(new_w, out_size))
        new_h = max(1, min(new_h, out_size))

        pad_w = out_size - new_w
        pad_h = out_size - new_h
        pad_left = int(pad_w // 2)
        pad_right = int(pad_w - pad_left)
        pad_top = int(pad_h // 2)
        pad_bottom = int(pad_h - pad_top)

        # resize + pad RGB
        resized_rgb = rgb_img.resize((new_w, new_h), resample=Image.BILINEAR)
        canvas = Image.new("RGB", (out_size, out_size), color=(int(rgb_fill), int(rgb_fill), int(rgb_fill)))
        canvas.paste(resized_rgb, (pad_left, pad_top))
        rgb_img = canvas

        # resize + pad MSI
        if ms_tensor is not None:
            ms_tensor = _letterbox_ms_tensor(
                ms_tensor,
                out_size=out_size,
                scale=scale,
                pad_left=pad_left,
                pad_top=pad_top,
                pad_right=pad_right,
                pad_bottom=pad_bottom,
                pad_value=float(ms_fill),
            )

        # update target boxes in absolute xyxy (still before normalization)
        if target is not None:
            target = target.copy()
            if "boxes" in target:
                boxes = target["boxes"]
                if torch.is_tensor(boxes) and boxes.numel() > 0:
                    boxes = boxes * boxes.new_tensor([scale, scale, scale, scale])
                    boxes[:, 0::2] += float(pad_left)
                    boxes[:, 1::2] += float(pad_top)
                    target["boxes"] = boxes

            if "area" in target:
                area = target["area"]
                if torch.is_tensor(area) and area.numel() > 0:
                    target["area"] = area * float(scale * scale)

            target["size"] = torch.tensor([out_size, out_size], dtype=torch.int64)
            # 记录 letterbox 元信息（用于评测阶段把 pred/gt 从 letterbox 坐标反算回原图）
            target["letterbox_pad"] = torch.tensor([pad_left, pad_top], dtype=torch.float32)
            target["letterbox_scale"] = torch.tensor([scale], dtype=torch.float32)

            if "masks" in target:
                masks = target["masks"]
                if torch.is_tensor(masks) and masks.numel() > 0:
                    m = masks[:, None].float()
                    m = F_torch.interpolate(m, size=(new_h, new_w), mode="nearest")
                    m = F_torch.pad(
                        m,
                        (pad_left, pad_right, pad_top, pad_bottom),
                        mode="constant",
                        value=0.0,
                    )
                    target["masks"] = (m[:, 0] > 0.5)

        return rgb_img, ms_tensor, target

    def _apply_shared_transforms_letterbox(self, rgb_img, ms_tensor, target):
        """
        YOLO-style letterbox（保持宽高比 + padding 到方形）：
        - train: 可选 flip + letterbox
        - val/test: letterbox
        """
        out_size = int(self.output_hw[0])

        # flip
        if self.image_set == "train" and self.cfg.random_horizontal_flip and random.random() < self.cfg.flip_prob:
            rgb_img, target = det_transforms.hflip(rgb_img, target)
            if ms_tensor is not None:
                ms_tensor = torch.flip(ms_tensor, dims=[2])

        rgb_fill = int(getattr(self.cfg, "letterbox_fill_value", 114))
        ms_fill = float(getattr(self.cfg, "letterbox_ms_fill_value", 0.0))

        rgb_img, ms_tensor, target = self._letterbox_pair_abs(
            rgb_img,
            ms_tensor,
            target,
            out_size=out_size,
            rgb_fill=rgb_fill,
            ms_fill=ms_fill,
        )
        target = self._normalize_target_boxes(target, rgb_img.size[::-1])
        return rgb_img, ms_tensor, target

    @staticmethod
    def _target_to_targets_np(target) -> np.ndarray:
        if target is None:
            return np.zeros((0, 5), dtype=np.float32)
        boxes = target.get("boxes")
        labels = target.get("labels")
        if not (torch.is_tensor(boxes) and torch.is_tensor(labels)):
            return np.zeros((0, 5), dtype=np.float32)
        if boxes.numel() == 0:
            return np.zeros((0, 5), dtype=np.float32)
        boxes_np = boxes.detach().cpu().numpy().astype(np.float32).reshape(-1, 4)
        labels_np = labels.detach().cpu().numpy().astype(np.float32).reshape(-1, 1)
        if boxes_np.shape[0] != labels_np.shape[0]:
            raise ValueError(f"boxes/labels shape mismatch: boxes={boxes_np.shape}, labels={labels_np.shape}")
        return np.concatenate([labels_np, boxes_np], axis=1)

    @staticmethod
    def _update_target_from_targets_np(target, targets_np: np.ndarray, *, out_hw: Tuple[int, int]):
        if target is None:
            target = {}
        else:
            target = target.copy()

        h, w = [int(x) for x in out_hw]
        target["size"] = torch.tensor([h, w], dtype=torch.int64)

        if targets_np is None or targets_np.size == 0:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)
            return target

        if targets_np.ndim != 2 or targets_np.shape[1] != 5:
            raise ValueError(f"targets_np must have shape [N,5], got {targets_np.shape}")

        boxes = torch.from_numpy(targets_np[:, 1:5].astype(np.float32))
        labels = torch.from_numpy(targets_np[:, 0].astype(np.int64))
        wh = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0)
        area = wh[:, 0] * wh[:, 1]
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        return target

    def _apply_s2adet_flips(
        self,
        rgb_np: np.ndarray,
        ms_np: Optional[np.ndarray],
        targets_np: np.ndarray,
    ) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        if rgb_np.ndim != 3:
            raise ValueError(f"rgb_np must have shape [H,W,C], got {rgb_np.shape}")
        h, w = [int(x) for x in rgb_np.shape[:2]]
        flipud_prob = float(getattr(self.cfg, "flipud", 0.0))
        fliplr_prob = float(getattr(self.cfg, "fliplr", 0.5))

        if flipud_prob > 0 and random.random() < flipud_prob:
            rgb_np = np.flipud(rgb_np)
            if ms_np is not None:
                ms_np = np.flipud(ms_np)
            if targets_np.size:
                y1 = targets_np[:, 2].copy()
                y2 = targets_np[:, 4].copy()
                targets_np[:, 2] = float(h) - y2
                targets_np[:, 4] = float(h) - y1

        if fliplr_prob > 0 and random.random() < fliplr_prob:
            rgb_np = np.fliplr(rgb_np)
            if ms_np is not None:
                ms_np = np.fliplr(ms_np)
            if targets_np.size:
                x1 = targets_np[:, 1].copy()
                x2 = targets_np[:, 3].copy()
                targets_np[:, 1] = float(w) - x2
                targets_np[:, 3] = float(w) - x1

        rgb_np = np.ascontiguousarray(rgb_np)
        if ms_np is not None:
            ms_np = np.ascontiguousarray(ms_np)

        if targets_np.size:
            np.clip(targets_np[:, 1], 0.0, float(w), out=targets_np[:, 1])
            np.clip(targets_np[:, 3], 0.0, float(w), out=targets_np[:, 3])
            np.clip(targets_np[:, 2], 0.0, float(h), out=targets_np[:, 2])
            np.clip(targets_np[:, 4], 0.0, float(h), out=targets_np[:, 4])
            bw = targets_np[:, 3] - targets_np[:, 1]
            bh = targets_np[:, 4] - targets_np[:, 2]
            keep = (bw > 1.0) & (bh > 1.0)
            targets_np = targets_np[keep]

        return rgb_np, ms_np, targets_np

    def _apply_shared_transforms_s2adet(self, idx: int, rgb_img, ms_tensor, target):
        """
        S2ADet/YOLOv5 风格数据增强（同步作用于 RGB + MSI + bbox）：
        - train: mosaic(prob=mosaic) + random_perspective + HSV + flipud/fliplr
                 or (no mosaic) letterbox + HSV + flipud/fliplr
        - val/test: letterbox（确定性）
        """
        if self.cfg.include_masks:
            logger.warning("shared_transform=s2adet 暂不支持 masks，已回退到 letterbox。")
            return self._apply_shared_transforms_letterbox(rgb_img, ms_tensor, target)

        out_size = int(self.output_hw[0])
        rgb_fill = int(getattr(self.cfg, "letterbox_fill_value", 114))
        ms_fill = float(getattr(self.cfg, "letterbox_ms_fill_value", float(rgb_fill)))

        if self.image_set != "train":
            return self._apply_shared_transforms_letterbox(rgb_img, ms_tensor, target)

        mosaic_prob = float(getattr(self.cfg, "mosaic", 0.0))
        if mosaic_prob > 0 and random.random() < mosaic_prob:
            return self._apply_shared_transforms_s2adet_mosaic(idx, out_size=out_size, rgb_fill=rgb_fill, ms_fill=ms_fill)

        # no mosaic: letterbox (no extra random_perspective, consistent with third_party/S2ADet)
        rgb_img, ms_tensor, target = self._letterbox_pair_abs(
            rgb_img,
            ms_tensor,
            target,
            out_size=out_size,
            rgb_fill=rgb_fill,
            ms_fill=ms_fill,
        )

        rgb_np = np.asarray(rgb_img)
        if rgb_np.dtype != np.uint8:
            rgb_np = np.clip(rgb_np, 0, 255).astype(np.uint8)

        ms_np: Optional[np.ndarray] = None
        if ms_tensor is not None:
            ms_np = ms_tensor.detach().cpu().numpy()
            if ms_np.ndim != 3:
                raise ValueError(f"ms_tensor must have shape [C,H,W], got {tuple(ms_tensor.shape)}")
            ms_np = np.transpose(ms_np, (1, 2, 0))
            if ms_np.ndim == 2:
                ms_np = ms_np[..., None]

        targets_np = self._target_to_targets_np(target)

        _augment_hsv_rgb(
            rgb_np,
            hgain=float(getattr(self.cfg, "hsv_h", 0.0)),
            sgain=float(getattr(self.cfg, "hsv_s", 0.0)),
            vgain=float(getattr(self.cfg, "hsv_v", 0.0)),
        )
        if ms_np is not None and ms_np.ndim == 3 and ms_np.shape[2] == 3:
            ms_uint8 = np.clip(ms_np, 0, 255).astype(np.uint8)
            _augment_hsv_rgb(
                ms_uint8,
                hgain=float(getattr(self.cfg, "hsv_h", 0.0)),
                sgain=float(getattr(self.cfg, "hsv_s", 0.0)),
                vgain=float(getattr(self.cfg, "hsv_v", 0.0)),
            )
            ms_np = ms_uint8.astype(np.float32)

        rgb_np, ms_np, targets_np = self._apply_s2adet_flips(rgb_np, ms_np, targets_np)

        rgb_img = Image.fromarray(rgb_np)
        if ms_np is not None:
            ms_tensor = torch.from_numpy(np.transpose(ms_np, (2, 0, 1)).astype(np.float32))

        target = self._update_target_from_targets_np(target, targets_np, out_hw=rgb_np.shape[:2])
        target = self._normalize_target_boxes(target, rgb_img.size[::-1])
        return rgb_img, ms_tensor, target

    def _apply_shared_transforms_s2adet_mosaic(
        self,
        idx: int,
        *,
        out_size: int,
        rgb_fill: int,
        ms_fill: float,
    ):
        if out_size <= 0:
            raise ValueError(f"invalid img_size={out_size}")
        if cv2 is None:
            raise ImportError("需要安装 opencv-python 才能使用 shared_transform=s2adet 的 mosaic。")

        use_ms = bool(getattr(self.cfg, "use_msi_input", False))
        mosaic_border = (-out_size // 2, -out_size // 2)
        yc, xc = [int(random.uniform(-x, 2 * out_size + x)) for x in mosaic_border]

        indices = [idx] + random.choices(range(len(self.ids)), k=3)

        img4_rgb: Optional[np.ndarray] = None
        img4_ms: Optional[np.ndarray] = None
        labels_all: List[np.ndarray] = []
        base_image_id = None

        for i, index in enumerate(indices):
            rgb_i, ms_i, target_i = self._load_prepared_item(index)
            if base_image_id is None and target_i is not None:
                base_image_id = target_i.get("image_id")

            rgb_np = np.asarray(rgb_i)
            if rgb_np.dtype != np.uint8:
                rgb_np = np.clip(rgb_np, 0, 255).astype(np.uint8)
            h, w = [int(x) for x in rgb_np.shape[:2]]

            ms_np: Optional[np.ndarray] = None
            if use_ms and ms_i is not None:
                ms_np = ms_i.detach().cpu().numpy()
                ms_np = np.transpose(ms_np, (1, 2, 0))
                if ms_np.ndim == 2:
                    ms_np = ms_np[..., None]

            if i == 0:
                img4_rgb = np.full((out_size * 2, out_size * 2, rgb_np.shape[2]), rgb_fill, dtype=np.uint8)
                if use_ms and ms_np is not None:
                    img4_ms = np.full((out_size * 2, out_size * 2, ms_np.shape[2]), ms_fill, dtype=ms_np.dtype)

            if img4_rgb is None:
                raise RuntimeError("mosaic base image init failed")

            # place img in img4 (same as YOLOv5)
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, out_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(out_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:  # i == 3: bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, out_size * 2), min(out_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)

            img4_rgb[y1a:y2a, x1a:x2a] = rgb_np[y1b:y2b, x1b:x2b]
            if use_ms and img4_ms is not None and ms_np is not None:
                img4_ms[y1a:y2a, x1a:x2a] = ms_np[y1b:y2b, x1b:x2b]

            padw = x1a - x1b
            padh = y1a - y1b

            t = self._target_to_targets_np(target_i)
            if t.size:
                t[:, 1] += float(padw)
                t[:, 3] += float(padw)
                t[:, 2] += float(padh)
                t[:, 4] += float(padh)
                labels_all.append(t)

        targets = np.concatenate(labels_all, axis=0) if labels_all else np.zeros((0, 5), dtype=np.float32)
        if targets.size:
            np.clip(targets[:, 1:5], 0.0, float(out_size * 2), out=targets[:, 1:5])
            bw = targets[:, 3] - targets[:, 1]
            bh = targets[:, 4] - targets[:, 2]
            targets = targets[(bw > 1.0) & (bh > 1.0)]

        if use_ms and img4_ms is None:
            raise RuntimeError("mosaic requires ms modality but img4_ms is None")

        if use_ms:
            img_rgb, img_ms, targets = _random_perspective_pair(
                img4_rgb,
                img4_ms,
                targets,
                degrees=float(getattr(self.cfg, "degrees", 0.0)),
                translate=float(getattr(self.cfg, "translate", 0.0)),
                scale=float(getattr(self.cfg, "scale", 0.0)),
                shear=float(getattr(self.cfg, "shear", 0.0)),
                perspective=float(getattr(self.cfg, "perspective", 0.0)),
                border=mosaic_border,
                border_value_rgb=(rgb_fill, rgb_fill, rgb_fill),
                border_value_ms=float(ms_fill),
            )
        else:
            img_rgb, _, targets = _random_perspective_pair(
                img4_rgb,
                img4_rgb,
                targets,
                degrees=float(getattr(self.cfg, "degrees", 0.0)),
                translate=float(getattr(self.cfg, "translate", 0.0)),
                scale=float(getattr(self.cfg, "scale", 0.0)),
                shear=float(getattr(self.cfg, "shear", 0.0)),
                perspective=float(getattr(self.cfg, "perspective", 0.0)),
                border=mosaic_border,
                border_value_rgb=(rgb_fill, rgb_fill, rgb_fill),
                border_value_ms=float(rgb_fill),
            )
            img_ms = None

        _augment_hsv_rgb(
            img_rgb,
            hgain=float(getattr(self.cfg, "hsv_h", 0.0)),
            sgain=float(getattr(self.cfg, "hsv_s", 0.0)),
            vgain=float(getattr(self.cfg, "hsv_v", 0.0)),
        )
        if img_ms is not None and img_ms.ndim == 3 and img_ms.shape[2] == 3:
            ms_uint8 = np.clip(img_ms, 0, 255).astype(np.uint8)
            _augment_hsv_rgb(
                ms_uint8,
                hgain=float(getattr(self.cfg, "hsv_h", 0.0)),
                sgain=float(getattr(self.cfg, "hsv_s", 0.0)),
                vgain=float(getattr(self.cfg, "hsv_v", 0.0)),
            )
            img_ms = ms_uint8.astype(np.float32)

        img_rgb, img_ms_opt, targets = self._apply_s2adet_flips(img_rgb, img_ms, targets)

        rgb_img = Image.fromarray(img_rgb)
        ms_tensor = (
            torch.from_numpy(np.transpose(img_ms_opt, (2, 0, 1)).astype(np.float32))
            if use_ms and img_ms_opt is not None
            else None
        )

        out_target = {
            "image_id": base_image_id if base_image_id is not None else torch.tensor([int(self.ids[idx])]),
            "orig_size": torch.tensor([out_size, out_size], dtype=torch.int64),
        }
        out_target = self._update_target_from_targets_np(out_target, targets, out_hw=(out_size, out_size))
        out_target = self._normalize_target_boxes(out_target, rgb_img.size[::-1])
        return rgb_img, ms_tensor, out_target

    @staticmethod
    def _square_resize(img, target, *, size: int):
        # 参考 datasets.transforms.SquareResize 的逻辑，但固定 size，便于同步到 ms_tensor
        rescaled_img = F.resize(img, (size, size))
        w, h = rescaled_img.size
        if target is None:
            return rescaled_img, None
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_img.size, img.size))
        ratio_width, ratio_height = ratios

        target = target.copy()
        if "boxes" in target:
            boxes = target["boxes"]
            target["boxes"] = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        if "area" in target:
            target["area"] = target["area"] * (ratio_width * ratio_height)
        target["size"] = torch.tensor([h, w])
        if "masks" in target:
            target["masks"] = det_transforms.interpolate(
                target["masks"][:, None].float(), (h, w), mode="nearest"
            )[:, 0] > 0.5
        return rescaled_img, target

    def _random_size_crop(self, img, target, *, min_size: int, max_size: int):
        w = random.randint(min_size, min(img.width, max_size))
        h = random.randint(min_size, min(img.height, max_size))
        region = det_transforms.T.RandomCrop.get_params(img, [h, w])  # type: ignore[attr-defined]
        # datasets.transforms.crop expects (i, j, h, w)
        cropped_img, cropped_target = det_transforms.crop(img, target, region)
        return cropped_img, cropped_target, region

    def _apply_shared_transforms_coco(self, rgb_img, ms_tensor, target):
        """
        COCO 风格的共享几何增强（同步作用于 RGB + MSI）：
        - train: RandomHorizontalFlip + RandomSelect(SquareResize, Resize->Crop->SquareResize)
        - val/test: SquareResize
        """
        img_size = int(self.output_hw[0])
        square_div_64 = bool(getattr(self.cfg, "square_resize_div_64", False))
        patch_size = int(getattr(self.cfg, "patch_size", 16))
        num_windows = int(getattr(self.cfg, "num_windows", 4))
        multi_scale = bool(getattr(self.cfg, "multi_scale", False))
        expanded_scales = bool(getattr(self.cfg, "expanded_scales", False))
        do_random_resize_via_padding = bool(getattr(self.cfg, "do_random_resize_via_padding", False))

        # flip
        if self.image_set == "train" and self.cfg.random_horizontal_flip and random.random() < self.cfg.flip_prob:
            rgb_img, target = det_transforms.hflip(rgb_img, target)
            if ms_tensor is not None:
                ms_tensor = torch.flip(ms_tensor, dims=[2])

        if not square_div_64:
            # 暂时只强对齐 square_div_64 版本（与 DINOv2 window/patch 要求更匹配）。
            # 非 square 模式回退到 simple。
            return self._apply_shared_transforms_simple(rgb_img, ms_tensor, target)

        scales = [img_size]
        if multi_scale:
            scales = compute_multi_scale_scales(
                img_size, expanded_scales=expanded_scales, patch_size=patch_size, num_windows=num_windows
            )
            if not do_random_resize_via_padding:
                scales = [scales[-1]]

        subset = self.image_set
        if subset == "train":
            if random.random() < 0.5:
                size = int(random.choice(scales))
                rgb_img, target = self._square_resize(rgb_img, target, size=size)
                if ms_tensor is not None:
                    ms_tensor = _resize_ms_tensor(ms_tensor, (size, size))
            else:
                # branch2: resize -> crop -> square resize
                rgb_img, target = det_transforms.resize(rgb_img, target, size=random.choice([400, 500, 600]), max_size=1333)
                if ms_tensor is not None:
                    ms_tensor = _resize_ms_tensor(ms_tensor, rgb_img.size[::-1])

                rgb_img, target, region = self._random_size_crop(rgb_img, target, min_size=384, max_size=600)
                if ms_tensor is not None:
                    i, j, h, w = region
                    ms_tensor = ms_tensor[:, i : i + h, j : j + w]

                size = int(random.choice(scales))
                rgb_img, target = self._square_resize(rgb_img, target, size=size)
                if ms_tensor is not None:
                    ms_tensor = _resize_ms_tensor(ms_tensor, (size, size))
        else:
            size = int(img_size)
            rgb_img, target = self._square_resize(rgb_img, target, size=size)
            if ms_tensor is not None:
                ms_tensor = _resize_ms_tensor(ms_tensor, (size, size))

        target = self._normalize_target_boxes(target, rgb_img.size[::-1])
        return rgb_img, ms_tensor, target

    def _normalize_target_boxes(self, target, size_hw: Tuple[int, int]):
        if target is None or "boxes" not in target:
            return target
        boxes = target["boxes"]
        if boxes.numel() == 0:
            return target
        boxes = box_xyxy_to_cxcywh(boxes)
        h, w = size_hw
        scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
        target["boxes"] = boxes / scale
        return target


__all__ = [
    "build_rgb_only_dataset",
    "build_rgb_only_dataloader",
    "MultispectralDatasetConfig",
    "CocoRgbMultispectralDataset",
    "build_multispectral_dataset",
    "select_annotation_file",
]
