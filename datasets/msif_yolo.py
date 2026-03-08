from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import Dataset

from datasets.multispectral_coco import CocoRgbMultispectralDataset
from utils.box_ops import box_xyxy_to_cxcywh


class MsifYoloStyleDataset(Dataset):
    """
    轻量包装：将 CocoRgbMultispectralDataset 的输出转换为更接近 YOLO 的 target 形式（cxcywh 归一化）。
    不引入 Mosaic/HSV，只做格式对齐，便于 MSCFT 在保持现有管线的同时使用 YOLO 预训练/损失。
    """

    def __init__(self, base_dataset: CocoRgbMultispectralDataset, img_size: int):
        self.base = base_dataset
        self.img_size = img_size
        self.ids = getattr(base_dataset, "ids", list(range(len(base_dataset))))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        sample, target = self.base[idx]
        boxes = target.get("boxes")
        if boxes is not None and boxes.numel() > 0:
            # 如果是 xyxy 像素坐标，转换成归一化 cxcywh
            if float(boxes.max()) > 1.5:
                h, w = target.get("size", target.get("orig_size", None))
                if h is None or w is None:
                    h = w = self.img_size
                else:
                    h = int(h[0]) if isinstance(h, torch.Tensor) else int(h)
                    w = int(w[1]) if isinstance(target.get("size"), torch.Tensor) else int(w)
                cxcywh = box_xyxy_to_cxcywh(boxes)
                scale = torch.tensor([w, h, w, h], dtype=cxcywh.dtype, device=cxcywh.device)
                boxes = cxcywh / scale
            target["boxes"] = boxes
        return sample, target


def build_msif_yolo_dataset(image_set: str, args, img_size: int) -> MsifYoloStyleDataset:
    """构建基于 CocoRgbMultispectralDataset 的 YOLO 风格 Dataset。"""
    from datasets.multispectral_coco import build_multispectral_dataset

    base_dataset = build_multispectral_dataset(
        image_set,
        args,
        img_size,
        use_rgb_input=bool(getattr(args, "use_rgb_input", True)),
        use_msi_input=bool(getattr(args, "use_msi_input", True)),
    )
    return MsifYoloStyleDataset(base_dataset, img_size=img_size)
