from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

try:
    from pycocotools.coco import COCO  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError("需要 `pycocotools` 才能合并 COCO 数据集。") from exc

logger = logging.getLogger(__name__)


def _to_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return int(str(x))


def _normalize_categories(coco: COCO) -> List[Tuple[int, str]]:
    pairs = []
    for cat_id, meta in getattr(coco, "cats", {}).items():
        pairs.append((_to_int(cat_id), str(meta.get("name", "")).strip().lower()))
    pairs.sort(key=lambda t: t[0])
    return pairs


@dataclass(frozen=True)
class _CocoMergePlan:
    image_offset: int
    ann_offset: int


def merge_coco_apis(coco_list: Sequence[COCO]) -> Tuple[COCO, List[_CocoMergePlan]]:
    """
    合并多个 COCO API（用于 COCOEvaluator）：
    - 对每个子数据集的 image_id / ann_id 做 offset，避免冲突
    - categories 做并集（同一 id 的 name 不一致会 warning）
    """
    if not coco_list:
        raise ValueError("coco_list 不能为空。")

    categories_by_id: Dict[int, Dict[str, Any]] = {}
    for i, coco in enumerate(coco_list):
        for cat_id, meta in getattr(coco, "cats", {}).items():
            cid = _to_int(cat_id)
            name = str(meta.get("name", "")).strip()
            existing = categories_by_id.get(cid)
            if existing is None:
                categories_by_id[cid] = {"id": cid, "name": name, **{k: v for k, v in meta.items() if k not in {"id"}}}
                continue
            if str(existing.get("name", "")).strip().lower() != name.lower():
                logger.warning(
                    "合并 COCO 数据集时同一 category_id=%d 的 name 不一致（idx=%d）：%s vs %s",
                    cid,
                    i,
                    existing.get("name", ""),
                    name,
                )

    merged_dict: Dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": [categories_by_id[k] for k in sorted(categories_by_id.keys())],
    }

    plans: List[_CocoMergePlan] = []
    max_img_id = -1
    max_ann_id = -1

    for i, coco in enumerate(coco_list):
        img_offset = 0 if i == 0 else max_img_id + 1
        ann_offset = 0 if i == 0 else max_ann_id + 1
        plans.append(_CocoMergePlan(image_offset=img_offset, ann_offset=ann_offset))

        images = coco.dataset.get("images", [])  # type: ignore[attr-defined]
        annotations = coco.dataset.get("annotations", [])  # type: ignore[attr-defined]

        next_missing_ann_id = ann_offset
        for img in images:
            img = dict(img)
            img_id = _to_int(img.get("id"))
            new_img_id = img_id + img_offset
            img["id"] = new_img_id
            merged_dict["images"].append(img)
            max_img_id = max(max_img_id, new_img_id)

        for ann in annotations:
            ann = dict(ann)
            ann_id = ann.get("id", None)
            img_id = _to_int(ann.get("image_id"))
            ann["image_id"] = img_id + img_offset
            if ann_id is None:
                ann["id"] = next_missing_ann_id
                next_missing_ann_id += 1
            else:
                ann["id"] = _to_int(ann_id) + ann_offset
            max_ann_id = max(max_ann_id, _to_int(ann["id"]))
            merged_dict["annotations"].append(ann)

        if next_missing_ann_id > max_ann_id + 1:
            max_ann_id = next_missing_ann_id - 1

    merged = COCO()
    merged.dataset = merged_dict  # type: ignore[assignment]
    merged.createIndex()
    return merged, plans


class MultiCocoDataset(Dataset):
    """
    将多个“COCO 风格数据集实现”在 split 维度上拼接，并保持 COCOEvaluator 可用。

    要求：
    - 每个子 dataset 的 __getitem__ 返回 (sample, target)
    - target 中包含 image_id（tensor 或 int）
    - dataset 可通过 datasets.get_coco_api_from_dataset 获取 coco api（含 coco.dataset）
    """

    def __init__(self, datasets: Sequence[Dataset], coco_apis: Sequence[COCO]):
        if len(datasets) != len(coco_apis):
            raise ValueError("datasets 与 coco_apis 数量不一致。")
        if not datasets:
            raise ValueError("datasets 不能为空。")

        self.datasets = list(datasets)
        self.coco, plans = merge_coco_apis(list(coco_apis))
        self._image_offsets = [p.image_offset for p in plans]
        self._cumulative_sizes = self._build_cumulative_sizes(self.datasets)
        self.ids = sorted(list(getattr(self.coco, "imgs", {}).keys()))

    @staticmethod
    def _build_cumulative_sizes(datasets: Sequence[Dataset]) -> List[int]:
        cumulative: List[int] = []
        running = 0
        for ds in datasets:
            running += len(ds)
            cumulative.append(running)
        return cumulative

    def __len__(self) -> int:
        return self._cumulative_sizes[-1]

    def __getitem__(self, idx: int):
        if idx < 0:
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self._cumulative_sizes, idx)
        sample_idx = idx if dataset_idx == 0 else idx - self._cumulative_sizes[dataset_idx - 1]

        sample, target = self.datasets[dataset_idx][sample_idx]
        offset = int(self._image_offsets[dataset_idx])

        if target is not None and "image_id" in target:
            image_id = target["image_id"]
            if torch.is_tensor(image_id):
                target = dict(target)
                target["image_id"] = image_id + offset
            else:
                target = dict(target)
                target["image_id"] = _to_int(image_id) + offset
        return sample, target
