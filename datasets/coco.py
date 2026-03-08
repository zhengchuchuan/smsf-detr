# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import logging

import torch
import torch.utils.data
import torchvision
try:
    import pycocotools.mask as coco_mask
except Exception as exc:  # pragma: no cover
    coco_mask = None  # type: ignore[assignment]
    _pycocotools_import_error = exc

import datasets.transforms as T


def compute_multi_scale_scales(img_size, expanded_scales=False, patch_size=16, num_windows=4):
    # round to the nearest multiple of 4*patch_size to enable both patching and windowing
    base_num_patches_per_window = img_size // (patch_size * num_windows)
    offsets = [-3, -2, -1, 0, 1, 2, 3, 4] if not expanded_scales else [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    scales = [base_num_patches_per_window + offset for offset in offsets]
    proposed_scales = [scale * patch_size * num_windows for scale in scales]
    proposed_scales = [scale for scale in proposed_scales if scale >= patch_size * num_windows * 2]  # ensure minimum image size
    return proposed_scales


def convert_coco_poly_to_mask(segmentations, height, width):
    """Convert polygon segmentation to a binary mask tensor of shape [N, H, W].
    Requires pycocotools.
    """
    if coco_mask is None:  # pragma: no cover
        raise ImportError(
            "当前环境未安装 `pycocotools`，无法将 COCO segmentation 转为 mask。"
            "若你仅做 bbox 检测任务可忽略该函数；如需 segmentation，请先安装：pip install pycocotools"
        ) from _pycocotools_import_error
    masks = []
    for polygons in segmentations:
        if polygons is None or len(polygons) == 0:
            # empty segmentation for this instance
            masks.append(torch.zeros((height, width), dtype=torch.uint8))
            continue
        try:
            rles = coco_mask.frPyObjects(polygons, height, width)
        except:
            rles = polygons
        mask = coco_mask.decode(rles)
        if mask.ndim < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if len(masks) == 0:
        return torch.zeros((0, height, width), dtype=torch.uint8)
    return torch.stack(masks, dim=0)

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
        # polygon
        for poly in segmentation:
            if isinstance(poly, (list, tuple)) and len(poly) >= 6:
                return True
        # list[rle]
        for poly in segmentation:
            if isinstance(poly, dict) and "counts" in poly and "size" in poly:
                return True
        return False
    return False


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        include_masks=False,
        remap_mscoco_category=False,
        category_names=None,
        filter_annotations_without_masks: bool = True,
        drop_images_without_masks: bool = True,
    ):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.include_masks = include_masks
        cat_id_map = None
        normalized_names = []
        # 去除coco中的背景类别
        if category_names:
            normalized_names = [
                str(name).strip().lower()
                for name in category_names
                if name and str(name).strip().lower() not in {"background", "_background_"}
            ]
        if remap_mscoco_category or normalized_names:
            candidate_map = {}
            matched_names = set()
            if normalized_names:
                name_to_index = {name: idx for idx, name in enumerate(normalized_names)}
                for cat in self.coco.cats.values():
                    name = str(cat.get("name", "")).lower()
                    mapped_idx = name_to_index.get(name)
                    if mapped_idx is not None:
                        candidate_map[cat["id"]] = mapped_idx
                        matched_names.add(name)
                if len(matched_names) == len(normalized_names):
                    cat_id_map = candidate_map
                elif candidate_map:
                    logging.warning(
                        "Category names provided (%s) do not fully match dataset categories; "
                        "matched %d / %d. Falling back to default ordering.",
                        normalized_names,
                        len(matched_names),
                        len(normalized_names),
                    )
            if cat_id_map is None:
                filtered_ids = []
                for cat in self.coco.cats.values():
                    name = str(cat.get("name", "")).lower()
                    if name in {"background", "_background_"}:
                        continue
                    filtered_ids.append(cat["id"])
                for idx, cat_id in enumerate(sorted(filtered_ids)):
                    candidate_map[cat_id] = idx
                cat_id_map = candidate_map or None
        self.prepare = ConvertCoco(
            include_masks=include_masks,
            category_id_map=cat_id_map,
            filter_annotations_without_masks=filter_annotations_without_masks,
        )

        if include_masks and drop_images_without_masks:
            kept = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anns = self.coco.loadAnns(ann_ids)
                anns = [a for a in anns if a.get("iscrowd", 0) == 0]
                if filter_annotations_without_masks:
                    anns = [a for a in anns if _has_valid_segmentation(a.get("segmentation"))]
                if len(anns) == 0:
                    continue
                kept.append(img_id)
            self.ids = kept

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


class ConvertCoco(object):

    def __init__(self, include_masks=False, category_id_map=None, filter_annotations_without_masks: bool = True):
        self.include_masks = include_masks
        self.category_id_map = category_id_map
        self.filter_annotations_without_masks = filter_annotations_without_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        if self.category_id_map is not None:
            filtered = []
            for obj in anno:
                mapped = self.category_id_map.get(obj["category_id"])
                if mapped is None:
                    continue
                obj = obj.copy()
                obj["category_id"] = mapped
                filtered.append(obj)
            anno = filtered

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        if self.include_masks and self.filter_annotations_without_masks:
            anno = [obj for obj in anno if _has_valid_segmentation(obj.get("segmentation"))]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area_vals = []
        for obj in anno:
            area_val = obj.get("area", None)
            if area_val is None:
                bbox = obj.get("bbox", None)
                if bbox and len(bbox) == 4:
                    area_val = float(bbox[2]) * float(bbox[3])
                else:
                    area_val = 0.0
            area_vals.append(area_val)
        area = torch.tensor(area_vals, dtype=torch.float32)
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        # add segmentation masks if requested, otherwise ensure consistent key when include_masks=True
        if self.include_masks:
            segmentations = [obj.get("segmentation", None) for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)
            if masks.numel() > 0:
                target["masks"] = masks[keep]
            else:
                target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)

            target["masks"] = target["masks"].bool()

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, img_size, multi_scale=False, expanded_scales=False, skip_random_resize=False, patch_size=16, num_windows=4):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [img_size]
    if multi_scale:
        # scales = [448, 512, 576, 640, 704, 768, 832, 896]
        scales = compute_multi_scale_scales(img_size, expanded_scales, patch_size, num_windows)
        if skip_random_resize:
            scales = [scales[-1]]
        print(scales)

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([img_size], max_size=1333),
            normalize,
        ])
    if image_set == 'val_speed':
        return T.Compose([
            T.SquareResize([img_size]),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_coco_transforms_square_div_64(image_set, img_size, multi_scale=False, expanded_scales=False, skip_random_resize=False, patch_size=16, num_windows=4):
    """
    """

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    scales = [img_size]
    if multi_scale:
        # scales = [448, 512, 576, 640, 704, 768, 832, 896]
        scales = compute_multi_scale_scales(img_size, expanded_scales, patch_size, num_windows)
        if skip_random_resize:
            scales = [scales[-1]]
        print(scales)

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.SquareResize(scales),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.SquareResize(scales),
                ]),
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.SquareResize([img_size]),
            normalize,
        ])
    if image_set == 'test':
        return T.Compose([
            T.SquareResize([img_size]),
            normalize,
        ])
    if image_set == 'val_speed':
        return T.Compose([
            T.SquareResize([img_size]),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def build(image_set, args, img_size):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root /  "val2017", root / "annotations" / f'{mode}_val2017.json'),
        "test": (root / "test2017", root / "annotations" / f'image_info_test-dev2017.json'),
    }
    
    img_folder, ann_file = PATHS[image_set.split("_")[0]]
    
    try:
        square_resize = args.square_resize
    except:
        square_resize = False
    
    try:
        square_resize_div_64 = args.square_resize_div_64
    except:
        square_resize_div_64 = False

    
    remap = getattr(args, "remap_mscoco_category", False)

    if square_resize_div_64:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms_square_div_64(
            image_set,
            img_size,
            multi_scale=args.multi_scale,
            expanded_scales=args.expanded_scales,
            skip_random_resize=not args.do_random_resize_via_padding,
            patch_size=args.patch_size,
            num_windows=args.num_windows
        ), remap_mscoco_category=remap)
    else:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(
            image_set,
            img_size,
            multi_scale=args.multi_scale,
            expanded_scales=args.expanded_scales,
            skip_random_resize=not args.do_random_resize_via_padding,
            patch_size=args.patch_size,
            num_windows=args.num_windows
        ), remap_mscoco_category=remap)
    return dataset

def build_roboflow(image_set, args, img_size):
    root = Path(args.dataset_dir)
    assert root.exists(), f'provided Roboflow path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train", root / "train" / "_annotations.coco.json"),
        "val": (root /  "valid", root / "valid" / "_annotations.coco.json"),
        "test": (root / "test", root / "test" / "_annotations.coco.json"),
    }
    
    img_folder, ann_file = PATHS[image_set.split("_")[0]]
    
    try:
        square_resize = args.square_resize
    except:
        square_resize = False
    
    try:
        square_resize_div_64 = args.square_resize_div_64
    except:
        square_resize_div_64 = False
    
    try:
        include_masks = args.segmentation_head
    except:
        include_masks = False

    
    remap = getattr(args, "remap_mscoco_category", False)

    if square_resize_div_64:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms_square_div_64(
            image_set,
            img_size,
            multi_scale=args.multi_scale,
            expanded_scales=args.expanded_scales,
            skip_random_resize=not args.do_random_resize_via_padding,
            patch_size=args.patch_size,
            num_windows=args.num_windows
        ), include_masks=include_masks, remap_mscoco_category=remap)
    else:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(
            image_set,
            img_size,
            multi_scale=args.multi_scale,
            expanded_scales=args.expanded_scales,
            skip_random_resize=not args.do_random_resize_via_padding,
            patch_size=args.patch_size,
            num_windows=args.num_windows
        ), include_masks=include_masks, remap_mscoco_category=remap)
    return dataset
