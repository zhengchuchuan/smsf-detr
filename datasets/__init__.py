# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import torch.utils.data
import torchvision
import copy
import logging

from .coco import build as build_coco
from .coco import build_roboflow
from .o365 import build_o365
from .multispectral_coco import build_multispectral_dataset, build_rgb_only_dataset
try:
    from .multi_dataset import MultiCocoDataset
except ImportError:  # pragma: no cover
    MultiCocoDataset = None  # type: ignore[assignment]


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    coco_api = getattr(dataset, "coco", None)
    if coco_api is not None:
        return coco_api


def build_dataset(image_set, args, img_size):
    dataset_dirs = getattr(args, "dataset_dirs", None)
    if dataset_dirs:
        if MultiCocoDataset is None:
            raise ImportError("需要 `pycocotools` 才能合并 COCO 数据集，请先 `pip install pycocotools`。")
        try:
            from omegaconf import ListConfig  # type: ignore
        except Exception:  # pragma: no cover
            ListConfig = ()  # type: ignore

        if isinstance(dataset_dirs, ListConfig):
            dataset_dirs = list(dataset_dirs)
        elif isinstance(dataset_dirs, (list, tuple)):
            dataset_dirs = list(dataset_dirs)
        elif isinstance(dataset_dirs, (str, bytes)):
            raise TypeError("data.dataset_dirs 不能是字符串，请配置为 YAML 列表。")
        else:
            raise TypeError(f"data.dataset_dirs 必须是 list/tuple/ListConfig，当前为: {type(dataset_dirs)}")

        datasets = []
        coco_apis = []
        skipped = 0
        for root in dataset_dirs:
            sub_args = copy.copy(args)
            setattr(sub_args, "dataset_dirs", None)
            setattr(sub_args, "dataset_dir", root)
            try:
                ds = build_dataset(image_set, sub_args, img_size)
            except FileNotFoundError as exc:
                if getattr(args, "skip_missing_splits", False):
                    skipped += 1
                    logging.warning(
                        "构建多数据集 split=%s 时跳过 dataset_dir=%s（缺失文件）: %s",
                        image_set,
                        root,
                        exc,
                    )
                    continue
                raise
            datasets.append(ds)
            coco = get_coco_api_from_dataset(ds)
            if coco is None:
                raise ValueError(f"dataset_dir={root} 的数据集不提供 COCO API，无法合并评估。")
            coco_apis.append(coco)

        if not datasets:
            raise FileNotFoundError(f"多数据集构建失败：所有 dataset_dirs 在 split={image_set} 上均不可用。")
        if skipped:
            logging.warning("多数据集 split=%s：跳过=%d, 保留=%d", image_set, skipped, len(datasets))
        return MultiCocoDataset(datasets, coco_apis)

    # dataset_key 表示“数据集加载器类型”（例如 coco_rgb / coco_rgb_msi / coco_msi）。
    # 历史上使用过 dataset_file=coco_ms / coco_ms_rgb 这类别名，为避免语义歧义，这里不再自动映射。
    dataset_key = getattr(args, "dataset_key", None) or getattr(args, "dataset_file", None)
    if not dataset_key:
        raise ValueError("data.dataset_key/data.dataset_file 未配置，无法构建数据集。")
    if str(dataset_key) in {"coco_ms", "coco_ms_rgb"}:
        raise ValueError(
            f"dataset_key={dataset_key} 为旧命名且语义不清晰，请在 YAML 中改为："
            "coco_rgb_msi（RGB+MSI）/ coco_rgb（仅 RGB）/ coco_msi（仅 MSI）。"
        )
    remap_categories = getattr(args, "remap_mscoco_category", False)
    class_names_cfg = getattr(args, "class_names", None)
    if (
        class_names_cfg
        and str(dataset_key) in {"coco_rgb", "coco_msi", "coco_rgb_msi"}
        and not remap_categories
    ):
        # 如果提供了 class_names，但未显式声明 remap，则默认启用 remap：
        # - 训练侧：把 COCO category_id 映射为连续的 0..(K-1)
        # - 评测侧：BaseTrainer 会将模型输出 label 再映射回原 category_id
        remap_categories = True
        setattr(args, "remap_mscoco_category", True)

    if dataset_key == 'coco':
        return build_coco(image_set, args, img_size)
    if dataset_key == 'o365':
        return build_o365(image_set, args, img_size)
    if dataset_key == 'roboflow':
        return build_roboflow(image_set, args, img_size)
    if dataset_key == 'coco_msi':
        # 允许通过 args.use_rgb_input / args.use_msi_input 做消融；默认保持原行为（仅 MSI）。
        return build_multispectral_dataset(
            image_set,
            args,
            img_size,
            use_rgb_input=getattr(args, "use_rgb_input", False),
            use_msi_input=getattr(args, "use_msi_input", True),
        )
    if dataset_key == 'coco_rgb_msi':
        # 允许通过 args.use_rgb_input / args.use_msi_input 做消融；默认保持原行为（RGB+MSI）。
        return build_multispectral_dataset(
            image_set,
            args,
            img_size,
            use_rgb_input=getattr(args, "use_rgb_input", True),
            use_msi_input=getattr(args, "use_msi_input", True),
        )
    if dataset_key == 'coco_rgb':
        dataset_root = getattr(args, "dataset_dir", None) or getattr(args, "ms_dataset_dir", None)
        if dataset_root is None:
            raise ValueError("请通过 --dataset_dir 或 --ms_dataset_dir 指定数据根目录。")
        subset = image_set.split("_")[0]
        class_names = class_names_cfg if remap_categories else None
        return build_rgb_only_dataset(
            dataset_root=dataset_root,
            image_set=subset,
            img_size=img_size,
            include_masks=getattr(args, "segmentation_head", False),
            filter_annotations_without_masks=getattr(args, "filter_annotations_without_masks", True),
            drop_images_without_masks=getattr(args, "drop_images_without_masks", True),
            multi_scale=getattr(args, "multi_scale", False),
            expanded_scales=getattr(args, "expanded_scales", False),
            skip_random_resize=not getattr(args, "do_random_resize_via_padding", False),
            patch_size=getattr(args, "patch_size", 16),
            num_windows=getattr(args, "num_windows", 4),
            square_resize_div_64=getattr(args, "square_resize_div_64", False),
            remap_mscoco_category=remap_categories,
            category_names=class_names,
        )
    raise ValueError(f'dataset {args.dataset_file} not supported')
