"""
Filter a COCO detection annotation JSON by category names (or ids).

Primary use-case in this repo:
- Build "3cls" annotations for `data/oil_20260115` by dropping the rare "photovoltaic" class,
  so third_party/MRT-DETR can train with `num_classes=3`.

Examples
--------
1) From the msifp-detr Hydra data config (recommended):

    python utils/data_process/filter_coco_categories.py \
      --config configs/data/oil_rgb_msi_20260115_3cls.yaml \
      --drop-empty-images

This will read:
  - data.dataset_dir (e.g. data/oil_20260115)
  - data.class_names (e.g. ["oil", "building", "machine"])
and write filtered JSONs to:
  data/oil_20260115/annotations_3cls/{train,val,test}.json

2) Manual (no Hydra config):

    python utils/data_process/filter_coco_categories.py \
      --dataset-root data/oil_20260115 \
      --keep-names oil building machine \
      --drop-empty-images
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

import yaml


DEFAULT_SPLITS = ("train", "val", "test")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_yaml(path: Path) -> Dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise TypeError(f"YAML root must be a mapping: {path} (got {type(obj)})")
    return obj


def _norm_name(value: Any) -> str:
    return str(value or "").strip()


def _infer_from_msifp_data_config(cfg_path: Path) -> Tuple[Path, List[str]]:
    """
    Load `configs/data/*.yaml` (Hydra-style) as a plain YAML mapping and extract:
      - data.dataset_dir
      - data.class_names
    """
    cfg = _load_yaml(cfg_path)
    data = cfg.get("data", {}) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict at key 'data' in {cfg_path}, got {type(data)}")

    dataset_dir = _norm_name(data.get("dataset_dir"))
    if not dataset_dir:
        raise KeyError(f"Missing `data.dataset_dir` in {cfg_path}")

    class_names = data.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise KeyError(f"Missing/invalid `data.class_names` in {cfg_path} (got {type(class_names)})")

    keep_names = [_norm_name(x) for x in class_names]
    keep_names = [x for x in keep_names if x and x != "_background_"]
    if not keep_names:
        raise ValueError(f"`data.class_names` after filtering is empty in {cfg_path}")

    return Path(dataset_dir), keep_names


def _resolve_keep_ids(
    categories: Sequence[Mapping[str, Any]],
    *,
    keep_names: Sequence[str] | None,
    keep_ids: Sequence[int] | None,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """
    Return (keep_cat_ids, keep_categories).
    - keep_cat_ids are kept in an order suitable for writing to categories[].
    """
    cats = list(categories or [])
    id_to_cat: Dict[int, Dict[str, Any]] = {}
    name_to_id: Dict[str, int] = {}
    for cat in cats:
        try:
            cid = int(cat.get("id"))
        except Exception:
            continue
        name = _norm_name(cat.get("name"))
        id_to_cat[cid] = dict(cat)
        if name:
            name_to_id[name] = cid

    if keep_ids is not None and keep_names is not None:
        raise ValueError("Pass only one of --keep-ids or --keep-names.")

    if keep_ids is not None:
        ids = [int(x) for x in keep_ids]
        missing = [x for x in ids if x not in id_to_cat]
        if missing:
            raise KeyError(f"keep_ids not found in categories: {missing}")
        keep_cat_ids = ids
    else:
        if keep_names is None or not list(keep_names):
            raise ValueError("Missing keep category selector: provide --keep-names or --keep-ids.")
        missing_names = [n for n in keep_names if n not in name_to_id]
        if missing_names:
            raise KeyError(f"keep_names not found in categories: {missing_names}")
        keep_cat_ids = [name_to_id[n] for n in keep_names]

    keep_categories: List[Dict[str, Any]] = [id_to_cat[cid] for cid in keep_cat_ids]
    return keep_cat_ids, keep_categories


def filter_coco_dict(
    data: MutableMapping[str, Any],
    *,
    keep_category_names: Sequence[str] | None = None,
    keep_category_ids: Sequence[int] | None = None,
    drop_empty_images: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Filter a COCO dict and return (new_data, stats).
    """
    data = dict(data)
    categories_in = list(data.get("categories", []) or [])
    keep_cat_ids, keep_categories = _resolve_keep_ids(
        categories_in, keep_names=keep_category_names, keep_ids=keep_category_ids
    )
    keep_set: Set[int] = set(int(x) for x in keep_cat_ids)

    anns_in: List[Mapping[str, Any]] = list(data.get("annotations", []) or [])
    anns_out: List[Dict[str, Any]] = []
    kept_img_ids: Set[int] = set()
    dropped_anns = 0
    for ann in anns_in:
        cid = ann.get("category_id", None)
        try:
            cid_int = int(cid)
        except Exception:
            dropped_anns += 1
            continue
        if cid_int not in keep_set:
            dropped_anns += 1
            continue
        ann_copy = dict(ann)
        anns_out.append(ann_copy)
        try:
            kept_img_ids.add(int(ann_copy.get("image_id")))
        except Exception:
            # Keep the annotation anyway; image filtering will handle bad ids later.
            pass

    images_in: List[Mapping[str, Any]] = list(data.get("images", []) or [])
    if drop_empty_images:
        images_out: List[Dict[str, Any]] = []
        dropped_imgs = 0
        for img in images_in:
            try:
                img_id = int(img.get("id"))
            except Exception:
                dropped_imgs += 1
                continue
            if img_id not in kept_img_ids:
                dropped_imgs += 1
                continue
            images_out.append(dict(img))
    else:
        images_out = [dict(x) for x in images_in]
        dropped_imgs = 0

    kept_img_ids_final: Set[int] = set()
    for img in images_out:
        try:
            kept_img_ids_final.add(int(img.get("id")))
        except Exception:
            continue
    anns_out = [a for a in anns_out if int(a.get("image_id")) in kept_img_ids_final]

    data["categories"] = keep_categories
    data["annotations"] = anns_out
    data["images"] = images_out

    stats = {
        "images_in": len(images_in),
        "images_out": len(images_out),
        "images_dropped": int(dropped_imgs),
        "annotations_in": len(anns_in),
        "annotations_out": len(anns_out),
        "annotations_dropped": int(dropped_anns),
        "categories_in": len(categories_in),
        "categories_out": len(keep_categories),
    }
    return data, stats


def _iter_splits(values: Sequence[str] | None) -> List[str]:
    if not values:
        return list(DEFAULT_SPLITS)
    out: List[str] = []
    for s in values:
        s = _norm_name(s)
        if not s:
            continue
        out.append(s)
    if not out:
        return list(DEFAULT_SPLITS)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="msifp-detr data config yaml.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Dataset root (contains annotations/).")
    parser.add_argument(
        "--keep-names",
        nargs="+",
        default=None,
        help="Category names to keep (must exist in categories[].name).",
    )
    parser.add_argument(
        "--keep-ids",
        nargs="+",
        type=int,
        default=None,
        help="Category ids to keep (must exist in categories[].id).",
    )
    parser.add_argument(
        "--src-ann-dir",
        type=Path,
        default=None,
        help="Source annotation directory (default: <dataset-root>/annotations).",
    )
    parser.add_argument(
        "--dst-ann-dir",
        type=Path,
        default=None,
        help="Destination annotation directory (default: <dataset-root>/annotations_3cls).",
    )
    parser.add_argument("--splits", nargs="+", default=None, help="Splits to convert (default: train val test).")
    parser.add_argument("--drop-empty-images", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    keep_names = args.keep_names
    dataset_root = args.dataset_root

    if args.config is not None:
        inferred_root, inferred_keep_names = _infer_from_msifp_data_config(args.config)
        if dataset_root is None:
            dataset_root = inferred_root
        if keep_names is None:
            keep_names = inferred_keep_names

    if dataset_root is None:
        raise SystemExit("Missing --dataset-root (or provide --config to infer it).")
    dataset_root = Path(dataset_root)

    if args.src_ann_dir is None:
        src_ann_dir = dataset_root / "annotations"
    else:
        src_ann_dir = Path(args.src_ann_dir)

    if args.dst_ann_dir is None:
        # Keep naming stable for MRT-DETR dataset configs.
        dst_ann_dir = dataset_root / "annotations_3cls"
    else:
        dst_ann_dir = Path(args.dst_ann_dir)

    splits = _iter_splits(args.splits)
    drop_empty_images = bool(args.drop_empty_images)

    for split in splits:
        src = src_ann_dir / f"{split}.json"
        if not src.is_file():
            raise FileNotFoundError(f"Missing annotation file: {src}")
        dst = dst_ann_dir / f"{split}.json"
        if dst.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite: {dst} (pass --overwrite)")

        coco = _load_json(src)
        new_coco, stats = filter_coco_dict(
            coco,
            keep_category_names=keep_names,
            keep_category_ids=args.keep_ids,
            drop_empty_images=drop_empty_images,
        )
        _dump_json(new_coco, dst)

        kept = ", ".join(keep_names or []) if keep_names else ", ".join(str(x) for x in (args.keep_ids or []))
        print(
            f"[{split}] keep=({kept}) drop_empty_images={drop_empty_images} -> {dst}\n"
            f"  images: {stats['images_in']} -> {stats['images_out']} (dropped={stats['images_dropped']})\n"
            f"  anns:   {stats['annotations_in']} -> {stats['annotations_out']} (dropped={stats['annotations_dropped']})\n"
            f"  cats:   {stats['categories_in']} -> {stats['categories_out']}"
        )


if __name__ == "__main__":
    main()

