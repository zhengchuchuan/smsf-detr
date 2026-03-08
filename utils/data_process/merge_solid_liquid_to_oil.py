
"""
将 COCO 检测标注中的 solid/liquid 合并为一个新类别 oil，并重排类别 id。

期望的类别映射（按旧 category_id）：
  1: solid  -> 1: oil
  2: liquid -> 1: oil
  3: building -> 2: building
  4: machine  -> 3: machine
  5: photovoltaic -> 4: photovoltaic

会同步修改：
- data['categories']：替换为 4 个类别（id=1..4）
- data['annotations'][*]['category_id']：按映射重写

推荐用法（不覆盖原文件，生成带 suffix 的新 json）：
  python utils/data_process/merge_solid_liquid_to_oil.py --dataset-root data/oil

多文件/多数据集根目录：
  python utils/data_process/merge_solid_liquid_to_oil.py --dataset-roots data/oil data/oil_after_20250519

覆盖写回（不推荐，谨慎使用）：
  python utils/data_process/merge_solid_liquid_to_oil.py --dataset-root data/oil --overwrite
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


DEFAULT_FILES = ("train.json", "val.json", "test.json")


@dataclass(frozen=True)
class CategorySpec:
    id: int
    name: str
    supercategory: str = ""


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_name(value: Any) -> str:
    return str(value or "").strip().lower()


def _build_default_mapping(
    categories: Sequence[Mapping[str, Any]],
    *,
    solid_id: int,
    liquid_id: int,
    building_id: int,
    machine_id: int,
    photovoltaic_id: int,
    oil_name: str,
) -> Tuple[Dict[int, int], List[CategorySpec]]:
    """
    返回 (old_id -> new_id, new_categories)。
    new_categories 会尽量沿用原 categories 的 name/supercategory（oil 的 name 强制为 oil_name）。
    """
    id_to_cat: Dict[int, Mapping[str, Any]] = {}
    for cat in categories or []:
        try:
            cid = int(cat.get("id"))
        except Exception:
            continue
        id_to_cat[cid] = cat

    def take(old_id: int, *, new_id: int, default_name: str) -> CategorySpec:
        cat = id_to_cat.get(old_id, {})
        name = str(cat.get("name") or default_name).strip() or default_name
        supercat = str(cat.get("supercategory") or "").strip()
        return CategorySpec(id=new_id, name=name, supercategory=supercat)

    mapping = {
        int(solid_id): 1,
        int(liquid_id): 1,
        int(building_id): 2,
        int(machine_id): 3,
        int(photovoltaic_id): 4,
    }

    # 新 categories：oil 强制命名为 oil_name；其它尽量复用原名字
    building = take(building_id, new_id=2, default_name="building")
    machine = take(machine_id, new_id=3, default_name="machine")
    photovoltaic = take(photovoltaic_id, new_id=4, default_name="photovoltaic")
    oil_supercat = ""
    for candidate in (id_to_cat.get(solid_id), id_to_cat.get(liquid_id)):
        if not candidate:
            continue
        oil_supercat = str(candidate.get("supercategory") or "").strip()
        if oil_supercat:
            break
    new_categories = [
        CategorySpec(id=1, name=str(oil_name).strip() or "oil", supercategory=oil_supercat),
        building,
        machine,
        photovoltaic,
    ]
    return mapping, new_categories


def convert_coco_dict(
    data: MutableMapping[str, Any],
    *,
    solid_id: int = 1,
    liquid_id: int = 2,
    building_id: int = 3,
    machine_id: int = 4,
    photovoltaic_id: int = 5,
    oil_name: str = "oil",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    转换单个 COCO dict，返回新的 dict（不修改输入引用）。
    strict=True 时遇到未知 category_id 会直接报错；否则会丢弃该 annotation。
    """
    data = dict(data)
    categories = list(data.get("categories", []) or [])
    mapping, new_categories = _build_default_mapping(
        categories,
        solid_id=solid_id,
        liquid_id=liquid_id,
        building_id=building_id,
        machine_id=machine_id,
        photovoltaic_id=photovoltaic_id,
        oil_name=oil_name,
    )

    anns_in: List[Mapping[str, Any]] = list(data.get("annotations", []) or [])
    anns_out: List[Dict[str, Any]] = []
    dropped = 0
    for ann in anns_in:
        ann = dict(ann)
        raw_cid = ann.get("category_id", None)
        try:
            cid = int(raw_cid)
        except Exception as exc:
            if strict:
                raise ValueError(f"annotation.id={ann.get('id')} 的 category_id 非法: {raw_cid!r}") from exc
            dropped += 1
            continue

        new_id = mapping.get(cid)
        if new_id is None:
            if strict:
                raise KeyError(
                    f"发现未在映射表中的 category_id={cid}（annotation.id={ann.get('id')}）。"
                    f"若确实要丢弃这些标注，请加 --non-strict。"
                )
            dropped += 1
            continue

        ann["category_id"] = int(new_id)
        anns_out.append(ann)

    if dropped:
        data["annotations_dropped"] = int(dropped)

    data["annotations"] = anns_out
    data["categories"] = [
        {"id": c.id, "name": c.name, "supercategory": c.supercategory} for c in new_categories
    ]
    return data


def convert_file(
    src: Path,
    dst: Path,
    *,
    solid_id: int,
    liquid_id: int,
    building_id: int,
    machine_id: int,
    photovoltaic_id: int,
    oil_name: str,
    strict: bool,
) -> Tuple[int, int]:
    data = _load_json(src)
    anns_before = len(data.get("annotations", []) or [])
    converted = convert_coco_dict(
        data,
        solid_id=solid_id,
        liquid_id=liquid_id,
        building_id=building_id,
        machine_id=machine_id,
        photovoltaic_id=photovoltaic_id,
        oil_name=oil_name,
        strict=strict,
    )
    anns_after = len(converted.get("annotations", []) or [])
    _dump_json(converted, dst)
    return anns_before, anns_after


def _iter_annotation_files(dataset_root: Path, files: Sequence[str]) -> Iterable[Path]:
    ann_dir = dataset_root / "annotations"
    for name in files:
        path = ann_dir / name
        if path.exists():
            yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="单个数据集根目录（包含 annotations/ 与 rgb/{split}/）。",
    )
    parser.add_argument(
        "--dataset-roots",
        type=Path,
        nargs="*",
        default=None,
        help="多个数据集根目录，等价于多次执行 --dataset-root。",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="要转换的标注文件名（默认在 annotations/ 下查找 train/val/test.json）。",
    )
    parser.add_argument(
        "--suffix",
        default="_oil",
        help="不覆盖写入时，输出文件名追加的后缀（如 train_oil.json）。",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖写回原始 json（不推荐）。")
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="遇到未知 category_id 时丢弃该 annotation（默认严格模式会报错）。",
    )

    # id 映射参数（按你当前的约定给默认值）
    parser.add_argument("--solid-id", type=int, default=1)
    parser.add_argument("--liquid-id", type=int, default=2)
    parser.add_argument("--building-id", type=int, default=3)
    parser.add_argument("--machine-id", type=int, default=4)
    parser.add_argument("--photovoltaic-id", type=int, default=5)
    parser.add_argument("--oil-name", type=str, default="oil")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots: List[Path] = []
    if args.dataset_root is not None:
        roots.append(args.dataset_root)
    if args.dataset_roots:
        roots.extend(args.dataset_roots)
    if not roots:
        raise SystemExit("请提供 --dataset-root 或 --dataset-roots。")

    files = args.files or list(DEFAULT_FILES)
    strict = not bool(args.non_strict)

    for root in roots:
        root = root.expanduser()
        ann_dir = root / "annotations"
        if not ann_dir.is_dir():
            raise FileNotFoundError(f"未找到标注目录：{ann_dir}")

        targets = list(_iter_annotation_files(root, files))
        if not targets:
            raise FileNotFoundError(f"{ann_dir} 下未找到要转换的文件：{files}")

        for src in targets:
            if args.overwrite:
                dst = src
            else:
                dst = src.with_name(src.stem + args.suffix + src.suffix)
                if dst.exists():
                    raise FileExistsError(f"{dst} 已存在；如需覆盖请加 --overwrite 或调整 --suffix")

            before, after = convert_file(
                src,
                dst,
                solid_id=args.solid_id,
                liquid_id=args.liquid_id,
                building_id=args.building_id,
                machine_id=args.machine_id,
                photovoltaic_id=args.photovoltaic_id,
                oil_name=args.oil_name,
                strict=strict,
            )
            print(f"[OK] {src} -> {dst}  annotations: {before} -> {after}")


if __name__ == "__main__":
    main()

