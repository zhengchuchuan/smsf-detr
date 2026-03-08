from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class RunIndexItem:
    image: str
    npz: str | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="交互式选择“oil 对应的 cluster id”，并保存为 selected_clusters.json（用于二值化 oil vs non-oil）。"
    )
    parser.add_argument("--run-dir", type=str, required=True, help="repr run 目录（包含 meta.json / labels / clusters）。")
    parser.add_argument("--start", type=int, default=0, help="从第几张开始浏览（默认 0）。")
    parser.add_argument("--step", type=int, default=1, help="每次前进/后退多少张（默认 1）。")
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="输出 json 路径（默认 <run-dir>/selection/selected_clusters.json）。",
    )
    parser.add_argument(
        "--load",
        type=str,
        default="",
        help="加载已有的 selected_clusters.json（用于继续增量选择）。",
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="伪彩色 overlay 透明度（默认 0.45）。")
    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=0.45,
        help="选中 oil 区域 mask 的叠加透明度（默认 0.45）。",
    )
    return parser.parse_args()


def _load_palette(palette_path: Path) -> np.ndarray:
    obj = json.loads(palette_path.read_text(encoding="utf-8"))
    colors = obj.get("colors", {}) or {}
    if not isinstance(colors, dict):
        raise TypeError(f"palette.json 的 colors 必须是 dict，实际为: {type(colors)}")
    max_id = max(int(k) for k in colors.keys()) if colors else -1
    if max_id < 0:
        raise ValueError(f"palette.json 无有效 colors: {palette_path}")
    palette = np.zeros((max_id + 1, 3), dtype=np.uint8)
    for k, v in colors.items():
        idx = int(k)
        if not (isinstance(v, (list, tuple)) and len(v) == 3):
            raise ValueError(f"palette color[{k}] 非 RGB 三元组: {v}")
        palette[idx] = np.array([int(v[0]), int(v[1]), int(v[2])], dtype=np.uint8)
    return palette


def _load_run_index(meta_json: Path) -> list[RunIndexItem]:
    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    files = meta.get("files", []) or []
    items: list[RunIndexItem] = []
    for it in files:
        if not isinstance(it, dict):
            continue
        image = str(it.get("image", "")).strip()
        npz = str(it.get("npz", "")).strip() or None
        if not image:
            continue
        items.append(RunIndexItem(image=image, npz=npz))
    if not items:
        raise ValueError(f"meta.json 未包含有效 files: {meta_json}")
    return items


def _load_label_map(path: Path) -> np.ndarray:
    im = Image.open(path)
    arr = np.array(im)
    if arr.ndim != 2:
        raise ValueError(f"label map 期望二维数组，实际 shape={arr.shape} path={path}")
    return arr


def _resize_nearest(label_hw: np.ndarray, *, size_wh: tuple[int, int]) -> np.ndarray:
    w, h = size_wh
    pil = Image.fromarray(label_hw)
    up = pil.resize((int(w), int(h)), resample=Image.NEAREST)
    return np.array(up)


def _blend(a: np.ndarray, b: np.ndarray, *, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out = (1.0 - alpha) * a.astype(np.float32) + alpha * b.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_mask_rgb(rgb: np.ndarray, mask: np.ndarray, *, alpha: float) -> np.ndarray:
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    color = np.zeros_like(rgb)
    color[..., 0] = 255
    blended = _blend(rgb, color, alpha=alpha)
    out = rgb.copy()
    out[mask] = blended[mask]
    return out


def _load_selected(path: Path) -> set[int]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    clusters = obj.get("oil_clusters", obj.get("clusters", obj.get("selected_clusters", []))) or []
    out: set[int] = set()
    for x in clusters:
        try:
            out.add(int(x))
        except Exception:
            continue
    return out


def _save_selected(path: Path, *, selected: set[int], extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "oil_clusters": sorted(int(x) for x in selected),
        **extra,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser()
    meta_json = run_dir / "meta.json"
    labels_dir = run_dir / "labels"
    palette_json = run_dir / "clusters" / "palette.json"
    if not meta_json.is_file():
        raise FileNotFoundError(f"缺少 meta.json: {meta_json}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"缺少 labels/: {labels_dir}")
    if not palette_json.is_file():
        raise FileNotFoundError(f"缺少 clusters/palette.json: {palette_json}")

    items = _load_run_index(meta_json)
    palette = _load_palette(palette_json)

    out_path = Path(args.out).expanduser() if str(args.out).strip() else (run_dir / "selection" / "selected_clusters.json")
    selected: set[int] = set()
    if str(args.load).strip():
        selected |= _load_selected(Path(args.load).expanduser())
    elif out_path.is_file():
        selected |= _load_selected(out_path)

    start = int(args.start)
    step = max(1, int(args.step))
    idx = int(np.clip(start, 0, len(items) - 1))

    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib 不可用，无法进行交互选择。") from exc

    backend = str(matplotlib.get_backend() or "").lower()
    if backend == "agg" and not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "当前环境为 headless（DISPLAY 未设置）且 matplotlib backend=Agg，无法弹出交互窗口。\n"
            "- 若你在远程 SSH：请使用 X11 转发/VSCode Remote GUI 等方式启用图形界面；或设置可用的 GUI backend。\n"
            "- 或者改用自动选择：`python tools/repr_auto_select_oil_clusters.py --run-dir ...`"
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax0, ax1 = axes
    ax0.set_title("RGB")
    ax1.set_title("PseudoColor + OilMask")
    for ax in axes:
        ax.axis("off")

    state: dict[str, Any] = {"up_labels": None, "stem": None, "rgb": None}

    def _render() -> None:
        nonlocal idx
        item = items[idx]
        image_path = Path(item.image).expanduser()
        if not image_path.is_file():
            raise FileNotFoundError(f"原图不存在: {image_path}")
        stem = image_path.stem
        label_path = labels_dir / f"{stem}.png"
        if not label_path.is_file():
            raise FileNotFoundError(f"缺少 label map: {label_path}")

        img = Image.open(image_path).convert("RGB")
        rgb = np.array(img, dtype=np.uint8)
        label_map = _load_label_map(label_path)
        up = _resize_nearest(label_map, size_wh=img.size)

        color = palette[up.astype(np.int32)]
        overlay = _blend(rgb, color, alpha=float(args.alpha))
        mask = np.isin(up, np.array(sorted(selected), dtype=up.dtype))
        overlay = _apply_mask_rgb(overlay, mask, alpha=float(args.mask_alpha))

        ax0.imshow(rgb)
        ax1.imshow(overlay)
        title = (
            f"[{idx+1}/{len(items)}] {stem} | selected_oil_clusters={len(selected)} "
            f"(click toggle; n/p next/prev; s save; c clear; q quit)"
        )
        fig.suptitle(title)
        fig.canvas.draw_idle()

        state["up_labels"] = up
        state["stem"] = stem
        state["rgb"] = rgb

    def _toggle_at(x: int, y: int) -> None:
        up = state.get("up_labels")
        if up is None:
            return
        if y < 0 or x < 0 or y >= up.shape[0] or x >= up.shape[1]:
            return
        cid = int(up[int(y), int(x)])
        if cid in selected:
            selected.remove(cid)
            logging.info("[-] remove cluster %d", cid)
        else:
            selected.add(cid)
            logging.info("[+] add cluster %d", cid)
        _render()

    def _on_click(event) -> None:
        if event.inaxes not in {ax0, ax1}:
            return
        if event.xdata is None or event.ydata is None:
            return
        _toggle_at(int(event.xdata), int(event.ydata))

    def _on_key(event) -> None:
        nonlocal idx
        key = str(getattr(event, "key", "") or "").lower()
        if key in {"q", "escape"}:
            plt.close(fig)
            return
        if key == "s":
            _save_selected(
                out_path,
                selected=selected,
                extra={
                    "run_dir": str(run_dir),
                    "palette": str(palette_json),
                },
            )
            logging.info("saved: %s (clusters=%d)", out_path, len(selected))
            return
        if key == "c":
            selected.clear()
            _render()
            return
        if key in {"n", "right"}:
            idx = min(len(items) - 1, idx + step)
            _render()
            return
        if key in {"p", "left"}:
            idx = max(0, idx - step)
            _render()
            return

    fig.canvas.mpl_connect("button_press_event", _on_click)
    fig.canvas.mpl_connect("key_press_event", _on_key)

    _render()
    plt.show()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
