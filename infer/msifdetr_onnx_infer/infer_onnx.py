from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_CLASS_NAMES: list[str] = [
    "oil",
    "building",
    "machine",
    "photovoltaic",
]


@dataclass(frozen=True)
class DetItem:
    label: int
    score: float
    box_xyxy: list[float]
    class_name: str | None = None


def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _sanitize_dirname(name: str) -> str:
    name = str(name).strip()
    name = name.replace("/", "_").replace("\\", "_")
    name = "_".join([p for p in name.split() if p])
    return name or "data"


def _iter_images(input_path: Path, *, recursive: bool) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if input_path.is_file():
        return [input_path]
    if recursive:
        paths = [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    else:
        paths = [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(paths)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # 数值稳定：避免 exp 溢出
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _box_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return np.stack([x1, y1, x2, y2], axis=-1)


def _topk_flat(prob: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    prob: (Q, C)
    return: (scores[k], flat_indices[k]) where flat_indices in [0, Q*C)
    """
    flat = prob.reshape(-1)
    if k >= flat.shape[0]:
        order = np.argsort(-flat)
        return flat[order], order
    idx = np.argpartition(-flat, kth=k - 1)[:k]
    order = idx[np.argsort(-flat[idx])]
    return flat[order], order


def _preprocess_rgb_square(
    img: Image.Image,
    *,
    height: int,
    width: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((int(width), int(height)), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC, [0,1]
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean_arr) / std_arr
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return arr.astype(np.float32, copy=False)


def _load_class_names(path: str | Path | None) -> list[str]:
    if not path:
        return list(DEFAULT_CLASS_NAMES)
    p = _as_path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"class-names 文件不存在: {p}")
    names: list[str] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.lower() in {"background", "_background_", "bg"}:
                continue
            names.append(s)
    return names


def _load_font(font_size: int) -> ImageFont.ImageFont | None:
    if font_size <= 0:
        return None
    candidates = [
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=int(font_size))
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _draw_detections(
    image: Image.Image,
    dets: list[DetItem],
    *,
    font_size: int = 0,
    line_width: int = 3,
) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    if font_size <= 0:
        font_size = max(16, int(min(w, h) * 0.03))
    font = _load_font(font_size)
    pad = 2

    for det in dets:
        x1, y1, x2, y2 = det.box_xyxy
        color = (255, 0, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=int(line_width))
        name = det.class_name if det.class_name else f"class_{det.label}"
        title = f"{name} {det.score:.3f}"
        if font is not None:
            try:
                tbox0 = draw.textbbox((0, 0), title, font=font)
                text_w = float(tbox0[2] - tbox0[0])
                text_h = float(tbox0[3] - tbox0[1])

                tx = float(x1)
                tx = max(0.0, min(tx, float(w) - (text_w + 2 * pad)))
                ty_out = float(y1) - (text_h + 2 * pad)
                ty = ty_out if ty_out >= 0 else float(y1)
                ty = max(0.0, min(ty, float(h) - (text_h + 2 * pad)))

                bg = (int(tx), int(ty), int(tx + text_w + 2 * pad), int(ty + text_h + 2 * pad))
                draw.rectangle(bg, fill=(0, 0, 0))
            except Exception:
                tx, ty = float(x1), max(0.0, float(y1))
            draw.text((int(tx + pad), int(ty + pad)), title, fill=(255, 255, 255), font=font)
        else:
            tx = int(max(0.0, min(float(x1), float(w - 1))))
            ty = int(max(0.0, float(y1) - float(font_size) - 2 * pad))
            if ty <= 0:
                ty = int(max(0.0, min(float(y1), float(h - 1))))
            draw.text((tx, ty), title, fill=color)
    return img


def _create_session(model_path: Path, *, device: str):
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("缺少依赖 onnxruntime，请先 pip install onnxruntime 或 onnxruntime-gpu") from exc

    device = str(device).strip().lower()
    if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    try:
        sess = ort.InferenceSession(str(model_path), providers=providers)
    except Exception as exc:
        raise RuntimeError(
            f"加载 ONNX 失败: {model_path}\n"
            f"device={device}, providers={providers}\n"
            f"原始错误: {exc}"
        ) from exc
    return sess


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MSIF-DETR ONNX 推理（不依赖原工程代码）")
    parser.add_argument("--model", required=True, help="导出的 model.onnx 路径")
    parser.add_argument("--input", required=True, help="输入图片文件或目录")
    parser.add_argument("--recursive", action="store_true", help="递归扫描 input 目录")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="onnxruntime 执行设备")
    parser.add_argument("--height", type=int, default=0, help="模型输入高（0=从 ONNX 读取；需与导出时一致）")
    parser.add_argument("--width", type=int, default=0, help="模型输入宽（0=从 ONNX 读取；需与导出时一致）")
    parser.add_argument("--num-select", type=int, default=300, help="PostProcess top-K（默认 300）")
    parser.add_argument("--score-thr", type=float, default=0.3, help="置信度阈值（默认 0.3）")
    parser.add_argument("--max-dets", type=int, default=100, help="每张图最多保留多少检测（默认 100）")
    parser.add_argument(
        "--class-names",
        default=None,
        help="类别名文件（每行一个，不含背景类）；不传则使用脚本内置 DEFAULT_CLASS_NAMES。",
    )
    parser.add_argument("--save-vis", action="store_true", help="保存可视化图片")
    parser.add_argument("--vis-font-size", type=int, default=0, help="可视化字体大小（0=自动）")
    parser.add_argument("--vis-line-width", type=int, default=3, help="可视化框线宽度")
    parser.add_argument("--output-dir", default="outputs", help="输出根目录（默认 outputs）")
    parser.add_argument(
        "--run-name",
        default="",
        help='输出子目录名（默认自动："<输入文件夹名>-YYYYMMDD-HHMM"）',
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model_path = _as_path(args.model).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX 模型不存在: {model_path}")

    input_path = _as_path(args.input).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"input 不存在: {input_path}")

    img_paths = _iter_images(input_path, recursive=bool(args.recursive))
    if not img_paths:
        raise FileNotFoundError(f"未找到图片: {input_path}")

    output_root = _as_path(args.output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = str(args.run_name).strip()
    if not run_name:
        base = input_path.name if input_path.is_dir() else input_path.stem
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        run_name = f"{_sanitize_dirname(base)}-{stamp}"
    out_dir = output_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "vis"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    class_names = _load_class_names(args.class_names)

    sess = _create_session(model_path, device=str(args.device))
    inputs = sess.get_inputs()
    if len(inputs) != 1:
        raise RuntimeError(
            f"当前推理脚本仅支持单输入（RGB-only）ONNX；但模型 inputs={len(inputs)}: {[i.name for i in inputs]}"
        )
    input_name = inputs[0].name
    input_shape = list(inputs[0].shape or [])
    model_h = input_shape[2] if len(input_shape) >= 4 else None
    model_w = input_shape[3] if len(input_shape) >= 4 else None
    if isinstance(model_h, int) and isinstance(model_w, int) and model_h > 0 and model_w > 0:
        if int(args.height) <= 0 and int(args.width) <= 0:
            args.height = int(model_h)
            args.width = int(model_w)
        elif int(args.height) > 0 and int(args.width) > 0:
            if int(args.height) != int(model_h) or int(args.width) != int(model_w):
                raise ValueError(
                    f"输入尺寸与 ONNX 不一致：你传 (H,W)=({args.height},{args.width})，"
                    f"但 ONNX 期望 (H,W)=({model_h},{model_w})。"
                )
        else:
            raise ValueError("height/width 必须同时为 0（自动）或同时指定为正整数。")
    else:
        # 兜底：ONNX 输入为动态 H/W 时需要用户显式指定
        if int(args.height) <= 0 or int(args.width) <= 0:
            raise ValueError(
                f"无法从 ONNX 输入形状推断 H/W（shape={input_shape}），请显式传 --height/--width。"
            )

    outputs_meta = sess.get_outputs()
    output_names = [o.name for o in outputs_meta]

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    results: list[dict] = []
    num_select = max(1, int(args.num_select))
    score_thr = float(args.score_thr)
    max_dets = max(1, int(args.max_dets))

    logging.info("model=%s", model_path)
    logging.info("input=%s (N=%d)", input_path, len(img_paths))
    logging.info("output=%s", out_dir)
    logging.info("device=%s providers=%s", args.device, sess.get_providers())
    logging.info("input_name=%s output_names=%s", input_name, output_names)

    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        tqdm = None  # type: ignore

    iterator = img_paths
    if tqdm is not None:
        iterator = tqdm(img_paths, total=len(img_paths), desc="Infer", unit="img")

    for p in iterator:
        img = Image.open(p)
        if img.mode != "RGB":
            img = img.convert("RGB")
        orig_w, orig_h = img.size

        x = _preprocess_rgb_square(img, height=int(args.height), width=int(args.width), mean=mean, std=std)
        batch = np.expand_dims(x, axis=0)  # 1,3,H,W

        ort_outs = sess.run(None, {input_name: batch})
        out_map = {name: value for name, value in zip(output_names, ort_outs)}

        if "pred_boxes" not in out_map or "pred_logits" not in out_map:
            raise RuntimeError(f"ONNX 输出缺少 pred_boxes/pred_logits，实际输出: {list(out_map.keys())}")

        pred_boxes = np.asarray(out_map["pred_boxes"], dtype=np.float32)  # (1,Q,4)
        pred_logits = np.asarray(out_map["pred_logits"], dtype=np.float32)  # (1,Q,C)

        if pred_boxes.ndim != 3 or pred_logits.ndim != 3:
            raise RuntimeError(
                f"输出维度不符合预期: pred_boxes={pred_boxes.shape}, pred_logits={pred_logits.shape}"
            )

        q = int(pred_logits.shape[1])
        c = int(pred_logits.shape[2])
        if pred_boxes.shape[1] != q:
            raise RuntimeError(
                f"Q 不一致: pred_boxes={pred_boxes.shape}, pred_logits={pred_logits.shape}"
            )

        # 约定：最后一类为 background/no-object
        num_obj_classes = max(0, c - 1)

        prob = _sigmoid(pred_logits[0])  # (Q,C)
        k = min(num_select, q * c)
        scores_k, flat_idx = _topk_flat(prob, k)
        topk_boxes = (flat_idx // c).astype(np.int64)
        labels = (flat_idx % c).astype(np.int64)

        boxes_xyxy = _box_cxcywh_to_xyxy(pred_boxes[0])  # (Q,4) in [0,1]
        boxes_xyxy = boxes_xyxy[topk_boxes]  # (K,4)

        scale = np.asarray([orig_w, orig_h, orig_w, orig_h], dtype=np.float32).reshape(1, 4)
        boxes_xyxy = boxes_xyxy * scale
        boxes_xyxy[:, 0::2] = np.clip(boxes_xyxy[:, 0::2], 0.0, float(orig_w))
        boxes_xyxy[:, 1::2] = np.clip(boxes_xyxy[:, 1::2], 0.0, float(orig_h))

        keep = scores_k >= float(score_thr)
        if num_obj_classes > 0:
            keep = keep & (labels >= 0) & (labels < num_obj_classes)
        else:
            keep = keep & (labels != (c - 1))

        scores_k = scores_k[keep]
        labels = labels[keep]
        boxes_xyxy = boxes_xyxy[keep]

        if scores_k.shape[0] > max_dets:
            order = np.argsort(-scores_k)[:max_dets]
            scores_k = scores_k[order]
            labels = labels[order]
            boxes_xyxy = boxes_xyxy[order]

        dets: list[DetItem] = []
        for s, l, b in zip(scores_k.tolist(), labels.tolist(), boxes_xyxy.tolist()):
            name = class_names[l] if 0 <= int(l) < len(class_names) else None
            dets.append(
                DetItem(
                    label=int(l),
                    score=float(s),
                    box_xyxy=[float(x) for x in b],
                    class_name=name,
                )
            )

        results.append(
            {
                "file": str(p),
                "detections": [
                    {
                        "label": d.label,
                        "class_name": d.class_name,
                        "score": d.score,
                        "box_xyxy": d.box_xyxy,
                        "box_xywh": [
                            d.box_xyxy[0],
                            d.box_xyxy[1],
                            d.box_xyxy[2] - d.box_xyxy[0],
                            d.box_xyxy[3] - d.box_xyxy[1],
                        ],
                    }
                    for d in dets
                ],
            }
        )

        if args.save_vis:
            vis = _draw_detections(
                img,
                dets,
                font_size=int(args.vis_font_size),
                line_width=int(args.vis_line_width),
            )
            out_path = vis_dir / f"{p.stem}.png"
            vis.save(out_path)

    out_json = out_dir / "predictions.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": str(model_path),
                "input": str(input_path),
                "height": int(args.height),
                "width": int(args.width),
                "score_thr": float(score_thr),
                "max_dets": int(max_dets),
                "num_select": int(num_select),
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logging.info("完成：%d 张，已写入 %s", len(results), out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
