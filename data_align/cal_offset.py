"""基于统计的平移偏移估计工具：批量估计“相对于参考通道”的 (dx, dy)。

算法要点（仅考虑平移，不做旋转/缩放）：
- 对每个样本（RGB 图 + HDR 光谱）选定参考通道（默认 720nm；也可用 RGB/JPG）；
- 估计“参考→JPG”与“参考→各光谱波段”的平移量；
- 跨样本对每个通道的平移做稳健聚合（默认中位数），得到最终配置；
- 按拍摄日期分成两组：513 之前（< 20240513）和 513 之后（>= 20240513），分别写两份 align.conf。
- 默认推荐参数：ECC + 梯度域匹配 + 0.8 中心 ROI + 0.5 下采样，可直接满足多数场景。

实现细节：
- 预处理到 float32，并可选梯度域增强跨模态匹配稳定性（--use-gradient）；
- 支持相位相关（phase；较快）与 ECC（ecc；默认，更稳健但更慢）；
- 可选中心 ROI 与下采样以提升速度与鲁棒性。

输出配置文件的含义：
- 文件按顺序给出（jpg, 450, 550, 650, 720, 750, 800, 850）的 (dx, dy)；
- 本工具输出的 (dx, dy) 均表示“参考通道 → 该通道”的位移（参考通道行应为 0,0）。
- 输出文件位于 conf 目录，命名为 `align_before_513.conf` 与 `align_after_513.conf`；
- 可配合 `data_align/data_align.py --alignment-reference ...` 使用（例如 `jpg`）。

使用示例：
- 以 RGB/JPG 为基准估计并写入 `data/config/oil/my-conf`（采用默认推荐参数）：
    python data_align/cal_offset.py ./data/oil/train/feedback --conf-ref jpg
- 仅打印（不落盘）：
    python data_align/cal_offset.py ./data/oil/train/feedback --dry-run
- 使用相位相关（更快但可能稍弱）：
    python data_align/cal_offset.py ./data/oil/train/feedback --method phase --no-gradient
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import sys

import cv2
import numpy as np

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io.spectral_io import open_hdr_img, channel as SPECTRAL_CHANNELS


LOGGER = logging.getLogger(__name__)

DATE_PATTERN = re.compile(r"(?<!\d)(\d{8})(?!\d)")
DEFAULT_CUTOFF = datetime.strptime("20240513", "%Y%m%d")  # 513 当天及之后为“后”

CHANNEL_ORDER: Tuple[object, ...] = ("jpg", *SPECTRAL_CHANNELS)


# ----------------------------- 图像预处理与估计 ----------------------------- #


def percentile_normalize_to_float32(img: np.ndarray, clip_percentile: float) -> np.ndarray:
    """将图像按百分位裁剪并缩放到 [0,1] 的 float32。

    适用于 RGB 灰度与光谱单通道，能抑制极端值并增强对比。
    """
    img = img.astype(np.float32)
    valid = img[np.isfinite(img)]
    if valid.size == 0:
        return np.zeros_like(img, dtype=np.float32)
    lo = np.percentile(valid, clip_percentile)
    hi = np.percentile(valid, 100 - clip_percentile)
    if hi <= lo:
        lo = float(np.min(valid))
        hi = float(np.max(valid))
        if hi <= lo:
            return np.zeros_like(img, dtype=np.float32)
    out = (img - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def gradient_magnitude(u: np.ndarray) -> np.ndarray:
    """Sobel 梯度幅值（float32，归一化到 [0,1]）。"""
    if u.dtype != np.float32:
        u = u.astype(np.float32)
    gx = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(u, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mmax = float(mag.max()) if mag.size else 0.0
    if mmax <= 0:
        return np.zeros_like(mag, dtype=np.float32)
    return (mag / mmax).astype(np.float32)


def center_crop_frac(img: np.ndarray, frac: float) -> np.ndarray:
    """按比例做中心裁剪（frac∈(0,1]）。"""
    if frac >= 0.999:
        return img
    h, w = img.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return img[y0 : y0 + ch, x0 : x0 + cw]


def resize_frac(img: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


@dataclass
class ShiftResult:
    dx: float
    dy: float
    score: float


def estimate_shift_phasecorr(template: np.ndarray, input_img: np.ndarray) -> ShiftResult:
    """相位相关估计平移：返回“template -> input”的 (dx, dy) 及响应值。

    要求两图尺寸一致、float32。OpenCV 返回 (dx, dy)。
    """
    # 轻微平滑，提升鲁棒性
    template_b = cv2.GaussianBlur(template, (5, 5), 1.0)
    input_b = cv2.GaussianBlur(input_img, (5, 5), 1.0)
    (dx, dy), response = cv2.phaseCorrelate(template_b, input_b)
    return ShiftResult(float(dx), float(dy), float(response))


def estimate_shift_ecc(template: np.ndarray, input_img: np.ndarray, iters: int = 200, eps: float = 1e-6) -> ShiftResult:
    """ECC 估计平移：返回“template -> input”的 (dx, dy) 及相关系数。

    使用平移模型（MOTION_TRANSLATION）。要求两图尺寸一致、float32。
    """
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    try:
        cc, warp = cv2.findTransformECC(template, input_img, warp, cv2.MOTION_TRANSLATION, criteria)
        dx, dy = float(warp[0, 2]), float(warp[1, 2])
        return ShiftResult(dx, dy, float(cc))
    except cv2.error as e:
        LOGGER.debug("ECC failed: %s", e)
        return ShiftResult(0.0, 0.0, float("nan"))


def prepare_pair(img_a: np.ndarray, img_b: np.ndarray, *, clip_percentile: float,
                 use_gradient: bool, roi_frac: float, downscale: float) -> Tuple[np.ndarray, np.ndarray]:
    """将两幅图（可为 JPG/灰度/单波段）处理为对齐估计所需的 float32 同尺寸图像。

    约定：若为 3 通道则自动按 BGR 转灰度，否则按单通道处理；
    返回 (a, b)，可直接用于 `estimate_shift_* (template=a, input=b)`。
    """
    def to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    # 灰度化并归一化
    a_gray = to_gray(img_a)
    b_gray = to_gray(img_b)
    a = percentile_normalize_to_float32(a_gray, clip_percentile)
    b = percentile_normalize_to_float32(b_gray, clip_percentile)

    if use_gradient:
        a = gradient_magnitude(a)
        b = gradient_magnitude(b)

    # 中心 ROI
    a = center_crop_frac(a, roi_frac)
    b = center_crop_frac(b, roi_frac)

    # 下采样
    a = resize_frac(a, downscale)
    b = resize_frac(b, downscale)

    # 尺寸对齐（取二者最小公共尺寸中心裁剪避免插值误差）
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a = center_crop_frac(a, min(h / a.shape[0], w / a.shape[1]))
    b = center_crop_frac(b, min(h / b.shape[0], w / b.shape[1]))
    return a.astype(np.float32, copy=False), b.astype(np.float32, copy=False)


# ----------------------------- 主流程：批量估计与聚合 ----------------------------- #


def extract_capture_date(filename: str) -> datetime:
    m = DATE_PATTERN.search(filename)
    if not m:
        raise ValueError(f"无法从文件名中提取日期: {filename}")
    return datetime.strptime(m.group(1), "%Y%m%d")


def list_samples(dataset_root: Path) -> List[Tuple[Path, Path]]:
    images_dir = dataset_root / "images"
    spectral_dir = dataset_root / "spectral"
    if not images_dir.is_dir() or not spectral_dir.is_dir():
        raise FileNotFoundError("dataset_root 下需包含 images/ 与 spectral/ 目录")
    image_files = sorted(
        [p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    pairs: List[Tuple[Path, Path]] = []
    for img_path in image_files:
        hdr = spectral_dir / f"{img_path.stem}.hdr"
        if hdr.exists():
            pairs.append((img_path, hdr))
    return pairs


def aggregate_robust(shifts: List[Tuple[float, float]]) -> Tuple[float, float]:
    """对一组 (dx, dy) 做中位数聚合。"""
    if not shifts:
        return 0.0, 0.0
    arr = np.asarray(shifts, dtype=np.float32)
    dx = float(np.median(arr[:, 0]))
    dy = float(np.median(arr[:, 1]))
    return dx, dy


def estimate_offsets_for_group(
    samples: Sequence[Tuple[Path, Path]],
    *,
    conf_ref: object,
    method: str,
    clip_percentile: float,
    use_gradient: bool,
    roi_frac: float,
    downscale: float,
    min_quality: float,
    show_progress: bool,
    progress_desc: str | None = None,
) -> Dict[object, Tuple[float, float]]:
    """对一组样本（同一日期分组）估计“相对于 conf_ref”的平移并聚合。

    本函数输出的每一行 (dx, dy) 都表示 “conf_ref -> key” 的平移量：
    - 当 conf_ref == 'jpg' 时：JPG 行恒为 (0,0)，其余为 JPG->band；
    - 当 conf_ref 为某个波段时：该波段行恒为 (0,0)，其余为 ref->key（含 JPG）。
    """
    if downscale <= 0:
        raise ValueError("downscale must be positive.")

    per_band_shifts: Dict[object, List[Tuple[float, float]]] = {k: [] for k in CHANNEL_ORDER}
    inv_downscale = 1.0 / downscale

    reference_key: object = conf_ref
    if isinstance(reference_key, str):
        reference_key = reference_key.lower()
        if reference_key not in {"jpg"}:
            raise ValueError(f"未知 conf_ref: {conf_ref}")
    else:
        reference_key = int(reference_key)

    ref_idx: int | None = None
    if reference_key != "jpg":
        try:
            ref_idx = SPECTRAL_CHANNELS.index(int(reference_key))
        except ValueError as e:
            raise RuntimeError(f"SPECTRAL_CHANNELS 中缺少 {reference_key}nm，无法作为参考") from e

    iterator: Iterable[Tuple[Path, Path]] = samples
    if show_progress and tqdm is not None:
        iterator = tqdm(
            samples,
            total=len(samples),
            desc=progress_desc or "Estimate offsets",
            unit="sample",
        )

    for img_path, hdr_path in iterator:
        bgr = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if bgr is None:
            LOGGER.warning("跳过无法读取的 RGB: %s", img_path)
            continue
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        elif bgr.shape[2] == 4:
            bgr = bgr[:, :, :3]

        cube = open_hdr_img(str(hdr_path))
        if cube is None:
            LOGGER.warning("跳过无法读取的 HDR: %s", hdr_path)
            continue
        if cube.ndim == 2:
            cube = cube[..., np.newaxis]

        # 参考图（JPG 或某个光谱波段）
        ref_img: np.ndarray
        if reference_key == "jpg":
            ref_img = bgr
            per_band_shifts["jpg"].append((0.0, 0.0))
        else:
            if ref_idx is None:
                raise RuntimeError("ref_idx 未设置（reference_key != 'jpg'）")
            ref_img = cube[:, :, ref_idx]

            # JPG 行：估计 ref -> jpg
            ref_p, jpg_p = prepare_pair(
                ref_img,
                bgr,
                clip_percentile=clip_percentile,
                use_gradient=use_gradient,
                roi_frac=roi_frac,
                downscale=downscale,
            )
            if method == "phase":
                res_j = estimate_shift_phasecorr(ref_p, jpg_p)
            elif method == "ecc":
                res_j = estimate_shift_ecc(ref_p, jpg_p)
            else:
                raise ValueError(f"未知方法: {method}")

            score_ok = np.isfinite(res_j.score) and (res_j.score >= min_quality)
            if score_ok:
                per_band_shifts["jpg"].append((res_j.dx * inv_downscale, res_j.dy * inv_downscale))

        # 光谱行：估计 ref -> band（ref 自身为 0,0）
        for idx, band_nm in enumerate(SPECTRAL_CHANNELS):
            if reference_key != "jpg" and int(band_nm) == int(reference_key):
                per_band_shifts[band_nm].append((0.0, 0.0))
                continue

            band = cube[:, :, idx]
            ref_p, band_p = prepare_pair(
                ref_img,
                band,
                clip_percentile=clip_percentile,
                use_gradient=use_gradient,
                roi_frac=roi_frac,
                downscale=downscale,
            )

            if method == "phase":
                res = estimate_shift_phasecorr(ref_p, band_p)
            elif method == "ecc":
                res = estimate_shift_ecc(ref_p, band_p)
            else:
                raise ValueError(f"未知方法: {method}")

            score_ok = np.isfinite(res.score) and (res.score >= min_quality)
            if not score_ok:
                LOGGER.debug(
                    "样本质量不足，跳过 %s: ref=%s -> %s score=%.4f",
                    img_path.name,
                    reference_key,
                    band_nm,
                    getattr(res, "score", float("nan")),
                )
                continue

            per_band_shifts[band_nm].append((res.dx * inv_downscale, res.dy * inv_downscale))

    # 聚合
    aggregated: Dict[object, Tuple[float, float]] = {}
    for key in CHANNEL_ORDER:
        dx, dy = aggregate_robust(per_band_shifts[key])
        aggregated[key] = (dx, dy)
    return aggregated


def write_align_conf(conf_dir: Path, filename: str, offsets: Dict[object, Tuple[float, float]], *, dry_run: bool) -> Path:
    """将 offsets 写入 conf_dir/filename（.conf），按 CHANNEL_ORDER 顺序逐行写 (dx, dy)。"""
    conf_dir.mkdir(parents=True, exist_ok=True)
    conf_path = conf_dir / filename
    lines: List[str] = []
    for key in CHANNEL_ORDER:
        dx, dy = offsets.get(key, (0.0, 0.0))
        # 以整数为主，向最近整数取整
        lines.append(f"{int(round(dx))}, {int(round(dy))}\n")
    if dry_run:
        LOGGER.info("[dry-run] 将写入 %s:\n%s", conf_path, "".join(lines))
        return conf_path
    with conf_path.open("w", encoding="utf-8") as f:
        f.writelines(lines)
    LOGGER.info("已写出: %s", conf_path)
    return conf_path


# ----------------------------- CLI ----------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "从数据集中统计估计 JPG 与各光谱波段的平移偏移，分别生成 513 前/后的 align.conf。"
        )
    )
    # 默认参数已按推荐配置设置，可直接运行；需要调参时再覆盖对应选项。
    p.add_argument(
        "dataset_root",
        type=Path,
        help="包含 images/ 与 spectral/ 的数据集根目录",
    )
    p.add_argument(
        "--conf-ref",
        type=str,
        default="jpg",
        help="配置参考通道：'jpg'/'rgb' 或波段值（如 720）。该通道在输出中为 (0,0)。",
    )
    p.add_argument(
        "--method",
        choices=["phase", "ecc"],
        default="ecc",
        help="平移估计方法：phase(相位相关，快) 或 ecc(ECC，更稳健，默认)",
    )
    gradient_group = p.add_mutually_exclusive_group()
    gradient_group.add_argument(
        "--use-gradient",
        dest="use_gradient",
        action="store_true",
        help="使用梯度域进行匹配（默认启用）",
    )
    gradient_group.add_argument(
        "--no-gradient",
        dest="use_gradient",
        action="store_false",
        help="关闭梯度域匹配（如遇噪声过大的情况）",
    )
    p.set_defaults(use_gradient=True)
    p.add_argument(
        "--clip-percentile",
        type=float,
        default=1.0,
        help="预处理百分位裁剪比例（越大越强，默认 1.0）",
    )
    p.add_argument(
        "--roi-frac",
        type=float,
        default=0.8,
        help="中心 ROI 比例 (0,1]，默认 0.8 仅使用中心 80%%",
    )
    p.add_argument(
        "--downscale",
        type=float,
        default=0.5,
        help="估计前下采样比例（默认 0.5，可兼顾速度与稳健）",
    )
    p.add_argument(
        "--min-quality",
        type=float,
        default=0.1,
        help="样本质量阈值（phase: 响应值；ecc: 相关系数）",
    )
    p.add_argument(
        "--cutoff-date",
        default="20240513",
        help="日期分割线（含当日归为‘后’），格式 YYYYMMDD",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印拟写入内容，不落盘",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭进度条（默认开启；若未安装 tqdm 会自动关闭）",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default="data/config/oil/my-conf",
        help="对齐配置输出目录（默认: data/config/oil/my-conf）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.downscale <= 0:
        raise SystemExit("--downscale 必须大于 0。")

    # 解析 cutoff 日期
    try:
        cutoff = datetime.strptime(args.cutoff_date, "%Y%m%d")
    except Exception as e:
        raise SystemExit(f"无效的 --cutoff-date: {args.cutoff_date}") from e

    pairs = list_samples(args.dataset_root)
    if not pairs:
        LOGGER.warning("未找到可用样本（images/ 与 spectral/ 需同名配对）")
        return

    # 按日期分组：< cutoff 为“前”，>= cutoff 为“后”
    before: List[Tuple[Path, Path]] = []
    after: List[Tuple[Path, Path]] = []
    for img_path, hdr_path in pairs:
        try:
            d = extract_capture_date(img_path.name)
        except Exception:
            LOGGER.warning("文件名未包含日期，跳过：%s", img_path.name)
            continue
        if d < cutoff:
            before.append((img_path, hdr_path))
        else:
            after.append((img_path, hdr_path))

    LOGGER.info("样本统计：before=%d, after=%d (cutoff=%s)", len(before), len(after), args.cutoff_date)

    # 估计并聚合两组参数
    conf_ref = args.conf_ref.lower()
    if conf_ref in {"rgb", "jpg"}:
        conf_ref_key: object = "jpg"
    else:
        try:
            conf_ref_key = int(conf_ref)
        except ValueError as e:
            raise SystemExit("--conf-ref 需为 'jpg'/'rgb' 或波段整数（如 720）。") from e

    est_kwargs = dict(
        conf_ref=conf_ref_key,
        method=args.method,
        clip_percentile=args.clip_percentile,
        use_gradient=args.use_gradient,
        roi_frac=args.roi_frac,
        downscale=args.downscale,
        min_quality=args.min_quality,
        show_progress=not args.no_progress,
    )

    before_offsets = (
        estimate_offsets_for_group(before, progress_desc="Estimate offsets (before)", **est_kwargs) if before else None
    )
    after_offsets = (
        estimate_offsets_for_group(after, progress_desc="Estimate offsets (after)", **est_kwargs) if after else None
    )

    # 输出目录：默认写入 dataset_root/../../conf 下，并命名为 align_before_513.conf / align_after_513.conf
    conf_dir = args.out_dir or (args.dataset_root.parent.parent / "conf")
    if before_offsets is not None:
        write_align_conf(conf_dir, "align_before_513.conf", before_offsets, dry_run=args.dry_run)
    if after_offsets is not None:
        write_align_conf(conf_dir, "align_after_513.conf", after_offsets, dry_run=args.dry_run)

    LOGGER.info("完成偏移估计与配置生成。")


if __name__ == "__main__":
    main()

'''
python data_align/cal_offset.py /mnt/d/Project/master-graduation-project/data/oil/train/feedback  --conf-ref jpg --out-dir data/config/oil/my-conf_dev
'''
