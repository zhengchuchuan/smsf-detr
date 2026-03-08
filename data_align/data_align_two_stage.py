"""
Two-stage (coarse + fine) per-sample registration for RGB + multispectral cubes.

Goal:
- Do NOT use statistical/global alignment configs (align_before/after_513.conf).
- For each sample, estimate translations and warp channels into a common reference frame.
- Coarse: phase correlation on downscaled images (robust to large shifts).
- Fine: ECC translation refinement (subpixel).

Notes:
- This script assumes the dominant misalignment is translation (no rotation/scale).
- Subpixel accuracy depends on image content and modality; RGB<->MS can still be harder.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Literal

import sys

import cv2
import numpy as np
import tifffile

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io.spectral_io import (
    channel as SPECTRAL_CHANNELS,
    open_hdr_img,
    save_hdr_img,
)

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

@dataclass(frozen=True)
class ShiftResult:
    dx: float
    dy: float
    score: float


def estimate_shift_phasecorr(template: np.ndarray, input_img: np.ndarray) -> ShiftResult:
    """Phase correlation translation estimate: returns template->input (dx, dy) and response score."""
    template = template.astype(np.float32, copy=False)
    input_img = input_img.astype(np.float32, copy=False)
    template_b = cv2.GaussianBlur(template, (5, 5), 1.0)
    input_b = cv2.GaussianBlur(input_img, (5, 5), 1.0)
    (dx, dy), response = cv2.phaseCorrelate(template_b, input_b)
    return ShiftResult(float(dx), float(dy), float(response))


LOGGER = logging.getLogger(__name__)

CHANNEL_ORDER: Tuple[object, ...] = ("jpg", *SPECTRAL_CHANNELS)
RGB_PLACEHOLDER_BANDS: Tuple[int, int, int] = (0, 1, 2)

MotionModel = Literal["translation", "affine", "homography"]
FineMethod = Literal["ecc", "orb"]


@dataclass(frozen=True)
class PairWarp:
    warp_ref_to_target: List[List[float]]  # 3x3 (homogeneous), maps ref -> target
    coarse_score: float
    fine_score: float


def _to_bgr_u8(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise RuntimeError(f"Failed to load RGB image: {path}")
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    elif bgr.shape[2] == 4:
        bgr = bgr[:, :, :3]
    elif bgr.shape[2] != 3:
        raise ValueError(f"Unsupported channel count {bgr.shape[2]} in {path}")
    return bgr.astype(np.uint8, copy=False)


def _warp2x3_to_3x3(warp: np.ndarray) -> np.ndarray:
    out = np.eye(3, dtype=np.float32)
    out[:2, :] = warp.astype(np.float32, copy=False)
    return out


def _warp3x3_to_2x3(warp: np.ndarray) -> np.ndarray:
    return warp[:2, :].astype(np.float32, copy=False)


def _scale_matrix(s: float) -> np.ndarray:
    return np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

def _translate_matrix(tx: float, ty: float) -> np.ndarray:
    return np.array([[1.0, 0.0, float(tx)], [0.0, 1.0, float(ty)], [0.0, 0.0, 1.0]], dtype=np.float32)


def _percentile_normalize_to_float32(img: np.ndarray, clip_percentile: float) -> np.ndarray:
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


def _gradient_magnitude(u: np.ndarray) -> np.ndarray:
    if u.dtype != np.float32:
        u = u.astype(np.float32)
    gx = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(u, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mmax = float(mag.max()) if mag.size else 0.0
    if mmax <= 0:
        return np.zeros_like(mag, dtype=np.float32)
    return (mag / mmax).astype(np.float32)


def _center_crop_to(img: np.ndarray, target_h: int, target_w: int) -> Tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    if target_h > h or target_w > w:
        raise ValueError("target crop larger than image")
    y0 = (h - target_h) // 2
    x0 = (w - target_w) // 2
    cropped = img[y0 : y0 + target_h, x0 : x0 + target_w]
    return cropped, x0, y0


def _resize_frac(img: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def _prepare_single_with_mat(
    img: np.ndarray,
    *,
    clip_percentile: float,
    use_gradient: bool,
    roi_frac: float,
    downscale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (prepared_img, A) where A maps original pixel coords -> prepared coords."""
    if downscale <= 0:
        raise ValueError("downscale must be positive.")
    if not (0 < roi_frac <= 1.0):
        raise ValueError("roi_frac must be in (0,1].")

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    prepared = _percentile_normalize_to_float32(gray, clip_percentile)
    if use_gradient:
        prepared = _gradient_magnitude(prepared)

    # Track coordinate transform from original -> current.
    A = np.eye(3, dtype=np.float32)

    # Center ROI crop in original resolution.
    if roi_frac < 0.999:
        h, w = prepared.shape[:2]
        ch, cw = int(h * roi_frac), int(w * roi_frac)
        ch = max(1, min(ch, h))
        cw = max(1, min(cw, w))
        prepared, x0, y0 = _center_crop_to(prepared, ch, cw)
        A = _translate_matrix(-x0, -y0) @ A

    # Downscale.
    if abs(downscale - 1.0) > 1e-6:
        prepared = _resize_frac(prepared, downscale)
        A = _scale_matrix(downscale) @ A

    return prepared.astype(np.float32, copy=False), A


def prepare_pair_with_mats(
    img_a: np.ndarray,
    img_b: np.ndarray,
    *,
    clip_percentile: float,
    use_gradient: bool,
    roi_frac: float,
    downscale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Like cal_offset.prepare_pair, but also returns (A_a, A_b) mapping original -> prepared coords."""
    a, A_a = _prepare_single_with_mat(
        img_a,
        clip_percentile=clip_percentile,
        use_gradient=use_gradient,
        roi_frac=roi_frac,
        downscale=downscale,
    )
    b, A_b = _prepare_single_with_mat(
        img_b,
        clip_percentile=clip_percentile,
        use_gradient=use_gradient,
        roi_frac=roi_frac,
        downscale=downscale,
    )

    # Match sizes via centered crop (avoid interpolation).
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    if a.shape[0] != h or a.shape[1] != w:
        a, x0, y0 = _center_crop_to(a, h, w)
        A_a = _translate_matrix(-x0, -y0) @ A_a
    if b.shape[0] != h or b.shape[1] != w:
        b, x0, y0 = _center_crop_to(b, h, w)
        A_b = _translate_matrix(-x0, -y0) @ A_b

    return a.astype(np.float32, copy=False), b.astype(np.float32, copy=False), A_a, A_b


def _warp_full_to_scaled(warp_full: np.ndarray, scale: float) -> np.ndarray:
    s = _scale_matrix(scale)
    inv_s = _scale_matrix(1.0 / scale)
    return (s @ warp_full @ inv_s).astype(np.float32, copy=False)


def _warp_scaled_to_full(warp_scaled: np.ndarray, scale: float) -> np.ndarray:
    s = _scale_matrix(scale)
    inv_s = _scale_matrix(1.0 / scale)
    return (inv_s @ warp_scaled @ s).astype(np.float32, copy=False)


def _ecc_refine_warp(
    template: np.ndarray,
    input_img: np.ndarray,
    *,
    init_warp: np.ndarray,  # 3x3, ref->target (template->input)
    motion: MotionModel,
    iters: int,
    eps: float,
) -> Tuple[np.ndarray, float]:
    """ECC refinement. Returns (warp_ref_to_target(3x3), cc)."""
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(iters), float(eps))
    try:
        if motion == "translation":
            warp_in = _warp3x3_to_2x3(init_warp)
            cc, warp_out = cv2.findTransformECC(template, input_img, warp_in, cv2.MOTION_TRANSLATION, criteria)
            return _warp2x3_to_3x3(warp_out), float(cc)
        if motion == "affine":
            warp_in = _warp3x3_to_2x3(init_warp)
            cc, warp_out = cv2.findTransformECC(template, input_img, warp_in, cv2.MOTION_AFFINE, criteria)
            return _warp2x3_to_3x3(warp_out), float(cc)
        if motion == "homography":
            warp_in = init_warp.astype(np.float32, copy=True)
            cc, warp_out = cv2.findTransformECC(template, input_img, warp_in, cv2.MOTION_HOMOGRAPHY, criteria)
            return warp_out.astype(np.float32, copy=False), float(cc)
        raise ValueError(f"Unknown motion model: {motion}")
    except cv2.error as e:
        LOGGER.debug("ECC failed (%s): %s", motion, e)
        return init_warp, float("nan")

def _to_feature_u8(
    img: np.ndarray,
    *,
    clip_percentile: float,
    use_gradient: bool,
) -> np.ndarray:
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    u = _percentile_normalize_to_float32(gray, clip_percentile)
    if use_gradient:
        u = _gradient_magnitude(u)
    return np.clip(u * 255.0, 0.0, 255.0).astype(np.uint8)


def _estimate_orb_residual_warp(
    ref_u8: np.ndarray,
    tgt_aligned_u8: np.ndarray,
    *,
    motion: MotionModel,
    nfeatures: int,
    ratio: float,
    min_matches: int,
    ransac_reproj_threshold: float,
    ransac_max_iters: int,
    ransac_confidence: float,
) -> Optional[np.ndarray]:
    """Estimate residual warp Wr mapping ref -> tgt_aligned using ORB+RANSAC."""
    if motion == "translation":
        raise ValueError("ORB residual warp does not support pure translation motion.")

    orb = cv2.ORB_create(nfeatures=int(nfeatures))
    kp1, des1 = orb.detectAndCompute(ref_u8, None)
    kp2, des2 = orb.detectAndCompute(tgt_aligned_u8, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < float(ratio) * n.distance:
            good.append(m)
    if len(good) < int(min_matches):
        return None

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    if motion == "affine":
        M, inliers = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_reproj_threshold),
            maxIters=int(ransac_max_iters),
            confidence=float(ransac_confidence),
            refineIters=10,
        )
        if M is None:
            return None
        return _warp2x3_to_3x3(M)

    # homography
    H, inliers = cv2.findHomography(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_reproj_threshold),
        maxIters=int(ransac_max_iters),
        confidence=float(ransac_confidence),
    )
    if H is None:
        return None
    return H.astype(np.float32, copy=False)


def estimate_pair_warp_two_stage(
    ref_img: np.ndarray,
    target_img: np.ndarray,
    *,
    clip_percentile: float,
    use_gradient: bool,
    roi_frac: float,
    coarse_downscale: float,
    fine_pyramid: Sequence[float],
    motion: MotionModel,
    fine_method: FineMethod,
    ecc_iters: int,
    ecc_eps: float,
    min_coarse_score: float,
    min_fine_score: float,
    orb_nfeatures: int,
    orb_ratio: float,
    orb_min_matches: int,
    ransac_reproj_threshold: float,
    ransac_max_iters: int,
    ransac_confidence: float,
) -> Optional[PairWarp]:
    """Estimate warp for a pair; returns warp mapping ref -> target (template -> input)."""
    if coarse_downscale <= 0:
        raise ValueError("coarse_downscale must be positive.")
    if not fine_pyramid:
        raise ValueError("fine_pyramid must be non-empty.")
    if any(s <= 0 for s in fine_pyramid):
        raise ValueError("downscale must be positive.")

    # Coarse (phase correlation) at low resolution to capture large translation.
    ref_c, tgt_c, A_ref_c, A_tgt_c = prepare_pair_with_mats(
        ref_img,
        target_img,
        clip_percentile=clip_percentile,
        use_gradient=use_gradient,
        roi_frac=roi_frac,
        downscale=coarse_downscale,
    )
    coarse = estimate_shift_phasecorr(ref_c, tgt_c)
    if not np.isfinite(coarse.score) or coarse.score < min_coarse_score:
        return None

    # Coarse warp estimated in prepared coords -> convert to full-res coords.
    warp_prepared = _translate_matrix(coarse.dx, coarse.dy)
    warp_full = np.linalg.inv(A_tgt_c) @ warp_prepared @ A_ref_c

    best_score = float("nan")

    if fine_method == "ecc":
        for fine_downscale in fine_pyramid:
            ref_f, tgt_f, A_ref_f, A_tgt_f = prepare_pair_with_mats(
                ref_img,
                target_img,
                clip_percentile=clip_percentile,
                use_gradient=use_gradient,
                roi_frac=roi_frac,
                downscale=fine_downscale,
            )
            init_scaled = A_tgt_f @ warp_full @ np.linalg.inv(A_ref_f)
            refined_scaled, cc = _ecc_refine_warp(
                ref_f,
                tgt_f,
                init_warp=init_scaled,
                motion=motion,
                iters=ecc_iters,
                eps=ecc_eps,
            )
            if np.isfinite(cc):
                best_score = float(cc)
            warp_full = np.linalg.inv(A_tgt_f) @ refined_scaled @ A_ref_f

        if np.isfinite(best_score) and best_score < min_fine_score:
            return None

    elif fine_method == "orb":
        if motion == "translation":
            raise ValueError("--fine-method orb 需要 --motion affine 或 homography")

        # Use the finest pyramid level for ORB.
        fine_downscale = max(float(s) for s in fine_pyramid)
        ref_f, tgt_f, A_ref_f, A_tgt_f = prepare_pair_with_mats(
            ref_img,
            target_img,
            clip_percentile=clip_percentile,
            use_gradient=use_gradient,
            roi_frac=roi_frac,
            downscale=fine_downscale,
        )
        init_scaled = A_tgt_f @ warp_full @ np.linalg.inv(A_ref_f)

        # Warp target into ref coords (prepared) using coarse init, then estimate residual on smaller displacement.
        tgt_aligned = warp_to_reference(
            tgt_f,
            warp_ref_to_target=init_scaled,
            motion=motion,
            output_shape=ref_f.shape[:2],
            fill_value=0,
            interpolation=cv2.INTER_LINEAR,
        )
        ref_u8 = _to_feature_u8(ref_f, clip_percentile=0.0, use_gradient=False)
        tgt_u8 = _to_feature_u8(tgt_aligned, clip_percentile=0.0, use_gradient=False)
        Wr = _estimate_orb_residual_warp(
            ref_u8,
            tgt_u8,
            motion=motion,
            nfeatures=orb_nfeatures,
            ratio=orb_ratio,
            min_matches=orb_min_matches,
            ransac_reproj_threshold=ransac_reproj_threshold,
            ransac_max_iters=ransac_max_iters,
            ransac_confidence=ransac_confidence,
        )
        if Wr is not None:
            # aligned_target = warp(target -> ref) using init_scaled (Wc).
            # Residual Wr estimated between ref and aligned_target corresponds to: Wr ≈ Wc^{-1} * W*.
            # Therefore W* ≈ Wc * Wr.
            init_scaled = init_scaled @ Wr
            warp_full = np.linalg.inv(A_tgt_f) @ init_scaled @ A_ref_f

    else:
        raise ValueError(f"Unknown fine method: {fine_method}")

    return PairWarp(
        warp_ref_to_target=warp_full.astype(float).tolist(),
        coarse_score=float(coarse.score),
        fine_score=best_score,
    )


def estimate_pair_warp_with_fallbacks(
    ref_img: np.ndarray,
    target_img: np.ndarray,
    *,
    clip_percentile: float,
    use_gradient: bool,
    roi_frac: float,
    coarse_downscale: float,
    fine_pyramid: Sequence[float],
    motion: MotionModel,
    fine_method: FineMethod,
    ecc_iters: int,
    ecc_eps: float,
    min_coarse_score: float,
    min_fine_score: float,
    orb_nfeatures: int,
    orb_ratio: float,
    orb_min_matches: int,
    ransac_reproj_threshold: float,
    ransac_max_iters: int,
    ransac_confidence: float,
) -> Optional[PairWarp]:
    """Two-stage estimate with a couple of safe fallbacks for difficult modalities."""
    primary = estimate_pair_warp_two_stage(
        ref_img,
        target_img,
        clip_percentile=clip_percentile,
        use_gradient=use_gradient,
        roi_frac=roi_frac,
        coarse_downscale=coarse_downscale,
        fine_pyramid=fine_pyramid,
        motion=motion,
        fine_method=fine_method,
        ecc_iters=ecc_iters,
        ecc_eps=ecc_eps,
        min_coarse_score=min_coarse_score,
        min_fine_score=min_fine_score,
        orb_nfeatures=orb_nfeatures,
        orb_ratio=orb_ratio,
        orb_min_matches=orb_min_matches,
        ransac_reproj_threshold=ransac_reproj_threshold,
        ransac_max_iters=ransac_max_iters,
        ransac_confidence=ransac_confidence,
    )
    if primary is not None:
        return primary

    # Fallback 1: disable gradient (sometimes helps very low-SNR bands).
    if use_gradient:
        secondary = estimate_pair_warp_two_stage(
            ref_img,
            target_img,
            clip_percentile=clip_percentile,
            use_gradient=False,
            roi_frac=roi_frac,
            coarse_downscale=coarse_downscale,
            fine_pyramid=fine_pyramid,
            motion=motion,
            fine_method=fine_method,
            ecc_iters=ecc_iters,
            ecc_eps=ecc_eps,
            min_coarse_score=min_coarse_score,
            min_fine_score=min_fine_score,
            orb_nfeatures=orb_nfeatures,
            orb_ratio=orb_ratio,
            orb_min_matches=orb_min_matches,
            ransac_reproj_threshold=ransac_reproj_threshold,
            ransac_max_iters=ransac_max_iters,
            ransac_confidence=ransac_confidence,
        )
        if secondary is not None:
            return secondary

    # Fallback 2: expand ROI to full frame (helps when content drifts out of the center).
    if roi_frac < 0.999:
        tertiary = estimate_pair_warp_two_stage(
            ref_img,
            target_img,
            clip_percentile=clip_percentile,
            use_gradient=use_gradient,
            roi_frac=1.0,
            coarse_downscale=coarse_downscale,
            fine_pyramid=fine_pyramid,
            motion=motion,
            fine_method=fine_method,
            ecc_iters=ecc_iters,
            ecc_eps=ecc_eps,
            min_coarse_score=min_coarse_score,
            min_fine_score=min_fine_score,
            orb_nfeatures=orb_nfeatures,
            orb_ratio=orb_ratio,
            orb_min_matches=orb_min_matches,
            ransac_reproj_threshold=ransac_reproj_threshold,
            ransac_max_iters=ransac_max_iters,
            ransac_confidence=ransac_confidence,
        )
        if tertiary is not None:
            return tertiary

    return None


def warp_to_reference(
    image: np.ndarray,
    *,
    warp_ref_to_target: np.ndarray,  # 3x3, ref->target
    motion: MotionModel,
    output_shape: Tuple[int, int],
    fill_value: float = 0.0,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Warp `image` (target) into reference frame using inverse mapping of warp_ref_to_target.

    `warp_ref_to_target` maps ref -> target (template -> input), so to warp target -> ref we use WARP_INVERSE_MAP.
    """
    h, w = output_shape
    border = (
        (float(fill_value),) * image.shape[2]
        if image.ndim == 3
        else float(fill_value)
    )
    if motion in {"translation", "affine"}:
        matrix_2x3 = _warp3x3_to_2x3(warp_ref_to_target)
        return cv2.warpAffine(
            image,
            matrix_2x3,
            (int(w), int(h)),
            flags=interpolation | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border,
        )
    if motion == "homography":
        return cv2.warpPerspective(
            image,
            warp_ref_to_target.astype(np.float32, copy=False),
            (int(w), int(h)),
            flags=interpolation | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border,
        )
    raise ValueError(f"Unknown motion model: {motion}")


def normalize_to_uint8(band: np.ndarray, clip_percentile: float) -> np.ndarray:
    valid = band[np.isfinite(band)]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.uint8)
    lo = np.percentile(valid, clip_percentile)
    hi = np.percentile(valid, 100 - clip_percentile)
    if hi <= lo:
        lo = float(valid.min())
        hi = float(valid.max())
        if hi <= lo:
            return np.zeros_like(band, dtype=np.uint8)
    scaled = np.clip((band - lo) / (hi - lo), 0, 1)
    return (scaled * 255).astype(np.uint8)


def list_pairs(dataset_root: Path) -> List[Tuple[Path, Path]]:
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


def parse_reference(value: str) -> object:
    lowered = value.lower()
    if lowered in {"jpg", "rgb"}:
        return "jpg"
    numeric = int(value)
    if numeric not in SPECTRAL_CHANNELS:
        valid = ", ".join(str(b) for b in SPECTRAL_CHANNELS)
        raise argparse.ArgumentTypeError(
            f"Unsupported reference '{numeric}'. Use 'jpg' or one of: {valid}."
        )
    return numeric


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-sample two-stage alignment (phase -> ECC) without global configs."
    )
    p.add_argument("dataset_root", type=Path, help="包含 images/ 与 spectral/ 的数据集根目录")
    p.add_argument(
        "--reference",
        type=parse_reference,
        default="jpg",
        help="对齐参考通道：固定为 'jpg'（本脚本按 JPG 作为基准进行两阶段配准）。",
    )
    p.add_argument(
        "--output-full",
        type=Path,
        default=None,
        help="RGB+7band 输出目录（默认: dataset_root / 'aligned_full_tif_two_stage_<motion>').",
    )
    p.add_argument(
        "--output-nir",
        type=Path,
        default=None,
        help="7band 输出目录（默认: dataset_root / 'aligned_nir_tif_two_stage_<motion>').",
    )
    p.add_argument(
        "--full-format",
        choices=["tif", "hdr"],
        default="hdr",
        help="full 输出格式。",
    )
    p.add_argument(
        "--nir-format",
        choices=["tif", "hdr"],
        default="tif",
        help="nir 输出格式。",
    )
    p.add_argument(
        "--use-gradient",
        action="store_true",
        help="使用梯度域匹配（跨模态更稳健）。",
    )
    p.add_argument(
        "--no-gradient",
        dest="use_gradient",
        action="store_false",
        help="关闭梯度域匹配。",
    )
    p.set_defaults(use_gradient=True)
    p.add_argument("--clip-percentile", type=float, default=1.0)
    p.add_argument("--roi-frac", type=float, default=0.8, help="中心 ROI 比例 (0,1]，默认 0.8")
    p.add_argument("--coarse-downscale", type=float, default=0.25, help="粗配准下采样比例")
    p.add_argument(
        "--fine-pyramid",
        type=str,
        default="0.5,1.0",
        help="精配准金字塔下采样比例列表（逗号分隔，如 '0.5,1.0'）。",
    )
    p.add_argument(
        "--motion",
        choices=["translation", "affine", "homography"],
        default="homography",
        help="精配准变换模型：translation/affine/homography（默认 homography）。",
    )
    p.add_argument(
        "--fine-method",
        choices=["ecc", "orb"],
        default="ecc",
        help="第二阶段精配准算法：ecc 或 orb（默认 ecc）。",
    )
    p.add_argument("--ecc-iters", type=int, default=400, help="ECC 迭代次数")
    p.add_argument("--ecc-eps", type=float, default=1e-6, help="ECC 收敛阈值")
    p.add_argument("--min-coarse-score", type=float, default=0.1, help="phase 响应阈值")
    p.add_argument("--min-fine-score", type=float, default=0.6, help="ECC 相关系数阈值")
    p.add_argument("--orb-nfeatures", type=int, default=2000, help="ORB 关键点数量上限")
    p.add_argument("--orb-ratio", type=float, default=0.75, help="ORB knn ratio test 阈值")
    p.add_argument("--orb-min-matches", type=int, default=30, help="ORB 最少有效匹配数")
    p.add_argument("--ransac-reproj", type=float, default=3.0, help="RANSAC 重投影阈值（像素）")
    p.add_argument("--ransac-iters", type=int, default=2000, help="RANSAC 最大迭代次数")
    p.add_argument("--ransac-confidence", type=float, default=0.995, help="RANSAC 置信度")
    p.add_argument(
        "--save-offsets-jsonl",
        type=Path,
        default=None,
        help="将每个样本估计到的偏移写入 JSONL（可用于排查/复现）。",
    )
    p.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="输出文件名后缀（默认自动生成，如 20251228-2316）。",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    if args.reference != "jpg":
        raise SystemExit("--reference 当前仅支持 'jpg'。")

    try:
        fine_pyramid = [float(x.strip()) for x in str(args.fine_pyramid).split(",") if x.strip()]
    except Exception as e:
        raise SystemExit(f"无效的 --fine-pyramid: {args.fine_pyramid}") from e

    motion: MotionModel = args.motion
    fine_method: FineMethod = args.fine_method

    pairs = list_pairs(args.dataset_root)
    if not pairs:
        LOGGER.warning("未找到可用样本（images/ 与 spectral/ 需同名配对）")
        return

    # Match data_align.py style but avoid clobbering by default; include motion model in folder name.
    output_full = args.output_full or (args.dataset_root / f"aligned_full_tif_two_stage_{fine_method}_{motion}")
    output_nir = args.output_nir or (args.dataset_root / f"aligned_nir_tif_two_stage_{fine_method}_{motion}")
    output_full.mkdir(parents=True, exist_ok=True)
    output_nir.mkdir(parents=True, exist_ok=True)

    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d-%H%M")

    offsets_fp = args.save_offsets_jsonl
    offsets_fh = None
    if offsets_fp is not None:
        offsets_fp.parent.mkdir(parents=True, exist_ok=True)
        offsets_fh = offsets_fp.open("a" if offsets_fp.exists() else "w", encoding="utf-8")

    iterator: Iterable[Tuple[Path, Path]] = pairs
    if not args.no_progress and tqdm is not None:
        iterator = tqdm(pairs, total=len(pairs), desc="Align(two-stage)", unit="sample")

    for rgb_path, hdr_path in iterator:
        stem = rgb_path.stem
        full_out = output_full / f"{stem}_{run_tag}.{'hdr' if args.full_format == 'hdr' else 'tif'}"
        nir_out = output_nir / f"{stem}_{run_tag}.{'hdr' if args.nir_format == 'hdr' else 'tif'}"
        if not args.overwrite and (full_out.exists() or nir_out.exists()):
            continue

        try:
            bgr = _to_bgr_u8(rgb_path)
            cube = open_hdr_img(str(hdr_path))
            if cube is None:
                raise RuntimeError(f"Failed to open HDR cube: {hdr_path}")
            if cube.ndim == 2:
                cube = cube[..., np.newaxis]

            if cube.shape[2] != len(SPECTRAL_CHANNELS):
                raise ValueError(
                    f"Unexpected band count {cube.shape[2]} for {hdr_path}, expected {len(SPECTRAL_CHANNELS)}"
                )

            # Reference is always JPG for this script.
            ref_img = bgr

            # Estimate per-channel warps (jpg -> band).
            warps: Dict[object, PairWarp] = {}
            warps["jpg"] = PairWarp(np.eye(3, dtype=float).tolist(), 1.0, 1.0)

            for idx, band_nm in enumerate(SPECTRAL_CHANNELS):
                band = cube[:, :, idx]
                band_warp = estimate_pair_warp_with_fallbacks(
                    ref_img,
                    band,
                    clip_percentile=args.clip_percentile,
                    use_gradient=args.use_gradient,
                    roi_frac=args.roi_frac,
                    coarse_downscale=args.coarse_downscale,
                    fine_pyramid=fine_pyramid,
                    motion=motion,
                    fine_method=fine_method,
                    ecc_iters=args.ecc_iters,
                    ecc_eps=args.ecc_eps,
                    min_coarse_score=args.min_coarse_score,
                    min_fine_score=args.min_fine_score,
                    orb_nfeatures=args.orb_nfeatures,
                    orb_ratio=args.orb_ratio,
                    orb_min_matches=args.orb_min_matches,
                    ransac_reproj_threshold=args.ransac_reproj,
                    ransac_max_iters=args.ransac_iters,
                    ransac_confidence=args.ransac_confidence,
                )
                if band_warp is None:
                    LOGGER.warning("Warp estimate failed for %s (band %s); fallback to identity.", rgb_path.name, band_nm)
                    warps[band_nm] = PairWarp(np.eye(3, dtype=float).tolist(), float("nan"), float("nan"))
                else:
                    warps[band_nm] = band_warp

            # Warp all bands into JPG frame (target -> jpg): negate (jpg->target).
            ref_h, ref_w = bgr.shape[:2]
            bgr_aligned = bgr

            aligned_bands_raw: List[np.ndarray] = []
            for idx, band_nm in enumerate(SPECTRAL_CHANNELS):
                band = cube[:, :, idx]
                w = np.asarray(warps[band_nm].warp_ref_to_target, dtype=np.float32)
                band_aligned = warp_to_reference(
                    band,
                    warp_ref_to_target=w,
                    motion=motion,
                    output_shape=(ref_h, ref_w),
                    fill_value=0,
                    interpolation=cv2.INTER_LINEAR,
                )
                aligned_bands_raw.append(band_aligned)

            spectral = np.stack(aligned_bands_raw, axis=2)
            if bgr_aligned.dtype != spectral.dtype:
                # When saving to uint16 stacks, plain astype(uint16) keeps values in 0..255 which
                # many viewers render as near-black. Expand 8-bit to 16-bit by bit replication:
                # v16 = (v8 << 8) | v8 == v8 * 257, so 255 maps to 65535 (full range).
                if bgr_aligned.dtype == np.uint8 and spectral.dtype == np.uint16:
                    bgr_u16 = bgr_aligned.astype(np.uint16, copy=False)
                    bgr_aligned = (bgr_u16 << 8) | bgr_u16
                else:
                    bgr_aligned = bgr_aligned.astype(spectral.dtype, copy=False)
            full = np.concatenate([bgr_aligned, spectral], axis=2)

            if args.nir_format == "hdr":
                save_hdr_img(
                    spectral,
                    str(nir_out),
                    list(SPECTRAL_CHANNELS),
                    band_names=[f"{nm}nm" for nm in SPECTRAL_CHANNELS],
                )
                LOGGER.info("Saved %s", nir_out)
            else:
                planar_nir = np.ascontiguousarray(spectral.transpose(2, 0, 1))
                tifffile.imwrite(
                    str(nir_out),
                    planar_nir,
                    dtype=planar_nir.dtype,
                    photometric="MINISBLACK",
                    planarconfig="SEPARATE",
                    metadata={
                        "axes": "SYX",
                        "ChannelNames": [f"{nm}nm" for nm in SPECTRAL_CHANNELS],
                    },
                )
                LOGGER.info("Saved %s", nir_out)

            if args.full_format == "hdr":
                save_hdr_img(
                    full,
                    str(full_out),
                    [*RGB_PLACEHOLDER_BANDS, *SPECTRAL_CHANNELS],
                    band_names=["B", "G", "R", *[f"{nm}nm" for nm in SPECTRAL_CHANNELS]],
                )
                LOGGER.info("Saved %s", full_out)
            else:
                planar_full = np.ascontiguousarray(full.transpose(2, 0, 1))
                tifffile.imwrite(
                    str(full_out),
                    planar_full,
                    dtype=planar_full.dtype,
                    photometric="MINISBLACK",
                    planarconfig="SEPARATE",
                    metadata={
                        "axes": "SYX",
                        "ChannelNames": ["B", "G", "R", *[f"{nm}nm" for nm in SPECTRAL_CHANNELS]],
                    },
                )
                LOGGER.info("Saved %s", full_out)

            if offsets_fh is not None:
                payload = {
                    "stem": stem,
                    "reference": "jpg",
                    "motion": motion,
                    "warps": {str(k): asdict(v) for k, v in warps.items()},
                }
                offsets_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
                offsets_fh.flush()

        except Exception as exc:
            LOGGER.exception("Failed to align %s: %s", rgb_path.name, exc)
            continue

    if offsets_fh is not None:
        offsets_fh.close()


if __name__ == "__main__":
    main()
