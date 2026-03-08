from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class MatchAnythingSettings:
    repo_dir: Path
    config_path: Path
    ckpt_path: Path
    method: str  # "matchanything_eloftr" or "matchanything_roma"
    estimator: str = "similarity"  # "similarity" (estimateAffinePartial2D) or "affine" (estimateAffine2D)
    device: str = "cuda"
    imgresize: int = 832  # longer-edge resize before matching; <=0 disables.
    min_confidence: float = 0.1
    max_matches: int = 5000
    ransac_reproj_threshold: float = 3.0
    min_inliers: int = 8
    min_scale: float = 0.7
    max_scale: float = 1.3


@dataclass(frozen=True)
class MatchAnythingAffineResult:
    """Outputs from MatchAnything matching + RANSAC fitting.

    All keypoints are in the *original preserved* coordinates of the inputs.
    """

    matrix: np.ndarray  # 2x3 float32, maps target -> reference
    mkpts_target: np.ndarray  # Nx2 float32
    mkpts_reference: np.ndarray  # Nx2 float32
    confidence: np.ndarray  # (N,) float32
    inliers: np.ndarray  # (N,) bool


def _resize_keep_aspect(
    image: np.ndarray,
    *,
    max_dim: int,
    interpolation: int,
) -> Tuple[np.ndarray, float, float]:
    height, width = image.shape[:2]
    if max_dim <= 0:
        return image, 1.0, 1.0

    # MatchAnything's official evaluation rescales so that the longer edge becomes `max_dim`
    # (even if this upsamples).
    scale = float(max_dim) / float(max(height, width))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    sx = float(width) / float(new_w)
    sy = float(height) / float(new_h)
    return resized, sx, sy


def _ensure_repo_on_path(repo_dir: Path) -> None:
    import sys

    repo_str = str(repo_dir.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


@lru_cache(maxsize=2)
def _get_matchanything_matcher(
    repo_dir: str,
    config_path: str,
    ckpt_path: str,
    method: str,
    device: str,
    imgresize: int,
):
    import torch
    from contextlib import contextmanager

    repo = Path(repo_dir)
    cfg = Path(config_path)
    ckpt = Path(ckpt_path)
    _ensure_repo_on_path(repo)
    roma_dir = repo / "third_party" / "ROMA"
    if roma_dir.is_dir():
        import sys

        roma_str = str(roma_dir.resolve())
        roma_parent = str(roma_dir.parent.resolve())
        # Ensure both ROMA and its parent are importable.
        if roma_parent not in sys.path:
            sys.path.insert(0, roma_parent)
        if roma_str not in sys.path:
            sys.path.insert(0, roma_str)

    try:
        from src.config.default import get_cfg_defaults  # type: ignore
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) == "yacs":
            raise ModuleNotFoundError(
                "MatchAnything dependency missing: 'yacs'. Install it with `pip install yacs`."
            ) from exc
        raise
    try:
        from yacs.config import CfgNode as CN  # type: ignore
    except ModuleNotFoundError:
        # yacs is required by MatchAnything configs; the import error is handled above.
        CN = None  # type: ignore[assignment]

    class _PassThroughProfiler:
        @staticmethod
        @contextmanager
        def profile(action_name: str):
            yield action_name

    def _lower_config(node):
        # Convert yacs CfgNode to a nested dict with lowercase keys (MatchAnything expects this).
        if CN is not None and isinstance(node, CN):
            return {str(k).lower(): _lower_config(v) for k, v in node.items()}
        return node

    config = get_cfg_defaults()
    config.merge_from_file(str(cfg))
    config.METHOD = str(method)

    # MatchAnything's ELoFTR config enables RoPE; the implementation expects a 4-tuple `NPE`
    # (train_res_H, train_res_W, test_res_H, test_res_W). The official evaluation script sets:
    #   NPE = [832, 832, args.imgresize, args.imgresize]
    # so we mirror that here.
    if config.DATASET.NPE_NAME is not None:
        test_res = int(imgresize) if int(imgresize) > 0 else 832
        config.LOFTR.COARSE.NPE = [832, 832, test_res, test_res]

    config_l = _lower_config(config)
    if str(method) == "matchanything_eloftr":
        from src.loftr import LoFTR  # type: ignore

        module = LoFTR(config=config_l["loftr"], profiler=_PassThroughProfiler())
    elif str(method) == "matchanything_roma":
        try:
            from ROMA.roma.matchanything_roma_model import MatchAnything_Model  # type: ignore
        except ModuleNotFoundError as exc:
            missing = getattr(exc, "name", None)
            if missing == "kornia":
                raise ModuleNotFoundError(
                    "MatchAnything ROMA requires 'kornia'. Install it with `pip install kornia`."
                ) from exc
            if missing == "ROMA":
                raise ModuleNotFoundError(
                    "Failed to import ROMA package. Ensure MatchAnything repo has "
                    "third_party/ROMA and that it is on PYTHONPATH."
                ) from exc
            raise

        module = MatchAnything_Model(config=config_l["roma"], test_mode=True)
    else:
        raise ValueError(f"Unsupported MatchAnything method: {method}")

    # Load checkpoint (Lightning-style .ckpt contains {'state_dict': ...}).
    ckpt_obj = torch.load(str(ckpt), map_location="cpu")
    state_dict = ckpt_obj.get("state_dict", ckpt_obj)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unexpected checkpoint format: {type(ckpt_obj)}")

    # Most MatchAnything checkpoints prefix parameters with "matcher." (because it was saved from a LightningModule).
    if any(str(k).startswith("matcher.") for k in state_dict.keys()):
        state_dict = {
            (str(k)[len("matcher.") :] if str(k).startswith("matcher.") else str(k)): v
            for k, v in state_dict.items()
        }

    module.load_state_dict(state_dict, strict=False)
    module = module.eval()
    module = module.to(torch.device(device))
    return module, config


def _pad_bottom_right_to_multiple(
    image: np.ndarray,
    *,
    multiple: int,
    value: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad image to have H/W divisible by `multiple`, return (padded, valid_mask).

    valid_mask is bool with True for original pixels and False for padded region.
    """
    if multiple <= 1:
        mask = np.ones(image.shape[:2], dtype=bool)
        return image, mask

    height, width = image.shape[:2]
    pad_h = (-height) % int(multiple)
    pad_w = (-width) % int(multiple)
    if pad_h == 0 and pad_w == 0:
        mask = np.ones((height, width), dtype=bool)
        return image, mask

    pad_spec = ((0, pad_h), (0, pad_w))
    if image.ndim == 3:
        pad_spec = (*pad_spec, (0, 0))

    padded = np.pad(image, pad_spec, mode="constant", constant_values=int(value))
    mask = np.zeros(padded.shape[:2], dtype=bool)
    mask[:height, :width] = True
    return padded, mask


def _matchanything_get_matches(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    ref_mask: Optional[np.ndarray],
    tgt_mask: Optional[np.ndarray],
    settings: MatchAnythingSettings,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Run MatchAnything and return filtered matches in preserved coordinates.

    Inputs are expected to be 2D uint8 grayscale arrays in the same coordinate frame
    as you will later warp (e.g. "preserved" coordinates in data_align scripts).
    """
    if reference.ndim != 2 or target.ndim != 2:
        raise ValueError("MatchAnything expects 2D grayscale images for this wrapper.")
    if reference.dtype != np.uint8 or target.dtype != np.uint8:
        raise ValueError(
            "MatchAnything expects uint8 grayscale; call normalize_to_uint8 first."
        )

    matcher, config = _get_matchanything_matcher(
        str(settings.repo_dir),
        str(settings.config_path),
        str(settings.ckpt_path),
        str(settings.method),
        str(settings.device),
        int(settings.imgresize),
    )
    device = next(matcher.parameters()).device  # type: ignore[attr-defined]

    ref_small, sx_ref, sy_ref = _resize_keep_aspect(
        reference,
        max_dim=int(settings.imgresize),
        interpolation=cv2.INTER_AREA,
    )
    tgt_small, sx_tgt, sy_tgt = _resize_keep_aspect(
        target,
        max_dim=int(settings.imgresize),
        interpolation=cv2.INTER_AREA,
    )
    ref_h_resized, ref_w_resized = ref_small.shape[:2]
    tgt_h_resized, tgt_w_resized = tgt_small.shape[:2]

    # Work around MatchAnything ELoFTR's PAN aggregation which assumes coarse feature maps
    # are divisible by `pool_size`. For stride=8 this means input H/W must be divisible by 32.
    if settings.method == "matchanything_eloftr":
        try:
            coarse_stride = int(config.LOFTR.RESOLUTION[0])
        except Exception:
            coarse_stride = 8
        try:
            pool_size = int(config.LOFTR.COARSE.POOl_SIZE)
            pool_size2 = int(config.LOFTR.COARSE.POOl_SIZE2)
        except Exception:
            pool_size = 1
            pool_size2 = 1
        required_multiple = max(1, coarse_stride * max(pool_size, pool_size2))
        ref_small, _ = _pad_bottom_right_to_multiple(
            ref_small, multiple=required_multiple, value=0
        )
        tgt_small, _ = _pad_bottom_right_to_multiple(
            tgt_small, multiple=required_multiple, value=0
        )

    import torch

    batch: dict[str, torch.Tensor] = {}
    if settings.method == "matchanything_eloftr":
        batch["image0"] = (
            torch.from_numpy(tgt_small.astype(np.float32) / 255.0)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )
        batch["image1"] = (
            torch.from_numpy(ref_small.astype(np.float32) / 255.0)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )
    elif settings.method == "matchanything_roma":
        tgt_rgb = np.repeat(tgt_small[:, :, None], 3, axis=2)
        ref_rgb = np.repeat(ref_small[:, :, None], 3, axis=2)
        batch["image0_rgb_origin"] = (
            torch.from_numpy(tgt_rgb.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )
        batch["image1_rgb_origin"] = (
            torch.from_numpy(ref_rgb.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )
    else:
        raise ValueError(f"Unsupported MatchAnything method: {settings.method}")

    with torch.no_grad():
        # MatchAnything's internal code uses autocast('cuda') in a few places; keep it enabled only on CUDA.
        use_cuda_amp = (
            bool(getattr(config, "LOFTR", None) and config.LOFTR.FP16)
            and device.type == "cuda"
        )
        with torch.autocast(enabled=use_cuda_amp, device_type=device.type):
            matcher(batch)  # type: ignore[misc]

    mkpts0 = batch.get("mkpts0_f")
    mkpts1 = batch.get("mkpts1_f")
    mconf = batch.get("mconf")
    if mkpts0 is None or mkpts1 is None or mconf is None:
        return None

    k0 = mkpts0.detach().cpu().numpy().astype(np.float32)  # target (image0)
    k1 = mkpts1.detach().cpu().numpy().astype(np.float32)  # reference (image1)
    c = mconf.detach().cpu().numpy().astype(np.float32)

    if k0.shape[0] < 8:
        return None

    # Filter out matches that fall into padded region (if we padded).
    # Use the pre-pad resized sizes.
    in_tgt = (
        (k0[:, 0] >= 0)
        & (k0[:, 0] <= float(tgt_w_resized - 1))
        & (k0[:, 1] >= 0)
        & (k0[:, 1] <= float(tgt_h_resized - 1))
    )
    in_ref = (
        (k1[:, 0] >= 0)
        & (k1[:, 0] <= float(ref_w_resized - 1))
        & (k1[:, 1] >= 0)
        & (k1[:, 1] <= float(ref_h_resized - 1))
    )
    keep = in_tgt & in_ref
    k0 = k0[keep]
    k1 = k1[keep]
    c = c[keep]
    if k0.shape[0] < 8:
        return None

    keep = c >= float(settings.min_confidence)
    k0 = k0[keep]
    k1 = k1[keep]
    c = c[keep]
    if k0.shape[0] < 8:
        return None

    max_m = int(settings.max_matches)
    if max_m > 0 and k0.shape[0] > max_m:
        idx = np.argsort(-c)[:max_m]
        k0 = k0[idx]
        k1 = k1[idx]

    # Scale matches back to original preserved coords.
    k0[:, 0] *= float(sx_tgt)
    k0[:, 1] *= float(sy_tgt)
    k1[:, 0] *= float(sx_ref)
    k1[:, 1] *= float(sy_ref)

    # Mask filtering in original preserved coordinates.
    if tgt_mask is not None:
        tgtm = (tgt_mask > 0).astype(bool)
        x0 = np.clip(np.round(k0[:, 0]).astype(np.int32), 0, tgtm.shape[1] - 1)
        y0 = np.clip(np.round(k0[:, 1]).astype(np.int32), 0, tgtm.shape[0] - 1)
        keep0 = tgtm[y0, x0]
    else:
        keep0 = np.ones((k0.shape[0],), dtype=bool)

    if ref_mask is not None:
        refm = (ref_mask > 0).astype(bool)
        x1 = np.clip(np.round(k1[:, 0]).astype(np.int32), 0, refm.shape[1] - 1)
        y1 = np.clip(np.round(k1[:, 1]).astype(np.int32), 0, refm.shape[0] - 1)
        keep1 = refm[y1, x1]
    else:
        keep1 = np.ones((k1.shape[0],), dtype=bool)

    keep = keep0 & keep1
    k0 = k0[keep]
    k1 = k1[keep]
    c = c[keep]
    if k0.shape[0] < 8:
        return None

    return k0, k1, c


def estimate_matchanything_affine_transform_and_matches(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    ref_mask: Optional[np.ndarray],
    tgt_mask: Optional[np.ndarray],
    settings: MatchAnythingSettings,
) -> Optional[MatchAnythingAffineResult]:
    """Estimate affine transform mapping `target -> reference` using MatchAnything.

    Returns both the fitted matrix and the underlying keypoint correspondences.
    """
    matches = _matchanything_get_matches(
        reference,
        target,
        ref_mask=ref_mask,
        tgt_mask=tgt_mask,
        settings=settings,
    )
    if matches is None:
        return None
    k0, k1, c = matches  # target, reference

    if settings.estimator == "similarity":
        matrix, inliers = cv2.estimateAffinePartial2D(
            k0.reshape(-1, 1, 2),
            k1.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=float(settings.ransac_reproj_threshold),
            maxIters=5000,
            confidence=0.999,
            refineIters=10,
        )
    elif settings.estimator == "affine":
        matrix, inliers = cv2.estimateAffine2D(
            k0.reshape(-1, 1, 2),
            k1.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=float(settings.ransac_reproj_threshold),
            maxIters=5000,
            confidence=0.999,
            refineIters=10,
        )
    else:
        raise ValueError(f"Unsupported MatchAnything estimator: {settings.estimator}")
    if matrix is None or inliers is None:
        return None

    inlier_count = int(np.count_nonzero(inliers))
    if inlier_count < int(settings.min_inliers):
        return None

    linear = matrix[:2, :2].astype(np.float64, copy=False)
    try:
        singular_values = np.linalg.svd(linear, compute_uv=False)
    except np.linalg.LinAlgError:
        return None
    min_s = float(np.min(singular_values))
    max_s = float(np.max(singular_values))
    if not (
        float(settings.min_scale) <= min_s <= float(settings.max_scale)
        and float(settings.min_scale) <= max_s <= float(settings.max_scale)
    ):
        return None

    inlier_mask = inliers.astype(bool).reshape(-1)
    return MatchAnythingAffineResult(
        matrix=matrix.astype(np.float32),
        mkpts_target=k0,
        mkpts_reference=k1,
        confidence=c,
        inliers=inlier_mask,
    )


def estimate_matchanything_affine_transform(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    ref_mask: Optional[np.ndarray],
    tgt_mask: Optional[np.ndarray],
    settings: MatchAnythingSettings,
) -> Optional[np.ndarray]:
    """Backwards-compatible wrapper returning only the affine matrix."""
    result = estimate_matchanything_affine_transform_and_matches(
        reference,
        target,
        ref_mask=ref_mask,
        tgt_mask=tgt_mask,
        settings=settings,
    )
    if result is None:
        return None
    return result.matrix
