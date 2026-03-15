from __future__ import annotations

import logging
from argparse import Namespace
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch.nn as nn

from engines.models.base import BaseDetector

from .rtdetrv4_detector import RTDETRv4Detector
from .composite_criterion import RtmsfDetrCriterionWithMasks


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_rtdetrv4_config(path_like: str | Path | None) -> Path:
    """
    解析 RT-DETRv4 的 YAML 配置路径：
    - 允许传入绝对路径；
    - 允许传入相对项目根目录的路径；
    - 缺省使用 rtv4_hgnetv2_s_coco.yml。
    """
    if path_like is None:
        return (
            _repo_root()
            / "engines"
            / "models"
            / "rtmsfdetr"
            / "rtdetrv4"
            / "configs"
            / "rtv4"
            / "rtv4_hgnetv2_s_coco.yml"
        )
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = _repo_root() / path
    return path


def _build_rtdetrv4_overrides(args: Namespace, *, cfg_path: Path) -> Dict[str, Any]:
    num_classes = int(getattr(args, "num_classes", 80))
    img_size = int(getattr(args, "img_size", 640))
    disable_distill = bool(getattr(args, "disable_distill", True))

    rgb_channels = int(getattr(args, "rgb_input_channels", 3))
    ms_channels = int(getattr(args, "ms_input_channels", 0))
    input_channels = int(getattr(args, "input_channels", rgb_channels + ms_channels))

    hgnet_pretrained = bool(getattr(args, "hgnet_pretrained", False))
    hgnet_local_model_dir = getattr(args, "hgnet_local_model_dir", None)

    dual_stream_backbone = bool(getattr(args, "dual_stream_backbone", False))
    backbone_output_merge_raw = getattr(args, "backbone_output_merge", "add")
    backbone_output_merge: Any = "add" if backbone_output_merge_raw is None else backbone_output_merge_raw
    if isinstance(backbone_output_merge, str):
        backbone_output_merge = str(backbone_output_merge or "add")
        if backbone_output_merge.lower() == "avg":
            logging.warning("backbone_output_merge=avg 已弃用，将自动改为 wavg（可学习加权平均，初始等价于 avg）。")
            backbone_output_merge = "wavg"
        backbone_output_merge = str(backbone_output_merge or "add")
    elif isinstance(backbone_output_merge, Mapping):
        backbone_output_merge = dict(backbone_output_merge)
        if any(str(v).strip().lower() == "avg" for v in backbone_output_merge.values()):
            logging.warning(
                "backbone_output_merge 中包含 avg（已弃用），将自动改为 wavg（可学习加权平均，初始等价于 avg）。"
            )
            backbone_output_merge = {
                k: ("wavg" if str(v).strip().lower() == "avg" else v) for k, v in backbone_output_merge.items()
            }
    else:
        raise TypeError(f"Unsupported backbone_output_merge type: {type(backbone_output_merge)}")

    fusion_cfg_raw = getattr(args, "backbone_fusion", None)
    fusion_cfg: Dict[str, Any] = {}
    if isinstance(fusion_cfg_raw, dict):
        fusion_cfg = dict(fusion_cfg_raw)
    elif isinstance(fusion_cfg_raw, Mapping):
        fusion_cfg = dict(fusion_cfg_raw)
    elif fusion_cfg_raw is not None and hasattr(fusion_cfg_raw, "items"):
        fusion_cfg = {k: v for k, v in fusion_cfg_raw.items()}  # type: ignore[assignment]

    fusion_type = str(fusion_cfg.get("type", "coattention") or "coattention").lower()
    fusion_position_raw = fusion_cfg.get(
        "position",
        fusion_cfg.get("fusion_position", fusion_cfg.get("insertion", fusion_cfg.get("fusion_pos", None))),
    )
    # Fusion insertion point inside HGNetv2DualStream stage loop.
    # - pre_downsample: fuse at stage entry, then run stage.downsample + stage.blocks.
    # - pre_block (default): stage.downsample, then fuse, then stage.blocks.
    # - post_block: stage.downsample + stage.blocks, then fuse.
    fusion_position_norm = "pre_block"
    if fusion_position_raw is not None and str(fusion_position_raw).strip() != "":
        pos = str(fusion_position_raw).strip().lower()
        if pos in {"pre_downsample", "pre-ds", "preds", "pre_stage", "stage_start", "before_downsample"}:
            fusion_position_norm = "pre_downsample"
        elif pos in {"pre", "pre_block", "before", "before_block", "before_blocks"}:
            fusion_position_norm = "pre_block"
        elif pos in {
            "post",
            "post_block",
            "after",
            "after_block",
            "after_blocks",
            "post_stage",
            "after_stage",
            "stage_end",
        }:
            fusion_position_norm = "post_block"
        elif dual_stream_backbone:
            raise ValueError(
                "Unsupported backbone_fusion.position; expected pre_block/post_block (aliases: post_stage), "
                f"got {fusion_position_raw}"
            )
    fusion_d_model = int(fusion_cfg.get("d_model", 128) or 128)
    fusion_nhead = int(fusion_cfg.get("nhead", 8) or 8)
    fusion_dropout = float(fusion_cfg.get("dropout", 0.0) or 0.0)
    fusion_alpha_init = float(fusion_cfg.get("alpha_init", 0.0) or 0.0)
    fusion_n_layer = int(fusion_cfg.get("n_layer", 8) or 8)
    fusion_block_exp = int(fusion_cfg.get("block_exp", 4) or 4)
    fusion_vert_anchors = int(fusion_cfg.get("vert_anchors", 8) or 8)
    fusion_horz_anchors = int(fusion_cfg.get("horz_anchors", 8) or 8)
    fusion_num_stages = int(fusion_cfg.get("num_stages", fusion_cfg.get("mrt_num_stages", 2)) or 2)
    fusion_use_pos_encoding = bool(fusion_cfg.get("use_pos_encoding", fusion_cfg.get("pos_encoding", True)))
    fusion_writeback_merge = str(
        fusion_cfg.get("writeback_merge", fusion_cfg.get("residual_merge", fusion_cfg.get("merge", "add"))) or "add"
    )
    if fusion_writeback_merge.lower() == "avg":
        logging.warning("backbone_fusion.writeback_merge=avg 已弃用，将自动改为 wavg（可学习加权平均，初始等价于 avg）。")
        fusion_writeback_merge = "wavg"
    fusion_kv_stride = fusion_cfg.get("kv_stride", None)
    fusion_fuse_stage_idx = fusion_cfg.get("fuse_stage_idx", None)
    fusion_c2former_groups = fusion_cfg.get("groups", None)
    fusion_c2former_cca_stride = fusion_cfg.get("cca_stride", fusion_cfg.get("stride", None))
    fusion_c2former_offset_range_factor = fusion_cfg.get(
        "offset_range_factor",
        fusion_cfg.get("offset_range", None),
    )
    fusion_c2former_no_offset = fusion_cfg.get("no_offset", fusion_cfg.get("no_off", None))
    fusion_c2former_attn_drop = fusion_cfg.get(
        "attn_drop",
        fusion_cfg.get("attn_drop_rate", fusion_cfg.get("attention_dropout", None)),
    )
    fusion_c2former_proj_drop = fusion_cfg.get(
        "proj_drop",
        fusion_cfg.get("drop_rate", fusion_cfg.get("proj_drop_rate", None)),
    )
    fusion_c2former_offset_kernel_size = fusion_cfg.get(
        "offset_kernel_size",
        fusion_cfg.get("kernel_size", None),
    )
    fusion_c2former_padding_mode = fusion_cfg.get("padding_mode", None)
    fusion_c2former_align_corners = fusion_cfg.get("align_corners", None)
    fusion_c2former_offset_on = fusion_cfg.get("offset_on", fusion_cfg.get("align_on", "ms"))
    fusion_c2former_global_kv = bool(fusion_cfg.get("global_kv", False))
    fusion_c2former_global_vert_anchors = int(fusion_cfg.get("global_vert_anchors", 8) or 8)
    fusion_c2former_global_horz_anchors = int(fusion_cfg.get("global_horz_anchors", 8) or 8)
    fusion_c2former_use_pos_encoding = bool(fusion_cfg.get("c2former_use_pos_encoding", False))
    fusion_c2former_pos_temperature = float(fusion_cfg.get("c2former_pos_temperature", 10000.0) or 10000.0)

    post_fusion_cfg_raw = getattr(args, "backbone_post_fusion", None)
    post_fusion_cfg: Dict[str, Any] = {}
    if isinstance(post_fusion_cfg_raw, dict):
        post_fusion_cfg = dict(post_fusion_cfg_raw)
    elif isinstance(post_fusion_cfg_raw, Mapping):
        post_fusion_cfg = dict(post_fusion_cfg_raw)
    elif post_fusion_cfg_raw is not None and hasattr(post_fusion_cfg_raw, "items"):
        post_fusion_cfg = {k: v for k, v in post_fusion_cfg_raw.items()}  # type: ignore[assignment]

    post_fusion_type = str(post_fusion_cfg.get("type", "none") or "none").lower()
    post_fusion_d_model = int(post_fusion_cfg.get("d_model", fusion_d_model) or fusion_d_model)
    post_fusion_num_stages = int(post_fusion_cfg.get("num_stages", 2) or 2)
    post_fusion_fused_channels = int(post_fusion_cfg.get("fused_channels", post_fusion_cfg.get("proj_channels", 64)) or 64)
    post_fusion_norm = str(post_fusion_cfg.get("norm", "gn") or "gn")
    post_fusion_use_pos_encoding = bool(post_fusion_cfg.get("use_pos_encoding", True))
    # CFT/GPT-style post fusion (before HybridEncoder). Keep defaults aligned with backbone_fusion where possible.
    post_fusion_cft_nhead = int(post_fusion_cfg.get("nhead", fusion_nhead) or fusion_nhead)
    post_fusion_cft_n_layer = int(post_fusion_cfg.get("n_layer", fusion_n_layer) or fusion_n_layer)
    post_fusion_cft_block_exp = int(post_fusion_cfg.get("block_exp", fusion_block_exp) or fusion_block_exp)
    post_fusion_cft_vert_anchors = int(post_fusion_cfg.get("vert_anchors", fusion_vert_anchors) or fusion_vert_anchors)
    post_fusion_cft_horz_anchors = int(post_fusion_cfg.get("horz_anchors", fusion_horz_anchors) or fusion_horz_anchors)
    post_fusion_cft_dropout = float(post_fusion_cfg.get("dropout", fusion_dropout) or fusion_dropout)
    post_fusion_cft_alpha_init = float(post_fusion_cfg.get("alpha_init", fusion_alpha_init) or fusion_alpha_init)
    post_fusion_cft_writeback_merge = str(
        post_fusion_cfg.get(
            "writeback_merge",
            post_fusion_cfg.get("residual_merge", post_fusion_cfg.get("merge", fusion_writeback_merge)),
        )
        or fusion_writeback_merge
    )
    if post_fusion_cft_writeback_merge.lower() == "avg":
        logging.warning(
            "backbone_post_fusion.writeback_merge=avg 已弃用，将自动改为 wavg（可学习加权平均，初始等价于 avg）。"
        )
        post_fusion_cft_writeback_merge = "wavg"

    align_cfg_raw = getattr(args, "backbone_align", None)
    align_cfg: Dict[str, Any] = {}
    if isinstance(align_cfg_raw, bool):
        align_cfg = {"enabled": bool(align_cfg_raw)}
    elif isinstance(align_cfg_raw, dict):
        align_cfg = dict(align_cfg_raw)
    elif isinstance(align_cfg_raw, Mapping):
        align_cfg = dict(align_cfg_raw)
    elif align_cfg_raw is not None and hasattr(align_cfg_raw, "items"):
        align_cfg = {k: v for k, v in align_cfg_raw.items()}  # type: ignore[assignment]

    align_enabled = bool(align_cfg.get("enabled", align_cfg.get("enable", False)))
    align_stage_idx_raw = align_cfg.get("stage_idx", align_cfg.get("align_stage_idx", None))
    align_num_keypoints = int(align_cfg.get("num_keypoints", align_cfg.get("k", 5)) or 5)
    align_offset_scale = float(align_cfg.get("offset_scale", 6.0) or 6.0)
    align_offset_enabled = bool(align_cfg.get("offset_enabled", align_cfg.get("use_offset", True)))
    align_per_channel_offset = bool(
        align_cfg.get(
            "per_channel_offset",
            align_cfg.get("offset_per_channel", align_cfg.get("per_channel", False)),
        )
    )
    align_attention_norm = str(align_cfg.get("attention_norm", "sigmoid") or "sigmoid")
    align_position = str(
        align_cfg.get(
            "position",
            align_cfg.get("align_position", align_cfg.get("align_pos", "pre_block")),
        )
        or "pre_block"
    )
    align_padding_mode = str(align_cfg.get("padding_mode", "border") or "border")
    align_align_corners = bool(align_cfg.get("align_corners", True))
    align_loss_weight_raw = align_cfg.get("loss_weight", align_cfg.get("weight", align_cfg.get("align_loss_weight", None)))
    align_loss_weight: float | None = None
    if align_loss_weight_raw is not None and align_loss_weight_raw != "":
        if isinstance(align_loss_weight_raw, str) and align_loss_weight_raw.strip().lower() in {"none", "null"}:
            align_loss_weight = None
        else:
            align_loss_weight = float(align_loss_weight_raw)
    align_input_enabled = bool(
        align_cfg.get("input_enabled", align_cfg.get("input_align", align_cfg.get("align_input", False)))
    )
    align_loss_on = bool(align_enabled or align_input_enabled)
    if align_loss_on and align_loss_weight is None:
        # MRT-DETR default: treat alignment loss as an extra supervised term when alignment is enabled.
        align_loss_weight = 1.0
    align_loss_type = str(align_cfg.get("loss_type", align_cfg.get("loss", "cosine")) or "cosine")
    align_loss_downsample_raw = align_cfg.get("loss_downsample", align_cfg.get("downsample", None))
    align_loss_downsample: float | None = None
    if align_loss_downsample_raw is not None and align_loss_downsample_raw != "":
        if isinstance(align_loss_downsample_raw, str) and align_loss_downsample_raw.strip().lower() in {"none", "null"}:
            align_loss_downsample = None
        else:
            align_loss_downsample = float(align_loss_downsample_raw)
    align_nce_num_patches = int(align_cfg.get("nce_num_patches", align_cfg.get("nce_samples", 64)) or 64)
    align_nce_patch_size = int(align_cfg.get("nce_patch_size", align_cfg.get("nce_patch", 5)) or 5)
    align_nce_tau = float(align_cfg.get("nce_tau", align_cfg.get("nce_temperature", 0.2)) or 0.2)
    align_affine_enabled = bool(align_cfg.get("affine_enabled", align_cfg.get("affine", False)))
    align_affine_scale = float(align_cfg.get("affine_scale", 0.1) or 0.1)
    align_affine_init_identity = bool(
        align_cfg.get("affine_init_identity", align_cfg.get("affine_init", True))
    )
    align_affine_per_channel = bool(
        align_cfg.get("affine_per_channel", align_cfg.get("affine_per_chan", False))
    )
    align_affine_type = str(align_cfg.get("affine_type", align_cfg.get("affine_mode", "affine")) or "affine")
    align_ref_mode_raw = align_cfg.get("ref_mode", align_cfg.get("align_ref_mode", align_cfg.get("ref", None)))
    if align_ref_mode_raw is None or align_ref_mode_raw == "":
        align_ref_mode = "ms_mean" if (align_enabled or align_input_enabled) else "rgb"
    else:
        align_ref_mode = str(align_ref_mode_raw)
    align_ref_channel = int(align_cfg.get("ref_channel", align_cfg.get("align_ref_channel", 0)) or 0)

    align_input_proj = str(align_cfg.get("input_proj", align_cfg.get("input_projection", "rgb_to_ms")) or "rgb_to_ms")
    align_input_ref_mode_raw = align_cfg.get("input_ref_mode", align_cfg.get("input_ref", None))
    if align_input_ref_mode_raw is None or align_input_ref_mode_raw == "":
        align_input_ref_mode = align_ref_mode
    else:
        align_input_ref_mode = str(align_input_ref_mode_raw)
    align_input_ref_channel = int(
        align_cfg.get("input_ref_channel", align_cfg.get("input_ref_ch", align_ref_channel)) or align_ref_channel
    )
    align_input_affine_only = bool(align_cfg.get("input_affine_only", align_cfg.get("input_affine", False)))
    align_input_offset_enabled = bool(
        align_cfg.get("input_offset_enabled", align_cfg.get("input_use_offset", True))
    )
    if align_input_affine_only:
        align_input_offset_enabled = False
    align_input_affine_enabled = bool(
        align_cfg.get("input_affine_enabled", align_cfg.get("input_affine", align_affine_enabled))
    )
    if align_input_affine_only:
        align_input_affine_enabled = True
    align_input_num_keypoints = int(align_cfg.get("input_num_keypoints", align_num_keypoints) or align_num_keypoints)
    align_input_offset_scale = float(align_cfg.get("input_offset_scale", align_offset_scale) or align_offset_scale)
    align_input_per_channel_offset = bool(
        align_cfg.get("input_per_channel_offset", align_per_channel_offset)
    )
    align_input_attention_norm = str(align_cfg.get("input_attention_norm", align_attention_norm) or align_attention_norm)
    align_input_padding_mode = str(align_cfg.get("input_padding_mode", align_padding_mode) or align_padding_mode)
    align_input_align_corners = bool(align_cfg.get("input_align_corners", align_align_corners))
    align_input_loss_type = str(align_cfg.get("input_loss_type", align_loss_type) or align_loss_type)
    align_input_loss_downsample_raw = align_cfg.get("input_loss_downsample", align_cfg.get("input_downsample", None))
    align_input_loss_downsample: float | None = align_loss_downsample
    if align_input_loss_downsample_raw is not None and align_input_loss_downsample_raw != "":
        if isinstance(align_input_loss_downsample_raw, str) and align_input_loss_downsample_raw.strip().lower() in {
            "none",
            "null",
        }:
            align_input_loss_downsample = None
        else:
            align_input_loss_downsample = float(align_input_loss_downsample_raw)
    align_input_nce_num_patches = int(align_cfg.get("input_nce_num_patches", align_nce_num_patches) or align_nce_num_patches)
    align_input_nce_patch_size = int(align_cfg.get("input_nce_patch_size", align_nce_patch_size) or align_nce_patch_size)
    align_input_nce_tau = float(align_cfg.get("input_nce_tau", align_nce_tau) or align_nce_tau)
    align_input_affine_scale = float(align_cfg.get("input_affine_scale", align_affine_scale) or align_affine_scale)
    align_input_affine_init_identity = bool(
        align_cfg.get("input_affine_init_identity", align_affine_init_identity)
    )
    align_input_affine_per_channel = bool(
        align_cfg.get("input_affine_per_channel", align_affine_per_channel)
    )
    align_input_affine_type = str(
        align_cfg.get("input_affine_type", align_cfg.get("input_affine_mode", align_affine_type)) or align_affine_type
    )

    group_align_cfg_raw = getattr(args, "backbone_group_align", None)
    group_align_cfg: Dict[str, Any] | None = None
    if group_align_cfg_raw is not None:
        if isinstance(group_align_cfg_raw, dict):
            group_align_cfg = dict(group_align_cfg_raw)
        elif isinstance(group_align_cfg_raw, Mapping):
            group_align_cfg = dict(group_align_cfg_raw)
        elif hasattr(group_align_cfg_raw, "items"):
            group_align_cfg = {k: v for k, v in group_align_cfg_raw.items()}  # type: ignore[assignment]

    ms_band_sep_cfg_raw = getattr(args, "backbone_ms_band_sep", None)
    ms_band_sep_cfg: Dict[str, Any] | None = None
    if ms_band_sep_cfg_raw is not None:
        if isinstance(ms_band_sep_cfg_raw, dict):
            ms_band_sep_cfg = dict(ms_band_sep_cfg_raw)
        elif isinstance(ms_band_sep_cfg_raw, Mapping):
            ms_band_sep_cfg = dict(ms_band_sep_cfg_raw)
        elif hasattr(ms_band_sep_cfg_raw, "items"):
            ms_band_sep_cfg = {k: v for k, v in ms_band_sep_cfg_raw.items()}  # type: ignore[assignment]

    ms_residual_stem_cfg_raw = getattr(args, "backbone_ms_residual_stem", None)
    ms_residual_stem_cfg: Dict[str, Any] | None = None
    if ms_residual_stem_cfg_raw is not None:
        if isinstance(ms_residual_stem_cfg_raw, dict):
            ms_residual_stem_cfg = dict(ms_residual_stem_cfg_raw)
        elif isinstance(ms_residual_stem_cfg_raw, Mapping):
            ms_residual_stem_cfg = dict(ms_residual_stem_cfg_raw)
        elif hasattr(ms_residual_stem_cfg_raw, "items"):
            ms_residual_stem_cfg = {k: v for k, v in ms_residual_stem_cfg_raw.items()}  # type: ignore[assignment]

    eemsa_cfg_raw = getattr(args, "backbone_eemsa", None)
    eemsa_cfg: Dict[str, Any] | None = None
    if eemsa_cfg_raw is not None:
        if isinstance(eemsa_cfg_raw, dict):
            eemsa_cfg = dict(eemsa_cfg_raw)
        elif isinstance(eemsa_cfg_raw, Mapping):
            eemsa_cfg = dict(eemsa_cfg_raw)
        elif hasattr(eemsa_cfg_raw, "items"):
            eemsa_cfg = {k: v for k, v in eemsa_cfg_raw.items()}  # type: ignore[assignment]

    overrides: Dict[str, Any] = {
        "num_classes": num_classes,
        # 禁用 RT-DETRv4 自己的 MSCOCO remap（本工程由 BaseTrainer 统一处理 label->category_id 映射）
        "remap_mscoco_category": False,
        # 保持 eval_spatial_size 与当前训练输入一致，避免某些 buffer/anchor 逻辑使用错误尺寸
        "eval_spatial_size": [img_size, img_size],
        "HGNetv2": {
            "pretrained": hgnet_pretrained,
            "freeze_at": int(getattr(args, "hgnet_freeze_at", -1)),
            "freeze_norm": bool(getattr(args, "hgnet_freeze_norm", False)),
            # 支持堆叠输入：RGB(3)+MSI(n) -> 3+n 通道输入（如 10 通道）。
            # 注：仅当 backbone 为 HGNetv2 时生效，其它 backbone 会忽略该字段。
            "in_chs": input_channels,
        },
    }
    if hgnet_local_model_dir is not None:
        overrides["HGNetv2"]["local_model_dir"] = str(hgnet_local_model_dir)
    if eemsa_cfg is not None:
        overrides["HGNetv2"]["eemsa"] = eemsa_cfg
    if (not dual_stream_backbone) and ms_band_sep_cfg is not None:
        if rgb_channels == 0 and ms_channels > 0:
            overrides["HGNetv2"]["ms_band_sep"] = ms_band_sep_cfg
        elif bool(ms_band_sep_cfg.get("enabled", ms_band_sep_cfg.get("enable", False))):
            logging.warning(
                "检测到 backbone_ms_band_sep，但当前不是 MSI-only 单流输入(rgb_channels=%d, ms_channels=%d)，"
                "将忽略该配置。",
                rgb_channels,
                ms_channels,
            )
    if (not dual_stream_backbone) and ms_residual_stem_cfg is not None:
        if rgb_channels == 0 and ms_channels > 0:
            overrides["HGNetv2"]["ms_residual_stem"] = ms_residual_stem_cfg
        elif bool(ms_residual_stem_cfg.get("enabled", ms_residual_stem_cfg.get("enable", False))):
            logging.warning(
                "检测到 backbone_ms_residual_stem，但当前不是 MSI-only 单流输入(rgb_channels=%d, ms_channels=%d)，"
                "将忽略该配置。",
                rgb_channels,
                ms_channels,
            )
    if (not dual_stream_backbone) and group_align_cfg is not None:
        single_stream_group_align_on = bool(group_align_cfg.get("enabled", group_align_cfg.get("enable", False))) or bool(
            group_align_cfg.get("input_enabled", group_align_cfg.get("input_enable", False))
        )
        if rgb_channels == 0 and ms_channels > 0:
            overrides["HGNetv2"]["ms_group_align"] = group_align_cfg
        elif single_stream_group_align_on:
            logging.warning(
                "检测到 backbone_group_align，但当前不是 MSI-only 单流输入(rgb_channels=%d, ms_channels=%d)，"
                "将忽略该配置。",
                rgb_channels,
                ms_channels,
            )

    if dual_stream_backbone:
        # 需要从 vendored YAML 中读取 HGNetv2 的基础超参（name/return_idx/use_lab），避免硬编码各尺度配置。
        from .rtdetrv4.engine.core.yaml_utils import load_config  # type: ignore

        base_cfg = load_config(str(cfg_path))
        rtv4_cfg = base_cfg.get("RTv4", {}) or {}
        if rtv4_cfg.get("backbone") != "HGNetv2":
            raise ValueError(
                "当前 dual_stream_backbone 仅支持 HGNetv2 backbone；"
                f"但 RTv4.backbone={rtv4_cfg.get('backbone')} (cfg={cfg_path})"
            )

        hg_cfg = base_cfg.get("HGNetv2", {}) or {}
        hg_name = hg_cfg.get("name")
        if not hg_name:
            raise ValueError(f"未在 RT-DETRv4 配置中找到 HGNetv2.name (cfg={cfg_path})")

        return_idx = hg_cfg.get("return_idx", [1, 2, 3])
        use_lab = bool(hg_cfg.get("use_lab", False))
        freeze_stem_only = bool(hg_cfg.get("freeze_stem_only", True))
        local_model_dir = str(hg_cfg.get("local_model_dir", "weight/hgnetv2/"))
        if hgnet_local_model_dir is not None:
            local_model_dir = str(hgnet_local_model_dir)

        def _parse_fuse_stage_idx(value: Any, *, default: list[int]) -> list[int]:
            if value is None:
                return list(default)
            if isinstance(value, int):
                return [int(value)]
            if isinstance(value, str):
                parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
                value = parts
            # OmegaConf ListConfig / any other iterable sequence
            if not isinstance(value, (list, tuple)) and isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
                value = list(value)
            if isinstance(value, (list, tuple)):
                out: list[int] = []
                for item in value:
                    if isinstance(item, str):
                        key = item.strip().lower()
                        if key in {"c2", "p2"}:
                            out.append(0)
                        elif key in {"c3", "p3"}:
                            out.append(1)
                        elif key in {"c4", "p4"}:
                            out.append(2)
                        elif key in {"c5", "p5"}:
                            out.append(3)
                        else:
                            out.append(int(key))
                    else:
                        out.append(int(item))
                # 保持顺序去重
                dedup: list[int] = []
                for i in out:
                    if i not in dedup:
                        dedup.append(i)
                return dedup
            raise TypeError(f"Unsupported fuse_stage_idx type: {type(value)}")

        ms_channel_attn = str(getattr(args, "ms_channel_attn", "none") or "none")
        ms_channel_attn_position = str(getattr(args, "ms_channel_attn_position", "pre_fuse") or "pre_fuse")
        ms_channel_attn_stage_idx_raw = getattr(args, "ms_channel_attn_stage_idx", None)
        ms_channel_attn_stage_idx: list[int] | None = None
        if ms_channel_attn_stage_idx_raw is not None:
            if isinstance(ms_channel_attn_stage_idx_raw, str) and ms_channel_attn_stage_idx_raw.strip().lower() in {"none", "null"}:
                ms_channel_attn_stage_idx = None
            else:
                # Supports [1,2,3] / ["c3","c4"] / "c3,c4,c5".
                ms_channel_attn_stage_idx = _parse_fuse_stage_idx(ms_channel_attn_stage_idx_raw, default=[])

        ms_eca_kernel_size_raw = getattr(args, "ms_eca_kernel_size", None)
        ms_eca_kernel_size: int | None = None
        if ms_eca_kernel_size_raw is not None and ms_eca_kernel_size_raw != "":
            if isinstance(ms_eca_kernel_size_raw, str) and ms_eca_kernel_size_raw.strip().lower() in {"none", "null"}:
                ms_eca_kernel_size = None
            else:
                ms_eca_kernel_size = int(ms_eca_kernel_size_raw)

        ms_se_reduction = int(getattr(args, "ms_se_reduction", 16) or 16)

        align_stage_idx: list[int] | None = None
        if align_stage_idx_raw is not None:
            if isinstance(align_stage_idx_raw, str) and align_stage_idx_raw.strip().lower() in {"none", "null"}:
                align_stage_idx = None
            else:
                # Supports [1,2,3] / ["c3","c4"] / "c3,c4,c5".
                align_stage_idx = _parse_fuse_stage_idx(align_stage_idx_raw, default=[])

        fuse_stage_idx = _parse_fuse_stage_idx(fusion_fuse_stage_idx, default=list(return_idx))
        fusion_type_norm = str(fusion_type).strip().lower()
        if fusion_type_norm in {"none", "off", "disable", "disabled"}:
            fusion_type_norm = "none"
            fuse_stage_idx = []
        elif fusion_type_norm in {"coattention", "co-attention", "co_attention", "coattn"}:
            fusion_type_norm = "coattention"
        elif fusion_type_norm in {"c2former", "c2f"}:
            fusion_type_norm = "c2former"
        elif fusion_type_norm in {"gpt", "cft", "fusion_transformer", "cft_gpt"}:
            fusion_type_norm = "gpt"
        elif fusion_type_norm in {"mrt", "cross_spectrum", "cross-spectrum", "crossspectrum"}:
            fusion_type_norm = "mrt"
        else:
            raise ValueError(f"Unsupported backbone_fusion.type={fusion_type} for dual_stream_backbone.")

        overrides["RTv4"] = {"backbone": "HGNetv2DualStream"}
        overrides["HGNetv2DualStream"] = {
            "name": hg_name,
            "fusion_type": fusion_type_norm,
            "fusion_position": fusion_position_norm,
            "output_fusion_type": post_fusion_type,
            "rgb_in_chs": rgb_channels,
            "ms_in_chs": ms_channels,
            "use_lab": use_lab,
            "return_idx": list(return_idx),
            "fuse_stage_idx": fuse_stage_idx,
            "output_merge": backbone_output_merge,
            "ms_channel_attn": ms_channel_attn,
            "ms_channel_attn_position": ms_channel_attn_position,
            "ms_channel_attn_stage_idx": ms_channel_attn_stage_idx,
            "ms_eca_kernel_size": ms_eca_kernel_size,
            "ms_se_reduction": ms_se_reduction,
            "align_enabled": align_enabled,
            "align_stage_idx": align_stage_idx,
            "align_num_keypoints": align_num_keypoints,
            "align_offset_scale": align_offset_scale,
            "align_offset_enabled": align_offset_enabled,
            "align_per_channel_offset": align_per_channel_offset,
            "align_attention_norm": align_attention_norm,
            "align_position": align_position,
            "align_padding_mode": align_padding_mode,
            "align_align_corners": align_align_corners,
            "align_loss_type": align_loss_type,
            "align_loss_downsample": align_loss_downsample,
            "align_nce_num_patches": align_nce_num_patches,
            "align_nce_patch_size": align_nce_patch_size,
            "align_nce_tau": align_nce_tau,
            "align_affine_enabled": align_affine_enabled,
            "align_affine_scale": align_affine_scale,
            "align_affine_init_identity": align_affine_init_identity,
            "align_affine_per_channel": align_affine_per_channel,
            "align_affine_type": align_affine_type,
            "align_ref_mode": align_ref_mode,
            "align_ref_channel": align_ref_channel,
            "align_input_enabled": align_input_enabled,
            "align_input_proj": align_input_proj,
            "align_input_ref_mode": align_input_ref_mode,
            "align_input_ref_channel": align_input_ref_channel,
            "align_input_num_keypoints": align_input_num_keypoints,
            "align_input_offset_enabled": align_input_offset_enabled,
            "align_input_offset_scale": align_input_offset_scale,
            "align_input_per_channel_offset": align_input_per_channel_offset,
            "align_input_attention_norm": align_input_attention_norm,
            "align_input_padding_mode": align_input_padding_mode,
            "align_input_align_corners": align_input_align_corners,
            "align_input_loss_type": align_input_loss_type,
            "align_input_loss_downsample": align_input_loss_downsample,
            "align_input_nce_num_patches": align_input_nce_num_patches,
            "align_input_nce_patch_size": align_input_nce_patch_size,
            "align_input_nce_tau": align_input_nce_tau,
            "align_input_affine_enabled": align_input_affine_enabled,
            "align_input_affine_scale": align_input_affine_scale,
            "align_input_affine_init_identity": align_input_affine_init_identity,
            "align_input_affine_per_channel": align_input_affine_per_channel,
            "align_input_affine_type": align_input_affine_type,
            "ms_band_sep": ms_band_sep_cfg,
            "ms_residual_stem": ms_residual_stem_cfg,
            "ms_group_align": group_align_cfg,
            "eemsa": eemsa_cfg,
            "fusion_d_model": fusion_d_model,
            "fusion_nhead": fusion_nhead,
            "fusion_block_exp": fusion_block_exp,
            "fusion_n_layer": fusion_n_layer,
            "fusion_vert_anchors": fusion_vert_anchors,
            "fusion_horz_anchors": fusion_horz_anchors,
            "fusion_dropout": fusion_dropout,
            "fusion_alpha_init": fusion_alpha_init,
            "fusion_kv_stride": fusion_kv_stride,
            "fusion_num_stages": fusion_num_stages,
            "fusion_use_pos_encoding": fusion_use_pos_encoding,
            "fusion_c2former_groups": fusion_c2former_groups,
            "fusion_c2former_cca_stride": fusion_c2former_cca_stride,
            "fusion_c2former_offset_range_factor": fusion_c2former_offset_range_factor,
            "fusion_c2former_no_offset": fusion_c2former_no_offset,
            "fusion_c2former_attn_drop": fusion_c2former_attn_drop,
            "fusion_c2former_proj_drop": fusion_c2former_proj_drop,
            "fusion_c2former_offset_kernel_size": fusion_c2former_offset_kernel_size,
            "fusion_c2former_padding_mode": fusion_c2former_padding_mode,
            "fusion_c2former_align_corners": fusion_c2former_align_corners,
            "fusion_c2former_offset_on": fusion_c2former_offset_on,
            "fusion_c2former_global_kv": fusion_c2former_global_kv,
            "fusion_c2former_global_vert_anchors": fusion_c2former_global_vert_anchors,
            "fusion_c2former_global_horz_anchors": fusion_c2former_global_horz_anchors,
            "fusion_c2former_use_pos_encoding": fusion_c2former_use_pos_encoding,
            "fusion_c2former_pos_temperature": fusion_c2former_pos_temperature,
            "fusion_writeback_merge": fusion_writeback_merge,
            "output_mrt_d_model": post_fusion_d_model,
            "output_mrt_num_stages": post_fusion_num_stages,
            "output_mrt_fused_channels": post_fusion_fused_channels,
            "output_mrt_norm": post_fusion_norm,
            "output_mrt_use_pos_encoding": post_fusion_use_pos_encoding,
            "output_cft_d_model": post_fusion_d_model,
            "output_cft_nhead": post_fusion_cft_nhead,
            "output_cft_block_exp": post_fusion_cft_block_exp,
            "output_cft_n_layer": post_fusion_cft_n_layer,
            "output_cft_vert_anchors": post_fusion_cft_vert_anchors,
            "output_cft_horz_anchors": post_fusion_cft_horz_anchors,
            "output_cft_dropout": post_fusion_cft_dropout,
            "output_cft_alpha_init": post_fusion_cft_alpha_init,
            "output_cft_writeback_merge": post_fusion_cft_writeback_merge,
            "freeze_stem_only": freeze_stem_only,
            "freeze_at": int(getattr(args, "hgnet_freeze_at", -1)),
            "freeze_norm": bool(getattr(args, "hgnet_freeze_norm", False)),
            "pretrained": hgnet_pretrained,
            "local_model_dir": local_model_dir,
        }

        # Enable MRT-DETR style alignment loss (computed in backbone and picked up by RTv4Criterion).
        if align_loss_on and align_loss_weight is not None:
            crit_cfg = overrides.setdefault("RTv4Criterion", {})
            weight_dict = crit_cfg.setdefault("weight_dict", {})
            if isinstance(weight_dict, Mapping):
                # Merge new weight into existing dict.
                weight_dict = dict(weight_dict)
            weight_dict["loss_deform_align"] = float(align_loss_weight)
            crit_cfg["weight_dict"] = weight_dict
            overrides["RTv4Criterion"] = crit_cfg

    if disable_distill:
        # 关闭 teacher 蒸馏相关逻辑：
        # - HybridEncoder 不再输出 distill_student_output；
        # - Criterion losses 去掉 distill，避免 CPU 训练时 loss_distill 被硬编码到 cuda。
        overrides["HybridEncoder"] = {"distill_teacher_dim": 0}
        overrides["RTv4Criterion"] = {"losses": ["mal", "boxes", "local"]}

    return overrides


def _abspath_under_repo(path_like: str | Path) -> str:
    """
    将路径规范化为绝对路径：
    - 绝对路径保持不变；
    - 相对路径视为相对仓库根目录（避免依赖 cwd）。
    """
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = _repo_root() / path
    return str(path)


def build_model_and_processors(
    args: Namespace,
) -> Tuple[BaseDetector, nn.Module, nn.Module]:
    """
    构建 RT-DETRv4 模型 + 损失 + 后处理（供 RtmsfDetrTrainer 使用）。

    返回：
    - model: BaseDetector 兼容封装（RTDETRv4Detector）
    - criterion: RTv4Criterion（vendored `engine`）
    - postprocessor: PostProcessor（vendored `engine`）
    """
    # 触发 RT-DETRv4 vendored `engine` 的注册逻辑（按需导入，避免全量 import 带来的可选依赖）。
    from .rtdetrv4.engine import core as _core  # noqa: F401
    from .rtdetrv4.engine import optim as _optim  # noqa: F401
    from .rtdetrv4.engine import rtv4 as _rtv4  # noqa: F401
    from .rtdetrv4.engine.backbone import hgnetv2 as _hgnetv2  # noqa: F401
    if bool(getattr(args, "dual_stream_backbone", False)):
        from .rtdetrv4.engine.backbone import hgnetv2_dualstream as _hgnetv2_dualstream  # noqa: F401
    from .rtdetrv4.engine.core import YAMLConfig  # type: ignore

    cfg_path = _resolve_rtdetrv4_config(getattr(args, "rtdetrv4_config", None))
    if not cfg_path.is_file():
        raise FileNotFoundError(f"未找到 RT-DETRv4 配置文件：{cfg_path}")

    disable_distill = bool(getattr(args, "disable_distill", True))
    overrides = _build_rtdetrv4_overrides(args, cfg_path=cfg_path)
    logging.info("加载 RT-DETRv4 配置: %s", cfg_path)
    yaml_cfg = YAMLConfig(str(cfg_path), **overrides)
    if disable_distill and "teacher_model" in getattr(yaml_cfg, "yaml_cfg", {}):
        logging.info(
            "已设置 disable_distill=true：将忽略 RT-DETRv4 配置中的 teacher_model / DINOv3 权重加载。"
        )

    teacher_model = None
    if not disable_distill:
        teacher_cfg = getattr(yaml_cfg, "yaml_cfg", {}).get("teacher_model")
        if not isinstance(teacher_cfg, dict):
            raise ValueError(
                "已启用蒸馏（disable_distill=false），但 RT-DETRv4 配置中缺少 `teacher_model` 字段。"
            )

        # 允许从上层（Hydra）覆盖 teacher 的本地资产路径，避免改 vendored YAML。
        override_repo_path = getattr(args, "teacher_repo_path", None)
        override_weights_path = getattr(args, "teacher_weights_path", None)
        if override_repo_path:
            teacher_cfg["dinov3_repo_path"] = str(override_repo_path)
        if override_weights_path:
            teacher_cfg["dinov3_weights_path"] = str(override_weights_path)

        # 规范化 teacher 的本地路径，避免依赖运行时 cwd。
        repo_path = teacher_cfg.get("dinov3_repo_path")
        weights_path = teacher_cfg.get("dinov3_weights_path")
        if repo_path is not None:
            teacher_cfg["dinov3_repo_path"] = _abspath_under_repo(str(repo_path))
        if weights_path is not None:
            teacher_cfg["dinov3_weights_path"] = _abspath_under_repo(str(weights_path))

        repo_dir = Path(teacher_cfg.get("dinov3_repo_path", "")).expanduser()
        weights_file = Path(teacher_cfg.get("dinov3_weights_path", "")).expanduser()
        if not repo_dir.is_dir():
            raise FileNotFoundError(
                "DINOv3 teacher repo 目录不存在，无法启用蒸馏："
                f" dinov3_repo_path={repo_dir}. "
                "请准备本地 dinov3 仓库目录，并通过 RTv4 YAML 或后续 Hydra override 指定路径。"
            )
        if not weights_file.is_file():
            raise FileNotFoundError(
                "DINOv3 teacher 权重文件不存在，无法启用蒸馏："
                f" dinov3_weights_path={weights_file}. "
                "请准备本地权重文件，并通过 RTv4 YAML 或后续 Hydra override 指定路径。"
            )

        logging.info(
            "启用 RT-DETRv4 蒸馏：teacher_repo=%s teacher_weights=%s",
            repo_dir,
            weights_file,
        )
        teacher_model = yaml_cfg.teacher_model
        # teacher 不参与训练，保持冻结。
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        device = getattr(args, "device", None)
        if device:
            teacher_model = teacher_model.to(device)

    raw_model = yaml_cfg.model
    criterion = yaml_cfg.criterion
    postprocessor = yaml_cfg.postprocessor

    # Optional instance segmentation head (mask head).
    # Note: dataset side is controlled by args.segmentation_head (BaseTrainer._ensure_data_args),
    # so we keep model behavior driven by the same flag to avoid mismatched targets.
    if bool(getattr(args, "segmentation_head", False)):
        from engines.models.msifdetr.common.segmentation_head import SegmentationHead

        decoder = getattr(raw_model, "decoder", None)
        if decoder is None:
            raise AttributeError("RT-DETRv4 model has no decoder; cannot attach segmentation head.")

        hidden_dim = getattr(decoder, "hidden_dim", None)
        num_layers = getattr(decoder, "num_layers", None)
        if hidden_dim is None or num_layers is None:
            raise AttributeError(
                "RT-DETRv4 decoder missing hidden_dim/num_layers; cannot build SegmentationHead. "
                f"decoder={type(decoder)}"
            )

        mask_downsample_ratio = int(getattr(args, "mask_downsample_ratio", 8))
        mask_feature_level = int(getattr(args, "mask_feature_level", 0))
        mask_aux_loss = bool(getattr(args, "mask_aux_loss", False))

        decoder.segmentation_head = SegmentationHead(int(hidden_dim), int(num_layers), downsample_ratio=mask_downsample_ratio)
        decoder.mask_feature_level = mask_feature_level
        decoder.mask_aux_loss = mask_aux_loss

        # Wrap detection criterion with mask losses.
        criterion = RtmsfDetrCriterionWithMasks(
            criterion,
            mask_point_sample_ratio=int(getattr(args, "mask_point_sample_ratio", 16)),
            mask_ce_loss_coef=float(getattr(args, "mask_ce_loss_coef", 5.0)),
            mask_dice_loss_coef=float(getattr(args, "mask_dice_loss_coef", 5.0)),
            mask_aux_loss=mask_aux_loss,
        )

    rgb_channels = int(getattr(args, "rgb_input_channels", 3))
    ms_channels = int(getattr(args, "ms_input_channels", 0))
    input_channels = int(getattr(args, "input_channels", rgb_channels + ms_channels))

    detector = RTDETRv4Detector(
        raw_model,
        teacher_model=teacher_model,
        expected_input_channels=input_channels,
        rgb_channels=rgb_channels,
        input_denormalize=bool(getattr(args, "input_denormalize", True)),
        rgb_mean=tuple(getattr(args, "rgb_mean", (0.485, 0.456, 0.406))),
        rgb_std=tuple(getattr(args, "rgb_std", (0.229, 0.224, 0.225))),
        clamp_after_denormalize=bool(getattr(args, "clamp_after_denormalize", True)),
    )

    return detector, criterion, postprocessor


def build_model(args: Namespace) -> BaseDetector:
    model, _, _ = build_model_and_processors(args)
    return model


def build_criterion_and_postprocessors(args: Namespace):
    _, criterion, postprocessor = build_model_and_processors(args)
    return criterion, postprocessor


__all__ = [
    "build_model",
    "build_criterion_and_postprocessors",
    "build_model_and_processors",
]
