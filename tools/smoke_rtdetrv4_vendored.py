from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path
import sys

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test for vendored RT-DETRv4 under engines/models/rtmsfdetr/rtdetrv4/ (engine/ + configs/)."
    )
    parser.add_argument(
        "--rtdetrv4-config",
        type=str,
        default="engines/models/rtmsfdetr/rtdetrv4/configs/rtv4/rtv4_hgnetv2_s_coco.yml",
        help="Path to RT-DETRv4 YAML config (vendored).",
    )
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--rgb-ch", type=int, default=3)
    parser.add_argument("--ms-ch", type=int, default=0)
    parser.add_argument("--dual-stream-backbone", action="store_true")
    parser.add_argument(
        "--backbone-output-merge",
        type=str,
        default="avg",
        choices=["avg", "add", "concat1x1"],
        help="How to merge rgb/ms outputs back to a single feature pyramid.",
    )
    parser.add_argument(
        "--segmentation-head",
        action="store_true",
        help="Enable instance segmentation head (predict pred_masks + postprocess masks).",
    )
    parser.add_argument("--mask-downsample-ratio", type=int, default=8)
    parser.add_argument("--mask-feature-level", type=int, default=0)
    args = parser.parse_args()

    repo_root = _repo_root()
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from engines.models.rtmsfdetr.rtdetrv4 import engine

    engine_file = Path(getattr(engine, "__file__", "")).resolve()
    if "engines" not in engine_file.parts or "rtmsfdetr" not in engine_file.parts:
        raise RuntimeError(f"RT-DETRv4 vendored engine 路径异常：{engine_file}")

    cfg_path = Path(args.rtdetrv4_config)
    if not cfg_path.is_absolute():
        cfg_path = repo_root / cfg_path
    if not cfg_path.is_file():
        raise FileNotFoundError(f"未找到 RT-DETRv4 YAML 配置：{cfg_path}")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    from engines.models.rtmsfdetr.builder import build_model_and_processors
    from utils.misc import NestedTensor

    build_args = Namespace(
        num_classes=int(args.num_classes),
        img_size=int(args.img_size),
        disable_distill=True,
        hgnet_pretrained=False,
        hgnet_local_model_dir=None,
        hgnet_freeze_at=-1,
        hgnet_freeze_norm=False,
        input_denormalize=False,
        clamp_after_denormalize=True,
        rgb_mean=(0.485, 0.456, 0.406),
        rgb_std=(0.229, 0.224, 0.225),
        rtdetrv4_config=str(cfg_path),
        rgb_input_channels=int(args.rgb_ch),
        ms_input_channels=int(args.ms_ch),
        input_channels=int(args.rgb_ch) + int(args.ms_ch),
        dual_stream_backbone=bool(args.dual_stream_backbone),
        backbone_output_merge=str(args.backbone_output_merge),
        backbone_fusion={
            "type": "coattention",
            "d_model": 128,
            "nhead": 8,
            "dropout": 0.0,
            "alpha_init": 0.0,
            "kv_stride": {"c3": 8, "c4": 4, "c5": 2},
        },
        segmentation_head=bool(args.segmentation_head),
        mask_downsample_ratio=int(args.mask_downsample_ratio),
        mask_feature_level=int(args.mask_feature_level),
        mask_aux_loss=False,
        # loss knobs (only used when segmentation_head=true)
        mask_point_sample_ratio=16,
        mask_ce_loss_coef=1.0,
        mask_dice_loss_coef=1.0,
    )

    model, _, postprocessor = build_model_and_processors(build_args)
    model.to(device)
    model.eval()

    in_ch = int(args.rgb_ch) + int(args.ms_ch)
    if in_ch <= 0:
        raise ValueError(f"Invalid rgb/ms channels: rgb={args.rgb_ch} ms={args.ms_ch}")
    x = torch.rand(1, in_ch, args.img_size, args.img_size, device=device)
    mask = torch.zeros((1, args.img_size, args.img_size), dtype=torch.bool, device=device)
    samples = NestedTensor(x, mask)

    with torch.no_grad():
        outputs = model(samples)

    logits = outputs.get("pred_logits")
    boxes = outputs.get("pred_boxes")
    if logits is None or boxes is None:
        raise RuntimeError(f"输出缺少 pred_logits/pred_boxes，keys={sorted(outputs.keys())}")

    if args.segmentation_head:
        masks = outputs.get("pred_masks")
        if masks is None:
            raise RuntimeError(f"已启用 segmentation_head，但输出缺少 pred_masks，keys={sorted(outputs.keys())}")
        sizes = torch.tensor([[args.img_size, args.img_size]], dtype=torch.int64, device=device)
        results = postprocessor(outputs, sizes)
        if not results or "masks" not in results[0]:
            raise RuntimeError("postprocessor 未输出 masks（segm 后处理失败）")
        print(f"[OK] pred_masks={tuple(masks.shape)} post_masks={tuple(results[0]['masks'].shape)}")

    print(f"[OK] engine={engine_file}")
    print(f"[OK] config={cfg_path}")
    print(
        f"[OK] dual_stream_backbone={bool(args.dual_stream_backbone)} rgb_ch={int(args.rgb_ch)} ms_ch={int(args.ms_ch)} "
        f"output_merge={str(args.backbone_output_merge)}"
    )
    print(f"[OK] pred_logits={tuple(logits.shape)} pred_boxes={tuple(boxes.shape)} device={device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
