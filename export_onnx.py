from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Mapping, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from engines.core.parse_config import load_config
from engines.trainer.base_trainer import (
    _filter_compatible_state_dict,
    _register_yolo_pickle_alias,
    _remap_ultralytics_state_dict,
)


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    model_state = checkpoint.get("model", checkpoint)
    if hasattr(model_state, "state_dict"):
        model_state = model_state.state_dict()
    if not isinstance(model_state, Mapping):
        raise TypeError(f"无法从 checkpoint 解析 state_dict，得到类型={type(model_state)}")
    model_state = _remap_ultralytics_state_dict(model_state)
    return model_state


def _load_checkpoint(path: Path, *, weights_only: bool) -> Any:
    _register_yolo_pickle_alias()
    return torch.load(path, map_location="cpu", weights_only=weights_only)


def _infer_hw(cfg: Any, *, height: int | None, width: int | None) -> Tuple[int, int]:
    if height is not None and width is not None:
        return int(height), int(width)

    img_size = None
    try:
        img_size = cfg.get("model", {}).get("img_size", None)
    except Exception:
        img_size = getattr(getattr(cfg, "model", None), "img_size", None)

    if isinstance(img_size, (list, tuple)) and img_size:
        img_size = img_size[0]
    if img_size is None:
        img_size = 640

    h = int(height if height is not None else img_size)
    w = int(width if width is not None else img_size)
    return h, w


def _align_hw_for_backbone(cfg: Any, h: int, w: int) -> Tuple[int, int]:
    """
    Windowed DINOv2 / ViT backbone 通常要求输入 H/W 能被 (patch_size * num_windows) 整除。
    训练/推理脚本里会做对齐；导出 ONNX 时也保持一致，避免触发 assert。
    """
    patch_size = int(getattr(getattr(cfg, "model", None), "patch_size", 16))
    num_windows = int(getattr(getattr(cfg, "model", None), "num_windows", 4))
    divisor = max(1, patch_size * num_windows)
    ah = int((int(h) + divisor - 1) // divisor * divisor)
    aw = int((int(w) + divisor - 1) // divisor * divisor)
    if (ah, aw) != (int(h), int(w)):
        logging.warning(
            "输入分辨率需要能被 patch_size*num_windows=%d 整除；已自动将 (H,W)=%s 对齐到 %s。",
            divisor,
            (int(h), int(w)),
            (ah, aw),
        )
    return ah, aw


def _load_config_any(
    config: str | Path, *, config_dir: str | Path, overrides: list[str] | None
) -> Any:
    """
    支持两种输入：
    1) configs/ 目录内的 Hydra 配置（走 load_config + defaults 组合）
    2) outputs/**/config.yaml 这类“已保存的完整配置”（直接 OmegaConf.load）
    """
    config_path = _as_path(config).expanduser()
    if config_path.is_file():
        cfg = OmegaConf.load(str(config_path))
        OmegaConf.set_struct(cfg, False)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
            OmegaConf.set_struct(cfg, False)
        return cfg
    return load_config(config, config_dir=_as_path(config_dir), overrides=overrides)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export checkpoint to ONNX (RFDETR/MSIFDETR).")
    parser.add_argument(
        "--config",
        required=True,
        help="训练时使用的配置：既支持 configs/ 下的 Hydra 配置，也支持 outputs/**/config.yaml。",
    )
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Hydra 搜索配置的根目录（默认 configs）。",
    )
    parser.add_argument(
        "--opts",
        nargs="*",
        default=None,
        help="可选覆盖项：KEY=VALUE（会在 load_config 时按顺序生效）。",
    )
    parser.add_argument("--checkpoint", required=True, help="训练保存的 checkpoint.pth / checkpoint_best.pth。")
    parser.add_argument("--output", default=None, help="导出的 .onnx 路径（默认与 checkpoint 同名）。")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="导出时使用的设备。")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset 版本（默认 17）。")
    parser.add_argument("--batch", type=int, default=1, help="dummy input batch size（默认 1）。")
    parser.add_argument("--height", type=int, default=640, help="dummy input 高（会自动按 patch_size*num_windows 对齐）。")
    parser.add_argument("--width", type=int, default=640, help="dummy input 宽（会自动按 patch_size*num_windows 对齐）。")
    parser.add_argument("--channels", type=int, default=None, help="dummy input 通道数（默认从配置推导）。")
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="给 batch 维打上动态轴标记（H/W 默认仍为静态）。",
    )
    parser.add_argument(
        "--dynamo",
        action="store_true",
        help="使用 torch.export/dynamo 路径导出（PyTorch 2.9 默认开启；本项目建议保持关闭以提高兼容性）。",
    )
    parser.add_argument(
        "--external-data",
        action="store_true",
        help="将参数导出为外部权重文件（会写到 output 同目录）。默认关闭以生成单个 .onnx 文件。",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="若 checkpoint 中包含 ema 权重，则优先导出 ema（否则回退 model）。",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="以 weights_only=True 方式加载 checkpoint（更安全，但可能无法读取包含非张量对象的 ckpt）。",
    )
    parser.add_argument("--verbose", action="store_true", help="打开 torch.onnx.export verbose。")
    args = parser.parse_args()

    config_root = _as_path(args.config_dir).resolve()
    cfg = _load_config_any(args.config, config_dir=config_root, overrides=list(args.opts or []))
    with open_dict(cfg):
        cfg.mode = "test"
        if "runtime" not in cfg:
            cfg.runtime = {}
        cfg.runtime.device = str(args.device)

    ckpt_path = _as_path(args.checkpoint).expanduser()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    out_path = _as_path(args.output).expanduser() if args.output else ckpt_path.with_suffix(".onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trainer_cfg = getattr(cfg, "trainer", None)
    if trainer_cfg is None:
        raise KeyError("配置中缺少 trainer 节点，无法实例化训练器。")

    trainer = instantiate(trainer_cfg, cfg)
    model = trainer.build_model()

    checkpoint = _load_checkpoint(ckpt_path, weights_only=bool(args.weights_only))
    if args.use_ema and isinstance(checkpoint, Mapping) and checkpoint.get("ema") is not None:
        model_state = checkpoint["ema"]
    else:
        model_state = _extract_state_dict(checkpoint)

    compatible_state = _filter_compatible_state_dict(model, model_state)
    skipped = len(model_state) - len(compatible_state)
    if skipped:
        logging.warning("checkpoint 有 %d 个参数形状不匹配，已跳过加载。", skipped)
    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    if missing or unexpected:
        logging.warning("加载 state_dict：missing=%d, unexpected=%d", len(missing), len(unexpected))

    model.eval()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model.to(device)

    if not hasattr(model, "export") or not callable(getattr(model, "export")):
        raise RuntimeError(
            "当前模型未实现 export()/forward_export，脚本暂只支持 RFDETR/MSIFDETR 系列。"
        )
    model.export()

    data_args = trainer._ensure_data_args()
    dual_stream = bool(getattr(data_args, "dual_stream_output", False))

    rgb_channels = int(getattr(data_args, "rgb_input_channels", 3))
    ms_channels = int(getattr(data_args, "ms_input_channels", 0))
    use_ms = bool(getattr(data_args, "use_msi_input", False)) and ms_channels > 0

    h, w = _infer_hw(cfg, height=args.height, width=args.width)
    h, w = _align_hw_for_backbone(cfg, h, w)
    batch = int(args.batch)

    input_names = ["images"]
    model_inputs: Any

    if dual_stream and use_ms:
        dummy_rgb = torch.zeros((batch, rgb_channels, int(h), int(w)), dtype=torch.float32, device=device)
        dummy_ms = torch.zeros((batch, ms_channels, int(h), int(w)), dtype=torch.float32, device=device)
        model_inputs = (dummy_rgb, dummy_ms)
        input_names = ["rgb", "ms"]
    else:
        inferred_channels = args.channels
        if inferred_channels is None:
            inferred_channels = getattr(data_args, "input_channels", None)
        if inferred_channels is None:
            # 典型场景：dual_stream_output=True 但未启用 MSI，此时仍可按 RGB-only 导出
            if dual_stream and not use_ms:
                inferred_channels = rgb_channels
            else:
                raise ValueError(
                    "无法从配置推导 input_channels；如果你在用 dual_stream_output=True，"
                    "请确保 use_msi_input=True 且 ms_input_channels>0，或手动传 --channels。"
                )
        dummy = torch.zeros((batch, int(inferred_channels), int(h), int(w)), dtype=torch.float32, device=device)
        model_inputs = dummy

    has_masks = getattr(model, "segmentation_head", None) is not None
    output_names = ["pred_boxes", "pred_logits"] + (["pred_masks"] if has_masks else [])

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {name: {0: "batch"} for name in input_names}
        for name in output_names:
            dynamic_axes[name] = {0: "batch"}

    torch.onnx.export(
        model,
        model_inputs,
        str(out_path),
        opset_version=int(args.opset),
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=bool(args.verbose),
        dynamo=bool(args.dynamo),
        external_data=bool(args.external_data),
        artifacts_dir=str(out_path.parent),
    )

    print(f"ONNX 导出完成: {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torch.set_grad_enabled(False)
    main()
