import datetime
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping
from types import SimpleNamespace, ModuleType

import torch
import torch.nn.functional as F
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
)

import numpy as np

from datasets import get_coco_api_from_dataset
from datasets.coco import compute_multi_scale_scales
from datasets.coco_eval import CocoEvaluator
from engines.core.metrics.metrics import CompositeCriterion, _LossItem, build_loss
from engines.core.parse_config import get_config
from utils.misc import (
    MetricLogger,
    NestedTensor,
    SmoothedValue,
    get_world_size,
    is_main_process,
    reduce_dict,
)
from engines.models.mscft.utils.torch_utils import ModelEMA
from utils.metrics import (
    MetricsPlotSink,
    save_detection_combined_curves,
    save_detection_metric_charts,
    save_detection_per_class_curves_from_coco_eval,
    save_detection_confusion_matrix,
    save_coco_dataset_distribution_charts,
    save_yolo_labels_correlogram,
    save_yolo_labels_extra_charts,
    save_yolo_labels_per_class_heatmaps,
    save_yolo_labels_cooccurrence_heatmap,
    save_inference_profile_chart,
    save_detection_visual_samples,
)
from utils.csv_metrics import CsvMetricsSink
from utils.flops import try_compute_gflops
from utils.wandb import WandbMetricLogger

try:
    # 优先使用 torch.amp.* 新接口，旧版本退回 torch.cuda.amp。
    from torch.amp.grad_scaler import GradScaler  # type: ignore
    from torch.amp.autocast_mode import autocast  # type: ignore
    _AMP_DEVICE_KW = {"device_type": "cuda"} if torch.cuda.is_available() else {"device_type": "cpu"}
except Exception:  # pragma: no cover
    from torch.cuda.amp import GradScaler, autocast  # type: ignore
    _AMP_DEVICE_KW = {"device_type": "cuda"} if torch.cuda.is_available() else {}


def _make_grad_scaler(enabled: bool) -> GradScaler:
    """
    创建 GradScaler，优先传入 device 以满足 torch.amp 推荐用法，旧版本不支持时回退。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return GradScaler(enabled=enabled, device=device)
    except TypeError:
        return GradScaler(enabled=enabled)


def _register_yolo_pickle_alias():
    """
    兼容 YOLOv5/MSCFT ckpt 中 pickled 的 models.*、utils.* 模块。
    这些 ckpt 常序列化 Model 对象（而非 state_dict），会引用如 models.common/experimental/yolo
    等模块路径；在本工程中需要手动注册别名以通过反序列化。
    """
    try:
        from engines.models.mscft import common as mscft_common
        from engines.models.mscft import experimental as mscft_experimental
        from engines.models.mscft import yolo as mscft_yolo
        from engines.models.mscft.utils import torch_utils as mscft_utils_torch_utils
        from engines.models.mscft.utils import general as mscft_utils_general
        from engines.models.mscft.utils import loss as mscft_utils_loss
        from engines.models.mscft.utils import metrics as mscft_utils_metrics
    except Exception:
        return

    def _ensure(parent: str):
        mod = sys.modules.get(parent)
        if mod is None:
            mod = ModuleType(parent)
            sys.modules[parent] = mod
        return mod

    models_mod = _ensure("models")
    utils_mod = _ensure("utils")

    alias_map = {
        "models.common": mscft_common,
        "models.experimental": mscft_experimental,
        "models.yolo": mscft_yolo,
        "utils.torch_utils": mscft_utils_torch_utils,
        "utils.general": mscft_utils_general,
        "utils.loss": mscft_utils_loss,
        # 避免覆盖项目内 utils.metrics，使用独立别名
        "models.utils.metrics": mscft_utils_metrics,
    }
    for name, module in alias_map.items():
        if name not in sys.modules:
            sys.modules[name] = module
        # 将子模块挂到父模块，确保 pickle 能通过 getattr 查找到
        parent, attr = name.split(".", 1)
        parent_mod = _ensure(parent)
        setattr(parent_mod, attr, module)
        # 同步顶级模块自身，方便重复导入
        if parent == "models" and "models" not in sys.modules:
            sys.modules["models"] = parent_mod
        if parent == "utils" and "utils" not in sys.modules:
            sys.modules["utils"] = parent_mod


def _remap_ultralytics_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    """
    将 Ultralytics/YOLOv5 风格的 key（如 model.0.conv.weight）重命名为本仓库 MSCFT
    模型的命名（model.model.0.conv.conv.weight 等）。
    仅在检测到 key 前缀为 model.<idx>.（idx 为数字）时应用，避免误伤其它模型（如 RT-DETRv4 wrapper 的 model.backbone.*）。
    """
    if not state_dict:
        return state_dict
    sample_key = next(iter(state_dict))
    if sample_key.startswith("model.model."):
        return state_dict
    # 仅匹配 model.<digits>.xxx 的 YOLO 风格；例如 model.0.conv.weight
    import re
    if re.match(r"^model\\.\\d+\\.", sample_key) is None:
        return state_dict

    def _remap_key(key: str) -> str:
        new_key = f"model.{key}"
        # Focus 层（0 与 3）在 MSCFT 实现中多包了一层 conv/BN
        for idx in ("0", "3"):
            prefix = f"model.model.{idx}."
            if new_key.startswith(prefix):
                new_key = new_key.replace(f"{prefix}conv.", f"{prefix}conv.conv.")
                new_key = new_key.replace(f"{prefix}bn.", f"{prefix}conv.bn.")
        return new_key

    remapped = { _remap_key(k): v for k, v in state_dict.items() }
    return remapped


def _filter_compatible_state_dict(model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    """
    仅保留 shape 与当前模型一致的参数，避免 strict=False 仍因 shape mismatch 抛错。
    """
    model_sd = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        if k not in model_sd:
            continue
        if model_sd[k].shape != v.shape:
            continue
        filtered[k] = v
    return filtered


class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.runtime_cfg = get_config(self.cfg, "runtime", {})
        self.device = get_config(self.runtime_cfg, "device", get_config(self.cfg, "device", "cpu"))
        self.device_ids = get_config(self.runtime_cfg, "device_ids", [])
        self.world_size = get_config(self.runtime_cfg, "world_size", 1)
        self.distributed = False
        self.mode = get_config(self.cfg, "mode", None) or self.cfg.get("mode", None)

        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        self.train_dataloader = None
        self.validation_dataloader = None
        self.test_dataloader = None

        self.data_args: SimpleNamespace | None = None

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.metric_plot_sink: MetricsPlotSink | None = None
        self.csv_metric_sink: CsvMetricsSink | None = None
        self.best_map = -1.0
        self.best_map_ema = -1.0
        self.ema: ModelEMA | None = None
        self._base_lrs: List[float] = []
        self._warmup_steps: int = 0
        self._global_step: int = 0
        self._last_eval_artifacts: Dict[str, Any] = {}

    def _save_last_eval_metric_charts(
        self,
        *,
        output_dir: Path,
        prefix: str,
        results_json: Dict[str, Any] | None = None,
    ) -> None:
        """
        将“验证阶段产生的可视化”统一写入 output_dir/metric_charts/<split>/。

        说明：
        - results_json：来自 _validate_epoch 的 coco_extended_metrics 结果，用于保存 PR/F1/overall/per-class 指标图；
        - 其它产物（confusion_matrix、per-class 曲线）来自 self._last_eval_artifacts，由 _validate_epoch 缓存。
        """
        train_cfg: Dict[str, Any] = get_config(self.cfg, "train", {}) or {}
        chart_cfg = get_config(train_cfg, "metric_charts", {}) or {}
        split_dir = prefix.replace(" ", "_") or "metrics"
        save_subdir = f"metric_charts/{split_dir}"

        if results_json:
            try:
                save_detection_metric_charts(
                    results_json,
                    output_dir,
                    prefix=prefix,
                    chart_cfg=chart_cfg,
                    save_subdir=save_subdir,
                )
            except Exception as exc:
                logging.warning("保存指标可视化失败: %s", exc)

        artifacts = self._last_eval_artifacts or {}
        # instance segmentation 额外输出（可选）：{prefix}_segm_*.png
        segm_cfg = get_config(chart_cfg, "segm", {}) or {}
        if bool(get_config(segm_cfg, "enabled", False)):
            segm_results = artifacts.get("results_json_segm")
            segm_eval = artifacts.get("coco_eval_segm")
            if segm_results is not None:
                try:
                    save_detection_metric_charts(
                        segm_results,
                        output_dir,
                        prefix=f"{prefix}_segm",
                        chart_cfg=chart_cfg,
                        save_subdir=save_subdir,
                    )
                except Exception as exc:
                    logging.warning("保存 segm 指标可视化失败: %s", exc)
            if segm_results is not None and segm_eval is not None:
                try:
                    save_detection_combined_curves(
                        segm_results,
                        segm_eval,
                        output_dir,
                        prefix=f"{prefix}_segm",
                        save_subdir=save_subdir,
                    )
                except Exception as exc:
                    logging.warning("保存 segm 合并曲线图失败: %s", exc)

        coco_eval_bbox = artifacts.get("coco_eval_bbox")
        if coco_eval_bbox is not None and results_json:
            try:
                save_detection_combined_curves(
                    results_json,
                    coco_eval_bbox,
                    output_dir,
                    prefix=prefix,
                    save_subdir=save_subdir,
                )
            except Exception as exc:
                logging.warning("保存合并曲线图失败: %s", exc)
        elif coco_eval_bbox is not None:
            try:
                save_detection_per_class_curves_from_coco_eval(
                    coco_eval_bbox,
                    output_dir,
                    prefix=prefix,
                    save_subdir=save_subdir,
                )
            except Exception as exc:
                logging.warning("保存每类曲线图失败: %s", exc)

        confusion = artifacts.get("confusion_matrix")
        cm_class_names = artifacts.get("confusion_class_names")
        normalize = bool(artifacts.get("confusion_normalize", True))
        if confusion is not None and cm_class_names is not None:
            try:
                save_detection_confusion_matrix(
                    confusion,
                    cm_class_names,
                    output_dir,
                    prefix=prefix,
                    normalize=normalize,
                    save_subdir=save_subdir,
                )
            except Exception as exc:
                logging.warning("保存混淆矩阵失败: %s", exc)

        # 推理 profile 图（FPS/延迟/参数量）
        profile = artifacts.get("inference_profile")
        if profile is not None:
            try:
                train_cfg: Dict[str, Any] = get_config(self.cfg, "train", {}) or {}
                chart_cfg = get_config(train_cfg, "metric_charts", {}) or {}
                infer_cfg = get_config(chart_cfg, "inference", {}) or {}
                if bool(get_config(infer_cfg, "enabled", False)):
                    save_inference_profile_chart(
                        profile,
                        output_dir,
                        prefix=prefix,
                        save_subdir=save_subdir,
                    )
            except Exception as exc:
                logging.warning("保存推理 profile 图失败: %s", exc)

    def build_wandb(self):
        # DDP 下只允许 rank0 初始化 wandb，否则会出现多进程重复建 run/写文件的问题。
        if not is_main_process():
            return None
        try:
            import wandb
        except ModuleNotFoundError:
            print("wandb is not installed; skip W&B logging.")
            return None

        # 优先使用模型名称作为 W&B project，缺省时回退到 train.project。
        model_cfg = self.cfg.get('model', {}) if self.cfg is not None else {}
        project = None

        project = get_config(model_cfg, 'model_alias')
        if not project:
            project = get_config(model_cfg, 'model_name')
        if not project:
            project = get_config(model_cfg, 'msifpdetr')


        run_name = None
        train_cfg = self.cfg.get('train', {}) if self.cfg is not None else {}
        if hasattr(train_cfg, "get"):
            run_name = train_cfg.get('run') or train_cfg.get('run_name')
        else:
            run_name = getattr(train_cfg, 'run', None) or getattr(train_cfg, 'run_name', None)

        config_dict = None
        try:
            from omegaconf import OmegaConf
            config_dict = OmegaConf.to_container(self.cfg, resolve=True)
        except Exception:
            try:
                config_dict = dict(self.cfg)
            except Exception:
                config_dict = None

        try:
            run = wandb.init(project=project, name=run_name, config=config_dict)
        except Exception as exc:
            print(f"Failed to initialize wandb: {exc}")
            return None

        self.wandb_logger = WandbMetricLogger(run)
        return self.wandb_logger

    def _maybe_wrap_ddp_model(self) -> None:
        """
        Wrap self.model with torch.nn.parallel.DistributedDataParallel (DDP) when torch.distributed is initialized.

        Notes:
        - True multi-GPU training requires multi-process DDP (torchrun). DataParallel is intentionally not used.
        - This is a no-op for single-process training, or when already wrapped.
        """
        if self.model is None:
            return
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return

        runtime_cfg = self.runtime_cfg or {}
        sync_bn = bool(get_config(runtime_cfg, "sync_bn", False))
        find_unused_parameters = bool(get_config(runtime_cfg, "find_unused_parameters", False))
        broadcast_buffers = bool(get_config(runtime_cfg, "ddp_broadcast_buffers", True))

        # Ensure model is on the correct device before wrapping.
        if sync_bn:
            try:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            except Exception as exc:
                logging.warning("SyncBatchNorm 转换失败，将继续使用普通 BN：%s", exc)

        ddp_kwargs: Dict[str, Any] = {
            "find_unused_parameters": find_unused_parameters,
            "broadcast_buffers": broadcast_buffers,
        }
        if getattr(self.device, "type", None) == "cuda" and torch.cuda.is_available():
            device_id = int(self.device.index) if self.device.index is not None else int(os.environ.get("LOCAL_RANK", 0))
            ddp_kwargs["device_ids"] = [device_id]
            ddp_kwargs["output_device"] = device_id

        rank = int(torch.distributed.get_rank())
        world_size = int(torch.distributed.get_world_size())
        try:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, **ddp_kwargs)
        except TypeError:
            # Backward compatibility for older torch versions that don't support some kwargs.
            ddp_kwargs.pop("broadcast_buffers", None)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, **ddp_kwargs)
        logging.info("DDP wrap 完成: rank=%d/%d, device=%s", rank, world_size, self.device)

    @staticmethod
    def _prune_numbered_checkpoints(output_dir: Path) -> int:
        """
        删除形如 checkpoint_<epoch>.pth 的周期性权重，仅保留 checkpoint.pth / checkpoint_best.pth 等非数字命名文件。
        返回删除的文件数。
        """
        deleted = 0
        for path in output_dir.glob("checkpoint_*.pth"):
            name = path.name
            tag = name[len("checkpoint_") : -len(".pth")]
            if not tag.isdigit():
                continue
            try:
                path.unlink()
                deleted += 1
            except FileNotFoundError:
                continue
            except Exception as exc:
                logging.warning("删除旧 checkpoint 失败（忽略）: %s (%s)", path, exc)
        return deleted

    @staticmethod
    def _collect_fusion_gate_stats(model: torch.nn.Module) -> dict:
        """
        收集融合/门控相关的标量参数，便于确认“MSI 分支是否被用到”。

        约定：
        - gate 参数命名包含 `.fusion_alpha.`（例如 `...fusion_alpha.value`），与优化器分组逻辑保持一致；
        - 仅记录标量参数，避免日志爆炸。
        """
        stats: dict[str, float] = {}
        for name, param in model.named_parameters():
            if not torch.is_tensor(param) or param.numel() != 1:
                continue
            if (".fusion_alpha." not in name) and (".gpt_alpha_" not in name) and (".progressive_alpha_" not in name):
                continue
            try:
                value = float(param.detach().float().cpu().item())
            except Exception:
                continue
            key_base = name.replace(".value", "").replace(".", "/")
            stats[f"gate/{key_base}"] = value
            stats[f"gate_tanh/{key_base}"] = float(torch.tanh(torch.tensor(value)).item())
        return stats

    @staticmethod
    def _collect_ms_fusion_runtime_stats(model: torch.nn.Module) -> dict:
        """
        收集 decoder-MS 融合的运行期统计（来自 TransformerDecoderLayer 的 buffer 累计）。

        目的：解释“强制 alpha 也无提升”的原因：是 MSI 分支输出幅度很小，还是被网络学会抵消。
        """
        stats: dict[str, float] = {}
        for module_name, module in model.named_modules():
            has_ms = getattr(module, "ms_cross_attn", None) is not None
            if not has_ms:
                continue
            count = getattr(module, "_ms_fusion_count", None)
            if count is None:
                continue
            try:
                cnt = float(count.detach().float().cpu().item())
            except Exception:
                continue
            if cnt <= 0:
                continue
            sum_rgb = getattr(module, "_ms_fusion_sum_rgb_abs", None)
            sum_ms = getattr(module, "_ms_fusion_sum_ms_abs", None)
            sum_ratio = getattr(module, "_ms_fusion_sum_ratio", None)
            if sum_rgb is None or sum_ms is None or sum_ratio is None:
                continue
            rgb_abs = float((sum_rgb / count).detach().float().cpu().item())
            ms_abs = float((sum_ms / count).detach().float().cpu().item())
            ratio = float((sum_ratio / count).detach().float().cpu().item())
            key_base = module_name.replace(".value", "").replace(".", "/")
            stats[f"ms_fusion_abs/rgb/{key_base}"] = rgb_abs
            stats[f"ms_fusion_abs/ms/{key_base}"] = ms_abs
            stats[f"ms_fusion_ratio/{key_base}"] = ratio
        return stats

    @staticmethod
    def _collect_ms_ref_shift_stats(model: torch.nn.Module) -> dict:
        """
        收集 MSI reference shift（dx,dy）参数，便于判断是否学到了“合理的对齐方向/幅度”。

        说明：
        - shift_param 是未限幅的自由参数；
        - shift_eff = shift_scale * tanh(shift_param) 才是实际加到 reference_points 上的偏移（归一化坐标）。
        """
        stats: dict[str, float] = {}
        for module_name, module in model.named_modules():
            shift = getattr(module, "ms_ref_shift", None)
            if shift is None or (not torch.is_tensor(shift)) or shift.numel() != 2:
                continue
            scale = float(getattr(module, "ms_ref_shift_scale", 0.0))
            try:
                shift_xy = shift.detach().float().cpu()
                dx, dy = float(shift_xy[0].item()), float(shift_xy[1].item())
                eff = (scale * torch.tanh(shift.detach())).detach().float().cpu()
                dx_eff, dy_eff = float(eff[0].item()), float(eff[1].item())
            except Exception:
                continue
            key_base = module_name.replace(".", "/")
            stats[f"ms_ref_shift/dx/{key_base}"] = dx
            stats[f"ms_ref_shift/dy/{key_base}"] = dy
            stats[f"ms_ref_shift_eff/dx/{key_base}"] = dx_eff
            stats[f"ms_ref_shift_eff/dy/{key_base}"] = dy_eff
        return stats

    def init_device(self, device=None):
        """
        根据 runtime 配置自动选择单卡/多卡设备。
        - 优先读取 runtime.device / runtime.device_ids / runtime.world_size；
        - CUDA 不可用时回退到 CPU；
        - 注意：DistributedSampler 需要 torch.distributed init_process_group；
          仅指定多个 device_ids 并不会自动初始化分布式训练。
        """
        runtime_cfg = self.runtime_cfg or {}
        device = device or get_config(runtime_cfg, "device", "cpu")
        device_ids = get_config(runtime_cfg, "device_ids", None)

        if device_ids is None:
            device_ids = []

        # CUDA 不可用直接回退
        if device == "cpu" or not torch.cuda.is_available():
            if device != "cpu":
                logging.warning("配置要求使用 CUDA，但当前环境不可用，自动回退到 CPU。")
            self.device = torch.device("cpu")
            self.device_ids = []
            self.world_size = 1
            self.distributed = False
            return self.device

        available_cnt = torch.cuda.device_count()
        if not device_ids:
            device_ids = list(range(available_cnt))
        # 仅保留有效的 GPU id
        device_ids = [idx for idx in device_ids if idx < available_cnt]
        if not device_ids:
            raise RuntimeError("未找到可用的 GPU，检查 runtime.device_ids 配置。")

        # 仅当 torch.distributed 已初始化且 world_size>1 时，才视为“真正的分布式训练”。
        ddp_ready = torch.distributed.is_available() and torch.distributed.is_initialized()
        ddp_world_size = int(torch.distributed.get_world_size()) if ddp_ready else 1
        ddp_rank = int(torch.distributed.get_rank()) if ddp_ready else 0

        # 单进程/非分布式场景下，固定使用 device_ids[0]；如需 DDP，请用 torchrun 启动并初始化进程组。
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        primary_idx = device_ids[local_rank % len(device_ids)] if ddp_world_size > 1 else int(device_ids[0])
        torch.cuda.set_device(primary_idx)

        self.device = torch.device(f"cuda:{primary_idx}")
        self.device_ids = device_ids
        self.world_size = ddp_world_size
        self.distributed = ddp_world_size > 1

        if self.distributed:
            logging.info(
                "使用分布式训练: rank=%s, world_size=%s, local_rank=%s, device=%s, device_ids=%s",
                ddp_rank, ddp_world_size, local_rank, self.device, device_ids,
            )
        else:
            if len(device_ids) > 1:
                logging.warning(
                    "检测到多 GPU (runtime.device_ids=%s) 但未初始化 torch.distributed，将以单进程单卡 %s 运行。"
                    " 若要多卡 DDP，请使用 torchrun 启动并在代码中 init_process_group。",
                    device_ids,
                    self.device,
                )
            else:
                logging.info("使用单卡训练: device=%s", self.device)

        return self.device

    def _ensure_data_args(self) -> SimpleNamespace:
        """
        将 data/train/model 配置合并成可复用的 namespace，避免在各个阶段反复构建。
        """
        if self.data_args is not None:
            self.data_args.distributed = bool(self.distributed)
            return self.data_args

        data_cfg = get_config(self.cfg, "data", {}) or {}
        train_cfg = get_config(self.cfg, "train", {}) or {}
        model_cfg = get_config(self.cfg, "model", {}) or {}

        sentinel = object()

        def pick(key, default=None):
            # 按 data > train > model 的优先级取值，未命中则使用默认。
            for cfg in (data_cfg, train_cfg, model_cfg):
                val = get_config(cfg, key, sentinel)
                if val is not sentinel:
                    return val
            return default

        args = SimpleNamespace()
        if hasattr(data_cfg, "items"):
            for key, value in data_cfg.items():
                setattr(args, key, value)
        for cfg in (train_cfg, model_cfg):
            if hasattr(cfg, "items"):
                for key, value in cfg.items():
                    if not hasattr(args, key):
                        setattr(args, key, value)

        # 模态开关，若未配置则默认同时启用 RGB+MSI
        args.use_rgb_input = pick("use_rgb_input", True)
        args.use_msi_input = pick("use_msi_input", True)

        # 训练阶段控制项
        args.batch_size = pick("batch_size", 1)
        args.num_workers = pick("num_workers", 0)
        args.grad_accum_steps = pick("grad_accum_steps", 1)
        args.min_train_batches = pick("min_train_batches", 5)
        args.multi_scale = pick("multi_scale", False)
        args.expanded_scales = pick("expanded_scales", False)
        args.do_random_resize_via_padding = pick("do_random_resize_via_padding", False)
        args.segmentation_head = pick("segmentation_head", False)
        args.class_names = pick("class_names", None)
        args.remap_mscoco_category = pick("remap_mscoco_category", False)
        # 与 datasets.build_dataset 的自动 remap 策略保持一致：
        # 如果提供了 class_names 且是 COCO 风格数据集，默认开启 remap_mscoco_category，
        # 以保证训练(连续 label)与评测(映射回 category_id)一致。
        dataset_key = pick("dataset_key", None) or pick("dataset_type", None)
        dataset_file = pick("dataset_file", None)
        if dataset_key and dataset_file and str(dataset_key) != str(dataset_file):
            logging.warning(
                "data.dataset_key=%s 与 data.dataset_file=%s 同时存在且不一致，将优先使用 dataset_key。",
                dataset_key,
                dataset_file,
            )
        if dataset_key and not dataset_file:
            dataset_file = dataset_key
        dataset_key = dataset_file
        if (
            args.class_names
            and dataset_key in {"coco_rgb", "coco_msi", "coco_rgb_msi"}
            and not args.remap_mscoco_category
        ):
            args.remap_mscoco_category = True

        # 模型相关尺寸/窗口
        args.img_size = pick("img_size", 640)
        args.patch_size = pick("patch_size", 16)
        args.num_windows = pick("num_windows", 4)

        # 输入通道与堆叠划分
        # 注意：即使配置里写了 ms_input_channels=7，只要 use_msi_input=False，就必须把 ms 通道视为 0，
        # 否则会导致模型按 10 通道构建，但数据实际只输出 3 通道，从而触发 DINOv2 输入通道校验失败。
        rgb_cfg_ch = int(pick("rgb_input_channels", 3))
        ms_cfg_ch = int(pick("ms_input_channels", int(pick("ms_expected_channels", 0))))
        rgb_ch = rgb_cfg_ch if args.use_rgb_input else 0
        ms_ch = ms_cfg_ch if args.use_msi_input else 0

        args.rgb_input_channels = rgb_ch
        args.ms_input_channels = ms_ch

        channel_splits = pick("channel_splits", None)
        if channel_splits is None:
            channel_splits = (rgb_ch, ms_ch)
        else:
            # 保证与实际启用模态一致，避免“配置残留”导致通道数推导错误
            channel_splits = (rgb_ch, ms_ch)
        args.channel_splits = channel_splits

        # 若使用堆叠输出，推导总体 input_channels，方便后续模型构建
        dual_stream = bool(pick("dual_stream_output", False))
        if not dual_stream:
            args.input_channels = int(sum(channel_splits))

        # 目录兼容性：coco_path 与 dataset_dir 保持一致，若缺省则报错
        dataset_file = pick("dataset_file", None)
        dataset_dir = pick("dataset_dir", None)
        coco_path = pick("coco_path", None)
        if coco_path is None and dataset_dir is not None:
            coco_path = dataset_dir
        if dataset_dir is None and coco_path is not None:
            dataset_dir = coco_path
        args.dataset_file = dataset_file
        args.dataset_key = dataset_key
        args.dataset_dir = dataset_dir
        args.coco_path = coco_path

        if not args.dataset_file:
            raise ValueError("data.dataset_file 未配置，无法构建数据集。")

        args.distributed = bool(self.distributed)
        self.data_args = args
        return self.data_args

    def _make_sampler(self, dataset, shuffle: bool):
        # 分布式场景下使用 DistributedSampler，其余根据 shuffle 选择随机/顺序。
        if self.distributed:
            return DistributedSampler(dataset, shuffle=shuffle)
        return RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

    def _build_train_loader(self, dataset, args, collate_fn):
        distributed = bool(getattr(args, "distributed", False))
        if distributed:
            sampler = DistributedSampler(dataset)
        else:
            sampler = RandomSampler(dataset)

        grad_accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))
        batch_size = int(getattr(args, "batch_size", 1))
        effective_batch_size = batch_size * grad_accum_steps
        num_workers = int(getattr(args, "num_workers", 0))
        min_batches = max(1, int(getattr(args, "min_train_batches", 5)))

        pin_memory = torch.cuda.is_available()
        # 数据量太小时使用带替换的采样，避免过早耗尽
        if len(dataset) < effective_batch_size * min_batches:
            sampler = RandomSampler(
                dataset, replacement=True, num_samples=effective_batch_size * min_batches
            )
            try:
                return DataLoader(
                    dataset,
                    batch_size=effective_batch_size,
                    sampler=sampler,
                    num_workers=num_workers,
                    collate_fn=collate_fn,
                    pin_memory=pin_memory,
                )
            except PermissionError as exc:
                logging.warning("DataLoader 创建失败（num_workers=%d），回退到单进程: %s", num_workers, exc)
                return DataLoader(
                    dataset,
                    batch_size=effective_batch_size,
                    sampler=sampler,
                    num_workers=0,
                    collate_fn=collate_fn,
                    pin_memory=pin_memory,
                )

        batch_sampler = BatchSampler(sampler, effective_batch_size, drop_last=True)
        try:
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=pin_memory,
            )
        except PermissionError as exc:
            logging.warning("DataLoader 创建失败（num_workers=%d），回退到单进程: %s", num_workers, exc)
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=0,
                collate_fn=collate_fn,
                pin_memory=pin_memory,
            )

    def _build_eval_loader(self, dataset, args, *, split: str, collate_fn):
        # 评估阶段不打乱，分布式使用 DistributedSampler 以均分样本。
        if self.distributed:
            sampler = DistributedSampler(dataset, shuffle=False)
        else:
            sampler = SequentialSampler(dataset)
        batch_size = int(getattr(args, "batch_size", 1))
        num_workers = int(getattr(args, "num_workers", 0))
        pin_memory = torch.cuda.is_available()
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=False,
            pin_memory=pin_memory,
        )

    @staticmethod
    def _has_coco_style_split(root: Path, *, split: str, prefer_bbox: bool) -> bool:
        """检查自定义 COCO 数据集是否存在某个 split 的标注文件（train/val/test）。"""
        ann_dir = root / "annotations"
        if not ann_dir.is_dir():
            return False

        candidate_roots = [ann_dir]
        if prefer_bbox:
            candidate_roots.extend([ann_dir / "detection", ann_dir / "segmentation"])
            candidate_names = [f"{split}_bbox.json", f"{split}.json"]
        else:
            candidate_roots.extend([ann_dir / "segmentation", ann_dir / "detection"])
            candidate_names = [f"{split}.json", f"{split}_bbox.json"]

        seen = set()
        for root_dir in candidate_roots:
            for name in candidate_names:
                path = root_dir / name
                key = str(path)
                if key in seen:
                    continue
                seen.add(key)
                if path.is_file():
                    return True
        return False

    @staticmethod
    def _infer_default_test_split(args) -> str:
        """
        默认策略：优先使用真实 test split（如果存在对应标注文件），否则回退到 val。

        说明：
        - 通过 `data.test_split` 显式覆盖（例如设置为 val 用于“用 val 当 test”）。
        - 兼容多数据集（data.dataset_dirs）：默认要求所有子数据集都具备 test（若启用 skip_missing_splits 则只需任一具备）。
        """
        dataset_key = getattr(args, "dataset_key", None) or getattr(args, "dataset_file", None) or ""
        dataset_key = str(dataset_key)
        prefer_bbox = not bool(getattr(args, "segmentation_head", False))
        skip_missing = bool(getattr(args, "skip_missing_splits", False))

        roots = []
        dataset_dirs = getattr(args, "dataset_dirs", None)
        if dataset_dirs:
            try:
                from omegaconf import ListConfig  # type: ignore
            except Exception:  # pragma: no cover
                ListConfig = ()  # type: ignore
            if isinstance(dataset_dirs, ListConfig):
                roots = list(dataset_dirs)
            elif isinstance(dataset_dirs, (list, tuple)):
                roots = list(dataset_dirs)
        else:
            dataset_root = getattr(args, "ms_dataset_dir", None) or getattr(args, "dataset_dir", None)
            if dataset_root:
                roots = [dataset_root]

        if not roots:
            # 未能推断数据根目录，保持旧行为回退到 val，避免误判导致 FileNotFoundError。
            return "val"

        def _root_has_test(root: Path) -> bool:
            # Roboflow 标注文件固定为 test/_annotations.coco.json
            if dataset_key == "roboflow":
                return (root / "test" / "_annotations.coco.json").is_file()
            # COCO 风格自定义数据集：annotations/test(.json|_bbox.json)
            if dataset_key in {"coco_rgb", "coco_msi", "coco_rgb_msi"}:
                return BaseTrainer._has_coco_style_split(root, split="test", prefer_bbox=prefer_bbox)
            return BaseTrainer._has_coco_style_split(root, split="test", prefer_bbox=prefer_bbox)

        flags = []
        for item in roots:
            try:
                root = Path(str(item)).expanduser()
            except Exception:
                flags.append(False)
                continue
            if not root.exists():
                flags.append(False)
                continue
            flags.append(_root_has_test(root))

        if not flags:
            return "val"
        has_test = any(flags) if skip_missing else all(flags)
        return "test" if has_test else "val"

    def build_dataset(self):
        """
        依据配置自动选择数据集类型（coco/roboflow/多光谱等）并构建 train/val/test dataloader。
        """
        from datasets import build_dataset as build_legacy_dataset
        from utils.misc import collate_fn

        args = self._ensure_data_args()
        img_size = getattr(args, "img_size", 640)
        # 允许将“验证集”指向非 val split（例如诊断性地与 test 对调）。
        val_split = getattr(args, "val_split", None) or "val"
        test_split = getattr(args, "test_split", None)
        if not test_split:
            test_split = self._infer_default_test_split(args)
            logging.info("data.test_split 未配置，自动选择 test_split=%s", test_split)

        logging.info("数据集 split 选择：val_split=%s, test_split=%s", val_split, test_split)
        logging.info("开始构建数据集: dataset_file=%s, img_size=%s", args.dataset_file, img_size)
        # build_dataset 会根据 args.dataset_file 自动选择数据集实现。
        self.train_dataset = build_legacy_dataset("train", args, img_size)
        self.validation_dataset = build_legacy_dataset(val_split, args, img_size)
        self.test_dataset = build_legacy_dataset(test_split, args, img_size)

        batch_size = int(getattr(args, "batch_size", 1))
        # 训练 DataLoader 采用梯度累积合并后的 batch_size，评估阶段不 drop_last。
        self.train_dataloader = self._build_train_loader(self.train_dataset, args, collate_fn)
        self.validation_dataloader = self._build_eval_loader(
            self.validation_dataset, args, split="val", collate_fn=collate_fn
        )
        self.test_dataloader = self._build_eval_loader(
            self.test_dataset, args, split="test", collate_fn=collate_fn
        )

        num_workers = int(getattr(args, "num_workers", 0))
        logging.info(
            "数据集与 DataLoader 构建完成: train=%d, val=%d, test=%d, batch_size=%d, workers=%d",
            len(self.train_dataset),
            len(self.validation_dataset),
            len(self.test_dataset),
            batch_size,
            num_workers,
        )
        return {
            "train": self.train_dataset,
            "val": self.validation_dataset,
            "test": self.test_dataset,
        }

    def build_train_dataloader(self):
        if self.train_dataloader is None:
            self.build_dataset()
        return self.train_dataloader

    def build_val_dataloader(self):
        if self.validation_dataloader is None:
            self.build_dataset()
        return self.validation_dataloader

    def build_test_dataloader(self):
        if self.test_dataloader is None:
            self.build_dataset()
        return self.test_dataloader

    def build_model(self):
        raise NotImplementedError

    def build_criterion(self):
        """
        按配置装配损失函数：
        - 从 train.criterion.losses 读取注册名与权重；
        - 通过注册表构建各子损失，再用 CompositeCriterion 归一化权重求和。
        """
        train_cfg: Dict[str, Any] = get_config(self.cfg, "train", {}) or {}
        criterion_cfg: Dict[str, Any] = get_config(train_cfg, "criterion", {}) or {}
        losses_cfg: List[Dict[str, Any]] = criterion_cfg.get("losses") or []

        if not losses_cfg:
            # 默认只使用 set_criterion
            losses_cfg = [{"name": "set_criterion", "weight": 1.0}]

        loss_items: List[_LossItem] = []
        extras: Dict[str, Any] = {}

        for loss_conf in losses_cfg:
            name = loss_conf.get("name")
            if not name:
                raise ValueError("loss 配置缺少 name 字段。")
            weight = float(loss_conf.get("weight", 1.0))
            params = loss_conf.get("params", {}) or {}

            criterion_module, extra = build_loss(name=name, cfg=self.cfg, loss_cfg=loss_conf, **params)
            loss_items.append(_LossItem(name=name, weight=weight, module=criterion_module))
            extras[name] = extra or {}

        # 归一化聚合，并打印日志
        total_weight = sum(item.weight for item in loss_items)
        for item in loss_items:
            norm = item.weight / total_weight if total_weight > 0 else 1.0 / len(loss_items)
            logging.info("组装损失: name=%s, weight=%.4f, normalized=%.4f", item.name, item.weight, norm)

        criterion = CompositeCriterion(loss_items)

        # 放到设备
        device_str = getattr(self.device, "type", None) or get_config(self.runtime_cfg, "device", "cpu")
        device = torch.device(device_str)
        criterion.to(device)

        self.criterion = criterion
        self.criterion_extras = extras  # 预留给 matcher/postprocess 等
        return self.criterion

    def build_optimizer(self):
        if self.model is None:
            raise RuntimeError("请先构建模型再创建优化器。")

        train_cfg: Dict[str, Any] = get_config(self.cfg, "train", {}) or {}
        model_cfg: Dict[str, Any] = get_config(self.cfg, "model", {}) or {}

        optimizer_type = str(get_config(train_cfg, "optimizer", "adamw")).lower()

        model_for_params = self.model.module if hasattr(self.model, "module") else self.model
        named_parameters = list(model_for_params.named_parameters())

        lr = get_config(train_cfg, "lr", None)
        if lr is None:
            lr = get_config(model_cfg, "lr", None)
        model_hyp = getattr(model_for_params, "hyp", {}) or {}
        lr = float(lr if lr is not None else model_hyp.get("lr0", 1e-4))
        lr_encoder = float(get_config(train_cfg, "lr_encoder", lr))
        lr_component_decay = float(get_config(train_cfg, "lr_component_decay", 1.0))
        lr_vit_layer_decay = float(get_config(train_cfg, "lr_vit_layer_decay", 1.0))
        weight_decay = float(get_config(train_cfg, "weight_decay", 1e-4))
        momentum = float(get_config(train_cfg, "momentum", model_hyp.get("momentum", 0.9)))
        ms_lr_scale = float(get_config(train_cfg, "ms_lr_scale", get_config(model_cfg, "ms_lr_scale", 1.0)))

        # 构建 backbone 参数组（包含分层 lr 与单独权重衰减）。
        param_groups: List[Dict[str, Any]] = []
        backbone_param_names = set()
        backbone = getattr(model_for_params, "backbone", None)
        backbone_impl = None
        if backbone is not None:
            try:
                backbone_impl = backbone[0] if hasattr(backbone, "__getitem__") else backbone
            except Exception:
                backbone_impl = backbone

        if backbone_impl is not None and hasattr(backbone_impl, "get_named_param_lr_pairs"):
            args = SimpleNamespace(
                lr=lr,
                lr_encoder=lr_encoder,
                lr_vit_layer_decay=lr_vit_layer_decay,
                lr_component_decay=lr_component_decay,
                weight_decay=weight_decay,
                out_feature_indexes=get_config(model_cfg, "out_feature_indexes", []),
                ms_lr_scale=ms_lr_scale,
            )
            try:
                named_pairs = backbone_impl.get_named_param_lr_pairs(args, prefix="backbone.0")
                backbone_param_names = set(named_pairs.keys())
                param_groups.extend(named_pairs.values())
            except Exception as exc:
                logging.warning("构建 backbone 参数组失败，回退默认 lr：%s", exc)

        decoder_key = "transformer.decoder"

        # 双流/融合实验常用的 gate/shift 参数不应施加 weight decay，
        # 否则会被 AdamW 系统性“压到 0”，导致融合路径/对齐补偿被动关闭（实验结论失真）。
        alpha_named_params = [
            (name, p)
            for name, p in named_parameters
            if p.requires_grad
            and name not in backbone_param_names
            and (
                ".fusion_alpha." in name
                or ".gpt_alpha_" in name
                or ".progressive_alpha_" in name
                or ".ms_ref_shift" in name
            )
        ]
        alpha_param_names = {name for name, _ in alpha_named_params}
        if alpha_named_params:
            param_groups.append(
                {
                    "params": [p for _, p in alpha_named_params],
                    "lr": lr,
                    "weight_decay": 0.0,
                }
            )

        decoder_params = [
            p
            for name, p in named_parameters
            if decoder_key in name
            and name not in backbone_param_names
            and name not in alpha_param_names
            and p.requires_grad
        ]
        if decoder_params:
            param_groups.extend(
                {"params": p, "lr": lr * lr_component_decay} for p in decoder_params
            )

        other_params = [
            p
            for name, p in named_parameters
            if name not in backbone_param_names
            and decoder_key not in name
            and name not in alpha_param_names
            and p.requires_grad
        ]
        if other_params:
            param_groups.extend({"params": p, "lr": lr} for p in other_params)

        filtered_groups: List[Dict[str, Any]] = []
        for group in param_groups:
            params = group.get("params")
            if params is None:
                continue
            if isinstance(params, torch.nn.Parameter):
                if not params.requires_grad:
                    continue
                filtered_groups.append(group)
            else:
                params_list = [p for p in params if getattr(p, "requires_grad", False)]
                if not params_list:
                    continue
                new_group = dict(group)
                new_group["params"] = params_list
                filtered_groups.append(new_group)

        if not filtered_groups:
            logging.warning("未找到需训练的参数，使用模型全部可训练参数创建优化器。")
            filtered_groups = [{"params": [p for p in model_for_params.parameters() if p.requires_grad]}]

        if optimizer_type == "sgd":
            for group in filtered_groups:
                group.setdefault("lr", lr)
                group.setdefault("weight_decay", weight_decay)
            self.optimizer = torch.optim.SGD(
                filtered_groups,
                lr=lr,  # 兼容 PyTorch 对 group 中 lr 的缩放
                momentum=momentum,
                nesterov=True,
            )
            logging.info(
                "优化器创建完成: type=SGD, lr=%.6f, momentum=%.4f, weight_decay=%.6f, param_groups=%d",
                lr,
                momentum,
                weight_decay,
                len(filtered_groups),
            )
        else:
            self.optimizer = torch.optim.AdamW(
                filtered_groups,
                lr=lr,
                weight_decay=weight_decay,
            )
            logging.info(
                "优化器创建完成: type=AdamW, lr=%.6f, weight_decay=%.6f, param_groups=%d",
                lr,
                weight_decay,
                len(filtered_groups),
            )
        return self.optimizer

    def build_scheduler(self):
        if self.optimizer is None:
            raise RuntimeError("请先创建优化器再构建学习率调度器。")

        if self.train_dataset is None:
            raise RuntimeError("train_dataset 未构建，无法计算学习率调度步数。")

        train_cfg: Dict[str, Any] = get_config(self.cfg, "train", {}) or {}

        batch_size = int(get_config(train_cfg, "batch_size", 1))
        grad_accum_steps = max(1, int(get_config(train_cfg, "grad_accum_steps", 1)))
        epochs = int(get_config(train_cfg, "epochs", 1))
        warmup_epochs = int(get_config(train_cfg, "warmup_epochs", 0))
        lr_scheduler_type = str(get_config(train_cfg, "lr_scheduler", "step"))
        lr_drop = int(get_config(train_cfg, "lr_drop", epochs))
        lr_min_factor = float(get_config(train_cfg, "lr_min_factor", 0.0))

        world_size = max(1, int(get_world_size()))
        total_batch_size_for_lr = batch_size * world_size * grad_accum_steps
        steps_per_epoch = (len(self.train_dataset) + total_batch_size_for_lr - 1) // total_batch_size_for_lr
        steps_per_epoch = max(1, steps_per_epoch)

        total_steps = steps_per_epoch * epochs
        warmup_steps = steps_per_epoch * warmup_epochs

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            if lr_scheduler_type == "cosine":
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return lr_min_factor + (1 - lr_min_factor) * 0.5 * (1 + math.cos(math.pi * progress))

            if lr_scheduler_type == "step":
                return 1.0 if current_step < lr_drop * steps_per_epoch else 0.1

            return 1.0

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        logging.info(
            "学习率调度器创建完成: type=%s, total_steps=%d, warmup_steps=%d, steps_per_epoch=%d",
            lr_scheduler_type, total_steps, warmup_steps, steps_per_epoch,
        )
        return self.scheduler

    @staticmethod
    def _move_samples_to_device(samples, device):
        if isinstance(samples, dict):
            return {k: v.to(device) for k, v in samples.items()}
        if hasattr(samples, "to"):
            return samples.to(device)
        return samples

    @staticmethod
    def _slice_samples(samples, start: int, end: int):
        if isinstance(samples, dict):
            return {
                k: NestedTensor(
                    v.tensors[start:end],
                    v.mask[start:end] if getattr(v, "mask", None) is not None else None,
                )
                for k, v in samples.items()
            }
        mask = samples.mask[start:end] if getattr(samples, "mask", None) is not None else None
        return NestedTensor(samples.tensors[start:end], mask)

    @staticmethod
    def _resize_samples(samples, size_hw):
        def _resize_single(nested: NestedTensor):
            tensors = F.interpolate(nested.tensors, size=size_hw, mode="bilinear", align_corners=False)
            mask = None
            if getattr(nested, "mask", None) is not None:
                mask = (
                    F.interpolate(nested.mask.unsqueeze(1).float(), size=size_hw, mode="nearest")
                    .squeeze(1)
                    .bool()
                )
            return NestedTensor(tensors, mask)

        if isinstance(samples, dict):
            return {k: _resize_single(v) for k, v in samples.items()}
        return _resize_single(samples)

    @staticmethod
    def _align_samples_to_block(samples, block_size: int):
        """将输入尺寸对齐到 block_size 的整数倍（向下取整，至少为 block_size）。"""
        def _align_single(nested: NestedTensor):
            h, w = nested.tensors.shape[-2:]
            new_h = max(block_size, (h // block_size) * block_size)
            new_w = max(block_size, (w // block_size) * block_size)
            if new_h == h and new_w == w:
                return nested
            return BaseTrainer._resize_samples(nested, (new_h, new_w))

        if isinstance(samples, dict):
            return {k: _align_single(v) for k, v in samples.items()}
        return _align_single(samples)

    @staticmethod
    def _sum_loss_dict(loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 优先使用 loss_total（通常带梯度）
        if "loss_total" in loss_dict and torch.is_tensor(loss_dict["loss_total"]):
            return loss_dict["loss_total"]

        losses = []
        for key, value in loss_dict.items():
            if not torch.is_tensor(value):
                continue
            if "loss" in key:
                losses.append(value)
        if not losses:
            return torch.tensor(0.0, device=next(iter(loss_dict.values())).device)
        return sum(losses)

    @staticmethod
    def _sanitize_bn_running_stats(model: torch.nn.Module | None) -> Dict[str, Any]:
        """
        修复 BatchNorm running stats 中的 NaN/Inf，避免验证阶段因为坏掉的统计量直接崩溃。

        背景：
        - 在训练后期若某些分支出现数值异常，BN 的 running_mean/running_var 可能写入非有限值；
        - 训练态 BN 使用当前 batch 统计，问题不一定立刻暴露；
        - 验证/测试态依赖 running stats，可能出现指标突降并锁死。
        """
        summary: Dict[str, Any] = {
            "fixed_modules": 0,
            "fixed_values": 0,
            "examples": [],
        }
        if model is None:
            return summary

        bn_base = torch.nn.modules.batchnorm._BatchNorm
        with torch.no_grad():
            for module_name, module in model.named_modules():
                if not isinstance(module, bn_base):
                    continue
                fixed_this_module = 0

                running_mean = getattr(module, "running_mean", None)
                if torch.is_tensor(running_mean):
                    bad = ~torch.isfinite(running_mean)
                    bad_count = int(bad.sum().item())
                    if bad_count > 0:
                        running_mean.data = torch.nan_to_num(
                            running_mean.data,
                            nan=0.0,
                            posinf=1.0e4,
                            neginf=-1.0e4,
                        )
                        fixed_this_module += bad_count
                    running_mean.data.clamp_(min=-1.0e4, max=1.0e4)

                running_var = getattr(module, "running_var", None)
                if torch.is_tensor(running_var):
                    bad = ~torch.isfinite(running_var)
                    bad_count = int(bad.sum().item())
                    if bad_count > 0:
                        running_var.data = torch.nan_to_num(
                            running_var.data,
                            nan=1.0,
                            posinf=1.0e4,
                            neginf=1.0,
                        )
                        fixed_this_module += bad_count
                    eps = float(getattr(module, "eps", 1.0e-5))
                    running_var.data.clamp_(min=eps, max=1.0e4)

                if fixed_this_module > 0:
                    summary["fixed_modules"] += 1
                    summary["fixed_values"] += fixed_this_module
                    if len(summary["examples"]) < 5:
                        summary["examples"].append(module_name or "<root>")

        return summary

    def train(self):
        logging.info("BaseTrainer train function called.")
        train_cfg: Dict[str, Any] = get_config(self.cfg, "train", {}) or {}

        self.init_device(self.device)
        self.build_dataset()
        # 确保 DataLoader 已创建，子类可直接使用
        self.build_train_dataloader()
        self.build_val_dataloader()
        self.build_test_dataloader()

        self.build_model()
        self._maybe_wrap_ddp_model()
        self.build_optimizer()
        self.build_scheduler()
        self.build_criterion()
        # 简易数据自检：快速统计首个 batch 中的正样本，避免“全背景”导致训练停滞。
        try:
            sample_batch = next(iter(self.train_dataloader))
            sample_targets = sample_batch[1] if isinstance(sample_batch, (list, tuple)) and len(sample_batch) > 1 else None
            if sample_targets:
                total_boxes = sum(int(t.get("boxes", torch.empty(0)).shape[0]) for t in sample_targets)
                logging.info("数据自检：首个 batch 正样本框总数=%d", total_boxes)
        except Exception as exc:
            logging.warning("数据自检失败（忽略继续训练）：%s", exc)
        # anchor 诊断（仅 MSCFT 提供），可通过 train.anchor_check 控制
        try:
            anchor_check = bool(get_config(train_cfg, "anchor_check", False))
            if anchor_check and hasattr(self, "_anchor_diagnostics"):
                self._anchor_diagnostics()
        except Exception as exc:
            logging.warning("anchor 诊断失败（忽略继续训练）：%s", exc)
        # 仅 MSCFT/YOLO 流水线需要 EMA，配置可关闭。
        if bool(get_config(train_cfg, "use_ema", False)):
            decay = float(get_config(train_cfg, "ema_decay", 0.9999))
            tau = int(get_config(train_cfg, "ema_tau", 0))
            try:
                self.ema = ModelEMA(self.model, decay=decay, updates=tau)
                logging.info("EMA 已启用: decay=%.5f, updates=%d", decay, tau)
            except Exception as exc:
                logging.warning("初始化 EMA 失败，将禁用 EMA: %s", exc)
                self.ema = None
        else:
            self.ema = None
        # 记录 warmup 相关超参
        steps_per_epoch = len(self.train_dataloader) if self.train_dataloader is not None else 0
        warmup_epochs = int(get_config(train_cfg, "warmup_epochs", 0))
        self._warmup_steps = max(0, warmup_epochs * max(1, steps_per_epoch))
        self._base_lrs = [group.get("lr", 0.0) for group in (self.optimizer.param_groups if self.optimizer else [])]

        # 日志与保存配置
        output_root = get_config(self.runtime_cfg, "output_root", None)
        if output_root is None:
            raise KeyError("runtime.output_root 缺失，请在主程序中设置。")
        output_dir = Path(output_root).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        self.metric_plot_sink = MetricsPlotSink(str(output_dir))
        save_metrics_csv = bool(get_config(train_cfg, "save_metrics_csv", True))
        metrics_csv_name = str(get_config(train_cfg, "metrics_csv_name", "metrics.csv"))
        self.csv_metric_sink = CsvMetricsSink(output_dir / metrics_csv_name) if save_metrics_csv else None
        wandb_enabled = bool(get_config(train_cfg, "wandb", False))
        self.wandb_logger = self.build_wandb() if wandb_enabled else None

        # 数据分布可视化（只生成一次）
        try:
            chart_cfg = get_config(train_cfg, "metric_charts", {}) or {}
            dataset_cfg = get_config(chart_cfg, "dataset", {}) or {}
            yolo_labels_cfg = get_config(chart_cfg, "yolo_labels", {}) or {}

            dataset_enabled = bool(get_config(dataset_cfg, "enabled", False))
            yolo_enabled = bool(get_config(yolo_labels_cfg, "enabled", False))

            if is_main_process() and (dataset_enabled or yolo_enabled):
                from types import SimpleNamespace

                split_to_ds = {
                    "train": self.train_dataset,
                    "val": self.validation_dataset,
                    "test": self.test_dataset,
                }

                split_to_coco = {}
                for split_name, ds in split_to_ds.items():
                    if ds is None:
                        continue
                    try:
                        coco_api = get_coco_api_from_dataset(ds)
                    except Exception:
                        coco_api = None
                    if coco_api is not None and hasattr(coco_api, "dataset"):
                        split_to_coco[split_name] = coco_api

                if split_to_coco:
                    # 合并为 all（train/val/test 合并），供数据分布与 YOLO 标签图复用
                    try:
                        first_ds = next(iter(split_to_coco.values())).dataset or {}
                        categories = first_ds.get("categories") or []
                    except Exception:
                        categories = []

                    all_anns = []
                    all_images = []
                    for coco_api in split_to_coco.values():
                        try:
                            ds = coco_api.dataset or {}
                            anns = ds.get("annotations") or []
                            imgs = ds.get("images") or []
                        except Exception:
                            anns = []
                            imgs = []
                        if isinstance(anns, list):
                            all_anns.extend(anns)
                        if isinstance(imgs, list):
                            all_images.extend(imgs)

                    # 尽量按 image id 去重，避免重复条目影响归一化映射
                    uniq_images = []
                    try:
                        by_id = {}
                        for img in all_images:
                            if not isinstance(img, dict) or "id" not in img:
                                continue
                            try:
                                by_id[int(img["id"])] = img
                            except Exception:
                                continue
                        uniq_images = list(by_id.values())
                    except Exception:
                        uniq_images = all_images

                    merged_coco = SimpleNamespace(
                        dataset={
                            "categories": categories,
                            "images": uniq_images,
                            "annotations": all_anns,
                        }
                    )

                    if dataset_enabled:
                        prefix = str(get_config(dataset_cfg, "prefix", "dataset"))
                        small_thr = int(get_config(dataset_cfg, "small_thr", 32))
                        medium_thr = int(get_config(dataset_cfg, "medium_thr", 96))
                        max_annotate_classes = int(get_config(dataset_cfg, "max_annotate_classes", 60))
                        annotate = bool(get_config(dataset_cfg, "annotate", True))
                        by_split = bool(get_config(dataset_cfg, "by_split", True))

                        save_coco_dataset_distribution_charts(
                            merged_coco,
                            output_dir,
                            prefix=prefix,
                            small_thr=small_thr,
                            medium_thr=medium_thr,
                            max_annotate_classes=max_annotate_classes,
                            annotate=annotate,
                            title_suffix="all",
                        )

                        # 再生成 train/val/test 各自的 size ratio（可通过 by_split 关闭）
                        if by_split:
                            for split_name, coco_api in split_to_coco.items():
                                save_coco_dataset_distribution_charts(
                                    coco_api,
                                    output_dir,
                                    prefix=f"{prefix}_{split_name}",
                                    small_thr=small_thr,
                                    medium_thr=medium_thr,
                                    max_annotate_classes=max_annotate_classes,
                                    annotate=annotate,
                                    save_class_counts=False,
                                    save_size_ratio=True,
                                    title_suffix=split_name,
                                )

                    if yolo_enabled:
                        split = str(get_config(yolo_labels_cfg, "split", "train")).strip().lower()
                        save_subdir = str(get_config(yolo_labels_cfg, "save_subdir", "metric_charts"))
                        max_points = int(get_config(yolo_labels_cfg, "max_points", 20000))
                        seed = int(get_config(yolo_labels_cfg, "seed", 0))
                        extra_cfg = get_config(yolo_labels_cfg, "extra", {}) or {}
                        extra_enabled = bool(get_config(extra_cfg, "enabled", False))
                        extra_filename = str(get_config(extra_cfg, "filename", "labels_extra.jpg"))
                        extra_bins = int(get_config(extra_cfg, "bins", 60))
                        extra_max_boxes = int(get_config(extra_cfg, "max_boxes", 200000))
                        per_class_cfg = get_config(yolo_labels_cfg, "per_class", {}) or {}
                        per_class_enabled = bool(get_config(per_class_cfg, "enabled", False))
                        per_class_top_k = int(get_config(per_class_cfg, "top_k", 16))
                        per_class_bins = int(get_config(per_class_cfg, "bins", 50))
                        per_class_max_boxes = int(get_config(per_class_cfg, "max_boxes_per_class", 20000))
                        per_class_centers_filename = str(
                            get_config(per_class_cfg, "centers_filename", "labels_centers_topk.jpg")
                        )
                        per_class_wh_filename = str(
                            get_config(per_class_cfg, "wh_filename", "labels_wh_topk.jpg")
                        )
                        coocc_cfg = get_config(yolo_labels_cfg, "cooccurrence", {}) or {}
                        coocc_enabled = bool(get_config(coocc_cfg, "enabled", False))
                        coocc_filename = str(get_config(coocc_cfg, "filename", "labels_cooccurrence.jpg"))
                        coocc_metric = str(get_config(coocc_cfg, "metric", "jaccard"))
                        coocc_top_k = int(get_config(coocc_cfg, "top_k", 30))
                        coocc_annotate = bool(get_config(coocc_cfg, "annotate", False))
                        coocc_max_annotate = int(get_config(coocc_cfg, "max_annotate_classes", 20))

                        if split == "all":
                            coco_for_plot = merged_coco
                        else:
                            coco_for_plot = split_to_coco.get(split)

                        if coco_for_plot is None:
                            logging.warning("生成 labels_correlogram.jpg 失败：未找到 split=%s 的 COCO 标注。", split)
                        else:
                            try:
                                save_yolo_labels_correlogram(
                                    coco_for_plot,
                                    output_dir,
                                    save_subdir=save_subdir,
                                    max_points=max_points,
                                    seed=seed,
                                )
                            except Exception as exc:
                                logging.warning("生成 labels_correlogram.jpg 失败（忽略继续）：%s", exc)
                            if extra_enabled:
                                try:
                                    save_yolo_labels_extra_charts(
                                        coco_for_plot,
                                        output_dir,
                                        save_subdir=save_subdir,
                                        filename=extra_filename,
                                        bins=extra_bins,
                                        max_boxes=extra_max_boxes,
                                        seed=seed,
                                    )
                                except Exception as exc:
                                    logging.warning("生成 labels_extra.jpg 失败（忽略继续）：%s", exc)
                            if per_class_enabled:
                                try:
                                    save_yolo_labels_per_class_heatmaps(
                                        coco_for_plot,
                                        output_dir,
                                        save_subdir=save_subdir,
                                        top_k=per_class_top_k,
                                        bins=per_class_bins,
                                        max_boxes_per_class=per_class_max_boxes,
                                        seed=seed,
                                        centers_filename=per_class_centers_filename,
                                        wh_filename=per_class_wh_filename,
                                    )
                                except Exception as exc:
                                    logging.warning("生成 per-class heatmaps 失败（忽略继续）：%s", exc)
                            if coocc_enabled:
                                try:
                                    save_yolo_labels_cooccurrence_heatmap(
                                        coco_for_plot,
                                        output_dir,
                                        save_subdir=save_subdir,
                                        filename=coocc_filename,
                                        metric=coocc_metric,
                                        top_k=coocc_top_k,
                                        annotate=coocc_annotate,
                                        max_annotate_classes=coocc_max_annotate,
                                    )
                                except Exception as exc:
                                    logging.warning("生成 labels_cooccurrence.jpg 失败（忽略继续）：%s", exc)
        except Exception as exc:
            logging.warning("生成数据分布图失败（忽略继续）：%s", exc)

        start_epoch = int(get_config(train_cfg, "start_epoch", 0))
        resume_path = get_config(train_cfg, "resume", "") 
        if resume_path:
            ckpt_path = Path(resume_path).expanduser()
            if ckpt_path.is_file():
                logging.info("加载断点: %s", ckpt_path)
                _register_yolo_pickle_alias()
                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                model_state = state.get("model", state)
                if hasattr(model_state, "state_dict"):
                    model_state = model_state.state_dict()
                model_state = _remap_ultralytics_state_dict(model_state)
                compatible_state = _filter_compatible_state_dict(self.model, model_state)
                skipped = len(model_state) - len(compatible_state)
                if skipped:
                    logging.warning("预训练权重有 %d 个参数形状不匹配，已跳过加载。", skipped)
                model_state = compatible_state
                if not isinstance(model_state, Mapping):
                    raise TypeError(f"无法从 {ckpt_path} 解析 state_dict，类型为 {type(model_state)}")
                missing, unexpected = self.model.load_state_dict(model_state, strict=False)
                if missing or unexpected:
                    logging.warning("模型权重缺失 %d 项, 额外 %d 项", len(missing), len(unexpected))
                if "optimizer" in state and self.optimizer is not None and state["optimizer"] is not None:
                    try:
                        self.optimizer.load_state_dict(state["optimizer"])
                    except Exception as exc:
                        logging.warning("加载 optimizer state 失败，继续仅加载模型权重: %s", exc)
                if "lr_scheduler" in state and self.scheduler is not None and state["lr_scheduler"] is not None:
                    try:
                        self.scheduler.load_state_dict(state["lr_scheduler"])
                    except Exception as exc:
                        logging.warning("加载 scheduler state 失败，继续仅加载模型权重: %s", exc)
                if "ema" in state and self.ema is not None:
                    try:
                        self.ema.ema.load_state_dict(state["ema"])
                        logging.info("EMA 权重已恢复。")
                    except Exception as exc:
                        logging.warning("加载 EMA 权重失败，将重新开始 EMA: %s", exc)
                start_epoch = int(state.get("epoch", -1)) + 1
            else:
                logging.warning("未找到 resume 路径: %s", ckpt_path)

        epochs = int(get_config(train_cfg, "epochs", 1))
        clip_max_norm = float(get_config(train_cfg, "clip_max_norm", 0.0))
        checkpoint_interval = int(get_config(train_cfg, "checkpoint_interval", 1))
        keep_only_best_and_last = bool(get_config(train_cfg, "keep_only_best_and_last", False))
        dont_save = bool(get_config(train_cfg, "dont_save_weights", False))
        use_early_stop = bool(get_config(train_cfg, "early_stopping", False))
        patience = int(get_config(train_cfg, "early_stopping_patience", 10))
        min_delta = float(get_config(train_cfg, "early_stopping_min_delta", 0.0))
        metric_charts_on_best_only = bool(get_config(train_cfg, "metric_charts_on_best_only", True))

        # best checkpoint 选择策略：
        # - 默认延续旧行为：按 val_loss 越小越好；
        # - 可通过 train.best_monitor / train.best_mode / train.best_metric_weights 配置为按 mAP 等指标选择。
        best_monitor = str(get_config(train_cfg, "best_monitor", "val_loss"))
        best_mode_cfg = str(get_config(train_cfg, "best_mode", "auto")).lower()
        best_start_epoch = int(get_config(train_cfg, "best_start_epoch", 0))
        best_eps = float(get_config(train_cfg, "best_eps", 1e-9))
        best_fallback_to_val_loss = bool(get_config(train_cfg, "best_fallback_to_val_loss", True))
        best_metric_weights_cfg = get_config(train_cfg, "best_metric_weights", None)
        best_metric_weights: Dict[str, float] = {}
        if isinstance(best_metric_weights_cfg, Mapping):
            for k, v in best_metric_weights_cfg.items():
                try:
                    best_metric_weights[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue

        # 记录每次 best 更新（方便回溯最优 epoch 对应的指标）。
        save_best_updates = bool(get_config(train_cfg, "save_best_updates", True))
        best_updates_filename = str(get_config(train_cfg, "best_updates_filename", "best_updates.jsonl"))
        best_updates_path = output_dir / best_updates_filename

        def _append_best_update(*, epoch: int, score: float, val_stats: Mapping[str, Any] | None) -> None:
            if (not save_best_updates) or (not is_main_process()):
                return
            payload: Dict[str, Any] = {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "epoch": int(epoch),
                "best_score": float(score),
                "best_mode": str(resolved_best_mode),
                "best_monitor": str(best_monitor),
            }
            if best_metric_weights:
                payload["best_metric_weights"] = dict(best_metric_weights)
            if val_stats:
                # 仅落盘少量常用指标，避免把超长曲线塞到记录文件里。
                for k in ("val_loss", "val_map50", "val_map75", "val_map50_95", "loss", "map50", "map75", "map50_95", "precision", "recall"):
                    if k in val_stats:
                        payload[k] = val_stats.get(k)
            try:
                best_updates_path.parent.mkdir(parents=True, exist_ok=True)
                with best_updates_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            except Exception as exc:
                logging.warning("写入 best 更新记录失败: %s", exc)

        def _normalize_best_key(key: str) -> str:
            raw = str(key).strip()
            if not raw:
                return raw
            raw_lower = raw.lower()
            alias = {
                "loss": "val_loss",
                "val.loss": "val_loss",
                "val/loss": "val_loss",
                "val_loss": "val_loss",
                "map50": "map50",
                "map@50": "map50",
                "ap50": "map50",
                "map75": "map75",
                "map@75": "map75",
                "map@0.75": "map75",
                "ap75": "map75",
                "map50_95": "map50_95",
                "map50-95": "map50_95",
                "map@50:95": "map50_95",
                "map@0.5:0.95": "map50_95",
                "precision": "precision",
                "prec": "precision",
                "p": "precision",
                "recall": "recall",
                "rec": "recall",
                "r": "recall",
                "f1": "f1",
            }
            return alias.get(raw_lower, raw)

        def _compute_best_score(val_stats: Mapping[str, Any] | None) -> float | None:
            if not val_stats:
                return None

            # 1) 若提供了权重字典，则使用加权求和作为 score（忽略 best_monitor）
            if best_metric_weights:
                score = 0.0
                used = False
                for k, w in best_metric_weights.items():
                    key = _normalize_best_key(k)
                    if key not in val_stats:
                        continue
                    value = val_stats.get(key, None)
                    if value is None:
                        continue
                    try:
                        v = float(value)
                        # Early epochs may yield NaN precision/metrics (e.g., empty predictions); treat as 0 for fitness.
                        if math.isnan(v):
                            v = 0.0
                        if not math.isfinite(v):
                            continue
                        score += float(w) * v
                        used = True
                    except (TypeError, ValueError):
                        continue
                if used:
                    return score
                # 指定权重但当前没产出对应指标时，按需回退到 val_loss，保证仍能产出 checkpoint_best。
                if best_fallback_to_val_loss:
                    try:
                        return float(val_stats.get("val_loss"))
                    except (TypeError, ValueError):
                        return None
                return None

            # 2) 单一 monitor
            monitor_key = _normalize_best_key(best_monitor)
            value = val_stats.get(monitor_key, None)
            if value is None and monitor_key != best_monitor:
                value = val_stats.get(best_monitor, None)
            if value is None:
                if best_fallback_to_val_loss and monitor_key != "val_loss":
                    try:
                        return float(val_stats.get("val_loss"))
                    except (TypeError, ValueError):
                        return None
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        resolved_best_mode = best_mode_cfg
        if resolved_best_mode not in {"min", "max"}:
            monitor_key = _normalize_best_key(best_monitor)
            if best_metric_weights:
                resolved_best_mode = "max"
            elif monitor_key.endswith("loss"):
                resolved_best_mode = "min"
            else:
                resolved_best_mode = "max"
        best_ckpt_score = float("inf") if resolved_best_mode == "min" else float("-inf")

        # 仅保留 latest/best：禁用周期性 epoch checkpoint，并在每个 epoch 后清理已有的 checkpoint_*.pth。
        if keep_only_best_and_last:
            checkpoint_interval = 0

        best_val = float("inf")
        stale_epochs = 0

        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()
            if self.distributed and hasattr(self.train_dataloader, "sampler"):
                sampler = getattr(self.train_dataloader, "sampler", None)
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)

            train_stats = self.train_one_epoch(epoch=epoch, clip_max_norm=clip_max_norm)
            model_for_sanitize = self.model.module if hasattr(self.model, "module") else self.model
            bn_fix_model = self._sanitize_bn_running_stats(model_for_sanitize)
            bn_fix_ema = {"fixed_modules": 0, "fixed_values": 0, "examples": []}
            if self.ema is not None and getattr(self.ema, "ema", None) is not None:
                bn_fix_ema = self._sanitize_bn_running_stats(self.ema.ema)
            if bn_fix_model["fixed_modules"] > 0 or bn_fix_ema["fixed_modules"] > 0:
                logging.warning(
                    "检测到并修复 BN 非有限统计量: model(modules=%d, values=%d, examples=%s), "
                    "ema(modules=%d, values=%d, examples=%s)",
                    int(bn_fix_model["fixed_modules"]),
                    int(bn_fix_model["fixed_values"]),
                    bn_fix_model["examples"],
                    int(bn_fix_ema["fixed_modules"]),
                    int(bn_fix_ema["fixed_values"]),
                    bn_fix_ema["examples"],
                )
            val_stats = self._validate_epoch(epoch=epoch)
            current_best_score = _compute_best_score(val_stats)
            is_new_best = False
            if epoch >= best_start_epoch and current_best_score is not None:
                if resolved_best_mode == "min":
                    is_new_best = current_best_score < best_ckpt_score - best_eps
                else:
                    is_new_best = current_best_score > best_ckpt_score + best_eps
                if is_new_best:
                    best_ckpt_score = current_best_score
                    _append_best_update(epoch=epoch, score=current_best_score, val_stats=val_stats)

            model_for_log = self.model.module if hasattr(self.model, "module") else self.model
            gate_stats = self._collect_fusion_gate_stats(model_for_log) if model_for_log is not None else {}
            ms_runtime_stats = (
                self._collect_ms_fusion_runtime_stats(model_for_log) if model_for_log is not None else {}
            )
            ms_shift_stats = (
                self._collect_ms_ref_shift_stats(model_for_log) if model_for_log is not None else {}
            )

            prefixed_train = {}
            for k, v in (train_stats or {}).items():
                if k.startswith("train_"):
                    prefixed_train[k] = v
                else:
                    prefixed_train[f"train_{k}"] = v
            prefixed_val = {}
            for k, v in (val_stats or {}).items():
                if k.startswith("val_"):
                    prefixed_val[k] = v
                else:
                    prefixed_val[f"val_{k}"] = v

            log_stats = {
                "epoch": epoch,
                **gate_stats,
                **ms_runtime_stats,
                **ms_shift_stats,
                "bn_sanitize_model_modules": int(bn_fix_model["fixed_modules"]),
                "bn_sanitize_model_values": int(bn_fix_model["fixed_values"]),
                "bn_sanitize_ema_modules": int(bn_fix_ema["fixed_modules"]),
                "bn_sanitize_ema_values": int(bn_fix_ema["fixed_values"]),
                **prefixed_train,
                **prefixed_val,
                "epoch_time": str(datetime.timedelta(seconds=int(time.time() - epoch_start))),
            }

            if is_main_process():
                # 将结构化指标也写入 log.log，避免额外生成 log.txt。
                log_file = output_dir / "log.log"
                log_line = json.dumps(log_stats, ensure_ascii=False)
                try:
                    with log_file.open("a", encoding="utf-8") as f:
                        f.write(log_line + "\n")
                except Exception as exc:
                    logging.warning("写入日志文件失败: %s", exc)
                logging.info("Epoch metric summary: %s", log_line)
                if self.csv_metric_sink is not None:
                    try:
                        self.csv_metric_sink.update(log_stats)
                    except Exception as exc:
                        logging.warning("写入 metrics.csv 失败（忽略继续训练）：%s", exc)

            if self.wandb_logger:
                self.wandb_logger.update(log_stats)

            # 清空 ms_fusion 的 epoch 累计，避免跨 epoch 混叠。
            model_for_reset = model_for_log
            if model_for_reset is not None:
                for _, module in model_for_reset.named_modules():
                    if getattr(module, "ms_cross_attn", None) is None:
                        continue
                    for buf_name in (
                        "_ms_fusion_sum_rgb_abs",
                        "_ms_fusion_sum_ms_abs",
                        "_ms_fusion_sum_ratio",
                        "_ms_fusion_count",
                    ):
                        buf = getattr(module, buf_name, None)
                        if torch.is_tensor(buf):
                            buf.zero_()

            if (not dont_save) and is_main_process():
                model_to_save = self.model.module if hasattr(self.model, "module") else self.model
                # 防止保存已损坏（NaN/Inf）的 checkpoint，避免后续评估“指标全 0”但不易察觉。
                has_nonfinite = False
                for k, v in model_to_save.state_dict().items():
                    if torch.is_tensor(v) and v.dtype.is_floating_point and (not torch.isfinite(v).all()):
                        # RT-DETRv4/DFINETransformer 会用 +inf 作为无效 anchor 的哨兵值（按设计如此），
                        # 不应阻止 checkpoint 保存。仅在“全为有限或仅含 inf（无 NaN）且命中特定 key”时放行。
                        if k.endswith("decoder.anchors") and torch.isinf(v).any() and (not torch.isnan(v).any()):
                            continue
                        logging.error("检测到非有限权重，跳过保存 checkpoint：%s", k)
                        has_nonfinite = True
                        break
                if not has_nonfinite:
                    checkpoint = {
                        "model": model_to_save.state_dict(),
                        "optimizer": self.optimizer.state_dict() if self.optimizer else None,
                        "lr_scheduler": self.scheduler.state_dict() if self.scheduler else None,
                        "epoch": epoch,
                        "config": self.cfg,
                    }
                    if self.ema is not None:
                        checkpoint["ema"] = self.ema.ema.state_dict()
                    torch.save(checkpoint, output_dir / "checkpoint.pth")
                    if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
                        torch.save(checkpoint, output_dir / f"checkpoint_{epoch:04d}.pth")
                    if is_new_best:
                        torch.save(checkpoint, output_dir / "checkpoint_best.pth")
                    if keep_only_best_and_last:
                        self._prune_numbered_checkpoints(output_dir)

            # 绘制/保存指标曲线与可视化
            if is_main_process() and self.metric_plot_sink is not None:
                if (not metric_charts_on_best_only) or is_new_best:
                    results_json = (val_stats or {}).get("results_json") or (val_stats or {}).get("results_json_segm")
                    self._save_last_eval_metric_charts(output_dir=output_dir, prefix="val", results_json=results_json)
                try:
                    self.metric_plot_sink.update(log_stats)
                    self.metric_plot_sink.save()
                except Exception as exc:
                    logging.warning("保存训练曲线失败: %s", exc)

            if val_stats and "val_loss" in val_stats:
                current = val_stats["val_loss"]
                if current < best_val - min_delta:
                    best_val = current
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                if use_early_stop and stale_epochs >= patience:
                    logging.info("早停触发，在 epoch=%d 结束训练。", epoch)
                    break

        # 训练结束后可选：自动在 test split 上跑一次评测（通常用于最终汇报指标）。
        run_test_after_train = bool(get_config(train_cfg, "run_test_after_train", False))
        test_after_train_ckpt = str(get_config(train_cfg, "test_after_train_ckpt", "best") or "best").strip().lower()
        if run_test_after_train and self.test_dataloader is not None and self.test_dataset is not None:
            ckpt_path = None
            if test_after_train_ckpt in {"best", "auto"}:
                candidate = output_dir / "checkpoint_best.pth"
                if candidate.is_file():
                    ckpt_path = candidate
                else:
                    candidate = output_dir / "checkpoint.pth"
                    ckpt_path = candidate if candidate.is_file() else None
            elif test_after_train_ckpt in {"last", "latest"}:
                candidate = output_dir / "checkpoint.pth"
                ckpt_path = candidate if candidate.is_file() else None
            elif test_after_train_ckpt in {"none", "skip", "false", "0"}:
                ckpt_path = None
            else:
                candidate = Path(test_after_train_ckpt).expanduser()
                if not candidate.is_absolute():
                    candidate = (output_dir / candidate).expanduser()
                ckpt_path = candidate if candidate.is_file() else None

            if ckpt_path is not None:
                try:
                    logging.info("训练结束自动测试：加载权重 %s", ckpt_path)
                    _register_yolo_pickle_alias()
                    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    model_state = state.get("model", state)
                    if hasattr(model_state, "state_dict"):
                        model_state = model_state.state_dict()
                    model_state = _remap_ultralytics_state_dict(model_state)
                    model_ref = self.model.module if hasattr(self.model, "module") else self.model
                    compatible_state = _filter_compatible_state_dict(model_ref, model_state)
                    skipped = len(model_state) - len(compatible_state)
                    if skipped:
                        logging.warning("自动测试权重有 %d 个参数形状不匹配，已跳过加载。", skipped)
                    missing, unexpected = model_ref.load_state_dict(compatible_state, strict=False)
                    if missing or unexpected:
                        logging.info("自动测试：模型权重缺失 %d 项, 额外 %d 项", len(missing), len(unexpected))

                    if self.ema is not None and isinstance(state, Mapping) and state.get("ema") is not None:
                        ema_state = state.get("ema")
                        if hasattr(ema_state, "state_dict"):
                            ema_state = ema_state.state_dict()
                        if isinstance(ema_state, Mapping):
                            ema_state = _remap_ultralytics_state_dict(ema_state)
                            compatible_ema = _filter_compatible_state_dict(self.ema.ema, ema_state)
                            skipped_ema = len(ema_state) - len(compatible_ema)
                            if skipped_ema:
                                logging.warning("自动测试 EMA 权重有 %d 个参数形状不匹配，已跳过加载。", skipped_ema)
                            self.ema.ema.load_state_dict(compatible_ema, strict=False)
                except Exception as exc:
                    logging.warning("训练结束自动测试：加载权重失败，将使用当前模型继续评测: %s", exc)
            else:
                logging.info("训练结束自动测试：未找到可用 checkpoint，将使用当前模型继续评测。")

            test_stats = (
                self._validate_epoch(
                    epoch=None,
                    dataloader=self.test_dataloader,
                    dataset=self.test_dataset,
                    split_prefix="test",
                )
                or {}
            )

            prefixed_test = {}
            for k, v in (test_stats or {}).items():
                if str(k).startswith("test_"):
                    prefixed_test[str(k)] = v
                elif str(k).startswith("val_"):
                    prefixed_test[f"test_{str(k)[len('val_'):]}"] = v
                else:
                    prefixed_test[f"test_{k}"] = v

            log_stats = {"mode": "test", **prefixed_test}
            if is_main_process():
                try:
                    log_file = output_dir / "log.log"
                    with log_file.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats, ensure_ascii=False) + "\n")
                except Exception as exc:
                    logging.warning("写入自动测试日志失败: %s", exc)
                logging.info("Auto test metric summary: %s", json.dumps(log_stats, ensure_ascii=False))
                try:
                    metrics_file = output_dir / "metrics.json"
                    with metrics_file.open("w", encoding="utf-8") as f:
                        json.dump(prefixed_test, f, ensure_ascii=False, indent=2)
                except Exception as exc:
                    logging.warning("保存自动测试指标失败: %s", exc)
                results_json = test_stats.get("results_json") or test_stats.get("results_json_segm")
                self._save_last_eval_metric_charts(output_dir=output_dir, prefix="test", results_json=results_json)
                if self.wandb_logger:
                    self.wandb_logger.update(log_stats)

        if self.wandb_logger:
            self.wandb_logger.close()

    def train_one_epoch(self, *, epoch: int, clip_max_norm: float = 0.0):
        train_cfg: Dict[str, Any] = get_config(self.cfg, "train", {}) or {}
        data_args = self._ensure_data_args()
        grad_accum_steps = max(1, int(get_config(train_cfg, "grad_accum_steps", 1)))
        device_str = str(self.device)
        amp_enabled = bool(get_config(train_cfg, "amp", False)) and device_str.startswith("cuda") and torch.cuda.is_available()
        print_freq = int(get_config(train_cfg, "print_freq", 10))
        batch_size = int(get_config(train_cfg, "batch_size", 1)) * grad_accum_steps
        multi_scale = bool(get_config(train_cfg, "multi_scale", False))
        expanded_scales = bool(get_config(train_cfg, "expanded_scales", False))
        skip_random_resize = bool(get_config(train_cfg, "do_random_resize_via_padding", False))
        num_windows = int(get_config(data_args, "num_windows", 4))
        patch_size = int(get_config(data_args, "patch_size", 16))

        self.model.train()
        self.criterion.train()
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("class_error", SmoothedValue(window_size=1, fmt="{value:.2f}"))
        # 兼容 DETR/YOLO 两套损失命名：DETR 输出 loss_ce/loss_bbox/loss_giou，YOLO 输出 loss_obj/loss_cls 等。
        metric_logger.add_meter("loss_ce", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("loss_bbox", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("loss_giou", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("loss_mask_ce", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("loss_mask_dice", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("loss_obj", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("loss_cls", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("loss_distill", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        align_loss_keys = (
            "loss_deform_align",
            "loss_deform_offset",
            "loss_deform_attn",
            "loss_deform_attn_entropy",
            "loss_ms_group_align",
            "loss_ms_group_offset",
            "loss_ms_group_attn",
            "loss_ms_group_attn_entropy",
        )
        for key in align_loss_keys:
            metric_logger.add_meter(key, SmoothedValue(window_size=1, fmt="{value:.4f}"))

        scaler = _make_grad_scaler(enabled=amp_enabled)
        effective_batch = batch_size
        assert effective_batch % grad_accum_steps == 0, "batch_size 应能被 grad_accum_steps 整除。"
        sub_batch = effective_batch // grad_accum_steps

        train_loader = self.build_train_dataloader()
        # 预填充一次，避免首个打印时出现除零
        init_lr = 0.0
        if self.optimizer and self.optimizer.param_groups:
            init_lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
        metric_logger.update(lr=init_lr, class_error=0.0)
        header = f"Epoch: [{epoch}]"

        self.optimizer.zero_grad()
        for step, (samples, targets) in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
            # 手动记录 warmup 步进，配合 scheduler 提供的 warmup lambda。
            if self._warmup_steps > 0 and self._global_step < self._warmup_steps and self.optimizer:
                warmup_factor = float(self._global_step + 1) / float(self._warmup_steps)
                for base_lr, group in zip(self._base_lrs, self.optimizer.param_groups):
                    group["lr"] = base_lr * warmup_factor

            if multi_scale and not skip_random_resize:
                scales = compute_multi_scale_scales(
                    getattr(data_args, "img_size", 640),
                    expanded_scales=expanded_scales,
                    patch_size=patch_size,
                    num_windows=num_windows,
                )
                random.seed(epoch * len(train_loader) + step)
                scale = random.choice(scales)
                samples = self._resize_samples(samples, (scale, scale))

            # 确保输入尺寸满足窗口化 ViT 要求（patch_size*num_windows 的倍数）
            block_size = int(patch_size) * int(num_windows)
            samples = self._align_samples_to_block(samples, block_size)

            last_loss_dict: Dict[str, torch.Tensor] | None = None
            for i in range(grad_accum_steps):
                start_idx = i * sub_batch
                end_idx = start_idx + sub_batch
                if end_idx > len(targets):
                    break
                cur_samples = self._slice_samples(samples, start_idx, end_idx)
                cur_samples = self._move_samples_to_device(cur_samples, self.device)
                cur_targets = [
                    {k: v.to(self.device) for k, v in t.items()} for t in targets[start_idx:end_idx]
                ]

                with autocast(enabled=amp_enabled, **_AMP_DEVICE_KW):
                    outputs = self.model(cur_samples, cur_targets)
                    loss_dict = self.criterion(outputs, cur_targets)
                    loss = self._sum_loss_dict(loss_dict) / float(grad_accum_steps)

                scaler.scale(loss).backward()
                last_loss_dict = loss_dict

            if last_loss_dict is None:
                continue

            reduced = reduce_dict({k: v for k, v in last_loss_dict.items() if torch.is_tensor(v)})
            # 训练日志优先打印 loss_total，避免把各分项 + loss_total 重复相加导致数值虚高。
            if "loss_total" in reduced and torch.is_tensor(reduced["loss_total"]):
                loss_value = float(reduced["loss_total"].item())
            else:
                reduced_losses = {k: v for k, v in reduced.items() if "loss" in k}
                loss_value = float(sum(v for v in reduced_losses.values()).item()) if reduced_losses else 0.0

            if not math.isfinite(loss_value):
                logging.warning("Loss 非有限值，跳过该 batch: %s", {k: v.detach().cpu() for k, v in reduced.items()})
                self.optimizer.zero_grad(set_to_none=True)
                continue

            if clip_max_norm > 0:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_max_norm)

            scaler.step(self.optimizer)
            scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            if self.ema is not None:
                try:
                    self.ema.update(self.model)
                except Exception as exc:
                    logging.warning("EMA 更新失败，将暂时跳过本次 EMA: %s", exc)
            self.optimizer.zero_grad()

            metric_logger.update(loss=loss_value, train_loss=loss_value)
            class_err = reduced.get("class_error") or reduced.get("set_criterion.class_error")
            if class_err is not None:
                metric_logger.update(class_error=float(class_err))
            if self.optimizer.param_groups:
                metric_logger.update(lr=self.optimizer.param_groups[0].get("lr", 0.0))
            # 记录 DETR/YOLO 分项损失。CompositeCriterion 会加前缀，如 set_criterion.loss_ce。
            key_aliases = (
                ("loss_ce", ("loss_ce", "set_criterion.loss_ce")),
                ("loss_bbox", ("loss_bbox", "set_criterion.loss_bbox")),
                ("loss_giou", ("loss_giou", "set_criterion.loss_giou")),
                ("loss_mask_ce", ("loss_mask_ce", "set_criterion.loss_mask_ce")),
                ("loss_mask_dice", ("loss_mask_dice", "set_criterion.loss_mask_dice")),
                ("loss_obj", ("loss_obj", "yolo.loss_obj")),
                ("loss_cls", ("loss_cls", "yolo.loss_cls")),
                ("loss_distill", ("loss_distill",)),
            )
            for meter_name, candidates in key_aliases:
                for cand in candidates:
                    if cand in reduced:
                        metric_logger.update(**{meter_name: float(reduced[cand])})
                        break
            for key in align_loss_keys:
                if key in reduced:
                    metric_logger.update(**{key: float(reduced[key])})
            self._global_step += 1

        metric_logger.synchronize_between_processes()
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        stats.setdefault("train_loss", stats.get("loss", 0.0))
        if self.optimizer and self.optimizer.param_groups:
            stats["train_lr"] = self.optimizer.param_groups[0].get("lr", 0.0)
        return stats

    @torch.inference_mode()
    def _validate_epoch(
        self,
        *,
        epoch: int | None = None,
        dataloader=None,
        dataset=None,
        split_prefix: str | None = None,
    ):
        eval_loader = dataloader if dataloader is not None else self.validation_dataloader
        eval_dataset = dataset if dataset is not None else self.validation_dataset
        if eval_loader is None or eval_dataset is None or self.model is None or self.criterion is None:
            return {}
        train_cfg: Dict[str, Any] = get_config(self.cfg, "train", {}) or {}
        model_cfg: Dict[str, Any] = get_config(self.cfg, "model", {}) or {}
        print_freq = int(get_config(train_cfg, "val_print_freq", get_config(train_cfg, "print_freq", 10)))
        runtime_cfg: Dict[str, Any] = self.runtime_cfg or {}
        output_root = get_config(runtime_cfg, "output_root", None)
        output_dir = Path(output_root).expanduser() if output_root else None
        device_str = str(self.device)
        amp_enabled = bool(get_config(train_cfg, "amp", False)) and device_str.startswith("cuda") and torch.cuda.is_available()
        fp16_eval = bool(get_config(runtime_cfg, "fp16_eval", False))
        seg_head = bool(get_config(train_cfg, "segmentation_head", False))
        mode = str(get_config(self.cfg, "mode", "train"))
        if split_prefix is None:
            split_prefix = "test" if mode == "test" else "val"
        # 缓存本轮验证产物（metric_charts），由 train/test 决定何时落盘（例如仅在 best 时写图）。
        self._last_eval_artifacts = {"split_prefix": split_prefix}
        data_args = self._ensure_data_args()
        chart_cfg: Dict[str, Any] = get_config(train_cfg, "metric_charts", {}) or {}
        infer_cfg: Dict[str, Any] = get_config(chart_cfg, "inference", {}) or {}
        infer_enabled = bool(get_config(infer_cfg, "enabled", False))
        infer_warmup = int(get_config(infer_cfg, "warmup_batches", 5))
        infer_max_batches = int(get_config(infer_cfg, "max_batches", 50))
        infer_include_postprocess = bool(get_config(infer_cfg, "include_postprocess", True))
        infer_include_loss = bool(get_config(infer_cfg, "include_loss", False))
        infer_include_flops = bool(get_config(infer_cfg, "include_flops", True))
        infer_batches = 0
        infer_images = 0
        infer_total_s = 0.0
        infer_gflops = None
        infer_flops_input_hw = None
        infer_flops_tried = False

        confusion_cfg = get_config(train_cfg, "confusion_matrix", {}) or {}
        confusion_enabled = bool(get_config(confusion_cfg, "enabled", True))
        confusion_iou_thr = float(get_config(confusion_cfg, "iou_threshold", 0.5))
        confusion_score_thr = float(get_config(confusion_cfg, "score_threshold", 0.0))
        confusion_normalize = bool(get_config(confusion_cfg, "normalize", True))
        num_obj_classes = int(get_config(train_cfg, "num_classes", get_config(model_cfg, "num_classes", 0)) or 0)
        bg_index = num_obj_classes if num_obj_classes > 0 else 0
        confusion = None
        cm_class_names = None

        def _nan_to_none_list(array):
            sanitized = []
            for value in array:
                if value is None:
                    sanitized.append(None)
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    sanitized.append(None)
                    continue
                sanitized.append(None if np.isnan(numeric) else numeric)
            return sanitized

        def coco_extended_metrics(coco_eval):
            """
            Safe version: ignores the –1 sentinel entries so precision/F1 never explode.
            """

            iou_thrs, rec_thrs = coco_eval.params.iouThrs, coco_eval.params.recThrs

            def _find_iou_idx(target: float) -> int | None:
                matches = np.argwhere(np.isclose(iou_thrs, target))
                if matches.size <= 0:
                    return None
                return int(matches.reshape(-1)[0])

            iou50_idx = _find_iou_idx(0.50)
            iou75_idx = _find_iou_idx(0.75)
            area_idx, maxdet_idx = 0, 2

            P = coco_eval.eval.get("precision")
            S = coco_eval.eval.get("scores")

            if P is None or iou50_idx is None:
                return {
                    "class_map": [],
                    "map": float("nan"),
                    "map@75": float("nan"),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "f1": float("nan"),
                    "curves": {
                        "recall": [],
                        "precision": [],
                        "confidence": [],
                        "f1": []
                    }
                }

            prec_raw = P[iou50_idx, :, :, area_idx, maxdet_idx]

            prec = prec_raw.copy().astype(float)
            prec[prec < 0] = np.nan

            precision_curve = np.nanmean(prec, axis=1)
            recall_curve = rec_thrs.astype(float)

            if S is not None:
                scores_raw = S[iou50_idx, :, :, area_idx, maxdet_idx].astype(float)
                scores_raw[prec_raw < 0] = np.nan
                confidence_curve = np.nanmean(scores_raw, axis=1)
            else:
                confidence_curve = np.full_like(recall_curve, np.nan, dtype=np.float64)

            f1_cls   = 2 * prec * rec_thrs[:, None] / (prec + rec_thrs[:, None])
            f1_macro = np.nanmean(f1_cls, axis=1)

            def _closest_score_idx(scores, target: float | None) -> int | None:
                if scores is None or target is None:
                    return None
                try:
                    target_val = float(target)
                except (TypeError, ValueError):
                    return None
                if np.isnan(target_val):
                    return None
                score_arr = np.asarray(scores, dtype=float)
                if score_arr.size == 0:
                    return None
                diffs = np.abs(score_arr - target_val)
                diffs[np.isnan(diffs)] = np.inf
                if np.all(np.isinf(diffs)):
                    return None
                return int(np.argmin(diffs))

            score_mode = str(get_config(chart_cfg, "score_select", "f1")).strip().lower()
            use_score_target = score_mode in {"threshold", "score", "fixed"}
            score_target = None
            if use_score_target:
                score_target = get_config(chart_cfg, "score_threshold", 0.5)
                try:
                    score_target = float(score_target)
                except (TypeError, ValueError):
                    score_target = 0.5

            selected_j = None
            score_target_used = False
            if use_score_target and S is not None:
                selected_j = _closest_score_idx(confidence_curve, score_target)
                if selected_j is not None:
                    score_target_used = True

            if selected_j is None:
                selected_j = 0 if np.all(np.isnan(f1_macro)) else int(np.nanargmax(f1_macro))

            macro_precision = float(np.nanmean(prec[selected_j]))
            macro_recall    = float(rec_thrs[selected_j])
            denom_macro = macro_precision + macro_recall
            macro_f1 = float(2 * macro_precision * macro_recall / denom_macro) if denom_macro > 0 else float("nan")

            if score_target_used:
                score_thr = float(score_target)
            elif S is not None:
                score_vec = S[iou50_idx, selected_j, :, area_idx, maxdet_idx].astype(float)
                score_vec[prec_raw[selected_j] < 0] = np.nan
                score_thr = float(np.nanmean(score_vec))
            else:
                score_thr = float("nan")

            # COCOeval.stats:
            # [0] AP@[.5:.95] all, [1] AP@0.5, [2] AP@0.75, [3] APs, [4] APm, [5] APl,
            # [6] AR@1, [7] AR@10, [8] AR@100, [9] ARs, [10] ARm, [11] ARl
            map_50_95 = float(coco_eval.stats[0])
            map_50 = float(coco_eval.stats[1])
            map_75 = float(coco_eval.stats[2])
            map_s = float(coco_eval.stats[3])
            map_m = float(coco_eval.stats[4])
            map_l = float(coco_eval.stats[5])
            ar_1 = float(coco_eval.stats[6])
            ar_10 = float(coco_eval.stats[7])
            ar_100 = float(coco_eval.stats[8])
            ar_s = float(coco_eval.stats[9])
            ar_m = float(coco_eval.stats[10])
            ar_l = float(coco_eval.stats[11])

            per_class = []
            cat_ids = coco_eval.params.catIds
            cat_id_to_name = {c["id"]: c["name"] for c in coco_eval.cocoGt.loadCats(cat_ids)}
            for k, cid in enumerate(cat_ids):
                p_slice = P[:, :, k, area_idx, maxdet_idx]
                valid   = p_slice > -1
                ap_50_95 = float(p_slice[valid].mean()) if valid.any() else float("nan")
                ap_50    = float(p_slice[iou50_idx][p_slice[iou50_idx] > -1].mean()) if (p_slice[iou50_idx] > -1).any() else float("nan")
                ap_75    = (
                    float(p_slice[iou75_idx][p_slice[iou75_idx] > -1].mean())
                    if iou75_idx is not None and (p_slice[iou75_idx] > -1).any()
                    else float("nan")
                )

                score_k = None
                score_idx_k = None
                if S is not None:
                    try:
                        score_k = S[iou50_idx, :, k, area_idx, maxdet_idx].astype(float)
                        score_k[prec_raw[:, k] < 0] = np.nan
                    except Exception:
                        score_k = None
                if use_score_target:
                    score_idx_k = _closest_score_idx(score_k, score_target)
                if score_idx_k is None:
                    f1_k = f1_cls[:, k].astype(float)
                    if np.all(np.isnan(f1_k)):
                        continue
                    score_idx_k = int(np.nanargmax(f1_k))

                pc = float(prec[score_idx_k, k]) if prec_raw[score_idx_k, k] > -1 else float("nan")
                rc = float(rec_thrs[score_idx_k])

                if np.isnan(ap_50_95) or np.isnan(ap_50) or np.isnan(pc) or np.isnan(rc):
                    continue

                denom_k = pc + rc
                f1_value = float(2 * pc * rc / denom_k) if denom_k > 0 else float("nan")

                per_class.append({
                    "class"      : cat_id_to_name[int(cid)],
                    "map@50:95"  : ap_50_95,
                    "map@50"     : ap_50,
                    "map@75"     : ap_75,
                    "precision"  : pc,
                    "recall"     : rc,
                    "f1"         : f1_value,
                })

            denom_all = macro_precision + macro_recall
            macro_f1_all = float(2 * macro_precision * macro_recall / denom_all) if denom_all > 0 else float("nan")

            per_class.append({
                "class"     : "all",
                "map@50:95" : map_50_95,
                "map@50"    : map_50,
                "map@75"    : map_75,
                "precision" : macro_precision,
                "recall"    : macro_recall,
                "f1"        : macro_f1_all,
            })

            f1_curve = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-8)

            return {
                "class_map": per_class,
                "map"      : map_50,
                "map@50:95": map_50_95,
                "map@75": map_75,
                "map_s": map_s,
                "map_m": map_m,
                "map_l": map_l,
                "ar@1": ar_1,
                "ar@10": ar_10,
                "ar@100": ar_100,
                "ar_s": ar_s,
                "ar_m": ar_m,
                "ar_l": ar_l,
                "precision": macro_precision,
                "recall"   : macro_recall,
                "f1"       : macro_f1,
                "score_threshold": None if np.isnan(score_thr) else score_thr,
                "curves": {
                    "recall": _nan_to_none_list(recall_curve),
                    "precision": _nan_to_none_list(precision_curve),
                    "confidence": _nan_to_none_list(confidence_curve),
                    "f1": _nan_to_none_list(f1_curve),
                }
            }

        eval_model = self.ema.ema if getattr(self, "ema", None) is not None else self.model
        bn_fix_eval = self._sanitize_bn_running_stats(eval_model)
        if bn_fix_eval["fixed_modules"] > 0:
            logging.warning(
                "验证前修复 BN 非有限统计量: split=%s, modules=%d, values=%d, examples=%s",
                split_prefix,
                int(bn_fix_eval["fixed_modules"]),
                int(bn_fix_eval["fixed_values"]),
                bn_fix_eval["examples"],
            )
        eval_model.eval()
        self.criterion.eval()
        metric_logger = MetricLogger(delimiter="  ")
        align_loss_keys = (
            "loss_deform_align",
            "loss_deform_offset",
            "loss_deform_attn",
            "loss_deform_attn_entropy",
            "loss_ms_group_align",
            "loss_ms_group_offset",
            "loss_ms_group_attn",
            "loss_ms_group_attn_entropy",
        )
        for key in align_loss_keys:
            metric_logger.add_meter(key, SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("class_error", SmoothedValue(window_size=1, fmt="{value:.2f}"))

        postprocess = None
        weight_dict = {}
        extras = getattr(self, "criterion_extras", {}) or {}
        if isinstance(extras, dict):
            # 兼容 {"postprocess": xx} 或 {"set_criterion": {"postprocess": xx}}
            postprocess = extras.get("postprocess")
            weight_dict = extras.get("weight_dict", {}) or {}
            if postprocess is None:
                for val in extras.values():
                    if isinstance(val, dict) and val.get("postprocess") is not None:
                        postprocess = val.get("postprocess")
                    if isinstance(val, dict) and val.get("weight_dict"):
                        weight_dict = val.get("weight_dict")

        iou_types = ("bbox", "segm") if seg_head else ("bbox",)
        coco_evaluator = None
        inverse_category_mapping = None
        postprocess_fn = None
        if isinstance(postprocess, dict):
            postprocess_fn = postprocess.get("bbox") or next(iter(postprocess.values()), None)
        else:
            postprocess_fn = postprocess
        if postprocess_fn is None:
            logging.warning("未找到 postprocess，无法计算 COCO 指标。criterion_extras=%s", list(extras.keys()) if isinstance(extras, dict) else type(extras))

        if postprocess_fn is not None:
            base_ds = None
            try:
                base_ds = get_coco_api_from_dataset(eval_dataset)
            except Exception as exc:
                logging.warning("获取 COCO API 失败，跳过 mAP 计算: %s", exc)
            if base_ds is not None:
                try:
                    coco_evaluator = CocoEvaluator(base_ds, iou_types)
                except Exception as exc:
                    logging.warning("初始化 COCO 评估器失败，跳过 mAP 计算: %s", exc)
                else:
                    if getattr(data_args, "remap_mscoco_category", False) and hasattr(base_ds, "cats"):
                        class_names = getattr(data_args, "class_names", None) or []
                        ordered_names = [
                            str(name).strip().lower()
                            for name in class_names
                            if name and str(name).strip().lower() not in {"background", "_background_"}
                        ]
                        if ordered_names:
                            remapped_ids = []
                            for name in ordered_names:
                                matched = None
                                for cat_id, meta in base_ds.cats.items():
                                    if str(meta.get("name", "")).lower() == name:
                                        matched = int(cat_id)
                                        break
                                if matched is not None:
                                    remapped_ids.append(matched)
                            # Keep behavior consistent with datasets.coco.CocoDetection:
                            # only apply the user-provided ordering when all names are matched.
                            if remapped_ids and len(remapped_ids) == len(ordered_names):
                                inverse_category_mapping = torch.as_tensor(remapped_ids, dtype=torch.int64, device=self.device)
                            elif remapped_ids:
                                logging.warning(
                                    "class_names=%s does not fully match dataset categories; matched %d / %d. "
                                    "Falling back to default category ordering for evaluation.",
                                    ordered_names,
                                    len(remapped_ids),
                                    len(ordered_names),
                                )
                        if inverse_category_mapping is None:
                            filtered_ids = []
                            for cat_id, meta in base_ds.cats.items():
                                name = str(meta.get("name", "")).lower()
                                if name in {"background", "_background_"}:
                                    continue
                                filtered_ids.append(int(cat_id))
                            filtered_ids = sorted(filtered_ids)
                            if filtered_ids:
                                inverse_category_mapping = torch.as_tensor(filtered_ids, dtype=torch.int64, device=self.device)
                        # Keep COCOeval consistent with the remapped category subset:
                        # when class_names selects a subset, evaluate only those category_ids.
                        if inverse_category_mapping is not None and coco_evaluator is not None:
                            eval_cat_ids = [int(x) for x in inverse_category_mapping.detach().cpu().tolist()]
                            for coco_eval in coco_evaluator.coco_eval.values():
                                coco_eval.params.catIds = list(eval_cat_ids)

        samples_cfg: Dict[str, Any] = get_config(chart_cfg, "samples", {}) or {}
        samples_enabled = bool(get_config(samples_cfg, "enabled", False))
        samples_num = int(get_config(samples_cfg, "num_samples", 8))
        samples_score_thr = float(get_config(samples_cfg, "score_threshold", 0.3))
        samples_max_dets = int(get_config(samples_cfg, "max_dets", 50))
        # 多光谱可视化：默认取第 4 通道（1-based），作为灰度底图。
        samples_msi_channel = int(get_config(samples_cfg, "msi_channel", 4))
        samples_saved = 0
        samples_rgb_root = None
        samples_msi_root = None
        samples_msi_suffix = None
        samples_class_id_to_name = None
        if samples_enabled and samples_num > 0 and is_main_process() and output_dir is not None:
            try:
                # 从 dataset 推断图片根目录（支持 RGB-only / RGB+MSI）
                ds = eval_dataset
                for _ in range(10):
                    if isinstance(ds, torch.utils.data.Subset):
                        ds = ds.dataset
                rgb_dir = getattr(ds, "rgb_dir", None)
                msi_dir = getattr(ds, "msi_dir", None)
                if rgb_dir:
                    samples_rgb_root = Path(rgb_dir)
                if msi_dir:
                    samples_msi_root = Path(msi_dir)

                # 兜底：旧 COCO/RGB-only dataset 常用字段
                if samples_rgb_root is None:
                    for attr in ("root", "img_folder", "img_dir", "image_root"):
                        v = getattr(ds, attr, None)
                        if v:
                            samples_rgb_root = Path(v)
                            break

                # MSI 后缀：优先从数据集 cfg 读取，其次从 data_args
                samples_msi_suffix = getattr(getattr(ds, "cfg", None), "ms_suffix", None) or getattr(data_args, "ms_suffix", None)
                # 类别名优先用连续标签（与训练/可视化一致）；否则回退到 COCO category_id。
                name_list = getattr(data_args, "class_names", None) or []
                if name_list:
                    normalized = [
                        str(name).strip()
                        for name in name_list
                        if name and str(name).strip().lower() not in {"background", "_background_"}
                    ]
                    if normalized:
                        samples_class_id_to_name = {int(i): n for i, n in enumerate(normalized)}
                if samples_class_id_to_name is None and base_ds is not None and hasattr(base_ds, "cats"):
                    samples_class_id_to_name = {int(cid): str(meta.get("name", cid)) for cid, meta in base_ds.cats.items()}
            except Exception:
                samples_rgb_root = None
                samples_msi_root = None
                samples_msi_suffix = None
                samples_class_id_to_name = None

            # 初始化混淆矩阵（不依赖 COCO API，但可借用 cats/class_names 来生成标签名）
            if confusion_enabled and num_obj_classes > 0:
                import numpy as _np  # 局部导入，避免训练侧无需求时额外开销

                confusion = _np.zeros((num_obj_classes + 1, num_obj_classes + 1), dtype=_np.int64)
                names_cfg = getattr(data_args, "class_names", None) or []
                obj_names = []
                if names_cfg:
                    for name in names_cfg:
                        if name is None:
                            continue
                        name_str = str(name).strip()
                        if not name_str:
                            continue
                        if name_str.lower() in {"background", "_background_"}:
                            continue
                        obj_names.append(name_str)
                if obj_names and len(obj_names) != num_obj_classes:
                    logging.warning(
                        "confusion_matrix: class_names 数量=%d 与 num_classes=%d 不一致，将回退到自动命名。",
                        len(obj_names),
                        num_obj_classes,
                    )
                    obj_names = []

                if not obj_names and base_ds is not None and hasattr(base_ds, "cats"):
                    try:
                        if inverse_category_mapping is not None:
                            obj_names = []
                            for mapped_id in inverse_category_mapping.detach().cpu().tolist():
                                meta = base_ds.cats.get(int(mapped_id), {})
                                obj_names.append(str(meta.get("name", f"id_{int(mapped_id)}")))
                        else:
                            obj_names = []
                            for idx in range(num_obj_classes):
                                meta = base_ds.cats.get(int(idx), {})
                                obj_names.append(str(meta.get("name", f"id_{idx}")))
                    except Exception:
                        obj_names = []

                if not obj_names:
                    obj_names = [f"class_{i}" for i in range(num_obj_classes)]
                cm_class_names = [*obj_names, "background"]

        header_base = "Test" if split_prefix == "test" else "Val"
        header = header_base if epoch is None else f"{header_base}: [{epoch}]"

        def _is_letterbox_target(t: Dict[str, Any]) -> bool:
            return bool(
                isinstance(t, dict)
                and t.get("letterbox_pad") is not None
                and t.get("letterbox_scale") is not None
                and t.get("size") is not None
                and t.get("orig_size") is not None
            )

        def _de_letterbox_xyxy(boxes_xyxy: torch.Tensor, t: Dict[str, Any]) -> torch.Tensor:
            if boxes_xyxy.numel() == 0:
                return boxes_xyxy
            pad = t["letterbox_pad"].to(device=boxes_xyxy.device, dtype=boxes_xyxy.dtype).flatten()
            if pad.numel() < 2:
                raise ValueError(f"letterbox_pad must have 2 values, got {tuple(pad.shape)}")
            scale = t["letterbox_scale"].to(device=boxes_xyxy.device, dtype=boxes_xyxy.dtype).flatten()
            if scale.numel() < 1:
                raise ValueError(f"letterbox_scale must have 1 value, got {tuple(scale.shape)}")
            scale = scale[0].clamp_min(1e-12)

            boxes = boxes_xyxy.clone()
            boxes[:, 0::2] -= pad[0]
            boxes[:, 1::2] -= pad[1]
            boxes = boxes / scale

            orig_hw = t["orig_size"].detach().cpu().flatten()
            if orig_hw.numel() < 2:
                raise ValueError(f"orig_size must have 2 values, got {tuple(orig_hw.shape)}")
            h = float(orig_hw[0].item())
            w = float(orig_hw[1].item())
            boxes[:, 0::2].clamp_(0.0, w)
            boxes[:, 1::2].clamp_(0.0, h)
            return boxes

        for batch_idx, (samples, targets) in enumerate(metric_logger.log_every(eval_loader, print_freq, header)):
            samples = self._move_samples_to_device(samples, self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            # 验证/测试同样对齐到 block_size
            block_size = int(getattr(data_args, "patch_size", 16)) * int(getattr(data_args, "num_windows", 4))
            samples = self._align_samples_to_block(samples, block_size)

            if infer_enabled and infer_include_flops and (not infer_flops_tried) and is_main_process():
                infer_flops_tried = True
                try:
                    flops_samples = self._slice_samples(samples, 0, 1)
                    infer_gflops = try_compute_gflops(eval_model, flops_samples, per_image=True)
                    if infer_gflops is not None:
                        nested = next(iter(flops_samples.values())) if isinstance(flops_samples, dict) else flops_samples
                        infer_flops_input_hw = [int(x) for x in nested.tensors.shape[-2:]]
                except Exception as exc:
                    logging.warning("统计 FLOPs 失败（忽略继续验证）：%s", exc)
                    infer_gflops = None
                    infer_flops_input_hw = None

            if fp16_eval:
                if isinstance(samples, dict):
                    for nested in samples.values():
                        nested.tensors = nested.tensors.half()
                else:
                    samples.tensors = samples.tensors.half()

            # 推理 profile：默认统计 forward(+postprocess) 的时间（不含 loss / COCOeval update）
            do_profile = infer_enabled and infer_batches < infer_max_batches
            if do_profile and batch_idx < infer_warmup:
                do_profile = False

            if do_profile and device_str.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter() if do_profile else None

            with autocast(enabled=amp_enabled, **_AMP_DEVICE_KW):
                outputs = eval_model(samples)

            if do_profile and device_str.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter() if do_profile else None

            if fp16_eval:
                for key, value in list(outputs.items()):
                    if isinstance(value, dict):
                        outputs[key] = {k: v.float() for k, v in value.items()}
                    elif isinstance(value, list):
                        outputs[key] = [{k: v.float() for k, v in item.items()} for item in value]
                    elif torch.is_tensor(value):
                        outputs[key] = value.float()

            # profile 计时尽量排除 loss（包含 reduce_dict 的分布式同步），默认不计入推理耗时。
            reduced = None
            if infer_include_loss:
                loss_dict = self.criterion(outputs, targets)
                tensor_losses = {k: v for k, v in loss_dict.items() if torch.is_tensor(v)}
                reduced = reduce_dict(tensor_losses)

            if coco_evaluator is not None and postprocess_fn is not None:
                use_letterbox = bool(targets) and all(_is_letterbox_target(t) for t in targets)
                postprocess_sizes = (
                    torch.stack([t["size"] for t in targets], dim=0)
                    if use_letterbox
                    else torch.stack([t["orig_size"] for t in targets], dim=0)
                )
                results_all = postprocess_fn(outputs, postprocess_sizes)
                if use_letterbox:
                    # postprocess 产出的是 letterbox 坐标（像素），这里反算回原图坐标用于 COCO eval。
                    remapped = []
                    for t, pred in zip(targets, results_all):
                        boxes = pred.get("boxes")
                        if boxes is None:
                            remapped.append(pred)
                            continue
                        mapped = dict(pred)
                        mapped["boxes"] = _de_letterbox_xyxy(boxes, t)
                        remapped.append(mapped)
                    results_all = remapped

                if do_profile and infer_include_postprocess:
                    if device_str.startswith("cuda") and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    infer_total_s += float(t2 - float(t0)) if t0 is not None else 0.0
                    batch_n = int(len(targets))
                    infer_images += batch_n
                    infer_batches += 1
                elif do_profile and not infer_include_postprocess:
                    infer_total_s += float(t1 - float(t0)) if (t0 is not None and t1 is not None) else 0.0
                    batch_n = int(len(targets))
                    infer_images += batch_n
                    infer_batches += 1

                # 混淆矩阵（基于 IoU 匹配，包含 background 行/列）
                if confusion is not None and cm_class_names is not None:
                    try:
                        from utils.box_ops import box_cxcywh_to_xyxy, box_iou

                        for t, pred in zip(targets, results_all):
                            gt_labels = t.get("labels")
                            gt_boxes = t.get("boxes")
                            if gt_labels is None or gt_boxes is None:
                                continue
                            # targets 的 boxes 是归一化 cxcywh，转到原图绝对坐标 xyxy
                            gt_boxes = gt_boxes.detach().cpu()
                            gt_labels = gt_labels.detach().cpu().long()
                            if gt_boxes.numel() == 0:
                                gt_xyxy = gt_boxes.reshape(0, 4)
                            else:
                                gt_xyxy = box_cxcywh_to_xyxy(gt_boxes)
                                if _is_letterbox_target(t):
                                    h_in, w_in = [int(x) for x in t["size"].detach().cpu().tolist()]
                                    scale = torch.tensor([w_in, h_in, w_in, h_in], dtype=gt_xyxy.dtype)
                                    gt_xyxy = gt_xyxy * scale
                                    gt_xyxy = _de_letterbox_xyxy(gt_xyxy, t)
                                else:
                                    h, w = [int(x) for x in t["orig_size"].detach().cpu().tolist()]
                                    scale = torch.tensor([w, h, w, h], dtype=gt_xyxy.dtype)
                                    gt_xyxy = gt_xyxy * scale

                            pred_boxes = pred.get("boxes")
                            pred_labels = pred.get("labels")
                            pred_scores = pred.get("scores")
                            if pred_boxes is None or pred_labels is None or pred_scores is None:
                                continue
                            pred_boxes = pred_boxes.detach().cpu()
                            pred_labels = pred_labels.detach().cpu().long()
                            pred_scores = pred_scores.detach().cpu()

                            # 过滤 score/非法类别
                            if pred_scores.numel() > 0:
                                keep = pred_scores >= float(confusion_score_thr)
                                keep = keep & (pred_labels >= 0) & (pred_labels < int(num_obj_classes))
                                pred_boxes = pred_boxes[keep]
                                pred_labels = pred_labels[keep]

                            # 空 GT / 空 pred 情况
                            if gt_labels.numel() == 0:
                                for pl in pred_labels.tolist():
                                    confusion[bg_index, int(pl)] += 1
                                continue
                            if pred_labels.numel() == 0:
                                for gl in gt_labels.tolist():
                                    if 0 <= int(gl) < int(num_obj_classes):
                                        confusion[int(gl), bg_index] += 1
                                continue

                            ious, _ = box_iou(pred_boxes, gt_xyxy)  # [Np, Ng]
                            matched_pred = torch.zeros((pred_labels.shape[0],), dtype=torch.bool)
                            matched_gt = torch.zeros((gt_labels.shape[0],), dtype=torch.bool)
                            if ious.numel() > 0:
                                work = ious.clone()
                                while True:
                                    max_val = float(work.max().item())
                                    if max_val < float(confusion_iou_thr):
                                        break
                                    flat_idx = int(torch.argmax(work).item())
                                    ng = int(gt_labels.shape[0])
                                    pi = flat_idx // ng
                                    gi = flat_idx % ng
                                    if matched_pred[pi] or matched_gt[gi]:
                                        work[pi, gi] = -1.0
                                        continue
                                    gl = int(gt_labels[gi].item())
                                    pl = int(pred_labels[pi].item())
                                    if 0 <= gl < int(num_obj_classes) and 0 <= pl < int(num_obj_classes):
                                        confusion[gl, pl] += 1
                                    matched_pred[pi] = True
                                    matched_gt[gi] = True
                                    work[pi, :] = -1.0
                                    work[:, gi] = -1.0

                            # 未匹配的 GT -> background
                            for gi in torch.nonzero(~matched_gt, as_tuple=False).flatten().tolist():
                                gl = int(gt_labels[gi].item())
                                if 0 <= gl < int(num_obj_classes):
                                    confusion[gl, bg_index] += 1
                            # 未匹配的 pred -> background GT
                            for pi in torch.nonzero(~matched_pred, as_tuple=False).flatten().tolist():
                                pl = int(pred_labels[pi].item())
                                if 0 <= pl < int(num_obj_classes):
                                    confusion[bg_index, pl] += 1
                    except Exception as exc:
                        logging.warning("confusion_matrix 统计失败（忽略继续验证）：%s", exc)

                # 样例可视化（GT + Pred）
                if (
                    samples_enabled
                    and (samples_rgb_root is not None or samples_msi_root is not None)
                    and base_ds is not None
                    and samples_saved < samples_num
                ):
                    try:
                        from utils.box_ops import box_cxcywh_to_xyxy

                        for t, pred in zip(targets, results_all):
                            if samples_saved >= samples_num:
                                break
                            image_id = t.get("image_id")
                            if image_id is None:
                                continue
                            image_id_int = int(image_id.detach().cpu().item()) if torch.is_tensor(image_id) else int(image_id)
                            img_meta = None
                            try:
                                img_meta = base_ds.loadImgs([image_id_int])[0]
                            except Exception:
                                img_meta = None
                            file_name = (img_meta or {}).get("file_name") if isinstance(img_meta, dict) else None
                            if not file_name:
                                continue
                            rgb_path = None
                            if samples_rgb_root is not None:
                                rgb_path = (Path(samples_rgb_root) / str(file_name)).expanduser()

                            msi_path = None
                            if samples_msi_root is not None:
                                suffix = str(samples_msi_suffix or ".tif")
                                rel = Path(str(file_name))
                                if rel.suffix.lower() not in {".tif", ".tiff"}:
                                    rel = rel.with_suffix(suffix)
                                msi_candidate = (Path(samples_msi_root) / rel).expanduser()
                                if not msi_candidate.is_file():
                                    msi_candidate = (Path(samples_msi_root) / f"{Path(str(file_name)).stem}{suffix}").expanduser()
                                msi_path = msi_candidate if msi_candidate.is_file() else None

                            gt_boxes = t.get("boxes")
                            gt_labels = t.get("labels")
                            if gt_boxes is None or gt_labels is None:
                                continue
                            gt_boxes = gt_boxes.detach().cpu()
                            gt_labels = gt_labels.detach().cpu().long()
                            if gt_boxes.numel() == 0:
                                gt_xyxy = gt_boxes.reshape(0, 4)
                            else:
                                gt_xyxy = box_cxcywh_to_xyxy(gt_boxes)
                                if _is_letterbox_target(t):
                                    h_in, w_in = [int(x) for x in t["size"].detach().cpu().tolist()]
                                    scale = torch.tensor([w_in, h_in, w_in, h_in], dtype=gt_xyxy.dtype)
                                    gt_xyxy = gt_xyxy * scale
                                    gt_xyxy = _de_letterbox_xyxy(gt_xyxy, t)
                                else:
                                    h, w = [int(x) for x in t["orig_size"].detach().cpu().tolist()]
                                    scale = torch.tensor([w, h, w, h], dtype=gt_xyxy.dtype)
                                    gt_xyxy = gt_xyxy * scale

                            pred_boxes = pred.get("boxes")
                            pred_labels = pred.get("labels")
                            pred_scores = pred.get("scores")
                            if pred_boxes is None or pred_labels is None or pred_scores is None:
                                continue
                            pred_boxes = pred_boxes.detach().cpu().numpy()
                            pred_labels = pred_labels.detach().cpu().long().tolist()
                            pred_scores = pred_scores.detach().cpu().float().tolist()

                            out_dir = Path(output_dir) / "metric_charts" / str(split_prefix) / "samples"
                            epoch_tag = f"e{int(epoch):03d}_" if epoch is not None else ""
                            base_name = f"{split_prefix}_{epoch_tag}sample_{samples_saved:03d}_img{image_id_int}"

                            gt_masks = t.get("masks")
                            gt_masks_np = gt_masks.detach().cpu().numpy() if torch.is_tensor(gt_masks) else None
                            pred_masks = pred.get("masks")
                            pred_masks_np = pred_masks.detach().cpu().numpy() if torch.is_tensor(pred_masks) else None

                            if rgb_path is not None and rgb_path.is_file():
                                save_detection_visual_samples(
                                    image_path=rgb_path,
                                    out_path=out_dir / f"{base_name}_rgb.png",
                                    gt_boxes_xyxy=gt_xyxy.detach().cpu().numpy(),
                                    gt_labels=gt_labels.tolist(),
                                    pred_boxes_xyxy=pred_boxes,
                                    pred_labels=pred_labels,
                                    pred_scores=pred_scores,
                                    gt_masks=gt_masks_np,
                                    pred_masks=pred_masks_np,
                                    class_id_to_name=samples_class_id_to_name,
                                    score_threshold=samples_score_thr,
                                    max_dets=samples_max_dets,
                                )
                            if msi_path is not None and msi_path.is_file():
                                save_detection_visual_samples(
                                    image_path=msi_path,
                                    out_path=out_dir / f"{base_name}_msi_c{int(samples_msi_channel):02d}.png",
                                    msi_channel=int(samples_msi_channel),
                                    gt_boxes_xyxy=gt_xyxy.detach().cpu().numpy(),
                                    gt_labels=gt_labels.tolist(),
                                    pred_boxes_xyxy=pred_boxes,
                                    pred_labels=pred_labels,
                                    pred_scores=pred_scores,
                                    gt_masks=gt_masks_np,
                                    pred_masks=pred_masks_np,
                                    class_id_to_name=samples_class_id_to_name,
                                    score_threshold=samples_score_thr,
                                    max_dets=samples_max_dets,
                                )
                            samples_saved += 1
                    except Exception as exc:
                        logging.warning("保存样例可视化失败（忽略继续）：%s", exc)

                if inverse_category_mapping is not None:
                    max_valid_label = inverse_category_mapping.numel()
                    for output in results_all:
                        labels = output["labels"].long()
                        valid = labels < max_valid_label
                        if not torch.all(valid):
                            output["scores"] = output["scores"][valid]
                            output["boxes"] = output["boxes"][valid]
                            if "masks" in output:
                                output["masks"] = output["masks"][valid]
                            labels = labels[valid]
                        output["labels"] = inverse_category_mapping[labels] if labels.numel() > 0 else labels

                res = {
                    target["image_id"].item(): output
                    for target, output in zip(targets, results_all)
                }
                coco_evaluator.update(res)
            elif do_profile:
                # 没有 postprocess_fn 时，用 forward 时间作为 fallback
                infer_total_s += float(t1 - float(t0)) if (t0 is not None and t1 is not None) else 0.0
                infer_images += int(len(targets))
                infer_batches += 1

            if reduced is None:
                loss_dict = self.criterion(outputs, targets)
                tensor_losses = {k: v for k, v in loss_dict.items() if torch.is_tensor(v)}
                reduced = reduce_dict(tensor_losses)
            loss_value = self._sum_loss_dict(reduced)
            metric_logger.update(loss=loss_value.item(), val_loss=loss_value.item())
            class_err = reduced.get("class_error") or reduced.get("set_criterion.class_error")
            if class_err is not None:
                metric_logger.update(class_error=float(class_err))
            # 记录 DETR/YOLO/Mask 分项损失。CompositeCriterion 会加前缀，如 set_criterion.loss_ce。
            key_aliases = (
                ("loss_ce", ("loss_ce", "set_criterion.loss_ce")),
                ("loss_bbox", ("loss_bbox", "set_criterion.loss_bbox")),
                ("loss_giou", ("loss_giou", "set_criterion.loss_giou")),
                ("loss_mask_ce", ("loss_mask_ce", "set_criterion.loss_mask_ce")),
                ("loss_mask_dice", ("loss_mask_dice", "set_criterion.loss_mask_dice")),
                ("loss_obj", ("loss_obj", "yolo.loss_obj")),
                ("loss_cls", ("loss_cls", "yolo.loss_cls")),
            )
            for meter_name, candidates in key_aliases:
                for cand in candidates:
                    if cand in reduced:
                        metric_logger.update(**{meter_name: float(reduced[cand])})
                        break
            for key in align_loss_keys:
                if key in reduced:
                    metric_logger.update(**{key: float(reduced[key])})

        metric_logger.synchronize_between_processes()
        if coco_evaluator is not None and postprocess_fn is not None:
            coco_evaluator.synchronize_between_processes()
            # COCOeval 的 accumulate/summarize 会产生大量 stdout 输出；只需在 rank0 执行一次即可。
            if is_main_process():
                coco_evaluator.accumulate()
                coco_evaluator.summarize()
        else:
            logging.warning("验证阶段未进行 COCO 评估：postprocess_fn=%s, coco_evaluator=%s", bool(postprocess_fn), bool(coco_evaluator))

        # 聚合混淆矩阵（分布式下先 all_reduce 求和），并缓存产物供上层决定是否写图
        if confusion is not None and cm_class_names is not None:
            try:
                if self.distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
                    cm_tensor = torch.as_tensor(confusion, dtype=torch.long, device=self.device)
                    torch.distributed.all_reduce(cm_tensor, op=torch.distributed.ReduceOp.SUM)
                    confusion = cm_tensor.detach().cpu().numpy()
                self._last_eval_artifacts["confusion_matrix"] = confusion
                self._last_eval_artifacts["confusion_class_names"] = cm_class_names
                self._last_eval_artifacts["confusion_normalize"] = bool(confusion_normalize)
            except Exception as exc:
                logging.warning("confusion_matrix 聚合失败（忽略继续）：%s", exc)

        if not metric_logger.meters:
            logging.warning("验证阶段未产生有效 batch，跳过指标计算。")
            self.model.train()
            self.criterion.train()
            return {}

        self.model.train()
        self.criterion.train()
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        stats["val_loss"] = stats.get("loss", 0.0)

        # 推理 profile 结果缓存：由上层决定是否写图
        if infer_enabled and infer_images > 0 and infer_total_s > 1e-9:
            params = 0
            try:
                model_ref = eval_model
                params = int(sum(p.numel() for p in model_ref.parameters() if p is not None))
            except Exception:
                params = 0
            profile = {
                "fps": float(infer_images) / float(infer_total_s),
                "latency_ms": float(infer_total_s) * 1000.0 / float(infer_images),
                "params_m": float(params) / 1e6 if params > 0 else None,
                "gflops": float(infer_gflops) if infer_gflops is not None else None,
                "flops_input_hw": infer_flops_input_hw,
                "device": device_str,
                "batches": int(infer_batches),
                "images": int(infer_images),
            }
            self._last_eval_artifacts["inference_profile"] = profile
            stats["infer_fps"] = profile["fps"]
            stats["infer_latency_ms"] = profile["latency_ms"]
            if profile.get("gflops") is not None:
                stats["infer_gflops"] = profile["gflops"]
            if profile.get("params_m") is not None:
                stats["params_m"] = profile["params_m"]
        # COCO 指标（mAP/PR/F1 等）只在 rank0 计算与返回，避免多卡重复计算/打印。
        if coco_evaluator is not None and is_main_process():
            results_json = coco_extended_metrics(coco_evaluator.coco_eval["bbox"])
            stats["results_json"] = results_json
            self._last_eval_artifacts["results_json"] = results_json
            # Expose macro P/R/F1 at the selected operating point (see coco_extended_metrics),
            # so checkpoint_best can be scored with a fitness formula.
            stats["precision"] = results_json.get("precision")
            stats["recall"] = results_json.get("recall")
            stats["f1"] = results_json.get("f1")
            if "bbox" in iou_types and "bbox" in coco_evaluator.coco_eval:
                stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
                stats["map50"] = results_json.get("map")
                stats["map75"] = results_json.get("map@75") if "map@75" in results_json else results_json.get("map75")
                stats["map50_95"] = (
                    results_json.get("map@50:95")
                    if "map@50:95" in results_json
                    else results_json.get("map50_95")
                )
                # 缓存 bbox COCOeval，用于生成 per-class 曲线图（上层决定何时写盘）
                self._last_eval_artifacts["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"]
            if "segm" in iou_types and "segm" in coco_evaluator.coco_eval:
                results_json = coco_extended_metrics(coco_evaluator.coco_eval["segm"])
                stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
                stats["results_json_segm"] = results_json
                self._last_eval_artifacts["results_json_segm"] = results_json
                stats["map50_mask"] = results_json.get("map")
                stats["map75_mask"] = results_json.get("map@75") if "map@75" in results_json else results_json.get("map75")
                stats["map50_95_mask"] = (
                    results_json.get("map@50:95")
                    if "map@50:95" in results_json
                    else results_json.get("map50_95")
                )
                self._last_eval_artifacts["coco_eval_segm"] = coco_evaluator.coco_eval["segm"]
        # 终端打印关键信息，训练中即可看到 mAP。
        map50 = stats.get("map50")
        map75 = stats.get("map75")
        map50_95 = stats.get("map50_95")
        if map50 is not None and map50_95 is not None:
            summary_prefix = "Test" if split_prefix == "test" else "Val"
            loss_value = (
                stats.get(f"{split_prefix}_loss")
                or stats.get("val_loss")
                or stats.get("loss")
                or float("nan")
            )
            logging.info(
                "%s Summary - mAP@0.5: %.4f  mAP@0.75: %.4f  mAP@0.5:0.95: %.4f  loss: %.4f",
                summary_prefix,
                map50,
                float(map75) if map75 is not None else float("nan"),
                map50_95,
                float(loss_value),
            )
        return stats

    def test(self):
        logging.info("BaseTrainer test function called.")

        # 设备与数据/模型构建，与训练复用同一套流程，但不创建优化器/调度器/梯度。
        self.init_device(self.device)
        self.build_dataset()
        # 评估只需要验证/测试集的 DataLoader
        self.build_val_dataloader()
        self.build_test_dataloader()

        self.build_model()
        self._maybe_wrap_ddp_model()
        self.build_criterion()

        # 确定输出目录（与训练一致的层级结构）。
        output_root = get_config(self.runtime_cfg, "output_root", None)
        if output_root is None:
            raise KeyError("runtime.output_root 缺失，请在主程序中设置。")
        output_dir = Path(output_root).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        # 加载评估用的 checkpoint
        eval_ckpt = (
            get_config(self.runtime_cfg, "eval_ckpt", None)
            or get_config(self.cfg.train, "resume", "")
            or ""
        )
        if eval_ckpt:
            ckpt_path = Path(eval_ckpt).expanduser()
            if ckpt_path.is_file():
                logging.info("加载评估权重: %s", ckpt_path)
                _register_yolo_pickle_alias()
                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                model_state = state.get("model", state)
                if hasattr(model_state, "state_dict"):
                    model_state = model_state.state_dict()
                model_state = _remap_ultralytics_state_dict(model_state)
                model_ref = self.model.module if hasattr(self.model, "module") else self.model
                compatible_state = _filter_compatible_state_dict(model_ref, model_state)
                skipped = len(model_state) - len(compatible_state)
                if skipped:
                    logging.warning("评估权重有 %d 个参数形状不匹配，已跳过加载。", skipped)
                model_state = compatible_state
                if not isinstance(model_state, Mapping):
                    raise TypeError(f"无法从 {ckpt_path} 解析 state_dict，类型为 {type(model_state)}")
                missing, unexpected = model_ref.load_state_dict(model_state, strict=False)
                if missing or unexpected:
                    logging.warning("模型权重缺失 %d 项, 额外 %d 项", len(missing), len(unexpected))

                # 与训练/自动测试行为对齐：若 checkpoint 内带有 EMA 权重且允许使用 EMA，则用 EMA 做评估。
                eval_use_ema = bool(get_config(self.runtime_cfg, "eval_use_ema", True))
                has_ema_in_ckpt = isinstance(state, Mapping) and state.get("ema") is not None
                if eval_use_ema and has_ema_in_ckpt:
                    if self.ema is None:
                        decay = float(get_config(self.cfg.train, "ema_decay", 0.9999))
                        tau = int(get_config(self.cfg.train, "ema_tau", 0))
                        try:
                            self.ema = ModelEMA(self.model, decay=decay, updates=tau)
                        except Exception as exc:
                            logging.warning("初始化 EMA 失败，将改用 model 权重评估: %s", exc)
                            self.ema = None

                if self.ema is not None and has_ema_in_ckpt:
                    ema_state = state.get("ema")
                    if hasattr(ema_state, "state_dict"):
                        ema_state = ema_state.state_dict()
                    if isinstance(ema_state, Mapping):
                        ema_state = _remap_ultralytics_state_dict(ema_state)
                        compatible_ema = _filter_compatible_state_dict(self.ema.ema, ema_state)
                        skipped_ema = len(ema_state) - len(compatible_ema)
                        if skipped_ema:
                            logging.warning("评估 EMA 权重有 %d 个参数形状不匹配，已跳过加载。", skipped_ema)
                        self.ema.ema.load_state_dict(compatible_ema, strict=False)
            else:
                logging.warning("未找到评估权重路径: %s", ckpt_path)
        else:
            logging.warning("未提供评估权重，将使用随机初始化模型进行测试。")

        # 执行一次完整验证/测试流程
        test_stats = (
            self._validate_epoch(
                epoch=None,
                dataloader=self.test_dataloader,
                dataset=self.test_dataset,
                split_prefix="test",
            )
            or {}
        )

        # 写入日志与指标文件
        prefixed_test = {}
        for k, v in (test_stats or {}).items():
            if str(k).startswith("test_"):
                prefixed_test[str(k)] = v
            elif str(k).startswith("val_"):
                prefixed_test[f"test_{str(k)[len('val_'):]}"] = v
            else:
                prefixed_test[f"test_{k}"] = v

        log_stats = {
            "mode": "test",
            **prefixed_test,
        }
        if is_main_process():
            log_file = output_dir / "log.log"
            log_line = json.dumps(log_stats, ensure_ascii=False)
            try:
                with log_file.open("a", encoding="utf-8") as f:
                    f.write(log_line + "\n")
            except Exception as exc:
                logging.warning("写入测试日志失败: %s", exc)
            logging.info("Test metric summary: %s", log_line)

            # 保存结构化指标
            metrics_file = output_dir / "metrics.json"
            try:
                with metrics_file.open("w", encoding="utf-8") as f:
                    json.dump(prefixed_test, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                logging.warning("保存测试指标失败: %s", exc)

            # 如已有结果字典，生成指标可视化
            results_json = test_stats.get("results_json") or test_stats.get("results_json_segm")
            self._save_last_eval_metric_charts(output_dir=output_dir, prefix="test", results_json=results_json)

        return test_stats
