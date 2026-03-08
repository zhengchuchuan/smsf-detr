import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from argparse import Namespace
import torch

from engines.core.parse_config import get_config
from engines.trainer.base_trainer import BaseTrainer
from datasets.msif_yolo import build_msif_yolo_dataset
from datasets import build_dataset as build_legacy_dataset
from utils.misc import collate_fn
from utils.misc import NestedTensor


class MscftTrainer(BaseTrainer):
    """
    复用项目内通用训练逻辑，调用本地化的 MSCFT YOLO+CFT 模型。
    - 需要提供 COCO 风格的数据配置（data.dataset_file、data.dataset_dir/coco_path 等）；
    - 默认支持 RGB+MSI 叠加输入，暂不支持双路输出（dual_stream_output）。
    """

    def __init__(self, cfg: Any):
        super().__init__(cfg)
        self._pretrained_loaded = False

    # -------- YOLO 预训练权重适配工具，限定在 MSCFT 训练器内 --------
    @staticmethod
    def _register_yolo_pickle_alias():
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
            "models.utils.metrics": mscft_utils_metrics,
        }
        for name, module in alias_map.items():
            if name not in sys.modules:
                sys.modules[name] = module
            parent, attr = name.split(".", 1)
            parent_mod = _ensure(parent)
            setattr(parent_mod, attr, module)
            if parent == "models":
                sys.modules["models"] = parent_mod
            if parent == "utils":
                sys.modules["utils"] = parent_mod

    @staticmethod
    def _remap_ultralytics_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        if not state_dict:
            return state_dict
        sample_key = next(iter(state_dict))
        if not sample_key.startswith("model.") or sample_key.startswith("model.model."):
            return state_dict

        def _remap_key(key: str) -> str:
            new_key = f"model.{key}"
            for idx in ("0", "3"):
                prefix = f"model.model.{idx}."
                if new_key.startswith(prefix):
                    new_key = new_key.replace(f"{prefix}conv.", f"{prefix}conv.conv.")
                    new_key = new_key.replace(f"{prefix}bn.", f"{prefix}conv.bn.")
            return new_key

        return {_remap_key(k): v for k, v in state_dict.items()}

    def _filter_compatible_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        model_sd = self.model.state_dict()
        filtered = {}
        for k, v in state_dict.items():
            if k not in model_sd:
                continue
            if model_sd[k].shape != v.shape:
                continue
            filtered[k] = v
        return filtered

    def _load_pretrained_if_any(self):
        if self._pretrained_loaded:
            return
        train_cfg = get_config(self.cfg, "train", {}) or {}
        model_cfg = get_config(self.cfg, "model", {}) or {}
        resume_path = get_config(train_cfg, "resume", "") or ""
        # 兼容 pretrain_weights/pretrained/weights 字段，用于加载 YOLO 预训练
        pretrain_path = (
            get_config(model_cfg, "pretrain_weights", "")
            or get_config(model_cfg, "pretrained", "")
            or get_config(model_cfg, "weights", "")
        )
        ckpt_path_str = resume_path or pretrain_path
        if not ckpt_path_str:
            return
        ckpt_path = Path(ckpt_path_str).expanduser()
        if not ckpt_path.is_file():
            logging.warning("MSCFT 预训练权重未找到: %s", ckpt_path)
            return

        self._register_yolo_pickle_alias()
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_state = state.get("model", state)
        if hasattr(model_state, "state_dict"):
            model_state = model_state.state_dict()
        model_state = self._remap_ultralytics_state_dict(model_state)
        compatible = self._filter_compatible_state_dict(model_state)
        skipped = len(model_state) - len(compatible)
        missing, unexpected = self.model.load_state_dict(compatible, strict=False)
        logging.info(
            "MSCFT 预训练加载完成: 总参数=%d, 成功=%d, 跳过(形状不匹配)=%d, 缺失=%d, 额外=%d",
            len(model_state),
            len(compatible),
            skipped,
            len(missing),
            len(unexpected),
        )
        # 防止 BaseTrainer 再次按 resume 尝试加载
        try:
            train_cfg["resume"] = ""
            if hasattr(self.cfg, "train"):
                self.cfg.train.resume = ""
        except Exception:
            pass
        self._pretrained_loaded = True

    def _ensure_data_args(self):
        try:
            args = super()._ensure_data_args()
        except ValueError as exc:
            raise ValueError(
                "mscft_trainer 现已走本仓库的数据管线，请在 data.dataset_file 中显式指定数据类型（如 coco_rgb_msi），"
                "并提供 data.dataset_dir 或 data.coco_path。"
            ) from exc

        dataset_file = getattr(args, "dataset_file", None)
        if not dataset_file:
            raise ValueError("mscft_trainer 需要 data.dataset_file（示例：coco_rgb_msi、coco_msi、coco_rgb）。")

        dataset_root = getattr(args, "dataset_dir", None) or getattr(args, "coco_path", None)
        if not dataset_root:
            raise ValueError("mscft_trainer 需要 data.dataset_dir 或 data.coco_path 指向 COCO 风格数据根目录。")

        if getattr(args, "dual_stream_output", False):
            raise ValueError("mscft_trainer 暂不支持 dual_stream_output，请使用通道叠加方式输入多模态数据。")

        if not getattr(args, "use_rgb_input", False) and not getattr(args, "use_msi_input", False):
            raise ValueError("至少启用 use_rgb_input 或 use_msi_input 其中之一。")

        # 若未推导 input_channels，按通道拆分值补齐，避免模型构建时落回默认 3 通道。
        if not hasattr(args, "input_channels"):
            rgb_ch = int(getattr(args, "rgb_input_channels", 0))
            ms_ch = int(getattr(args, "ms_input_channels", 0))
            splits = getattr(args, "channel_splits", None)
            if splits:
                try:
                    rgb_ch, ms_ch = int(splits[0]), int(splits[1] if len(splits) > 1 else splits[0])
                except Exception:
                    logging.warning("解析 channel_splits 失败，将使用 rgb/ms_input_channels。")
            args.input_channels = rgb_ch + ms_ch

        # img_size 允许传入 [h, w]，这里转为单个 int（取最大边，确保为 square）。
        img_size = getattr(args, "img_size", None)
        if isinstance(img_size, (list, tuple)):
            try:
                vals = [int(v) for v in img_size]
                if len(vals) >= 2 and vals[0] != vals[1]:
                    logging.warning("mscft_trainer 仅支持等宽高 img_size，已取最大值 %s。", max(vals))
                args.img_size = int(max(vals))
            except Exception:
                logging.warning("解析 img_size 失败，回退为默认 640。")
                args.img_size = 640
        # 可选 YOLO 归一化开关：强制使用线性 0-1 归一化，便于对齐 YOLO 预训练/超参。
        if bool(get_config(self.cfg, "data", {}).get("use_yolo_norm", False)):
            if getattr(args, "rgb_normalize_mode", None) != "linear":
                logging.info("use_yolo_norm 已开启，强制使用 RGB 线性归一化。")
                args.rgb_normalize_mode = "linear"
            # 若未指定多光谱归一化，默认采用 tensor_minmax 并中心化到 0 附近
            if getattr(args, "ms_normalize_mode", None) is None:
                args.ms_normalize_mode = "tensor_minmax"
            args.ms_center_to_rgb_range = bool(getattr(args, "ms_center_to_rgb_range", True))

        # sandbox 环境无 CUDA 时，强制单进程 DataLoader 以避免权限问题
        if not torch.cuda.is_available():
            setattr(args, "num_workers", 0)

        return args

    def build_model(self):
        from engines.models.mscft.builder import build_model as build_mscft_model

        args = self._ensure_data_args()
        model_cfg = get_config(self.cfg, "model", {}) or {}
        train_cfg = get_config(self.cfg, "train", {}) or {}

        num_classes = int(get_config(train_cfg, "num_classes", get_config(model_cfg, "num_classes", 80)))
        input_channels = int(getattr(args, "input_channels", 3))
        channel_splits = getattr(args, "channel_splits", None)
        model_yaml = get_config(model_cfg, "model_cfg", None)
        hyp_path = get_config(model_cfg, "hyp", None) or get_config(train_cfg, "hyp", None)
        conf_thres = float(get_config(model_cfg, "conf_thres", 0.25))
        iou_thres = float(get_config(model_cfg, "iou_thres", 0.45))

        namespace = Namespace(
            model_cfg=model_yaml,
            num_classes=num_classes,
            input_channels=input_channels,
            channel_splits=channel_splits,
            hyp=hyp_path,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
        )

        logging.info(
            "构建 MSCFT 模型: cfg=%s, num_classes=%s, input_channels=%s",
            model_yaml,
            num_classes,
            input_channels,
        )
        self.model = build_mscft_model(namespace).to(self.device)
        if hyp_path:
            try:
                import yaml

                hyp_file = Path(str(hyp_path)).expanduser()
                if hyp_file.is_file():
                    with hyp_file.open("r", encoding="utf-8") as f:
                        hyp = yaml.safe_load(f) or {}
                    setattr(self.model, "hyp", hyp)
                    try:
                        setattr(getattr(self.model, "model", None), "hyp", hyp)
                    except Exception:
                        pass
            except Exception as exc:
                logging.warning("读取 MSCFT hyp 文件失败（将忽略继续）：%s", exc)
        # 如果配置了预训练路径，这里就地加载（仅 MSCFT 使用），避免污染通用 BaseTrainer
        self._load_pretrained_if_any()
        return self.model

    def build_criterion(self):
        from engines.models.mscft.builder import build_criterion_and_postprocessors

        args = self._ensure_data_args()
        model_cfg = get_config(self.cfg, "model", {}) or {}
        train_cfg = get_config(self.cfg, "train", {}) or {}
        num_classes = int(get_config(train_cfg, "num_classes", get_config(model_cfg, "num_classes", 80)))
        input_channels = int(getattr(args, "input_channels", 3))
        channel_splits = getattr(args, "channel_splits", None)
        model_yaml = get_config(model_cfg, "model_cfg", None)
        hyp_path = get_config(model_cfg, "hyp", None) or get_config(train_cfg, "hyp", None)
        conf_thres = float(get_config(model_cfg, "conf_thres", 0.25))
        iou_thres = float(get_config(model_cfg, "iou_thres", 0.45))
        namespace = Namespace(
            model_cfg=model_yaml,
            num_classes=num_classes,
            input_channels=input_channels,
            channel_splits=channel_splits,
            hyp=hyp_path,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
        )
        criterion, postprocessors = build_criterion_and_postprocessors(namespace, model=self.model)
        criterion.to(self.device)
        self.criterion = criterion
        self.criterion_extras = {"postprocess": postprocessors.get("bbox")}
        return self.criterion

    # ---------- 覆盖数据构建，支持 YOLO 风格 Dataset ----------
    def build_dataset(self):
        args = self._ensure_data_args()
        loader_mode = getattr(args, "loader", None) or getattr(get_config(self.cfg, "data", {}), "get", lambda x, d=None: None)("loader", None)
        img_size = getattr(args, "img_size", 640)

        if loader_mode == "mscft_yolo":
            logging.info("使用 MSCFT YOLO 风格数据管线: loader=%s, img_size=%s", loader_mode, img_size)
            # 训练集使用 YOLO 风格，验证/测试保持通用 COCO 流水线以兼容评估
            self.train_dataset = build_msif_yolo_dataset("train", args, img_size)
            self.validation_dataset = build_legacy_dataset("val", args, img_size)
            self.test_dataset = build_legacy_dataset("val", args, img_size)

            # 训练 DataLoader 仍使用基类的随机采样/梯度累积逻辑
            self.train_dataloader = self._build_train_loader(self.train_dataset, args, collate_fn)
            self.validation_dataloader = self._build_eval_loader(
                self.validation_dataset, args, split="val", collate_fn=collate_fn
            )
            self.test_dataloader = self._build_eval_loader(
                self.test_dataset, args, split="test", collate_fn=collate_fn
            )
            logging.info(
                "数据集与 DataLoader 构建完成: train=%d, val=%d, test=%d, batch_size=%d, workers=%d",
                len(self.train_dataset),
                len(self.validation_dataset),
                len(self.test_dataset),
                int(getattr(args, "batch_size", 1)),
                int(getattr(args, "num_workers", 0)),
            )
            return {
                "train": self.train_dataset,
                "val": self.validation_dataset,
                "test": self.test_dataset,
            }

        # 默认沿用基类 COCO 流水线
        return super().build_dataset()

    # ---------- 额外诊断工具 ----------
    def _anchor_diagnostics(self, max_samples: int = 256, thr: float = 4.0):
        """
        简易 anchor 覆盖检查，统计 anchors/target 与 BPR，便于发现 anchor 与数据不匹配的问题。
        """
        if self.train_dataset is None or self.model is None:
            return
        try:
            yolo_model = self.model.model if hasattr(self.model, "model") else None
            detect = yolo_model.model[-1] if yolo_model is not None and hasattr(yolo_model, "model") else None
        except Exception:
            detect = None
        if detect is None or not hasattr(detect, "anchor_grid"):
            logging.warning("未找到 Detect 层，跳过 anchor 诊断。")
            return
        anchors = detect.anchor_grid.clone().view(-1, 2).cpu()
        img_size = int(getattr(self.data_args, "img_size", 640) or 640)
        wh_list = []
        num_images = min(len(self.train_dataset), max_samples)
        for idx in range(num_images):
            try:
                _, target = self.train_dataset[idx]
            except Exception:
                break
            boxes = target.get("boxes")
            if boxes is None or boxes.numel() == 0:
                continue
            if float(boxes.max()) <= 1.5:
                wh = boxes[:, 2:4] * img_size
            else:
                # 兼容像素级 xyxy
                xyxy = boxes
                wh = xyxy[:, 2:4] - xyxy[:, 0:2]
            wh_list.append(wh)
        if not wh_list:
            logging.warning("anchor 诊断未收集到有效标注，跳过。")
            return
        wh = torch.cat(wh_list, dim=0).float()
        r = wh[:, None] / anchors[None]
        x = torch.min(r, 1.0 / r).min(2)[0]
        best = x.max(1)[0]
        aat = (x > 1.0 / thr).float().sum(1).mean().item()
        bpr = (best > 1.0 / thr).float().mean().item()
        logging.info("Anchor 覆盖诊断: anchors/target=%.2f, BPR=%.4f, samples=%d", aat, bpr, len(wh))
