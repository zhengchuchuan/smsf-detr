import argparse
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import torch

from engines.core.parse_config import get_config
from engines.trainer.base_trainer import BaseTrainer


class RtmsfDetrTrainer(BaseTrainer):
    """
    RTMSF-DETR Trainer：集成 RT-DETRv4（rtv4_hgnetv2_* baseline），跑通 RGB-only 目标检测。

    - 训练/验证/测试流程复用 BaseTrainer；
    - build_model/build_criterion 负责将 RT-DETRv4 的模型/损失/后处理接入本工程接口。
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._rtv4_criterion = None
        self._rtv4_postprocessor = None

    def _resolve_pretrain_weights(self, model_cfg: Dict) -> Path | None:
        weight_cfg = get_config(model_cfg, "pretrain_weights", None)
        if not weight_cfg:
            return None
        weight_path = Path(weight_cfg).expanduser()
        if not weight_path.is_file():
            logging.warning("预训练权重文件未找到，跳过加载: %s", weight_path)
            return None
        return weight_path

    def _load_pretrained(
        self,
        *,
        model: torch.nn.Module,
        path: Path,
        exclude_keys: Iterable[str] | None = None,
    ) -> None:
        exclude = tuple(exclude_keys or ())
        with torch.serialization.safe_globals([argparse.Namespace]):
            state = torch.load(path, map_location="cpu", weights_only=True)
        raw_state = state.get("model", state.get("state_dict", state))
        if hasattr(raw_state, "state_dict"):
            raw_state = raw_state.state_dict()

        if not isinstance(raw_state, dict):
            raise TypeError(f"无法从 {path} 解析 state_dict，类型为: {type(raw_state)}")

        # 兼容 DDP/torch.compile 保存的 module. 前缀，以及 wrapper 的 model. 前缀。
        def _normalize_key(key: str) -> str:
            if key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith("model."):
                key = key[len("model.") :]
            return key

        def _expand_input_conv_weight(
            tensor: torch.Tensor,
            target: torch.Tensor,
        ) -> torch.Tensor:
            """
            Expand a pretrained RGB input conv to match a different input-channel count.

            This is critical for MSI-only single-stream runs: otherwise `backbone.stem.stem1.conv`
            falls back to random init when `in_chs != 3`, which is especially harmful on small datasets.
            """
            if (
                tensor.ndim != 4
                or target.ndim != 4
                or tensor.shape[0] != target.shape[0]
                or tensor.shape[2:] != target.shape[2:]
                or tensor.shape[1] == target.shape[1]
            ):
                return tensor

            src_in = int(tensor.shape[1])
            dst_in = int(target.shape[1])
            if src_in <= 0 or dst_in <= 0:
                return tensor

            if dst_in < src_in:
                if dst_in == 1:
                    return tensor.mean(dim=1, keepdim=True)
                return tensor[:, :dst_in]

            expanded = target.new_zeros(target.shape)
            expanded[:, :src_in] = tensor
            if dst_in > src_in:
                mean = tensor.mean(dim=1, keepdim=True)
                expanded[:, src_in:] = mean.repeat(1, dst_in - src_in, 1, 1)
            return expanded

        def _expand_for_dualstream_backbone(
            state_dict: Mapping[str, torch.Tensor],
            *,
            model_state: Mapping[str, torch.Tensor],
            dst_model: torch.nn.Module,
        ) -> Dict[str, torch.Tensor]:
            """
            将 RTv4-HGNetv2 单流预训练权重中的 `backbone.*` 参数，映射到双流骨干：
            - backbone.rgb_backbone.*
            - backbone.ms_backbone.*（包含输入通道自适配：3ch -> ms_in_chs）

            其余参数（encoder/decoder 等）保持原 key。
            """
            backbone = getattr(dst_model, "backbone", None)
            if backbone is None:
                return dict(state_dict)
            if not hasattr(backbone, "rgb_backbone") or not hasattr(backbone, "ms_backbone"):
                return dict(state_dict)

            rgb_backbone = getattr(backbone, "rgb_backbone", None)
            ms_backbone = getattr(backbone, "ms_backbone", None)
            rgb_in_chs = int(getattr(backbone, "rgb_in_chs", 3))
            ms_in_chs = int(getattr(backbone, "ms_in_chs", 0))

            expanded: Dict[str, torch.Tensor] = {}
            for k, v in state_dict.items():
                k = _normalize_key(str(k))

                # 非 backbone 参数：原样保留。
                if not k.startswith("backbone."):
                    expanded[k] = v
                    continue

                # 目标已经是 dual-stream key 的 checkpoint：直接保留，避免重复映射。
                if k.startswith(("backbone.rgb_backbone.", "backbone.ms_backbone.")):
                    expanded[k] = v
                    continue

                suffix = k[len("backbone.") :]
                if rgb_backbone is not None and rgb_in_chs > 0:
                    expanded[f"backbone.rgb_backbone.{suffix}"] = v

                if ms_backbone is not None and ms_in_chs > 0:
                    dst_key = f"backbone.ms_backbone.{suffix}"
                    tensor = v
                    target = model_state.get(dst_key)
                    if (
                        isinstance(tensor, torch.Tensor)
                        and target is not None
                        and isinstance(target, torch.Tensor)
                        and tensor.ndim == 4
                        and target.ndim == 4
                        and tensor.shape[0] == target.shape[0]
                        and tensor.shape[2:] == target.shape[2:]
                        and target.shape[1] == ms_in_chs
                    ):
                        tensor = _expand_input_conv_weight(tensor, target)

                    expanded[dst_key] = tensor

            return expanded

        def _expand_for_single_stream_backbone(
            state_dict: Mapping[str, torch.Tensor],
            *,
            model_state: Mapping[str, torch.Tensor],
            dst_model: torch.nn.Module,
        ) -> Dict[str, torch.Tensor]:
            """
            Adapt single-stream backbone input conv weights when the target input channels differ
            from the pretrained 3-channel checkpoint.
            """
            backbone = getattr(dst_model, "backbone", None)
            if backbone is None:
                return dict(state_dict)
            if hasattr(backbone, "rgb_backbone") or hasattr(backbone, "ms_backbone"):
                return dict(state_dict)

            expanded: Dict[str, torch.Tensor] = {}
            for k, v in state_dict.items():
                k = _normalize_key(str(k))
                tensor = v
                target = model_state.get(k)
                if (
                    isinstance(tensor, torch.Tensor)
                    and target is not None
                    and isinstance(target, torch.Tensor)
                    and k.startswith("backbone.")
                ):
                    tensor = _expand_input_conv_weight(tensor, target)
                expanded[k] = tensor
            return expanded

        model_state = model.state_dict()
        raw_state = _expand_for_dualstream_backbone(raw_state, model_state=model_state, dst_model=model)
        raw_state = _expand_for_single_stream_backbone(raw_state, model_state=model_state, dst_model=model)
        filtered = {}
        skipped = []
        for key, tensor in raw_state.items():
            if exclude and key.startswith(exclude):
                skipped.append(key)
                continue
            k = _normalize_key(str(key))
            target = model_state.get(k)
            if target is None or target.shape != tensor.shape:
                skipped.append(key)
                continue
            filtered[k] = tensor

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        logging.info(
            "RTv4 预训练权重加载完成: 使用 %d 项, 跳过 %d 项, 缺失 %d 项, 额外 %d 项",
            len(filtered),
            len(skipped),
            len(missing),
            len(unexpected),
        )

    def build_model(self):
        from engines.models.rtmsfdetr.builder import build_model_and_processors

        args = self._ensure_data_args()
        model_cfg = get_config(self.cfg, "model", {}) or {}
        train_cfg = get_config(self.cfg, "train", {}) or {}

        args.device = str(self.device)
        args.num_classes = int(get_config(train_cfg, "num_classes", get_config(model_cfg, "num_classes", 1)))

        # RT-DETRv4 config 路径与开关（当前已 vendoring 到仓库内）
        args.rtdetrv4_config = get_config(model_cfg, "rtdetrv4_config", None)
        args.disable_distill = bool(get_config(model_cfg, "disable_distill", True))
        # DINOv3 teacher 资产路径（Phase 1 接入：可选，用于覆盖 RTv4 YAML 内的默认相对路径）
        args.teacher_repo_path = get_config(model_cfg, "teacher_repo_path", None)
        args.teacher_weights_path = get_config(model_cfg, "teacher_weights_path", None)

        # HGNetv2 预训练默认关闭（否则会触发 torch.distributed.get_rank 与在线下载）
        args.hgnet_pretrained = bool(get_config(model_cfg, "hgnet_pretrained", False))
        args.hgnet_local_model_dir = get_config(model_cfg, "hgnet_local_model_dir", None)
        args.hgnet_freeze_at = int(get_config(model_cfg, "hgnet_freeze_at", -1))
        args.hgnet_freeze_norm = bool(get_config(model_cfg, "hgnet_freeze_norm", False))

        # 输入反归一化（恢复到 0~1），更贴近 RT-DETRv4 原始训练分布
        args.input_denormalize = bool(get_config(model_cfg, "input_denormalize", True))
        args.clamp_after_denormalize = bool(get_config(model_cfg, "clamp_after_denormalize", True))
        # mean/std 通常来自 data.*（oil_rgb.yaml 已提供），这里确保存在
        args.rgb_mean = tuple(get_config(get_config(self.cfg, "data", {}) or {}, "rgb_mean", (0.485, 0.456, 0.406)))
        args.rgb_std = tuple(get_config(get_config(self.cfg, "data", {}) or {}, "rgb_std", (0.229, 0.224, 0.225)))

        model, criterion, postprocessor = build_model_and_processors(args)
        self.model = model.to(self.device)
        self._rtv4_criterion = criterion.to(self.device)
        self._rtv4_postprocessor = postprocessor

        pretrain_path = None
        if not get_config(model_cfg, "force_no_pretrain", False):
            pretrain_path = self._resolve_pretrain_weights(model_cfg)
        if pretrain_path is not None:
            exclude = get_config(model_cfg, "pretrain_exclude_keys", None)
            # 预训练权重（weights/rtdetr/*.pth）通常保存的是 RTv4 raw model state_dict，
            # 而本工程的 self.model 是一层封装（RTDETRv4Detector），其 state_dict key 会多一个 `model.` 前缀。
            # 因此这里优先加载到 raw model 上，避免 key 不匹配导致“看似加载成功但实际未加载”。
            model_to_load = self.model.module if hasattr(self.model, "module") else self.model
            if hasattr(model_to_load, "model") and isinstance(getattr(model_to_load, "model"), torch.nn.Module):
                model_to_load = getattr(model_to_load, "model")
            self._load_pretrained(model=model_to_load, path=pretrain_path, exclude_keys=exclude)

        return self.model

    def build_criterion(self):
        if self._rtv4_criterion is None or self._rtv4_postprocessor is None:
            # 兜底：确保 build_model 已执行
            self.build_model()
        self.criterion = self._rtv4_criterion
        self.criterion_extras = {
            "postprocess": self._rtv4_postprocessor,
            "weight_dict": getattr(self._rtv4_criterion, "weight_dict", {}) or {},
        }
        return self.criterion

    def build_optimizer(self):
        """
        对齐 RT-DETRv4 的 optimizer regex 分组（参考 third_party YAMLConfig.get_optim_params）：

        - 支持在 Hydra 中配置 `train.optimizer_param_groups`（list of dict）：
          - `params`: 正则表达式（匹配 parameter name）
          - `lr` / `weight_decay`: 可选覆盖
        - 也兼容 `train.optimizer` 为 dict（third_party 风格），例如：
          - `train.optimizer.type` / `train.optimizer.lr` / `train.optimizer.weight_decay` / `train.optimizer.params`
        """
        if self.model is None:
            raise RuntimeError("请先构建模型再创建优化器。")

        train_cfg: Dict[str, Any] = get_config(self.cfg, "train", {}) or {}
        model_cfg: Dict[str, Any] = get_config(self.cfg, "model", {}) or {}

        optimizer_entry = get_config(train_cfg, "optimizer", "adamw")
        optimizer_cfg = (
            optimizer_entry
            if optimizer_entry is not None
            and hasattr(optimizer_entry, "items")
            and not isinstance(optimizer_entry, (str, bytes))
            else None
        )
        optimizer_type = (
            str(get_config(optimizer_cfg, "type", "adamw")).lower()
            if optimizer_cfg is not None
            else str(optimizer_entry).lower()
        )

        model_for_params = self.model.module if hasattr(self.model, "module") else self.model
        named_parameters = list(model_for_params.named_parameters())

        lr = get_config(train_cfg, "lr", None)
        if lr is None and optimizer_cfg is not None:
            lr = get_config(optimizer_cfg, "lr", None)
        if lr is None:
            lr = get_config(model_cfg, "lr", None)
        model_hyp = getattr(model_for_params, "hyp", {}) or {}
        lr = float(lr if lr is not None else model_hyp.get("lr0", 1e-4))

        weight_decay = get_config(train_cfg, "weight_decay", None)
        if weight_decay is None and optimizer_cfg is not None:
            weight_decay = get_config(optimizer_cfg, "weight_decay", None)
        weight_decay = float(weight_decay if weight_decay is not None else 1e-4)

        momentum = float(get_config(train_cfg, "momentum", model_hyp.get("momentum", 0.9)))

        betas = get_config(train_cfg, "betas", None)
        eps = get_config(train_cfg, "eps", None)
        if optimizer_cfg is not None:
            if betas is None:
                betas = get_config(optimizer_cfg, "betas", None)
            if eps is None:
                eps = get_config(optimizer_cfg, "eps", None)

        optimizer_param_groups = None
        if optimizer_cfg is not None:
            optimizer_param_groups = get_config(optimizer_cfg, "params", None) or get_config(optimizer_cfg, "param_groups", None)
        if optimizer_param_groups is None:
            optimizer_param_groups = get_config(train_cfg, "optimizer_param_groups", None) or get_config(train_cfg, "optimizer_params", None)

        def _pattern_match(pattern: str, name: str) -> bool:
            try:
                if re.findall(pattern, name):
                    return True
                # 兼容 wrapper 前缀 `model.` 导致的 `^backbone` 等锚点不匹配
                if name.startswith("model.") and re.findall(pattern, name[len("model.") :]):
                    return True
                return False
            except re.error as exc:
                raise ValueError(f"optimizer 参数分组正则非法: pattern={pattern}") from exc

        param_groups = []
        visited = set()
        if optimizer_param_groups:
            for pg in optimizer_param_groups:
                pattern = get_config(pg, "params", None)
                if not pattern:
                    raise ValueError("train.optimizer_param_groups 中每个分组必须包含 `params`（正则表达式）。")

                matched = [
                    (name, p)
                    for name, p in named_parameters
                    if p.requires_grad and _pattern_match(str(pattern), name)
                ]
                overlap = {name for name, _ in matched if name in visited}
                if overlap:
                    raise ValueError(
                        "optimizer 参数分组存在重叠匹配，请调整正则避免同一参数被多个分组命中。"
                        f" pattern={pattern}, overlap_count={len(overlap)}"
                    )
                visited.update(name for name, _ in matched)

                params = [p for _, p in matched]
                if not params:
                    logging.warning("optimizer 参数分组未命中任何参数，已忽略: pattern=%s", pattern)
                    continue

                group: Dict[str, Any] = {"params": params}
                # 复制额外配置（lr/weight_decay 等），但不覆盖 params
                if hasattr(pg, "items"):
                    for k, v in pg.items():
                        if k == "params":
                            continue
                        group[k] = v
                else:
                    for k, v in dict(pg).items():
                        if k == "params":
                            continue
                        group[k] = v

                group.setdefault("lr", lr)
                group.setdefault("weight_decay", weight_decay)
                group["lr"] = float(group["lr"])
                group["weight_decay"] = float(group["weight_decay"])

                param_groups.append(group)
                logging.info(
                    "optimizer 分组: pattern=%s, params=%d, lr=%.6f, weight_decay=%.6f",
                    pattern,
                    len(params),
                    float(group["lr"]),
                    float(group["weight_decay"]),
                )

        remaining = [(name, p) for name, p in named_parameters if p.requires_grad and name not in visited]
        if remaining:
            param_groups.append(
                {
                    "params": [p for _, p in remaining],
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
            )

        if not param_groups:
            logging.warning("未找到需训练的参数，使用模型全部可训练参数创建优化器。")
            param_groups = [{"params": [p for _, p in named_parameters if p.requires_grad], "lr": lr, "weight_decay": weight_decay}]

        if optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(
                param_groups,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=True,
            )
        else:
            adamw_kwargs: Dict[str, Any] = {}
            if betas is not None:
                adamw_kwargs["betas"] = tuple(float(x) for x in betas)
            if eps is not None:
                adamw_kwargs["eps"] = float(eps)
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=lr,
                weight_decay=weight_decay,
                **adamw_kwargs,
            )

        explicit_group_lrs = [
            float(pg.get("lr", lr))
            for pg in (optimizer_param_groups or [])
            if get_config(pg, "lr", None) is not None
        ]
        if optimizer_type == "adamw" and float(lr) >= 1.0e-3:
            min_explicit_lr = min(explicit_group_lrs) if explicit_group_lrs else float(lr)
            if float(lr) >= max(1.0e-3, 5.0 * float(min_explicit_lr)):
                logging.warning(
                    "检测到较大的 AdamW base lr=%.6f（显式分组最小 lr=%.6f）。"
                    " 若这是 RT-DETRv4/小数据集训练，需重点确认是否把 0.0004 误写成了 0.004。",
                    float(lr),
                    float(min_explicit_lr),
                )

        logging.info(
            "优化器创建完成: type=%s, lr=%.6f, weight_decay=%.6f, param_groups=%d",
            optimizer_type,
            lr,
            weight_decay,
            len(self.optimizer.param_groups) if self.optimizer else 0,
        )
        return self.optimizer

    def build_scheduler(self):
        """
        支持 RT-DETRv4 常用的 flat-cosine 迭代调度（warmup_iter + flat_epoch + cosine + no_aug_epoch）。
        其它 scheduler 类型则回退到 BaseTrainer 默认实现。
        """
        train_cfg: Dict[str, Any] = get_config(self.cfg, "train", {}) or {}
        scheduler_type = str(get_config(train_cfg, "lr_scheduler", "flatcosine")).lower()
        if scheduler_type not in {"flatcosine", "flat_cosine"}:
            return super().build_scheduler()

        if self.optimizer is None:
            raise RuntimeError("请先创建优化器再构建学习率调度器。")
        if self.train_dataloader is None:
            self.build_train_dataloader()

        steps_per_epoch = max(1, int(len(self.train_dataloader)))
        epochs = int(get_config(train_cfg, "epochs", 1))
        warmup_iter = int(get_config(train_cfg, "warmup_iter", 0))
        warmup_max_epochs = int(get_config(train_cfg, "warmup_max_epochs", 2))
        flat_epoch = int(get_config(train_cfg, "flat_epoch", 0))
        no_aug_epoch = int(get_config(train_cfg, "no_aug_epoch", 0))
        lr_gamma = float(get_config(train_cfg, "lr_gamma", 0.5))

        total_iter = steps_per_epoch * epochs
        if warmup_iter > 0 and warmup_max_epochs > 0:
            warmup_cap = steps_per_epoch * warmup_max_epochs
            if warmup_iter > warmup_cap:
                warmup_epochs_est = float(warmup_iter) / float(max(1, steps_per_epoch))
                logging.warning(
                    "warmup_iter=%d 过大（steps_per_epoch=%d，约 %.1f epochs），已自动裁剪为 %d（%d epochs）。"
                    " 如需关闭裁剪：设置 train.warmup_max_epochs=0。",
                    warmup_iter,
                    steps_per_epoch,
                    warmup_epochs_est,
                    warmup_cap,
                    warmup_max_epochs,
                )
                warmup_iter = warmup_cap
        flat_iter = steps_per_epoch * flat_epoch
        no_aug_iter = steps_per_epoch * no_aug_epoch

        def lr_lambda(current_iter: int) -> float:
            # 使用与 third_party FlatCosineLRScheduler 等价的 factor（相对 base_lr）
            if warmup_iter > 0 and current_iter <= warmup_iter:
                ratio = float(current_iter) / float(max(1, warmup_iter))
                return ratio * ratio

            if warmup_iter < current_iter <= flat_iter:
                return 1.0

            if no_aug_iter > 0 and current_iter >= total_iter - no_aug_iter:
                return lr_gamma

            denom = max(1, total_iter - flat_iter - no_aug_iter)
            progress = float(current_iter - flat_iter) / float(denom)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_gamma + (1.0 - lr_gamma) * cosine

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        logging.info(
            "学习率调度器创建完成: type=flatcosine, steps_per_epoch=%d, epochs=%d, warmup_iter=%d, flat_epoch=%d, no_aug_epoch=%d, lr_gamma=%.4f",
            steps_per_epoch,
            epochs,
            warmup_iter,
            flat_epoch,
            no_aug_epoch,
            lr_gamma,
        )
        return self.scheduler
