import logging
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from engines.models.base import BaseDetector
from engines.core.parse_config import get_config


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # 延迟导入以避免未安装时提前报错

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class MscftYoloDetector(BaseDetector):
    """
    将 third_party/mscft 的 YOLO 多光谱模型封装为 BaseDetector 兼容接口。
    - 训练模式返回原始预测（供 YOLO 损失计算）。
    - 推理模式返回推理结果与原始预测，后处理可在 PostProcess 中完成。
    """

    supported_modalities = ("rgb", "ms", "rgb_ms")

    def __init__(self, model: nn.Module, channel_splits: Tuple[int, int] | None = None):
        super().__init__()
        self.model = model
        self.channel_splits = channel_splits

    def forward(self, samples, targets=None):
        if hasattr(samples, "tensors"):
            images = samples.tensors
        else:
            images = samples

        # 若传入叠加通道的张量且配置了 channel_splits，则拆分为双流输入
        if self.channel_splits and isinstance(images, torch.Tensor):
            rgb_ch, ms_ch = self.channel_splits
            if rgb_ch > 0 and ms_ch > 0 and images.shape[1] == rgb_ch + ms_ch:
                rgb = images[:, :rgb_ch]
                ms = images[:, rgb_ch:rgb_ch + ms_ch]
                images = (rgb, ms)
        preds = self.model(images)  # training: List[p], eval: (det, raw)
        if self.training:
            return {"preds": preds, "images": images}
        if isinstance(preds, tuple) and len(preds) == 2:
            det, raw = preds
        else:
            det, raw = None, preds
        return {"detections": det, "preds": raw, "images": images}


class MscftCriterion(nn.Module):
    """
    使用原始 YOLO ComputeLoss 进行损失计算。将 COCO 风格 targets 转为 YOLO (img, cls, x, y, w, h) 格式。
    """

    def __init__(self, model: nn.Module, hyp: Dict[str, Any]):
        super().__init__()
        from engines.models.mscft.utils.loss import ComputeLoss

        self.hyp = hyp
        # YOLO 训练脚本会为 model 注入 hyp/gr，这里保持一致
        model.hyp = hyp
        model.gr = hyp.get("gr", 1.0)
        self.compute_loss = ComputeLoss(model)

    @staticmethod
    def _to_yolo_targets(targets: List[Dict[str, torch.Tensor]], images: torch.Tensor) -> torch.Tensor:
        # 支持 tuple/list 输入（RGB, MS），选取首个张量获取尺寸与设备
        if isinstance(images, (tuple, list)):
            first_tensor = None
            for img in images:
                if isinstance(img, torch.Tensor):
                    first_tensor = img
                    break
            if first_tensor is None:
                raise ValueError("images 中未找到有效的 Tensor。")
            images = first_tensor

        device = images.device
        bs, _, h, w = images.shape
        entries: List[torch.Tensor] = []
        for img_idx, tgt in enumerate(targets):
            boxes = tgt.get("boxes")
            labels = tgt.get("labels")
            if boxes is None or labels is None or boxes.numel() == 0:
                continue
            boxes = boxes.to(device)
            lbl = labels.to(device).float()

            # 训练管线会在 Dataset 中将标注转为归一化的 cxcywh（0~1），这里直接复用；
            # 若检测到像素级 xyxy（>1.5），再转换为归一化的 cxcywh 以兼容 YOLO 损失。
            if float(boxes.max()) <= 1.5:
                cxcywh = boxes
            else:
                xyxy = boxes
                cxcywh = xyxy.clone()
                cxcywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / (2.0 * w)
                cxcywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / (2.0 * h)
                cxcywh[:, 2] = (xyxy[:, 2] - xyxy[:, 0]) / w
                cxcywh[:, 3] = (xyxy[:, 3] - xyxy[:, 1]) / h

            entry = torch.stack(
                [
                    torch.full((cxcywh.shape[0],), float(img_idx), device=device),
                    lbl,
                    cxcywh[:, 0],
                    cxcywh[:, 1],
                    cxcywh[:, 2],
                    cxcywh[:, 3],
                ],
                dim=1,
            )
            entries.append(entry)
        if not entries:
            return torch.zeros((0, 6), device=device)
        return torch.cat(entries, dim=0)

    def forward(self, outputs: Dict[str, Any], targets: List[Dict[str, torch.Tensor]]):
        preds = outputs.get("preds")
        images = outputs.get("images")
        if preds is None or images is None:
            raise ValueError("MSCFTCriterion 需要 preds 与 images。")
        target_tensor = self._to_yolo_targets(targets, images)
        loss, items = self.compute_loss(preds, target_tensor)
        # items: [lbox, lobj, lcls, loss]（items 已 detach，在日志中使用）
        return {
            "loss_bbox": items[0].detach(),
            "loss_obj": items[1].detach(),
            "loss_cls": items[2].detach(),
            "loss_total": loss,
        }


class PostProcess:
    """
    简化版后处理：调用 YOLO 的 NMS，输出 boxes/scores/labels（已缩放回原图尺寸）。
    """

    def __init__(self, conf_thres: float = 0.25, iou_thres: float = 0.45):
        from engines.models.mscft.utils.general import non_max_suppression
        from engines.models.mscft.utils.general import scale_coords

        self.non_max_suppression = non_max_suppression
        self.scale_coords = scale_coords
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def __call__(self, outputs: Dict[str, Any], orig_target_sizes: torch.Tensor):
        det = outputs.get("detections")
        if det is None:
            return []
        nms_out = self.non_max_suppression(det, conf_thres=self.conf_thres, iou_thres=self.iou_thres)
        images = outputs.get("images")
        input_hw = None
        if isinstance(images, torch.Tensor):
            input_hw = images.shape[2:]
        elif isinstance(images, (tuple, list)):
            for img in images:
                if isinstance(img, torch.Tensor):
                    input_hw = img.shape[2:]
                    break
        if input_hw is None:
            raise ValueError("PostProcess 需要提供 Tensor 格式的 images 以进行尺度还原。")
        results = []
        for i, preds in enumerate(nms_out):
            if preds is None or len(preds) == 0:
                results.append({"boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,)), "labels": torch.zeros((0,), dtype=torch.long)})
                continue
            h0, w0 = orig_target_sizes[i].tolist()
            h_in, w_in = input_hw
            # 训练/验证管线对图像做等边缩放（无 letterbox），直接按比例还原到原图尺寸
            scale = torch.tensor([w0 / w_in, h0 / h_in, w0 / w_in, h0 / h_in], device=preds.device, dtype=preds.dtype)
            preds[:, :4] = preds[:, :4] * scale
            preds[:, 0::2].clamp_(min=0, max=w0)
            preds[:, 1::2].clamp_(min=0, max=h0)
            boxes = preds[:, :4]
            scores = preds[:, 4]
            labels = preds[:, 5].long()
            results.append({"boxes": boxes, "scores": scores, "labels": labels})
        return results


def build_model(args: Namespace):
    from engines.models.mscft.yolo import Model as YoloModel

    model_cfg = getattr(args, "model_cfg", None) or getattr(args, "cfg", None)
    if model_cfg is None:
        # 默认使用本地迁移的 FLIR 多模态 transformer 结构
        model_cfg = Path(__file__).resolve().parent / "transformer" / "yolov5l_fusion_transformer_FLIR.yaml"
    model_cfg = Path(model_cfg)
    if not model_cfg.is_file():
        raise FileNotFoundError(f"未找到 MSCFT 模型配置: {model_cfg}")

    num_classes = int(getattr(args, "num_classes", 80))
    input_channels = int(getattr(args, "input_channels", 3))
    splits = getattr(args, "channel_splits", None)
    if splits and len(splits) >= 2:
        rgb_ch, ms_ch = int(splits[0]), int(splits[1])
    else:
        rgb_ch, ms_ch = input_channels, 0
    # 双流：第一路 RGB，第二路 MS，保持原始通道数以便 GPT 拼接
    ch_list = [max(1, rgb_ch), max(1, ms_ch or rgb_ch)]

    model = YoloModel(cfg=str(model_cfg), ch=ch_list, nc=num_classes)
    detector = MscftYoloDetector(model, channel_splits=(rgb_ch, ms_ch))
    return detector


def build_criterion_and_postprocessors(args: Namespace, model: MscftYoloDetector | None = None):
    if model is None:
        model = build_model(args)
    yolo_model = model.model if isinstance(model, MscftYoloDetector) else model
    hyp_cfg = getattr(args, "hyp", None)
    if hyp_cfg is None:
        hyp_cfg = Path(__file__).resolve().parent / "hyp.scratch.yaml"
    hyp_cfg = Path(hyp_cfg)
    if not hyp_cfg.is_file():
        raise FileNotFoundError(f"未找到 MSCFT 超参文件: {hyp_cfg}")
    hyp = _load_yaml(hyp_cfg)

    criterion = MscftCriterion(yolo_model, hyp)
    postprocessors = {"bbox": PostProcess(conf_thres=getattr(args, "conf_thres", 0.25), iou_thres=getattr(args, "iou_thres", 0.45))}
    return criterion, postprocessors


class PostProcessWrapper:
    """
    兼容注册表返回的 PostProcess 类占位。
    """

    def __init__(self, conf_thres: float = 0.25, iou_thres: float = 0.45):
        self.processor = PostProcess(conf_thres=conf_thres, iou_thres=iou_thres)

    def __call__(self, outputs: Dict[str, Any], orig_target_sizes: torch.Tensor):
        return self.processor(outputs, orig_target_sizes)
