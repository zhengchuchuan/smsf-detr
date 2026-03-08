"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from ..core import register


__all__ = ['PostProcessor']


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class PostProcessor(nn.Module):
    __share__ = [
        'num_classes',
        'use_focal_loss',
        'num_top_queries',
        'remap_mscoco_category'
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        pred_masks = outputs.get('pred_masks', None)
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        # orig_target_sizes follows DETR convention: (h, w). xyxy scaling needs (w, h, w, h).
        if orig_target_sizes.ndim != 2 or orig_target_sizes.shape[1] != 2:
            raise ValueError(
                f"orig_target_sizes must have shape [B, 2] as (h, w), got {tuple(orig_target_sizes.shape)}"
            )
        img_h, img_w = orig_target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(bbox_pred.dtype).unsqueeze(1)
        bbox_pred = bbox_pred * scale_fct

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # TODO for older tensorrt
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            query_index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=query_index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(bbox_pred, dim=1, index=index.unsqueeze(-1).tile(1, 1, bbox_pred.shape[-1]))
                query_index = index
            else:
                boxes = bbox_pred
                query_index = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0).repeat(scores.shape[0], 1)

        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # TODO
        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for i, (lab, box, sco) in enumerate(zip(labels, boxes, scores)):
            result = dict(labels=lab, boxes=box, scores=sco)
            if pred_masks is not None:
                # pred_masks: [B, Q, Hm, Wm] -> select top-K queries and resize to original image size.
                idx = query_index[i]
                masks_i = pred_masks[i].index_select(0, idx)  # [K, Hm, Wm]
                h, w = orig_target_sizes[i].tolist()
                masks_i = F.interpolate(
                    masks_i.unsqueeze(1),
                    size=(int(h), int(w)),
                    mode='bilinear',
                    align_corners=False,
                )  # [K, 1, H, W]
                result["masks"] = masks_i > 0.0
            results.append(result)

        return results


    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
