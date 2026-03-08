"""
RF-DETR 损失与匹配器的轻量实现，便于在本项目中独立使用。
主要包含：
- box_ops: 常用框变换与 IoU 计算；
- HungarianMatcher: 匈牙利匹配，支持 mask 成本；
- SetCriterion: 组合分类/框/可选 mask 损失，附带辅助层与编码器输出支持；
- PostProcess: 推理后处理，选取 top-K 预测。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

try:  # 优先使用 SciPy 的匈牙利算法，缺省时回退到简单贪心。
    from scipy.optimize import linear_sum_assignment  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False
    linear_sum_assignment = None

# ----------------------------- box ops -----------------------------


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w.clamp(min=0.0)), (y_c - 0.5 * h.clamp(min=0.0)),
         (x_c + 0.5 * w.clamp(min=0.0)), (y_c + 0.5 * h.clamp(min=0.0))]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    # degenerate boxes gives inf / nan results
    # so do an early check
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


# ----------------------------- utils -----------------------------


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.Tensor]:
    """计算 top-k 准确率。"""
    if target.numel() == 0:
        return [torch.tensor(0.0, device=output.device)]

    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / target.numel()))
    return res


def is_dist_avail_and_initialized() -> bool:
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return torch.distributed.get_world_size()


# ----------------------------- matching -----------------------------


def point_sample(input: torch.Tensor, point_coords: torch.Tensor, **kwargs) -> torch.Tensor:
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    hw = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")
    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))
    return loss / hw


def _hungarian(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if _HAS_SCIPY and linear_sum_assignment is not None:  # type: ignore
        indices = [linear_sum_assignment(c) for c in cost.split(1, 0)]
        indices = [(torch.as_tensor(i[0]), torch.as_tensor(i[1])) for i in indices]
        return torch.cat([i for i, _ in indices]), torch.cat([j for _, j in indices])
    # 简单贪心回退
    logging.warning("SciPy 不可用，Hungarian 匹配使用贪心回退，结果可能次优。")
    indices = []
    for c in cost:
        c = c.clone()
        rows, cols = [], []
        while c.numel() > 0:
            min_idx = c.argmin()
            r = min_idx // c.shape[1]
            col = min_idx % c.shape[1]
            rows.append(r)
            cols.append(col)
            c = c[r != torch.arange(c.shape[0], device=c.device)][:, col != torch.arange(c.shape[1], device=c.device)]
            if c.numel() == 0:
                break
        indices.append((torch.tensor(rows), torch.tensor(cols)))
    return torch.cat([i for i, _ in indices]), torch.cat([j for _, j in indices])


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1,
                 focal_alpha: float = 0.25, mask_point_sample_ratio: int = 16,
                 cost_mask_ce: float = 1, cost_mask_dice: float = 1) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha
        self.mask_point_sample_ratio = mask_point_sample_ratio
        self.cost_mask_ce = cost_mask_ce
        self.cost_mask_dice = cost_mask_dice

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], group_detr: int = 1):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        flat_pred_logits = outputs["pred_logits"].flatten(0, 1)
        out_prob = flat_pred_logits.sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        masks_present = "masks" in targets[0] and "pred_masks" in outputs
        if masks_present:
            tgt_masks = torch.cat([v["masks"] for v in targets])
            out_masks = outputs["pred_masks"].flatten(0, 1)

        giou = generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost_giou = -giou

        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-F.logsigmoid(-flat_pred_logits))
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-F.logsigmoid(flat_pred_logits))
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        if masks_present:
            num_points = max(1, out_masks.shape[-2] * out_masks.shape[-1] // self.mask_point_sample_ratio)
            tgt_masks = tgt_masks.to(out_masks.dtype)
            point_coords = torch.rand(1, num_points, 2, device=out_masks.device)
            pred_masks_logits = point_sample(out_masks.unsqueeze(1), point_coords.repeat(out_masks.shape[0], 1, 1),
                                             align_corners=False).squeeze(1)
            tgt_masks_flat = point_sample(tgt_masks.unsqueeze(1), point_coords.repeat(tgt_masks.shape[0], 1, 1),
                                          align_corners=False, mode="nearest").squeeze(1)
            cost_mask_ce = batch_sigmoid_ce_loss(pred_masks_logits, tgt_masks_flat)
            cost_mask_dice = batch_dice_loss(pred_masks_logits, tgt_masks_flat)

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        if masks_present:
            C = C + self.cost_mask_ce * cost_mask_ce + self.cost_mask_dice * cost_mask_dice
        C = C.view(bs, num_queries, -1).float().cpu()

        # 防止 inf/nan 进入匈牙利算法，统一替换为有限大数。
        C = torch.nan_to_num(C, nan=1e6, posinf=1e6, neginf=1e6)

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        g_num_queries = num_queries // group_detr
        C_list = C.split(g_num_queries, dim=1)
        for g_i in range(group_detr):
            C_g = C_list[g_i]
            indices_g = [linear_sum_assignment(c[i]) if _HAS_SCIPY and linear_sum_assignment is not None
                         else (torch.arange(len(c[i])), torch.arange(c[i].shape[1])) for i, c in enumerate(C_g.split(sizes, -1))]
            if _HAS_SCIPY and linear_sum_assignment is not None:
                indices_g = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices_g]
            else:  # fallback: greedy diag
                indices_g = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices_g]
            if g_i == 0:
                indices = indices_g
            else:
                indices = [
                    (torch.cat([indice1[0], indice2[0] + g_num_queries * g_i]),
                     torch.cat([indice1[1], indice2[1]]))
                    for indice1, indice2 in zip(indices, indices_g)
                ]
        return indices


# ----------------------------- losses -----------------------------


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def sigmoid_varifocal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    focal_weight = targets * (targets > 0.0).float() + \
        (1 - alpha) * (prob - targets).abs().pow(gamma) * \
        (targets <= 0.0).float()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = ce_loss * focal_weight

    return loss.mean(1).sum() / num_boxes


def position_supervised_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = ce_loss * (torch.abs(targets - prob) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * (targets > 0.0).float() + (1 - alpha) * (targets <= 0.0).float()
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


def calculate_uncertainty(logits):
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio=3, importance_sample_ratio=0.75
):
    assert oversample_ratio >= 1
    assert 0 <= importance_sample_ratio <= 1
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords


# ----------------------------- Criterion -----------------------------


@dataclass
class MatcherLossConfig:
    set_cost_class: float
    set_cost_bbox: float
    set_cost_giou: float
    mask_point_sample_ratio: int = 16
    mask_ce_loss_coef: float = 1.0
    mask_dice_loss_coef: float = 1.0


class SetCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: Dict[str, float],
        focal_alpha: float,
        losses: List[str],
        group_detr: int = 1,
        sum_group_losses: bool = False,
        use_varifocal_loss: bool = False,
        use_position_supervised_loss: bool = False,
        ia_bce_loss: bool = False,
        mask_point_sample_ratio: int = 16,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.group_detr = group_detr
        self.sum_group_losses = sum_group_losses
        self.use_varifocal_loss = use_varifocal_loss
        self.use_position_supervised_loss = use_position_supervised_loss
        self.ia_bce_loss = ia_bce_loss
        self.mask_point_sample_ratio = mask_point_sample_ratio

    # ---------------- Loss definitions ----------------
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        if self.ia_bce_loss:
            alpha = self.focal_alpha
            gamma = 2
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            iou_targets = torch.diag(box_iou(
                box_cxcywh_to_xyxy(src_boxes.detach()),
                box_cxcywh_to_xyxy(target_boxes))[0])
            prob = src_logits.sigmoid()
            pos_weights = torch.zeros_like(src_logits)
            neg_weights = prob ** gamma
            pos_ind = [id for id in idx]
            pos_ind.append(target_classes_o)
            t = prob[pos_ind].pow(alpha) * iou_targets.pow(1 - alpha)
            t = torch.clamp(t, 0.01).detach()
            pos_weights[pos_ind] = t.to(pos_weights.dtype)
            neg_weights[pos_ind] = 1 - t.to(neg_weights.dtype)
            loss_ce = neg_weights * src_logits - F.logsigmoid(src_logits) * (pos_weights + neg_weights)
            loss_ce = loss_ce.sum() / num_boxes
        elif self.use_position_supervised_loss:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            iou_targets = torch.diag(box_iou(
                box_cxcywh_to_xyxy(src_boxes.detach()),
                box_cxcywh_to_xyxy(target_boxes))[0])
            cls_iou_func_targets = torch.zeros((src_logits.shape[0], src_logits.shape[1], self.num_classes),
                                               dtype=src_logits.dtype, device=src_logits.device)
            pos_ind = [id for id in idx]
            pos_ind.append(target_classes_o)
            cls_iou_func_targets[pos_ind] = iou_targets
            norm_targets = cls_iou_func_targets \
                / (cls_iou_func_targets.view(cls_iou_func_targets.shape[0], -1, 1).amax(1, True) + 1e-8)
            loss_ce = position_supervised_loss(src_logits, norm_targets, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        elif self.use_varifocal_loss:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            iou_targets = torch.diag(box_iou(
                box_cxcywh_to_xyxy(src_boxes.detach()),
                box_cxcywh_to_xyxy(target_boxes))[0])
            cls_iou_targets = torch.zeros((src_logits.shape[0], src_logits.shape[1], self.num_classes),
                                          dtype=src_logits.dtype, device=src_logits.device)
            pos_ind = [id for id in idx]
            pos_ind.append(target_classes_o)
            cls_iou_targets[pos_ind] = iou_targets
            loss_ce = sigmoid_varifocal_loss(src_logits, cls_iou_targets, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        else:
            target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o
            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                                dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]
            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}
        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return {'cardinality_error': card_err}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        assert 'pred_masks' in outputs, "pred_masks missing in model outputs"
        pred_masks = outputs['pred_masks']
        idx = self._get_src_permutation_idx(indices)
        src_masks = pred_masks[idx]
        if src_masks.numel() == 0:
            return {'loss_mask_ce': src_masks.sum(), 'loss_mask_dice': src_masks.sum()}
        target_masks = torch.cat([t['masks'][j] for t, (_, j) in zip(targets, indices)], dim=0)
        src_masks = src_masks.unsqueeze(1)
        target_masks = target_masks.unsqueeze(1).float()
        num_points = max(src_masks.shape[-2], src_masks.shape[-2] * src_masks.shape[-1] // self.mask_point_sample_ratio)
        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                num_points,
                3,
                0.75,
            )
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
                mode="nearest",
            ).squeeze(1)
        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)
        losses = {
            "loss_mask_ce": sigmoid_ce_loss(point_logits, point_labels, num_boxes),
            "loss_mask_dice": dice_loss(point_logits, point_labels, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }
        if loss not in loss_map:
            raise KeyError(f"不支持的 loss 类型：{loss}")
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        group_detr = self.group_detr if self.training else 1
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets, group_detr=group_detr)

        num_boxes = sum(len(t["labels"]) for t in targets)
        if not self.sum_group_losses:
            num_boxes = num_boxes * group_detr
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses: Dict[str, torch.Tensor] = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, group_detr=group_detr)
                for loss in self.losses:
                    kwargs = {'log': False} if loss == 'labels' else {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            indices = self.matcher(enc_outputs, targets, group_detr=group_detr)
            for loss in self.losses:
                kwargs = {'log': False} if loss == 'labels' else {}
                l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # 按 weight_dict 缩放（未在 CompositeCriterion 再加权），class_error 不在 weight_dict 中保持原值。
        scaled = {}
        for k, v in losses.items():
            weight = self.weight_dict.get(k, 1.0)
            if weight == 0:
                continue
            scaled[k] = v * weight
        return scaled


# ----------------------------- PostProcess -----------------------------


class PostProcess(nn.Module):
    def __init__(self, num_select: int = 300) -> None:
        super().__init__()
        self.num_select = num_select

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        out_masks = outputs.get('pred_masks', None)

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = []
        if out_masks is not None:
            for i in range(out_masks.shape[0]):
                res_i = {'scores': scores[i], 'labels': labels[i], 'boxes': boxes[i]}
                k_idx = topk_boxes[i]
                masks_i = torch.gather(out_masks[i], 0, k_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, out_masks.shape[-2], out_masks.shape[-1]))
                h, w = target_sizes[i].tolist()
                masks_i = F.interpolate(masks_i.unsqueeze(1), size=(int(h), int(w)), mode='bilinear', align_corners=False)
                res_i['masks'] = masks_i > 0.0
                results.append(res_i)
        else:
            results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results
