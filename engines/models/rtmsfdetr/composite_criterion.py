from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from engines.core.metrics.rfdetr_loss import (
    calculate_uncertainty,
    dice_loss,
    get_uncertain_point_coords_with_randomness,
    get_world_size,
    is_dist_avail_and_initialized,
    point_sample,
    sigmoid_ce_loss,
)


class RtmsfDetrCriterionWithMasks(nn.Module):
    """
    Wrapper criterion for RT-DETRv4 (vendored) that adds instance segmentation losses.

    - Delegates detection losses to the original RTv4Criterion.
    - Optionally computes mask BCE/Dice losses (PointRend-style point sampling) when `pred_masks`
      exists in model outputs.
    """

    def __init__(
        self,
        det_criterion: nn.Module,
        *,
        mask_point_sample_ratio: int = 16,
        mask_ce_loss_coef: float = 5.0,
        mask_dice_loss_coef: float = 5.0,
        mask_aux_loss: bool = False,
    ) -> None:
        super().__init__()
        self.det_criterion = det_criterion
        self.mask_point_sample_ratio = int(mask_point_sample_ratio)
        self.mask_ce_loss_coef = float(mask_ce_loss_coef)
        self.mask_dice_loss_coef = float(mask_dice_loss_coef)
        self.mask_aux_loss = bool(mask_aux_loss)

        # Expose a merged weight_dict for logging/inspection.
        base_weight_dict = getattr(det_criterion, "weight_dict", {}) or {}
        self.weight_dict: Dict[str, float] = dict(base_weight_dict)
        self.weight_dict.setdefault("loss_mask_ce", self.mask_ce_loss_coef)
        self.weight_dict.setdefault("loss_mask_dice", self.mask_dice_loss_coef)

    @staticmethod
    def _get_first_tensor_device(outputs: Dict[str, Any]) -> torch.device:
        for v in outputs.values():
            if torch.is_tensor(v):
                return v.device
        return torch.device("cpu")

    @staticmethod
    def _get_src_permutation_idx(indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _compute_num_boxes(self, outputs: Dict[str, Any], targets: List[Dict[str, Any]]) -> float:
        num_boxes = sum(len(t.get("labels", [])) for t in targets)
        device = self._get_first_tensor_device(outputs)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return float(num_boxes)

    def _loss_masks(
        self,
        outputs: Dict[str, Any],
        targets: List[Dict[str, Any]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: float,
    ) -> Dict[str, torch.Tensor]:
        if "pred_masks" not in outputs:
            raise KeyError("pred_masks missing in model outputs")
        pred_masks: torch.Tensor = outputs["pred_masks"]  # [B, Q, H, W]

        if any("masks" not in t for t in targets):
            raise KeyError("targets missing key 'masks' (did you enable dataset return masks?)")

        idx = self._get_src_permutation_idx(indices)
        src_masks = pred_masks[idx]  # [N, H, W]
        if src_masks.numel() == 0:
            return {
                "loss_mask_ce": src_masks.sum(),
                "loss_mask_dice": src_masks.sum(),
            }

        target_masks = torch.cat([t["masks"][j] for t, (_, j) in zip(targets, indices)], dim=0)  # [N, Ht, Wt]
        src_masks = src_masks.unsqueeze(1)
        target_masks = target_masks.unsqueeze(1).float()

        num_points = max(
            int(src_masks.shape[-2]),
            int(src_masks.shape[-2] * src_masks.shape[-1] // max(1, self.mask_point_sample_ratio)),
        )
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

        return {
            "loss_mask_ce": sigmoid_ce_loss(point_logits, point_labels, num_boxes),
            "loss_mask_dice": dice_loss(point_logits, point_labels, num_boxes),
        }

    def _get_main_match_indices(
        self, outputs: Dict[str, Any], targets: List[Dict[str, Any]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        cached = getattr(self.det_criterion, "_last_match_indices", None)
        if cached is not None:
            return cached
        matcher = getattr(self.det_criterion, "matcher", None)
        if matcher is None:
            raise AttributeError("det_criterion has no matcher; cannot compute mask losses")
        # Match based on detection heads (same contract as RTv4Criterion.forward).
        outputs_wo_aux = {k: v for k, v in outputs.items() if "aux" not in k}
        return matcher(outputs_wo_aux, targets)["indices"]

    def forward(self, outputs: Dict[str, Any], targets: List[Dict[str, Any]], **kwargs) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = self.det_criterion(outputs, targets, **kwargs)

        # Only compute mask losses when model provides pred_masks.
        if "pred_masks" not in outputs:
            return losses

        indices = self._get_main_match_indices(outputs, targets)
        num_boxes = self._compute_num_boxes(outputs, targets)
        mask_losses = self._loss_masks(outputs, targets, indices, num_boxes)
        losses["loss_mask_ce"] = mask_losses["loss_mask_ce"] * float(self.mask_ce_loss_coef)
        losses["loss_mask_dice"] = mask_losses["loss_mask_dice"] * float(self.mask_dice_loss_coef)

        if self.mask_aux_loss:
            matcher = getattr(self.det_criterion, "matcher", None)
            if matcher is None:
                raise AttributeError("det_criterion has no matcher; cannot compute aux mask losses")
            aux_outputs = outputs.get("aux_outputs") or []
            for i, aux in enumerate(aux_outputs):
                if not isinstance(aux, dict) or "pred_masks" not in aux:
                    continue
                indices_i = matcher(aux, targets)["indices"]
                aux_mask_losses = self._loss_masks(aux, targets, indices_i, num_boxes)
                losses[f"loss_mask_ce_aux_{i}"] = aux_mask_losses["loss_mask_ce"] * float(self.mask_ce_loss_coef)
                losses[f"loss_mask_dice_aux_{i}"] = aux_mask_losses["loss_mask_dice"] * float(self.mask_dice_loss_coef)

        return losses


__all__ = ["RtmsfDetrCriterionWithMasks"]

