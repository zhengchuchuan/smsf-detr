"""
Prototype bank utilities and dual-prototype matching blocks.

These modules implement Stage 2 of the oil-spill prototype plan: loading `.npz`
prototype files, projecting them to tensors, and providing reusable similarity /
contrastive heads for later integration into training or inference pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SimilarityType = Literal["cosine", "dot"]
ContrastiveMode = Literal["max_minus_max", "weighted_diff"]


@dataclass
class PrototypeMeta:
    oil_count: int
    background_count: int
    feature_dim: int
    raw_meta: Optional[Dict[str, Any]] = None


class PrototypeBank(nn.Module):
    """Container that stores oil / background prototypes as buffers."""

    def __init__(
        self,
        oil: torch.Tensor,
        background: torch.Tensor,
        *,
        meta: Optional[PrototypeMeta] = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if oil.ndim != 2 or background.ndim != 2:
            raise ValueError("Oil/background prototypes must be 2D tensors.")
        if oil.size(1) != background.size(1):
            raise ValueError("Oil/background feature dimensions must match.")
        if normalize:
            oil = F.normalize(oil, dim=1)
            background = F.normalize(background, dim=1)
        self.register_buffer("oil", oil)
        self.register_buffer("background", background)
        self.feature_dim = oil.size(1)
        self.meta = meta

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        normalize: bool = True,
        device: Optional[torch.device] = None,
    ) -> "PrototypeBank":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"prototype file not found: {path}")
        payload = np.load(path, allow_pickle=True)
        oil = torch.from_numpy(payload["oil"]).float()
        background = torch.from_numpy(payload["background"]).float()
        meta_obj: Optional[PrototypeMeta] = None
        meta_raw = payload.get("meta")
        if meta_raw is not None:
            try:
                meta_dict = (
                    meta_raw if isinstance(meta_raw, dict) else json_safe_load(meta_raw)
                )
                meta_obj = PrototypeMeta(
                    oil_count=int(meta_dict.get("oil_count", oil.shape[0])),
                    background_count=int(meta_dict.get("background_count", background.shape[0])),
                    feature_dim=int(meta_dict.get("feature_dim", oil.shape[1])),
                    raw_meta=meta_dict,
                )
            except Exception:
                meta_obj = PrototypeMeta(
                    oil_count=oil.shape[0],
                    background_count=background.shape[0],
                    feature_dim=oil.shape[1],
                    raw_meta=None,
                )
        bank = cls(oil, background, meta=meta_obj, normalize=normalize)
        if device is not None:
            bank = bank.to(device)
        return bank

    def extra_repr(self) -> str:
        oil_n = self.oil.shape[0]
        bg_n = self.background.shape[0]
        return f"oil={oil_n}, background={bg_n}, dim={self.feature_dim}"


def json_safe_load(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, (bytes, str)):
        import json

        return json.loads(payload)
    raise TypeError(f"Unsupported meta payload: {type(payload)}")


class DualPrototypeMatcher(nn.Module):
    """Compute similarity maps between feature maps and prototype banks."""

    def __init__(
        self,
        bank: PrototypeBank,
        *,
        similarity: SimilarityType = "cosine",
        feature_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.bank = bank
        self.similarity = similarity
        self.feature_normalize = feature_normalize

    def forward(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feature_map: Tensor[B, C, H, W]

        Returns:
            (oil_sim, bg_sim):
                oil_sim -> [B, H, W, K_oil]
                bg_sim  -> [B, H, W, K_bg]
        """
        if feature_map.ndim != 4:
            raise ValueError("feature_map must be [B, C, H, W]")
        b, c, h, w = feature_map.shape
        if c != self.bank.feature_dim:
            raise ValueError(
                f"Feature dim mismatch: map has {c}, prototypes have {self.bank.feature_dim}"
            )
        feats = feature_map.permute(0, 2, 3, 1).reshape(-1, c)
        if self.feature_normalize or self.similarity == "cosine":
            feats = F.normalize(feats, dim=1)
        oil = self.bank.oil
        background = self.bank.background
        oil_sim = compute_similarity(feats, oil, self.similarity)
        bg_sim = compute_similarity(feats, background, self.similarity)
        oil_sim = oil_sim.view(b, h, w, -1)
        bg_sim = bg_sim.view(b, h, w, -1)
        return oil_sim, bg_sim


def compute_similarity(
    feats: torch.Tensor,
    proto: torch.Tensor,
    mode: SimilarityType,
) -> torch.Tensor:
    if mode == "cosine":
        proto_norm = F.normalize(proto, dim=1)
        return feats @ proto_norm.t()
    if mode == "dot":
        return feats @ proto.t()
    raise ValueError(f"unsupported similarity mode: {mode}")


class ContrastiveDualHead(nn.Module):
    """Contrastive scoring head that merges dual similarity maps."""

    def __init__(
        self,
        oil_channels: int,
        bg_channels: int,
        *,
        mode: ContrastiveMode = "max_minus_max",
    ) -> None:
        super().__init__()
        self.mode = mode
        if mode == "weighted_diff":
            self.weight_oil = nn.Parameter(torch.ones(oil_channels) / max(oil_channels, 1))
            self.weight_bg = nn.Parameter(torch.ones(bg_channels) / max(bg_channels, 1))
            self.lambda_bg = nn.Parameter(torch.tensor(1.0))

    def forward(self, oil_sim: torch.Tensor, bg_sim: torch.Tensor) -> torch.Tensor:
        """
        oil_sim: [B, H, W, K_oil]
        bg_sim:  [B, H, W, K_bg]
        Returns: [B, H, W] contrastive score map
        """
        if oil_sim.ndim != 4 or bg_sim.ndim != 4:
            raise ValueError("Similarity maps must be 4D tensors.")
        if self.mode == "max_minus_max":
            score_oil = oil_sim.max(dim=-1).values
            score_bg = bg_sim.max(dim=-1).values
            return score_oil - score_bg
        if self.mode == "weighted_diff":
            weight_oil = torch.softmax(self.weight_oil, dim=0)
            weight_bg = torch.softmax(self.weight_bg, dim=0)
            score_oil = (oil_sim * weight_oil).sum(dim=-1)
            score_bg = (bg_sim * weight_bg).sum(dim=-1)
            return score_oil - self.lambda_bg * score_bg
        raise ValueError(f"Unsupported contrastive mode: {self.mode}")
