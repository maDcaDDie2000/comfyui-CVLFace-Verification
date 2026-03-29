"""Custom data passed between nodes (plain objects; ComfyUI uses string type tags)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch


@dataclass
class FaceEmbedderHandle:
    """Loaded CVLFace KP-RPE model + runtime options."""

    model: Any
    device: torch.device
    dtype: torch.dtype
    model_path: str


@dataclass
class FaceMeta:
    """Debug / quality metadata for one aligned face."""

    face_index: int
    det_score: float
    align_mode: str
    yaw_deg: Optional[float]
    bbox: np.ndarray
    keypoints_112: np.ndarray
    landmark_count: int


@dataclass
class AlignedFaceBundle:
    """Aligned 112×112 crop as Comfy IMAGE tensor plus geometry for KP-RPE."""

    image_bhwc: torch.Tensor
    keypoints_112: np.ndarray
    meta: FaceMeta
    dense_landmarks_112: Optional[np.ndarray] = None


@dataclass
class FaceProfile:
    """One or more reference embeddings (L2-normalized) and optional quality weights."""

    embeddings: np.ndarray
    qualities: np.ndarray
    model_path: str
    align_mode: str
    extras: dict = field(default_factory=dict)
