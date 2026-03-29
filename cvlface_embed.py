"""CVLFace KP-RPE embedding extraction from a local checkpoint directory only."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from cvlface_types import FaceEmbedderHandle


def _pick_device(prefer_cuda: bool) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_embedder(local_model_dir: str, prefer_cuda: bool) -> FaceEmbedderHandle:
    from transformers import AutoModel

    path = os.path.abspath(local_model_dir)
    device = _pick_device(prefer_cuda)
    dtype = torch.float32

    prev_off = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        model = AutoModel.from_pretrained(
            path,
            trust_remote_code=True,
            torch_dtype=dtype,
            local_files_only=True,
        )
    finally:
        if prev_off is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_off
    model.eval()
    model.to(device)
    return FaceEmbedderHandle(model=model, device=device, dtype=dtype, model_path=path)


_EMBEDDER_CACHE: dict[tuple, FaceEmbedderHandle] = {}


def get_cached_embedder(local_model_dir: str, prefer_cuda: bool) -> FaceEmbedderHandle:
    key = (os.path.abspath(local_model_dir), prefer_cuda)
    if key not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[key] = load_embedder(key[0], prefer_cuda)
    return _EMBEDDER_CACHE[key]


def image_bhwc_to_model_input(image_bhwc: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """BHWC [0,1] RGB → BCHW normalized [-1,1]."""
    x = image_bhwc.to(device=device, dtype=dtype)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.permute(0, 3, 1, 2)
    x = x * 2.0 - 1.0
    return x


@torch.inference_mode()
def compute_embedding(
    handle: FaceEmbedderHandle,
    aligned_bhwc: torch.Tensor,
    keypoints_112: np.ndarray,
) -> np.ndarray:
    """
    aligned_bhwc: 112×112 RGB float BHWC on any device (moved to model device).
    keypoints_112: (5, 2) float in aligned-crop pixel space.
    """
    model = handle.model
    device = handle.device
    dtype = handle.dtype
    x = image_bhwc_to_model_input(aligned_bhwc, device, dtype)
    kps = torch.from_numpy(keypoints_112.astype(np.float32)).view(1, 5, 2).to(device=device, dtype=dtype)

    out = model(x, kps)
    if isinstance(out, torch.Tensor):
        emb = out
    elif isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
        emb = out[0]
    elif hasattr(out, "logits"):
        emb = out.logits
    else:
        raise RuntimeError(f"Unexpected model output type: {type(out)}")
    if emb.dim() > 2:
        emb = emb.view(emb.shape[0], -1)
    emb = F.normalize(emb.float(), dim=-1)
    return emb.detach().cpu().numpy().astype(np.float32)
