"""CVLFace KP-RPE embedding extraction from a local checkpoint directory only."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from cvlface_types import FaceEmbedderHandle


def _evict_checkpoint_models_modules(checkpoint_dir: str) -> None:
    """
    Drop ``models`` / ``models.*`` entries that were loaded from this checkpoint so a
    later ``import models`` resolves to another extension (e.g. comfyui-rmbg) again.

    Transformers may execute remote code from the HF modules cache (path differs from
    ``checkpoint_dir``); match also by the fixed checkpoint folder name.
    """
    from cvlface_paths import CVLFACE_CHECKPOINT_NAME

    ck = os.path.normcase(os.path.abspath(checkpoint_dir) + os.sep)
    tag = os.path.normcase(CVLFACE_CHECKPOINT_NAME)

    def _path_marks_cvlface(*parts: str) -> bool:
        for raw in parts:
            if not raw:
                continue
            n = os.path.normcase(str(raw))
            if n.startswith(ck) or tag in n:
                return True
        return False

    keys = [k for k in sys.modules if k == "models" or k.startswith("models.")]
    keys.sort(key=len, reverse=True)
    for key in keys:
        mod = sys.modules.get(key)
        if mod is None:
            continue
        under = False
        fn = getattr(mod, "__file__", None)
        if fn and _path_marks_cvlface(fn):
            under = True
        if not under:
            for p in getattr(mod, "__path__", []) or []:
                if _path_marks_cvlface(str(p), str(p) + os.sep):
                    under = True
                    break
        if under:
            del sys.modules[key]


def _pick_device(prefer_cuda: bool) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@contextmanager
def _checkpoint_import_isolation(checkpoint_dir: str):
    """
    CVLFace's remote wrapper does ``from models import get_model``. ComfyUI puts many
    custom_node folders on sys.path, so a different top-level ``models`` package (e.g.
    comfyui-rmbg/models) can shadow the checkpoint's ``models/``. Prepend the checkpoint
    root and chdir so imports and relative file opens match upstream expectations.
    """
    path = os.path.abspath(checkpoint_dir)
    old_cwd = os.getcwd()
    removed_from_path = False
    try:
        while path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
        removed_from_path = True
        os.chdir(path)
        yield
    finally:
        try:
            os.chdir(old_cwd)
        except OSError:
            pass
        if removed_from_path:
            try:
                sys.path.remove(path)
            except ValueError:
                pass


def load_embedder(local_model_dir: str, prefer_cuda: bool) -> FaceEmbedderHandle:
    from transformers import AutoModel

    path = os.path.abspath(local_model_dir)
    device = _pick_device(prefer_cuda)
    dtype = torch.float32

    prev_off = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        try:
            with _checkpoint_import_isolation(path):
                model = AutoModel.from_pretrained(
                    path,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    local_files_only=True,
                )
        finally:
            _evict_checkpoint_models_modules(path)
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
