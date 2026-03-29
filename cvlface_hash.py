"""Input fingerprints for ComfyUI IS_CHANGED caching."""

from __future__ import annotations

import hashlib
from typing import Any, Optional

import numpy as np
import torch


def tensor_digest(t: Optional[torch.Tensor]) -> str:
    if t is None:
        return "∅"
    arr = t.detach().cpu().contiguous().float().numpy()
    h = hashlib.sha256()
    h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    h.update(arr.tobytes())
    return h.hexdigest()


def digest_any(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        if p is None:
            h.update(b"none")
        elif isinstance(p, torch.Tensor):
            h.update(tensor_digest(p).encode())
        elif isinstance(p, (list, tuple)):
            h.update(repr(p).encode())
        else:
            h.update(repr(p).encode())
    return h.hexdigest()
