"""CVLFace KP-RPE embedding extraction from a local checkpoint directory only."""

from __future__ import annotations

import glob
import inspect
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


def _is_meta_device(d) -> bool:
    if d is None:
        return False
    if isinstance(d, torch.device):
        return d.type == "meta"
    return "meta" in str(d).lower()


def _restore_default_device_safe(prev) -> None:
    """
    Never restore PyTorch's default device to ``meta`` — CVLFace uses factory ops that
    follow the default device; ``meta`` breaks ``.item()`` and interrupts nodes like
    Face Compare that run forward after the Loader has finished.
    """
    if not hasattr(torch, "set_default_device"):
        return
    if prev is None:
        torch.set_default_device("cpu")
        return
    if _is_meta_device(prev):
        torch.set_default_device("cpu")
        return
    try:
        torch.set_default_device(prev)
    except Exception:
        torch.set_default_device("cpu")


@contextmanager
def _torch_default_device_for_inference(compute_device: torch.device):
    """Run CVLFace forward with a concrete default device (matches model weights)."""
    if not hasattr(torch, "set_default_device"):
        yield
        return
    prev = None
    try:
        prev = torch.get_default_device()
    except Exception:
        prev = None
    torch.set_default_device(compute_device)
    try:
        yield
    finally:
        _restore_default_device_safe(prev)


@contextmanager
def _cvlface_weight_path_patch(checkpoint_root: str):
    """
    CVLFace ``wrapper.py`` always loads ``pretrained_model/model.pt``. Hugging Face
    snapshots often ship ``model.safetensors`` at the repo root instead. Patch the
    checkpoint's ``load_state_dict_from_path``.

    Paths must be resolved with **absolute** paths under ``checkpoint_root``: Transformers
    may change ``os.getcwd()`` during ``from_pretrained``, so relative paths are unreliable.

    ``BaseModel.load_state_dict_from_path`` uses the module global in ``models.base``;
    patch both ``models.base`` and ``models.base.utils``.
    """
    import models.base as cvl_b
    import models.base.utils as cvl_u

    root = os.path.abspath(checkpoint_root)
    orig_fn = cvl_b.load_state_dict_from_path

    def _resolve_weights_path(p: str) -> str:
        key = p.replace("\\", "/")
        if key != "pretrained_model/model.pt":
            return p
        candidates = [
            os.path.join(root, "pretrained_model", "model.pt"),
            os.path.join(root, "pretrained_model", "model.safetensors"),
            os.path.join(root, "model.safetensors"),
            os.path.join(root, "pytorch_model.bin"),
            os.path.join(root, "pretrained_model", "pytorch_model.bin"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        lone = sorted(glob.glob(os.path.join(root, "*.safetensors")))
        if len(lone) == 1 and os.path.isfile(lone[0]):
            return lone[0]
        lone_pm = sorted(glob.glob(os.path.join(root, "pretrained_model", "*.safetensors")))
        if len(lone_pm) == 1 and os.path.isfile(lone_pm[0]):
            return lone_pm[0]
        hint = ""
        try:
            top = sorted(os.listdir(root))[:30]
            hint = f"\nCheckpoint folder (absolute): {root}\nTop-level entries: {top}\n"
        except OSError:
            hint = f"\nCheckpoint folder (absolute): {root}\n"
        raise FileNotFoundError(
            "CVLFace: no weight file found. Expected one of (under that folder):\n"
            "  pretrained_model/model.pt\n"
            "  pretrained_model/model.safetensors\n"
            "  model.safetensors\n"
            "  pytorch_model.bin\n"
            "  or a single *.safetensors at the checkpoint root or in pretrained_model/\n"
            f"{hint}"
        )

    def patched(p: str):
        return orig_fn(_resolve_weights_path(p))

    cvl_b.load_state_dict_from_path = patched
    cvl_u.load_state_dict_from_path = patched
    try:
        yield
    finally:
        cvl_b.load_state_dict_from_path = orig_fn
        cvl_u.load_state_dict_from_path = orig_fn


@contextmanager
def _cvlface_force_linspace_cpu():
    """
    CVLFace vit.py builds drop-path rates with ``torch.linspace(...).item()``. Transformers
    can still run ``__init__`` under a meta/accelerate context where the default device is
    ``meta``, so ``linspace`` returns meta tensors and ``.item()`` crashes. Force concrete
    CPU linspace whenever the call would otherwise use ``meta`` or omit ``device``.
    """
    real = torch.linspace

    def patched(*args, **kwargs):
        kwargs = dict(kwargs)
        d = kwargs.get("device")
        if d is None or _is_meta_device(d):
            kwargs["device"] = torch.device("cpu")
        return real(*args, **kwargs)

    torch.linspace = patched  # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.linspace = real  # type: ignore[method-assign]


def _ensure_post_init_for_tied_weights(model) -> None:
    """
    Transformers >= ~4.50 expects ``all_tied_weights_keys`` (set in ``post_init()``).
    CVLFace ``wrapper.py`` calls ``PreTrainedModel.__init__`` but never ``post_init()``,
    so ``from_pretrained`` fails in ``_finalize_model_loading``.
    """
    try:
        _ = model.all_tied_weights_keys  # noqa: F841
        return
    except AttributeError:
        pass
    post_init = getattr(model, "post_init", None)
    if callable(post_init):
        post_init()


@contextmanager
def _cvlface_transformers_compat_patch():
    """
    Patch PreTrainedModel finalize so remote-code checkpoints missing post_init still load.
    Scoped to the CVLFace ``from_pretrained`` call only.
    """
    try:
        from transformers.modeling_utils import PreTrainedModel
    except ImportError:
        yield
        return

    orig_finalize = PreTrainedModel._finalize_model_loading.__func__

    @classmethod
    def _finalize_with_post_init(cls, model, load_config, loading_info):
        _ensure_post_init_for_tied_weights(model)
        return orig_finalize(cls, model, load_config, loading_info)

    PreTrainedModel._finalize_model_loading = _finalize_with_post_init
    try:
        yield
    finally:
        PreTrainedModel._finalize_model_loading = classmethod(orig_finalize)


@contextmanager
def _cvlface_wrapper_post_init_patch():
    """
    Belt-and-suspenders: patch CVLFace ``wrapper.CVLFaceRecognitionModel.__init__`` so
    ``post_init()`` runs after the inner ViT is built when transformers imports wrapper
    from the checkpoint directory.
    """
    try:
        import wrapper as cvl_wrapper
    except ImportError:
        yield
        return

    cls = getattr(cvl_wrapper, "CVLFaceRecognitionModel", None)
    if cls is None:
        yield
        return

    orig_init = cls.__init__

    def patched_init(self, cfg, *args, **kwargs):
        orig_init(self, cfg, *args, **kwargs)
        _ensure_post_init_for_tied_weights(self)

    cls.__init__ = patched_init
    try:
        yield
    finally:
        cls.__init__ = orig_init


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

    # Transformers may use meta-device init when low_cpu_mem_usage=True; CVLFace vit.py
    # does torch.linspace(...).item() in __init__, which breaks on meta tensors.
    fp_kw: dict = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "local_files_only": True,
    }
    _sig = inspect.signature(AutoModel.from_pretrained)
    if "low_cpu_mem_usage" in _sig.parameters:
        fp_kw["low_cpu_mem_usage"] = False
    if "device_map" in _sig.parameters:
        fp_kw["device_map"] = None

    prev_dd = None
    _restore_dd = False
    if hasattr(torch, "set_default_device") and hasattr(torch, "get_default_device"):
        try:
            prev_dd = torch.get_default_device()
        except Exception:
            prev_dd = None
        torch.set_default_device("cpu")
        _restore_dd = True

    prev_off = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        try:
            with _checkpoint_import_isolation(path):
                with _cvlface_transformers_compat_patch():
                    with _cvlface_wrapper_post_init_patch():
                        with _cvlface_weight_path_patch(path):
                            with _cvlface_force_linspace_cpu():
                                model = AutoModel.from_pretrained(path, **fp_kw)
        finally:
            _evict_checkpoint_models_modules(path)
    finally:
        if prev_off is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_off
        if _restore_dd:
            _restore_default_device_safe(prev_dd)
    model.eval()
    model.to(device)
    return FaceEmbedderHandle(model=model, device=device, dtype=dtype, model_path=path)


_EMBEDDER_CACHE_VER = 7
_EMBEDDER_CACHE: dict[tuple, FaceEmbedderHandle] = {}


def get_cached_embedder(local_model_dir: str, prefer_cuda: bool) -> FaceEmbedderHandle:
    key = (os.path.abspath(local_model_dir), prefer_cuda, _EMBEDDER_CACHE_VER)
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
    if handle is None or handle.model is None:
        raise RuntimeError("Face embedder is not connected or failed to load; use CVLFace Loader first.")
    model = handle.model
    device = handle.device
    dtype = handle.dtype
    for p in model.parameters():
        if getattr(p, "is_meta", False):
            raise RuntimeError(
                "CVLFace weights are on the meta device; reload the model (restart ComfyUI, run CVLFace Loader)."
            )

    with _torch_default_device_for_inference(device):
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
