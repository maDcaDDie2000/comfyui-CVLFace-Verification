"""
Resolve model directories under ComfyUI's models folder only (no downloads, no HF hub).
Folder names are fixed by this pack so paths are unambiguous.
"""

from __future__ import annotations

import os

CVLFACE_SUBDIR = "cvlface"
# Must match directory under ComfyUI/models/cvlface/ where the user places the checkpoint.
CVLFACE_CHECKPOINT_NAME = "vit_kprpe_webface12m"
# ComfyUI Add Node → right-click menu group (ties UI to the cvlface models area).
CVLF_NODE_MENU_CATEGORY = "CVLFace"

INSIGHTFACE_SUBDIR = "insightface"
INSIGHTFACE_BUFFALO_NAME = "buffalo_l"


def get_comfy_models_dir() -> str:
    try:
        import folder_paths

        return os.path.abspath(folder_paths.models_dir)
    except Exception:
        base = os.environ.get("COMFYUI_MODELS_BASE", "").strip()
        if base:
            return os.path.abspath(base)
        raise RuntimeError(
            "ComfyUI folder_paths is not available and COMFYUI_MODELS_BASE is unset. "
            "Install this pack inside ComfyUI/custom_nodes."
        )


def cvlface_checkpoint_dir() -> str:
    """Absolute path: models/cvlface/vit_kprpe_webface12m/"""
    return os.path.abspath(
        os.path.join(get_comfy_models_dir(), CVLFACE_SUBDIR, CVLFACE_CHECKPOINT_NAME)
    )


def insightface_pack_root() -> str:
    """Absolute path: models/insightface/ — contains models/buffalo_l/"""
    return os.path.abspath(os.path.join(get_comfy_models_dir(), INSIGHTFACE_SUBDIR))


def assert_cvlface_checkpoint(path: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"CVLFace model directory does not exist:\n  {path}\n"
            f"Place the full checkpoint tree at:\n"
            f"  ComfyUI/models/{CVLFACE_SUBDIR}/{CVLFACE_CHECKPOINT_NAME}/\n"
            f"with config.json at that path (see README)."
        )
    cfg = os.path.join(path, "config.json")
    if not os.path.isfile(cfg):
        raise FileNotFoundError(
            f"Missing config.json in CVLFace model directory:\n  {path}"
        )


def assert_insightface_buffalo(path: str) -> None:
    buffalo = os.path.join(path, "models", INSIGHTFACE_BUFFALO_NAME)
    if not os.path.isdir(buffalo):
        raise FileNotFoundError(
            f"InsightFace buffalo_l folder not found:\n  {buffalo}\n"
            f"Place the buffalo_l ONNX pack at:\n"
            f"  ComfyUI/models/{INSIGHTFACE_SUBDIR}/models/{INSIGHTFACE_BUFFALO_NAME}/\n"
            f"(see README)."
        )
    onnx = [f for f in os.listdir(buffalo) if f.endswith(".onnx")]
    if not onnx:
        raise FileNotFoundError(f"No .onnx files in {buffalo}")


def checkpoint_mtime_key(path: str) -> str:
    """Cheap invalidation fingerprint for IS_CHANGED."""
    cfg = os.path.join(path, "config.json")
    try:
        return str(int(os.path.getmtime(cfg)))
    except OSError:
        return "missing"
