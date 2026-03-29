"""
ComfyUI custom nodes: CVLFace KP-RPE face verification (detection, alignment, embeddings).
Clone this repository into ComfyUI/custom_nodes/comfyui-cvlface-verification (or similar).
All weights are read from ComfyUI/models subfolders only — no downloads, no Hugging Face API.
"""

import os
import sys
import traceback

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

try:
    import folder_paths

    _models = folder_paths.models_dir
    for _name in ("cvlface", "insightface"):
        _p = os.path.join(_models, _name)
        os.makedirs(_p, exist_ok=True)
    _cvl = os.path.join(_models, "cvlface")
    if hasattr(folder_paths, "add_model_folder_path"):
        folder_paths.add_model_folder_path("cvlface", _cvl, False)
except Exception:
    pass

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    import cvlface_nodes

    NODE_CLASS_MAPPINGS = cvlface_nodes.NODE_CLASS_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS = cvlface_nodes.NODE_DISPLAY_NAME_MAPPINGS
except Exception:
    print(
        "[comfyui-CVLFace-Verification] Failed to import cvlface_nodes "
        "(see traceback; common cause: missing pip deps such as insightface, transformers, opencv):\n"
        + traceback.format_exc()
    )

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
