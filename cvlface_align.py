"""
Face detection + alignment via InsightFace (buffalo_l: 2d106 + 3d68).
Maps dense landmarks to five ArcFace-style points, then similarity warp to 112×112.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import cv2
import numpy as np
import torch

from cvlface_types import AlignedFaceBundle, FaceMeta

AlignMode = Literal["2d106", "3d68", "auto"]

# InsightFace coarse 106 layout (2d106det): derive five ArcFace-style control points.
# Nose / mouth indices follow common JD-106 diagrams; tune IDX_* if your pipeline drifts.
IDX_106_LEFT_EYE = list(range(33, 43))
IDX_106_RIGHT_EYE = list(range(87, 97))
IDX_106_NOSE_TIP = 57
IDX_106_MOUTH_LEFT = 76
IDX_106_MOUTH_RIGHT = 82


def _mean_points(pts: np.ndarray, indices: List[int]) -> np.ndarray:
    sel = pts[np.asarray(indices, dtype=np.int64)]
    return sel.mean(axis=0).astype(np.float32)


def landmark_106_to_arcface5(pts106: np.ndarray) -> np.ndarray:
    p = pts106.astype(np.float32)
    le = _mean_points(p, IDX_106_LEFT_EYE)
    re = _mean_points(p, IDX_106_RIGHT_EYE)
    nose = p[IDX_106_NOSE_TIP]
    ml = p[IDX_106_MOUTH_LEFT]
    mr = p[IDX_106_MOUTH_RIGHT]
    return np.stack([le, re, nose, ml, mr], axis=0)


def landmark_68_to_arcface5(pts68: np.ndarray) -> np.ndarray:
    """iBUG 68 → five points (x,y only; ignore z if present)."""
    p = pts68.astype(np.float32)
    le = p[36:42].mean(axis=0)
    re = p[42:48].mean(axis=0)
    nose = p[30]
    ml = p[48]
    mr = p[54]
    return np.stack([le, re, nose, ml, mr], axis=0)


def _ensure_insightface():
    import insightface  # noqa: F401
    from insightface.app import FaceAnalysis

    return FaceAnalysis


def _get_analyzer(name: Optional[str] = None, root: Optional[str] = None, providers=None):
    from cvlface_paths import INSIGHTFACE_BUFFALO_NAME

    name = name or INSIGHTFACE_BUFFALO_NAME
    if not root:
        from cvlface_paths import insightface_pack_root

        root = insightface_pack_root()
    FaceAnalysis = _ensure_insightface()
    allowed = ["detection", "landmark_2d_106", "landmark_3d_68"]
    app = FaceAnalysis(name=name, root=root, allowed_modules=allowed, providers=providers)
    return app


_ANALYZER = None
_ANALYZER_KEY = None


def get_face_analyzer(
    det_size: int = 640,
    det_thresh: float = 0.5,
    ctx_id: int = 0,
    root: Optional[str] = None,
    providers=None,
):
    global _ANALYZER, _ANALYZER_KEY
    from cvlface_paths import assert_insightface_buffalo

    if not root:
        from cvlface_paths import insightface_pack_root

        root = insightface_pack_root()
    assert_insightface_buffalo(root)
    key = (det_size, det_thresh, ctx_id, root, tuple(providers) if providers else None)
    if _ANALYZER is None or _ANALYZER_KEY != key:
        app = _get_analyzer(root=root, providers=providers)
        app.prepare(ctx_id=ctx_id, det_thresh=det_thresh, det_size=(det_size, det_size))
        _ANALYZER = app
        _ANALYZER_KEY = key
    return _ANALYZER


def comfy_bhwc_to_bgr_uint8(image_bhwc: torch.Tensor) -> np.ndarray:
    """ComfyUI IMAGE float [0,1] BHWC → uint8 BGR (single image)."""
    t = image_bhwc[0].detach().cpu().clamp(0.0, 1.0).numpy()
    rgb = (t * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_uint8_to_comfy_bhwc(bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).unsqueeze(0)


def _select_face_index(faces, policy: str, face_index: int) -> int:
    if not faces:
        raise RuntimeError("No face detected.")
    if policy == "largest_area":
        areas = []
        for f in faces:
            b = f.bbox
            areas.append(float((b[2] - b[0]) * (b[3] - b[1])))
        return int(np.argmax(np.array(areas)))
    if policy == "highest_score":
        scores = [float(f.det_score) for f in faces]
        return int(np.argmax(np.array(scores)))
    idx = int(face_index)
    if idx < 0 or idx >= len(faces):
        raise RuntimeError(f"face_index {idx} out of range (0..{len(faces) - 1}).")
    return idx


def _resolve_align_mode(mode: AlignMode, face) -> Tuple[str, np.ndarray, int]:
    """Returns (used_mode, five_points_image_space, landmark_count_for_meta)."""
    lmk106 = face.get("landmark_2d_106")
    lmk68 = face.get("landmark_3d_68")
    pose = face.get("pose")
    yaw = float(pose[1]) * 180.0 / np.pi if pose is not None else 0.0

    use_3d = False
    if mode == "3d68":
        use_3d = True
    elif mode == "auto":
        if abs(yaw) >= 35.0:
            use_3d = True
        elif lmk106 is None and lmk68 is not None:
            use_3d = True

    if use_3d:
        if lmk68 is None:
            raise RuntimeError("3D / auto alignment needs landmark_3d_68 (buffalo_l).")
        pts5 = landmark_68_to_arcface5(lmk68[:, :2])
        return "3d68", pts5, 68

    if lmk106 is None:
        if face.kps is None:
            raise RuntimeError("No 106 landmarks and no detector keypoints.")
        pts5 = face.kps.astype(np.float32)
        return "2d106_fallback_kps", pts5, 5

    pts5 = landmark_106_to_arcface5(lmk106)
    return "2d106", pts5, 106


def align_one_face(
    image_bhwc: torch.Tensor,
    align_mode: str,
    face_selection: str,
    face_index: int,
    det_size: int,
    det_thresh: float,
    ctx_id: int,
    insightface_root: str = "",
) -> AlignedFaceBundle:
    from insightface.utils import face_align

    if image_bhwc.shape[0] > 1:
        image_bhwc = image_bhwc[0:1]

    app = get_face_analyzer(
        det_size=det_size,
        det_thresh=det_thresh,
        ctx_id=ctx_id,
        root=(insightface_root.strip() or None),
    )
    img_bgr = comfy_bhwc_to_bgr_uint8(image_bhwc)
    faces = app.get(img_bgr)
    idx = _select_face_index(faces, face_selection, face_index)
    face = faces[idx]

    used_mode, pts5, lmk_n = _resolve_align_mode(align_mode, face)  # type: ignore[arg-type]
    warped_bgr, m_affine = face_align.norm_crop2(img_bgr, pts5, image_size=112, mode="arcface")
    kps112 = face_align.trans_points2d(pts5, m_affine)

    dense_112 = None
    l106 = face.get("landmark_2d_106")
    l68 = face.get("landmark_3d_68")
    if l106 is not None:
        dense_112 = face_align.trans_points2d(l106.astype(np.float32), m_affine)
    elif l68 is not None:
        dense_112 = face_align.trans_points2d(l68[:, :2].astype(np.float32), m_affine)

    yaw_deg = None
    if face.pose is not None:
        yaw_deg = float(face.pose[1]) * 180.0 / np.pi

    meta = FaceMeta(
        face_index=idx,
        det_score=float(face.det_score),
        align_mode=used_mode,
        yaw_deg=yaw_deg,
        bbox=face.bbox.astype(np.float32),
        keypoints_112=kps112.astype(np.float32),
        landmark_count=lmk_n,
    )
    return AlignedFaceBundle(
        image_bhwc=bgr_uint8_to_comfy_bhwc(warped_bgr),
        keypoints_112=kps112.astype(np.float32),
        meta=meta,
        dense_landmarks_112=dense_112,
    )


def draw_landmarks_on_crop(
    crop_bhwc: torch.Tensor,
    keypoints_112: np.ndarray,
    full_lmk_112: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """RGB preview: keypoints on aligned crop."""
    rgb = (crop_bhwc[0].detach().cpu().clamp(0, 1).numpy() * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if full_lmk_112 is not None:
        for (x, y) in full_lmk_112.astype(np.int32):
            cv2.circle(bgr, (int(x), int(y)), 1, (0, 255, 0), -1)
    for (x, y) in keypoints_112.astype(np.int32):
        cv2.circle(bgr, (int(x), int(y)), 3, (0, 0, 255), -1)
    return bgr_uint8_to_comfy_bhwc(bgr)


def transform_landmarks_to_112(pts_image: np.ndarray, m_affine) -> np.ndarray:
    from insightface.utils import face_align

    return face_align.trans_points2d(pts_image.astype(np.float32), m_affine)
