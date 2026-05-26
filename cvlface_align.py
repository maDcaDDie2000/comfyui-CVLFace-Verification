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

# Sentinel cosine score when detection/alignment fails (always below any real threshold).
NO_FACE_SCORE = -1.0

# InsightFace coarse 106 layout (2d106det): derive five ArcFace-style control points.
# Nose / mouth indices follow common JD-106 diagrams; tune IDX_* if your pipeline drifts.
IDX_106_LEFT_EYE = list(range(33, 43))
IDX_106_RIGHT_EYE = list(range(87, 97))
IDX_106_NOSE_TIP = 57
IDX_106_MOUTH_LEFT = 76
IDX_106_MOUTH_RIGHT = 82

# Standard InsightFace / ArcFace five-point template for 112×112 (mouth corners sit near y≈92).
_ARCFACE_DST_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# Pull template toward center (zoom out) and shift up so chin/jaw are not clipped on tilted frontals.
ALIGN_FACE_MARGIN_SCALE = 0.82
ALIGN_FACE_SHIFT_Y = -4.0


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


def truncate_image_batch(
    images: torch.Tensor,
    max_n: int,
    label: str,
    log_prefix: str = "[comfyui-CVLFace-Verification]",
) -> torch.Tensor:
    """Keep at most ``max_n`` images; warn when truncating."""
    n = int(images.shape[0])
    if n > max_n:
        print(f"{log_prefix} {label}: batch has {n} images; truncating to {max_n}.")
        return images[:max_n]
    return images


def _draw_grid_cell(
    canvas: np.ndarray,
    x0: int,
    y0: int,
    w: int,
    h: int,
    text: str,
    bg_bgr: tuple[int, int, int],
    fg_bgr: tuple[int, int, int] = (24, 24, 24),
    font_scale: float = 0.45,
) -> None:
    canvas[y0 : y0 + h, x0 : x0 + w] = bg_bgr
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    tx = x0 + max(2, (w - tw) // 2)
    ty = y0 + (h + th) // 2
    cv2.putText(canvas, text, (tx, ty), font, font_scale, fg_bgr, thickness, cv2.LINE_AA)


def render_no_pass_placeholder() -> torch.Tensor:
    """Single-frame placeholder when no targets pass (Save Image / PreviewImage need batch ≥ 1)."""
    w, h = 480, 96
    canvas = np.full((h, w, 3), 36, dtype=np.uint8)
    _draw_grid_cell(canvas, 0, 0, w, h, "NO TARGETS PASSED THRESHOLD", (36, 36, 36), (210, 210, 210), 0.65)
    return bgr_uint8_to_comfy_bhwc(canvas)


def _is_no_face_score(score: float) -> bool:
    return float(score) <= NO_FACE_SCORE + 1e-5


def _format_grid_score(score: float, match_threshold: float) -> tuple[str, bool]:
    if _is_no_face_score(score):
        return "N/F", False
    ok = float(score) >= float(match_threshold)
    return f"{score:.2f}", ok


def render_comparison_grids(
    score_matrix: np.ndarray,
    column_aggregates: np.ndarray,
    match_threshold: float,
    max_cols: int = 10,
    label_w: int = 60,
    cell_w: int = 88,
    cell_h: int = 52,
    skipped_ref_indices: Optional[list[int]] = None,
    no_face_target_indices: Optional[list[int]] = None,
) -> torch.Tensor:
    """
    Pass/fail grid: rows = references (1-based labels), columns = targets (1-based labels).
    Skipped refs and no-face targets are still shown as rows/columns marked N/F.
    """
    score_matrix = np.asarray(score_matrix, dtype=np.float32)
    column_aggregates = np.asarray(column_aggregates, dtype=np.float32)
    r_count, t_count = score_matrix.shape
    skipped_refs = set(skipped_ref_indices or [])
    no_face_targets = set(no_face_target_indices or [])
    if r_count == 0 or t_count == 0:
        blank = np.full((cell_h, label_w + cell_w, 3), 255, dtype=np.uint8)
        return bgr_uint8_to_comfy_bhwc(blank)

    pass_bg = (200, 235, 200)
    fail_bg = (190, 190, 255)
    header_bg = (225, 225, 225)
    label_bg = (215, 215, 215)
    skipped_bg = (210, 220, 235)
    agg_pass_bg = (130, 220, 130)
    agg_fail_bg = (130, 130, 235)
    warn_label = (40, 40, 180)

    panels: list[torch.Tensor] = []
    n_panels = (t_count + max_cols - 1) // max_cols
    for panel_i, col_start in enumerate(range(0, t_count, max_cols)):
        col_end = min(col_start + max_cols, t_count)
        n_cols = col_end - col_start
        n_rows_total = 1 + r_count + 1
        grid_h = n_rows_total * cell_h
        grid_w = label_w + n_cols * cell_w
        canvas = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)

        corner = "R/T" if n_panels == 1 else f"P{panel_i + 1}"
        _draw_grid_cell(canvas, 0, 0, label_w, cell_h, corner, label_bg, (40, 40, 40), 0.4)

        for c in range(n_cols):
            t_idx = col_start + c
            t_num = t_idx + 1
            col_label = f"T{t_num}" + (" N/F" if t_idx in no_face_targets else "")
            _draw_grid_cell(
                canvas,
                label_w + c * cell_w,
                0,
                cell_w,
                cell_h,
                col_label,
                header_bg,
                warn_label if t_idx in no_face_targets else (24, 24, 24),
                0.38 if t_idx in no_face_targets else 0.45,
            )

        for r in range(r_count):
            row_y = (1 + r) * cell_h
            r_num = r + 1
            row_skipped = r in skipped_refs
            row_label = f"R{r_num}" + (" N/F" if row_skipped else "")
            _draw_grid_cell(
                canvas,
                0,
                row_y,
                label_w,
                cell_h,
                row_label,
                label_bg,
                warn_label if row_skipped else (24, 24, 24),
                0.38 if row_skipped else 0.45,
            )
            for c in range(n_cols):
                t_idx = col_start + c
                score = float(score_matrix[r, t_idx])
                label, ok = _format_grid_score(score, match_threshold)
                cell_bg = skipped_bg if row_skipped or t_idx in no_face_targets else (pass_bg if ok else fail_bg)
                _draw_grid_cell(
                    canvas,
                    label_w + c * cell_w,
                    row_y,
                    cell_w,
                    cell_h,
                    label,
                    cell_bg,
                )

        agg_y = (1 + r_count) * cell_h
        _draw_grid_cell(canvas, 0, agg_y, label_w, cell_h, "AGG", label_bg, (20, 20, 20), 0.42)
        for c in range(n_cols):
            t_idx = col_start + c
            agg = float(column_aggregates[t_idx])
            label, ok = _format_grid_score(agg, match_threshold)
            _draw_grid_cell(
                canvas,
                label_w + c * cell_w,
                agg_y,
                cell_w,
                cell_h,
                label,
                agg_pass_bg if ok else agg_fail_bg,
                (10, 10, 10),
                0.48,
            )

        panels.append(bgr_uint8_to_comfy_bhwc(canvas))

    return torch.cat(panels, dim=0) if len(panels) > 1 else panels[0]


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


def _arcface_dst_template(
    image_size: int = 112,
    margin_scale: float = ALIGN_FACE_MARGIN_SCALE,
    shift_y: float = ALIGN_FACE_SHIFT_Y,
) -> np.ndarray:
    ratio = float(image_size) / 112.0
    dst = _ARCFACE_DST_112 * ratio
    if margin_scale != 1.0:
        cx = cy = (image_size - 1) / 2.0
        center = np.array([cx, cy], dtype=np.float32)
        dst = center + (dst - center) * float(margin_scale)
    if shift_y:
        dst[:, 1] += float(shift_y) * ratio
    return dst


def norm_crop2_with_chin_margin(
    img_bgr: np.ndarray,
    landmark: np.ndarray,
    image_size: int = 112,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ArcFace similarity warp with extra margin for chin/jaw (InsightFace default template
    places the mouth very low in 112×112, which clips tilted frontals).
    """
    src = landmark.astype(np.float32).reshape(5, 2)
    dst = _arcface_dst_template(image_size)
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        M, _ = cv2.estimateAffinePartial2D(src, dst)
    if M is None:
        raise RuntimeError("Failed to estimate face alignment transform.")
    warped = cv2.warpAffine(
        img_bgr,
        M,
        (image_size, image_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return warped, M


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
    warped_bgr, m_affine = norm_crop2_with_chin_margin(img_bgr, pts5, image_size=112)
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


def try_align_one_face(
    image_bhwc: torch.Tensor,
    align_mode: str,
    face_selection: str,
    face_index: int,
    det_size: int,
    det_thresh: float,
    ctx_id: int,
    insightface_root: str = "",
) -> Optional[AlignedFaceBundle]:
    """Like ``align_one_face`` but returns ``None`` when no face can be detected/aligned."""
    try:
        return align_one_face(
            image_bhwc,
            align_mode=align_mode,
            face_selection=face_selection,
            face_index=face_index,
            det_size=det_size,
            det_thresh=det_thresh,
            ctx_id=ctx_id,
            insightface_root=insightface_root,
        )
    except RuntimeError as exc:
        msg = str(exc)
        if (
            "No face detected" in msg
            or "face_index" in msg
            or "landmark" in msg.lower()
            or "keypoints" in msg.lower()
        ):
            return None
        raise


def render_no_face_aligned_placeholder() -> torch.Tensor:
    """112×112 aligned-crop placeholder when detection fails."""
    canvas = np.full((112, 112, 3), 42, dtype=np.uint8)
    _draw_grid_cell(canvas, 0, 0, 112, 112, "NO FACE", (42, 42, 42), (200, 200, 200), 0.45)
    return bgr_uint8_to_comfy_bhwc(canvas)


def render_compare_debug_preview(
    crop_bhwc: torch.Tensor,
    *,
    match: int,
    aggregate_score: float,
    aggregate_mode: str,
    match_threshold: float,
    scale: int = 4,
    banner_h: int = 100,
) -> torch.Tensor:
    """Readable debug card: upscaled face + dark caption strip (Comfy preview friendly)."""
    rgb = (crop_bhwc[0].detach().cpu().clamp(0, 1).numpy() * 255.0).astype(np.uint8)
    face_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = face_bgr.shape[:2]
    face_up = cv2.resize(face_bgr, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
    fw, fh = face_up.shape[1], face_up.shape[0]

    canvas = np.zeros((fh + banner_h, fw, 3), dtype=np.uint8)
    canvas[:fh] = face_up
    canvas[fh:] = (28, 28, 28)

    status = "MATCH" if match else "NO MATCH"
    status_color = (80, 220, 80) if match else (80, 80, 255)
    score_txt = "N/F" if _is_no_face_score(aggregate_score) else f"{aggregate_score:.3f}"
    lines = [
        (status, status_color, 0.85, 2),
        (f"score {score_txt}   threshold {match_threshold:.2f}", (240, 240, 240), 0.55, 1),
        (f"aggregate: {aggregate_mode}", (200, 200, 200), 0.5, 1),
    ]

    y = fh + 28
    for text, color, font_scale, thickness in lines:
        cv2.putText(
            canvas,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += int(28 * font_scale + 18)

    return bgr_uint8_to_comfy_bhwc(canvas)


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
