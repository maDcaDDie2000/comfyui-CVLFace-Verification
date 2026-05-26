"""
ComfyUI nodes: CVLFace KP-RPE verification pipeline.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

import numpy as np
import torch

from cvlface_align import (
    align_one_face,
    draw_landmarks_on_crop,
    render_compare_debug_preview,
    render_comparison_grids,
    truncate_image_batch,
)
from cvlface_embed import compute_embedding, get_cached_embedder
from cvlface_hash import digest_any, tensor_digest
from cvlface_paths import (
    CVLF_NODE_MENU_CATEGORY,
    assert_cvlface_checkpoint,
    checkpoint_mtime_key,
    cvlface_checkpoint_dir,
)
from cvlface_types import FaceEmbedderHandle, FaceProfile

LOG_PREFIX = "[comfyui-CVLFace-Verification]"
MAX_REF_IMAGES = 10
MAX_TARGET_IMAGES = 50


def _aggregate_scores(scores: np.ndarray, qualities: np.ndarray, how: str) -> float:
    if scores.size == 0:
        return 0.0
    if how == "max":
        return float(scores.max())
    if how == "mean":
        return float(scores.mean())
    if how == "quality_weighted_mean":
        w = np.maximum(qualities.astype(np.float64), 1e-6)
        return float((scores * w).sum() / w.sum())
    raise ValueError(how)


def _build_profile_tensor(
    ref_images: torch.Tensor,
    embedder: FaceEmbedderHandle,
    align_mode: str,
    face_selection: str,
    face_index: int,
    det_size: int,
    det_thresh: float,
    ctx_id: int,
) -> FaceProfile:
    ref_images = truncate_image_batch(ref_images, MAX_REF_IMAGES, "ref_image", LOG_PREFIX)
    embs = []
    quals = []

    for i in range(ref_images.shape[0]):
        bundle = align_one_face(
            ref_images[i : i + 1],
            align_mode=align_mode,
            face_selection=face_selection,
            face_index=face_index,
            det_size=det_size,
            det_thresh=det_thresh,
            ctx_id=ctx_id,
        )
        e = compute_embedding(embedder, bundle.image_bhwc, bundle.keypoints_112)
        embs.append(e[0])
        quals.append(bundle.meta.det_score)

    E = np.stack(embs, axis=0)
    Q = np.asarray(quals, dtype=np.float32)
    return FaceProfile(
        embeddings=E,
        qualities=Q,
        model_path=embedder.model_path,
        align_mode=align_mode,
        extras={},
    )


class CVLFaceLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("FACE_EMBEDDER",)
    RETURN_NAMES = ("face_embedder",)
    FUNCTION = "load"
    CATEGORY = CVLF_NODE_MENU_CATEGORY

    def load(self, device: str):
        path = cvlface_checkpoint_dir()
        assert_cvlface_checkpoint(path)
        prefer_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
        handle = get_cached_embedder(path, prefer_cuda)
        return (handle,)

    @classmethod
    def IS_CHANGED(cls, device: str):
        try:
            path = cvlface_checkpoint_dir()
            stamp = checkpoint_mtime_key(path)
        except Exception:
            stamp = "err"
        return digest_any(device, stamp)


class FaceAlign:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "align_mode": (["2d106", "3d68", "auto"], {"default": "2d106"}),
                "face_selection": (["largest_area", "highest_score", "index"], {"default": "largest_area"}),
                "face_index": ("INT", {"default": 0, "min": 0, "max": 63, "step": 1}),
                "det_thresh": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 0.99, "step": 0.01}),
                "det_size": ("INT", {"default": 640, "min": 320, "max": 1280, "step": 64}),
                "insightface_ctx": (["cuda", "cpu"], {"default": "cuda"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "FACE_META")
    RETURN_NAMES = ("aligned_face", "landmarks_preview", "face_meta")
    FUNCTION = "align"
    CATEGORY = CVLF_NODE_MENU_CATEGORY

    def align(
        self,
        image: torch.Tensor,
        align_mode: str,
        face_selection: str,
        face_index: int,
        det_thresh: float,
        det_size: int,
        insightface_ctx: str,
    ):
        ctx_id = 0 if insightface_ctx == "cuda" else -1
        bundle = align_one_face(
            image,
            align_mode=align_mode,
            face_selection=face_selection,
            face_index=face_index,
            det_size=det_size,
            det_thresh=det_thresh,
            ctx_id=ctx_id,
        )
        prev = draw_landmarks_on_crop(
            bundle.image_bhwc,
            bundle.keypoints_112,
            bundle.dense_landmarks_112,
        )
        return (bundle.image_bhwc, prev, bundle.meta)

    @classmethod
    def IS_CHANGED(
        cls,
        image: torch.Tensor,
        align_mode: str,
        face_selection: str,
        face_index: int,
        det_thresh: float,
        det_size: int,
        insightface_ctx: str,
    ):
        return digest_any(
            tensor_digest(image),
            align_mode,
            face_selection,
            face_index,
            det_thresh,
            det_size,
            insightface_ctx,
        )


class FaceReferenceProfile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_embedder": ("FACE_EMBEDDER",),
                "ref_image": ("IMAGE",),
                "align_mode": (["2d106", "3d68", "auto"], {"default": "2d106"}),
                "face_selection": (["largest_area", "highest_score", "index"], {"default": "largest_area"}),
                "face_index": ("INT", {"default": 0, "min": 0, "max": 63, "step": 1}),
                "det_thresh": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 0.99, "step": 0.01}),
                "det_size": ("INT", {"default": 640, "min": 320, "max": 1280, "step": 64}),
                "insightface_ctx": (["cuda", "cpu"], {"default": "cuda"}),
            },
        }

    RETURN_TYPES = ("FACE_PROFILE",)
    RETURN_NAMES = ("face_profile",)
    FUNCTION = "build"
    CATEGORY = CVLF_NODE_MENU_CATEGORY

    def build(
        self,
        face_embedder: Optional[FaceEmbedderHandle],
        ref_image: torch.Tensor,
        align_mode: str,
        face_selection: str,
        face_index: int,
        det_thresh: float,
        det_size: int,
        insightface_ctx: str,
    ):
        if face_embedder is None:
            raise RuntimeError(
                "Face Reference Profile: connect the output of **CVLFace Loader** (face_embedder)."
            )
        if ref_image.shape[0] < 1:
            raise RuntimeError("Face Reference Profile: ref_image batch must contain at least 1 image.")
        ctx_id = 0 if insightface_ctx == "cuda" else -1
        prof = _build_profile_tensor(
            ref_image,
            face_embedder,
            align_mode,
            face_selection,
            face_index,
            det_size,
            det_thresh,
            ctx_id,
        )
        return (prof,)

    @classmethod
    def IS_CHANGED(
        cls,
        face_embedder: Optional[FaceEmbedderHandle],
        ref_image: torch.Tensor,
        align_mode: str,
        face_selection: str,
        face_index: int,
        det_thresh: float,
        det_size: int,
        insightface_ctx: str,
    ):
        if face_embedder is None:
            return digest_any(
                "no_embedder",
                tensor_digest(ref_image),
                align_mode,
                face_selection,
                face_index,
                det_thresh,
                det_size,
                insightface_ctx,
            )
        return digest_any(
            face_embedder.model_path,
            str(face_embedder.device),
            tensor_digest(ref_image),
            align_mode,
            face_selection,
            face_index,
            det_thresh,
            det_size,
            insightface_ctx,
        )


def _passed_input_images(targets: torch.Tensor, matches: list[int]) -> torch.Tensor:
    """Full input images for targets that passed the threshold."""
    idx = [i for i, m in enumerate(matches) if m]
    if not idx:
        return torch.zeros((0, *targets.shape[1:]), dtype=targets.dtype, device=targets.device)
    return targets[idx]


class FaceCompareKPRPE:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_embedder": ("FACE_EMBEDDER",),
                "face_profile": ("FACE_PROFILE",),
                "target_image": ("IMAGE",),
                "align_mode": (["2d106", "3d68", "auto"], {"default": "2d106"}),
                "aggregate": (["max", "mean", "quality_weighted_mean"], {"default": "mean"}),
                "match_threshold": ("FLOAT", {"default": 0.35, "min": -1.0, "max": 1.0, "step": 0.01}),
                "face_selection": (["largest_area", "highest_score", "index"], {"default": "largest_area"}),
                "face_index": ("INT", {"default": 0, "min": 0, "max": 63, "step": 1}),
                "det_thresh": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 0.99, "step": 0.01}),
                "det_size": ("INT", {"default": 640, "min": 320, "max": 1280, "step": 64}),
                "insightface_ctx": (["cuda", "cpu"], {"default": "cuda"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "STRING", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "passed_images",
        "comparison_grids",
        "matches",
        "aggregate_scores",
        "scores_json",
        "debug_previews",
        "aligned_faces",
        "landmarks_previews",
    )
    FUNCTION = "compare"
    CATEGORY = CVLF_NODE_MENU_CATEGORY

    def compare(
        self,
        face_embedder: Optional[FaceEmbedderHandle],
        face_profile: Optional[FaceProfile],
        target_image: torch.Tensor,
        align_mode: str,
        aggregate: str,
        match_threshold: float,
        face_selection: str,
        face_index: int,
        det_thresh: float,
        det_size: int,
        insightface_ctx: str,
    ):
        if face_embedder is None:
            raise RuntimeError(
                "Face Compare KP-RPE: connect **CVLFace Loader** → face_embedder (same wire as your profile build)."
            )
        if face_profile is None:
            raise RuntimeError(
                "Face Compare KP-RPE: connect **Face Reference Profile** → face_profile."
            )
        if target_image.shape[0] < 1:
            raise RuntimeError("Face Compare KP-RPE: target_image batch must contain at least 1 image.")

        targets = truncate_image_batch(target_image, MAX_TARGET_IMAGES, "target_image", LOG_PREFIX)
        ctx_id = 0 if insightface_ctx == "cuda" else -1

        aligned_list = []
        preview_list = []
        debug_list = []
        q_embs = []

        for t in range(targets.shape[0]):
            bundle = align_one_face(
                targets[t : t + 1],
                align_mode=align_mode,
                face_selection=face_selection,
                face_index=face_index,
                det_size=det_size,
                det_thresh=det_thresh,
                ctx_id=ctx_id,
            )
            q_embs.append(compute_embedding(face_embedder, bundle.image_bhwc, bundle.keypoints_112)[0])
            aligned_list.append(bundle.image_bhwc)
            preview_list.append(
                draw_landmarks_on_crop(
                    bundle.image_bhwc,
                    bundle.keypoints_112,
                    bundle.dense_landmarks_112,
                )
            )

        refs = face_profile.embeddings
        q_matrix = np.stack(q_embs, axis=0)
        score_matrix = (refs @ q_matrix.T).astype(np.float32)

        per_target_agg: list[float] = []
        per_target_match: list[int] = []
        for t in range(score_matrix.shape[1]):
            col = score_matrix[:, t]
            agg = _aggregate_scores(col, face_profile.qualities, aggregate)
            per_target_agg.append(float(agg))
            per_target_match.append(1 if agg >= float(match_threshold) else 0)

        for t, agg in enumerate(per_target_agg):
            debug_list.append(
                render_compare_debug_preview(
                    aligned_list[t],
                    match=per_target_match[t],
                    aggregate_score=agg,
                    aggregate_mode=aggregate,
                    match_threshold=float(match_threshold),
                )
            )

        column_aggregates = np.asarray(per_target_agg, dtype=np.float32)
        comparison_grids = render_comparison_grids(
            score_matrix,
            column_aggregates,
            match_threshold=float(match_threshold),
        )

        payload = {
            "score_matrix": score_matrix.tolist(),
            "num_references": int(score_matrix.shape[0]),
            "num_targets": int(score_matrix.shape[1]),
            "aggregate_per_target": per_target_agg,
            "aggregate_mode": aggregate,
            "match_threshold": float(match_threshold),
            "matches": [bool(m) for m in per_target_match],
        }

        return (
            _passed_input_images(targets, per_target_match),
            comparison_grids,
            json.dumps(per_target_match),
            json.dumps(per_target_agg),
            json.dumps(payload, indent=2),
            torch.cat(debug_list, dim=0),
            torch.cat(aligned_list, dim=0),
            torch.cat(preview_list, dim=0),
        )

    @classmethod
    def IS_CHANGED(
        cls,
        face_embedder: Optional[FaceEmbedderHandle],
        face_profile: Optional[FaceProfile],
        target_image: torch.Tensor,
        align_mode: str,
        aggregate: str,
        match_threshold: float,
        face_selection: str,
        face_index: int,
        det_thresh: float,
        det_size: int,
        insightface_ctx: str,
    ):
        if face_embedder is None or face_profile is None:
            return digest_any(
                "disconnected",
                tensor_digest(target_image),
                align_mode,
                aggregate,
                match_threshold,
                face_selection,
                face_index,
                det_thresh,
                det_size,
                insightface_ctx,
            )
        ref_h = (
            hashlib.sha256(face_profile.embeddings.tobytes()).hexdigest()
            if face_profile.embeddings.size
            else "empty"
        )
        return digest_any(
            face_embedder.model_path,
            ref_h,
            tensor_digest(target_image),
            align_mode,
            aggregate,
            match_threshold,
            face_selection,
            face_index,
            det_thresh,
            det_size,
            insightface_ctx,
        )


NODE_CLASS_MAPPINGS = {
    "CVLFaceLoader": CVLFaceLoader,
    "FaceAlign": FaceAlign,
    "FaceReferenceProfile": FaceReferenceProfile,
    "FaceCompareKPRPE": FaceCompareKPRPE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CVLFaceLoader": "CVLFace Loader (KP-RPE)",
    "FaceAlign": "Face Align (InsightFace)",
    "FaceReferenceProfile": "Face Reference Profile",
    "FaceCompareKPRPE": "Face Compare KP-RPE",
}
