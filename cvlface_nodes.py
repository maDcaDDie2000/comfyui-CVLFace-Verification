"""
ComfyUI nodes: CVLFace KP-RPE verification pipeline.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

import numpy as np
import torch

from cvlface_align import align_one_face, draw_landmarks_on_crop
from cvlface_embed import compute_embedding, get_cached_embedder
from cvlface_hash import digest_any, tensor_digest
from cvlface_paths import (
    CVLF_NODE_MENU_CATEGORY,
    assert_cvlface_checkpoint,
    checkpoint_mtime_key,
    cvlface_checkpoint_dir,
)
from cvlface_types import FaceEmbedderHandle, FaceProfile


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
    ref_image: torch.Tensor,
    extra_ref_images: Optional[torch.Tensor],
    embedder: FaceEmbedderHandle,
    align_mode: str,
    face_selection: str,
    face_index: int,
    det_size: int,
    det_thresh: float,
    ctx_id: int,
) -> FaceProfile:
    embs = []
    quals = []

    def one(img: torch.Tensor):
        if img.shape[0] > 1:
            img = img[0:1]
        bundle = align_one_face(
            img,
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

    one(ref_image)
    if extra_ref_images is not None and extra_ref_images.shape[0] > 0:
        for i in range(extra_ref_images.shape[0]):
            one(extra_ref_images[i : i + 1])

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
            "optional": {
                "extra_ref_images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("FACE_PROFILE",)
    RETURN_NAMES = ("face_profile",)
    FUNCTION = "build"
    CATEGORY = CVLF_NODE_MENU_CATEGORY

    def build(
        self,
        face_embedder: FaceEmbedderHandle,
        ref_image: torch.Tensor,
        align_mode: str,
        face_selection: str,
        face_index: int,
        det_thresh: float,
        det_size: int,
        insightface_ctx: str,
        extra_ref_images: Optional[torch.Tensor] = None,
    ):
        ctx_id = 0 if insightface_ctx == "cuda" else -1
        prof = _build_profile_tensor(
            ref_image,
            extra_ref_images,
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
        face_embedder: FaceEmbedderHandle,
        ref_image: torch.Tensor,
        align_mode: str,
        face_selection: str,
        face_index: int,
        det_thresh: float,
        det_size: int,
        insightface_ctx: str,
        extra_ref_images: Optional[torch.Tensor] = None,
    ):
        ex = tensor_digest(extra_ref_images) if extra_ref_images is not None else "noextra"
        return digest_any(
            face_embedder.model_path,
            str(face_embedder.device),
            tensor_digest(ref_image),
            ex,
            align_mode,
            face_selection,
            face_index,
            det_thresh,
            det_size,
            insightface_ctx,
        )


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

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "FLOAT", "INT", "IMAGE")
    RETURN_NAMES = (
        "aligned_face",
        "landmarks_preview",
        "per_ref_scores_json",
        "aggregate_score",
        "match",
        "debug_preview",
    )
    FUNCTION = "compare"
    CATEGORY = CVLF_NODE_MENU_CATEGORY

    def compare(
        self,
        face_embedder: FaceEmbedderHandle,
        face_profile: FaceProfile,
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
        import cv2

        ctx_id = 0 if insightface_ctx == "cuda" else -1
        bundle = align_one_face(
            target_image,
            align_mode=align_mode,
            face_selection=face_selection,
            face_index=face_index,
            det_size=det_size,
            det_thresh=det_thresh,
            ctx_id=ctx_id,
        )
        q_emb = compute_embedding(face_embedder, bundle.image_bhwc, bundle.keypoints_112)[0]
        refs = face_profile.embeddings
        scores = (refs @ q_emb).astype(np.float32)
        agg = _aggregate_scores(scores, face_profile.qualities, aggregate)
        match = 1 if float(agg) >= float(match_threshold) else 0
        prev = draw_landmarks_on_crop(
            bundle.image_bhwc,
            bundle.keypoints_112,
            bundle.dense_landmarks_112,
        )

        rgb = (bundle.image_bhwc[0].detach().cpu().clamp(0, 1).numpy() * 255.0).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        txt = f"match={match} agg={agg:.3f} mode={aggregate}"
        cv2.putText(bgr, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        dbg = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).unsqueeze(0)

        payload = {
            "cosine_similarity": [float(x) for x in scores.tolist()],
            "aggregate": float(agg),
            "aggregate_mode": aggregate,
            "match_threshold": float(match_threshold),
            "match": bool(match),
        }
        return (
            bundle.image_bhwc,
            prev,
            json.dumps(payload, indent=2),
            float(agg),
            int(match),
            dbg,
        )

    @classmethod
    def IS_CHANGED(
        cls,
        face_embedder: FaceEmbedderHandle,
        face_profile: FaceProfile,
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
