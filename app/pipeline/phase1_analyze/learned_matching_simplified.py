"""Simplified geometric matcher for real-estate overlap + motion labeling.

Design goals:
- Single forward LoFTR pass (no reverse retry)
- Essential matrix + pose recovery only (no homography path)
- Minimal output: confidence in [0,1] and motion label
- Deterministic preprocessing and RANSAC seed
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import time
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


MotionLabel = str


@dataclass(frozen=True)
class SimplifiedMatcherConfig:
    checkpoint: str = "indoor"
    long_side: int = 768
    confidence_threshold: float = 0.5
    min_confident_matches: int = 30
    ransac_threshold_px: float = 1.0
    ransac_prob: float = 0.999
    rng_seed: int = 42
    # Motion decision thresholds.
    rotation_threshold_deg: float = 2.0
    translation_z_threshold: float = 0.15


class BaseLoFTRPairMatcher:
    """Reusable LoFTR model/preprocessing helpers for pair matchers."""

    _model_cache: Dict[str, Any] = {}

    def __init__(self, config: SimplifiedMatcherConfig) -> None:
        self.config = config

    @staticmethod
    def _preferred_torch_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and bool(mps_backend.is_available()):
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _same_device(a: torch.device, b: torch.device) -> bool:
        if a.type != b.type:
            return False
        if a.type in {"cpu", "mps"}:
            return True
        if a.type == "cuda":
            if a.index is None or b.index is None:
                return True
            return a.index == b.index
        return a == b

    def _load_matcher(self) -> tuple[Any, bool, float]:
        from kornia.feature import LoFTR

        checkpoint = self.config.checkpoint
        started_at = time.perf_counter()
        cache_hit = checkpoint in self._model_cache
        if cache_hit:
            matcher = self._model_cache[checkpoint]
            preferred = self._preferred_torch_device()
            try:
                current = next(matcher.parameters()).device
            except Exception:
                current = torch.device("cpu")
            if not self._same_device(current, preferred):
                matcher = matcher.to(preferred)
                matcher.eval()
                self._model_cache[checkpoint] = matcher
                logger.info(
                    "Moved simplified LoFTR matcher (%s) from %s to %s",
                    checkpoint,
                    current,
                    preferred,
                )
            return matcher, True, (time.perf_counter() - started_at)

        device = self._preferred_torch_device()
        matcher = LoFTR(pretrained=checkpoint)
        matcher = matcher.to(device)
        matcher.eval()
        self._model_cache[checkpoint] = matcher
        logger.info("Loaded simplified LoFTR matcher (%s) on %s", checkpoint, device)
        return matcher, False, (time.perf_counter() - started_at)

    @staticmethod
    def _resize_gray_with_padding(image: Image.Image, long_side: int) -> tuple[np.ndarray, Dict[str, int]]:
        rgb = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError("Invalid image dimensions for LoFTR preprocessing.")

        scale = float(long_side) / float(max(h, w))
        new_w = max(32, int(round(w * scale)))
        new_h = max(32, int(round(h * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(gray, (new_w, new_h), interpolation=interpolation)

        # LoFTR benefits from dimensions divisible by 8.
        pad_w = (8 - (new_w % 8)) % 8
        pad_h = (8 - (new_h % 8)) % 8
        if pad_w or pad_h:
            resized = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        meta = {
            "content_w": int(new_w),
            "content_h": int(new_h),
            "pad_w": int(pad_w),
            "pad_h": int(pad_h),
        }
        return resized, meta

    @staticmethod
    def _gray_to_tensor(gray: np.ndarray, device: torch.device) -> torch.Tensor:
        tensor = torch.from_numpy(gray).to(device=device, dtype=torch.float32)
        tensor = tensor / 255.0
        return tensor.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def _to_xy_points(raw_points: Any) -> np.ndarray:
        if raw_points is None:
            return np.empty((0, 2), dtype=np.float32)
        if isinstance(raw_points, torch.Tensor):
            arr = raw_points.detach().cpu().numpy()
        else:
            arr = np.asarray(raw_points)
        if arr.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 2)
        elif arr.ndim > 2:
            arr = arr.reshape(-1, arr.shape[-1])
        if arr.shape[-1] < 2:
            return np.empty((0, 2), dtype=np.float32)
        return np.ascontiguousarray(arr[:, :2], dtype=np.float32)

    @staticmethod
    def _to_score_vector(raw_scores: Any, fallback_len: int) -> np.ndarray:
        if raw_scores is None:
            return np.ones((fallback_len,), dtype=np.float32)
        if isinstance(raw_scores, torch.Tensor):
            arr = raw_scores.detach().cpu().numpy()
        else:
            arr = np.asarray(raw_scores)
        if arr.size == 0:
            return np.ones((fallback_len,), dtype=np.float32)
        arr = arr.reshape(-1).astype(np.float32, copy=False)
        if arr.shape[0] != fallback_len:
            # Keep behavior deterministic and conservative on malformed outputs.
            return np.ones((fallback_len,), dtype=np.float32)
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _extract_matches(self, correspondences: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points0 = self._to_xy_points(correspondences.get("keypoints0"))
        points1 = self._to_xy_points(correspondences.get("keypoints1"))
        if points0.shape[0] == 0 or points1.shape[0] == 0:
            points0 = self._to_xy_points(correspondences.get("mkpts0_f"))
            points1 = self._to_xy_points(correspondences.get("mkpts1_f"))
        count = int(min(points0.shape[0], points1.shape[0]))
        if count <= 0:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        points0 = points0[:count]
        points1 = points1[:count]
        raw_scores = (
            correspondences.get("confidence")
            if correspondences.get("confidence") is not None
            else correspondences.get("scores")
        )
        if raw_scores is None:
            raw_scores = correspondences.get("mconf")
        scores = self._to_score_vector(raw_scores, count)
        return points0, points1, scores

    @staticmethod
    def _summary(scores: np.ndarray) -> Dict[str, float]:
        if scores.size == 0:
            return {"count": 0.0, "mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": float(scores.size),
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "p95": float(np.percentile(scores, 95.0)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        }

    @staticmethod
    def _sample_normalized_matches(
        points0: np.ndarray,
        points1: np.ndarray,
        width0: int,
        height0: int,
        width1: int,
        height1: int,
        max_points: int,
    ) -> list[Dict[str, float]]:
        n = int(min(points0.shape[0], points1.shape[0]))
        if n <= 0:
            return []
        if max_points > 0:
            n = min(n, int(max_points))
        points0 = points0[:n]
        points1 = points1[:n]
        w0 = max(1.0, float(width0))
        h0 = max(1.0, float(height0))
        w1 = max(1.0, float(width1))
        h1 = max(1.0, float(height1))
        out: list[Dict[str, float]] = []
        for i in range(n):
            x0 = float(points0[i, 0] / w0)
            y0 = float(points0[i, 1] / h0)
            x1 = float(points1[i, 0] / w1)
            y1 = float(points1[i, 1] / h1)
            out.append(
                {
                    "x0": float(np.clip(x0, 0.0, 1.0)),
                    "y0": float(np.clip(y0, 0.0, 1.0)),
                    "x1": float(np.clip(x1, 0.0, 1.0)),
                    "y1": float(np.clip(y1, 0.0, 1.0)),
                    "dx": float(np.clip(x1 - x0, -1.0, 1.0)),
                    "dy": float(np.clip(y1 - y0, -1.0, 1.0)),
                }
            )
        return out

    @staticmethod
    def _segment_scores(inlier0: np.ndarray, inlier1: np.ndarray, width0: int, width1: int) -> Dict[str, float]:
        n = int(min(inlier0.shape[0], inlier1.shape[0]))
        if n <= 0:
            return {
                "from_left_25_50": 0.0,
                "from_right_50_75": 0.0,
                "to_left_25_50": 0.0,
                "to_right_50_75": 0.0,
                "cross_left_to_right": 0.0,
                "cross_right_to_left": 0.0,
                "cross_center_to_center": 0.0,
            }
        x0 = inlier0[:n, 0] / max(1.0, float(width0))
        x1 = inlier1[:n, 0] / max(1.0, float(width1))
        from_left = (x0 >= 0.25) & (x0 < 0.50)
        from_right = (x0 >= 0.50) & (x0 < 0.75)
        to_left = (x1 >= 0.25) & (x1 < 0.50)
        to_right = (x1 >= 0.50) & (x1 < 0.75)
        center0 = (x0 >= 0.40) & (x0 <= 0.60)
        center1 = (x1 >= 0.40) & (x1 <= 0.60)
        denom = float(max(1, n))
        return {
            "from_left_25_50": float(np.count_nonzero(from_left) / denom),
            "from_right_50_75": float(np.count_nonzero(from_right) / denom),
            "to_left_25_50": float(np.count_nonzero(to_left) / denom),
            "to_right_50_75": float(np.count_nonzero(to_right) / denom),
            "cross_left_to_right": float(np.count_nonzero(from_left & to_right) / denom),
            "cross_right_to_left": float(np.count_nonzero(from_right & to_left) / denom),
            "cross_center_to_center": float(np.count_nonzero(center0 & center1) / denom),
        }

    @staticmethod
    def _direction_from_label(label: MotionLabel) -> tuple[float, float]:
        mapping: Dict[MotionLabel, tuple[float, float]] = {
            "orbit_left": (-1.0, 0.0),
            "orbit_right": (1.0, 0.0),
            "tilt_up": (0.0, -1.0),
            "tilt_down": (0.0, 1.0),
            "push_in": (0.0, 0.0),
            "pull_out": (0.0, 0.0),
            "static": (0.0, 0.0),
            "uncertain": (0.0, 0.0),
        }
        return mapping.get(label, (0.0, 0.0))

    @staticmethod
    def _rotation_to_yaw_pitch_deg(rotation: np.ndarray) -> tuple[float, float]:
        # yaw around Y axis; pitch around X axis from rotation matrix.
        # Values are sufficient for discrete motion class labels.
        r00 = float(rotation[0, 0])
        r10 = float(rotation[1, 0])
        r20 = float(rotation[2, 0])
        r21 = float(rotation[2, 1])
        r22 = float(rotation[2, 2])
        yaw = math.degrees(math.atan2(r20, r00))
        pitch = math.degrees(math.atan2(-r21, math.sqrt(r00 * r00 + r10 * r10 + r22 * r22)))
        return yaw, pitch


class SimplifiedLoFTREssentialMatcher(BaseLoFTRPairMatcher):
    """Single-pass LoFTR + Essential geometry matcher."""

    def _classify_motion(self, rotation: np.ndarray, translation: np.ndarray) -> MotionLabel:
        yaw_deg, pitch_deg = self._rotation_to_yaw_pitch_deg(rotation)
        rot_thresh = float(self.config.rotation_threshold_deg)
        if max(abs(yaw_deg), abs(pitch_deg)) >= rot_thresh:
            if abs(yaw_deg) >= abs(pitch_deg):
                return "orbit_right" if yaw_deg > 0 else "orbit_left"
            return "tilt_down" if pitch_deg > 0 else "tilt_up"

        tz = float(translation[2, 0]) if translation.shape[0] >= 3 else 0.0
        if abs(tz) >= float(self.config.translation_z_threshold):
            # Scale is unknown; sign still provides direction in recovered camera frame.
            return "push_in" if tz < 0 else "pull_out"
        return "static"

    def match_pair(
        self,
        img1: Image.Image,
        img2: Image.Image,
        confidence_threshold: float | None = None,
        full_diagnostics: bool = False,
    ) -> Tuple[int, int, float, Tuple[float, float], Dict[str, Any]]:
        total_started = time.perf_counter()
        conf_threshold = (
            float(np.clip(confidence_threshold, 0.1, 1.0))
            if confidence_threshold is not None
            else float(self.config.confidence_threshold)
        )

        matcher, cache_hit, model_load_s = self._load_matcher()
        device = next(matcher.parameters()).device

        prep_started = time.perf_counter()
        img1_resized, meta0 = self._resize_gray_with_padding(img1, long_side=int(self.config.long_side))
        img2_resized, meta1 = self._resize_gray_with_padding(img2, long_side=int(self.config.long_side))
        resize_s = time.perf_counter() - prep_started

        tensor_started = time.perf_counter()
        tensor1 = self._gray_to_tensor(img1_resized, device=device)
        tensor2 = self._gray_to_tensor(img2_resized, device=device)
        tensor_s = time.perf_counter() - tensor_started

        loftr_started = time.perf_counter()
        batch = {"image0": tensor1, "image1": tensor2}
        with torch.inference_mode():
            output = matcher(batch)
            correspondences = output if isinstance(output, dict) else batch
        loftr_s = time.perf_counter() - loftr_started

        post_started = time.perf_counter()
        raw_points0, raw_points1, raw_scores = self._extract_matches(correspondences)
        raw_count = int(raw_scores.shape[0])
        keep_mask = raw_scores >= float(conf_threshold)
        points0 = raw_points0[keep_mask]
        points1 = raw_points1[keep_mask]
        scores = raw_scores[keep_mask]
        active_count = int(scores.shape[0])

        width0 = int(meta0["content_w"])
        height0 = int(meta0["content_h"])
        width1 = int(meta1["content_w"])
        height1 = int(meta1["content_h"])
        post_s = time.perf_counter() - post_started

        essential_started = time.perf_counter()
        cv2.setRNGSeed(int(self.config.rng_seed))
        geometry_model = "essential_ransac"
        inlier_mask = np.zeros((active_count,), dtype=bool)
        inlier_points0 = np.empty((0, 2), dtype=np.float32)
        inlier_points1 = np.empty((0, 2), dtype=np.float32)
        motion_label: MotionLabel = "uncertain"
        yaw_deg = 0.0
        pitch_deg = 0.0
        translation_z = 0.0

        if active_count >= int(self.config.min_confident_matches):
            focal = float(max(width0, height0, width1, height1))
            pp = (float(width0) * 0.5, float(height0) * 0.5)
            E, mask = cv2.findEssentialMat(
                points0,
                points1,
                focal=focal,
                pp=pp,
                method=cv2.RANSAC,
                prob=float(self.config.ransac_prob),
                threshold=float(self.config.ransac_threshold_px),
            )
            if E is not None and mask is not None:
                if E.shape == (3, 3):
                    E_use = E
                elif E.ndim == 2 and E.shape[0] % 3 == 0 and E.shape[1] == 3:
                    E_use = E[:3, :]
                elif E.ndim == 3 and E.shape[1:] == (3, 3):
                    E_use = E[0]
                else:
                    E_use = None

                if E_use is not None:
                    inlier_mask = np.asarray(mask).reshape(-1).astype(bool)
                    if inlier_mask.shape[0] != active_count:
                        inlier_mask = np.zeros((active_count,), dtype=bool)
                    if int(np.count_nonzero(inlier_mask)) >= 5:
                        _, R, t, pose_mask = cv2.recoverPose(
                            E_use,
                            points0,
                            points1,
                            focal=focal,
                            pp=pp,
                            mask=mask,
                        )
                        if pose_mask is not None:
                            pose_keep = np.asarray(pose_mask).reshape(-1).astype(bool)
                            if pose_keep.shape[0] == active_count:
                                inlier_mask = inlier_mask & pose_keep
                        if int(np.count_nonzero(inlier_mask)) >= 5:
                            motion_label = self._classify_motion(R, t)
                            yaw_deg, pitch_deg = self._rotation_to_yaw_pitch_deg(R)
                            translation_z = float(t[2, 0]) if t.shape[0] >= 3 else 0.0

        inlier_points0 = points0[inlier_mask]
        inlier_points1 = points1[inlier_mask]
        num_inliers = int(inlier_points0.shape[0])
        essential_s = time.perf_counter() - essential_started

        score_started = time.perf_counter()
        inlier_ratio = float(num_inliers) / max(1.0, float(active_count))
        # Mild support weighting only to avoid tiny-sample inflation.
        support = float(np.clip(float(active_count) / 80.0, 0.35, 1.0))
        confidence = float(np.clip(inlier_ratio * support, 0.0, 1.0))
        if active_count < int(self.config.min_confident_matches):
            confidence = 0.0
            motion_label = "uncertain"
        elif num_inliers < 5:
            confidence = 0.0
            motion_label = "uncertain"

        direction = self._direction_from_label(motion_label)
        segment_scores = self._segment_scores(inlier_points0, inlier_points1, width0=width0, width1=width1)
        score_s = time.perf_counter() - score_started

        raw_score_summary = self._summary(raw_scores)
        kept_score_summary = self._summary(scores)

        score_components: Dict[str, float] = {
            "inlier_ratio": float(inlier_ratio),
            "inlier_ratio_numerator": float(num_inliers),
            "inlier_ratio_denominator": float(active_count),
            "overlap_ratio": float(inlier_ratio),
            "robust_coverage": float(inlier_ratio),
            "combined_score": float(confidence),
            "final_score": float(confidence),
            "robust_score_valid": float(1.0 if active_count >= int(self.config.min_confident_matches) else 0.0),
            "motion_yaw_deg": float(yaw_deg),
            "motion_pitch_deg": float(pitch_deg),
            "motion_translation_z": float(translation_z),
            "motion_coherence": float(np.clip(max(abs(yaw_deg), abs(pitch_deg)) / 15.0, 0.0, 1.0)),
            "segment_strength": float(
                max(
                    segment_scores.get("cross_left_to_right", 0.0),
                    segment_scores.get("cross_right_to_left", 0.0),
                    segment_scores.get("cross_center_to_center", 0.0),
                )
            ),
        }

        threshold_trial = {
            "threshold": float(conf_threshold),
            "raw_matches": int(raw_count),
            "num_matches": int(active_count),
            "num_inliers": int(num_inliers),
            "inlier_ratio": float(inlier_ratio),
            "score": float(confidence),
            "geometry_model": geometry_model,
        }

        if full_diagnostics:
            raw_matches_payload = self._sample_normalized_matches(
                points0=raw_points0,
                points1=raw_points1,
                width0=width0,
                height0=height0,
                width1=width1,
                height1=height1,
                max_points=5000,
            )
            inlier_matches_payload = self._sample_normalized_matches(
                points0=inlier_points0,
                points1=inlier_points1,
                width0=width0,
                height0=height0,
                width1=width1,
                height1=height1,
                max_points=5000,
            )
        else:
            raw_matches_payload = []
            inlier_matches_payload = []

        total_s = time.perf_counter() - total_started
        diagnostics: Dict[str, Any] = {
            "matcher": "loftr_kornia_indoor_simplified",
            "checkpoint": f"kornia:{self.config.checkpoint}",
            "confidence_threshold": float(conf_threshold),
            "geometry_model": geometry_model,
            "raw_correspondence_count": int(raw_count),
            "threshold_match_count": int(active_count),
            "active_match_count": int(active_count),
            "ransac_reproj_threshold": float(self.config.ransac_threshold_px),
            "loftr_input_width": int(width0 + int(meta0["pad_w"])),
            "loftr_input_height": int(height0 + int(meta0["pad_h"])),
            "match_width": int(width0),
            "match_height": int(height0),
            "motion_label": motion_label,
            "direction": {"dx": float(direction[0]), "dy": float(direction[1])},
            "segment_scores": segment_scores,
            "score_components": score_components,
            "threshold_trials": [threshold_trial],
            "native_matching_scores": kept_score_summary,
            "native_matching_scores_raw": raw_score_summary,
            "raw_matches": raw_matches_payload,
            "inlier_matches": inlier_matches_payload,
            "timing": {
                "time_model_load_s": float(model_load_s),
                "time_resize_s": float(resize_s),
                "time_tensor_transfer_s": float(tensor_s),
                "time_loftr_s": float(loftr_s),
                "time_loftr_forward_main_s": float(loftr_s),
                "time_loftr_forward_reverse_s": 0.0,
                "time_postprocess_s": float(post_s),
                "time_f_s": 0.0,
                "time_h_s": 0.0,
                "time_scoring_s": float(score_s),
                "time_essential_s": float(essential_s),
                "time_pair_total_s": float(total_s),
                "time_reverse_pair_total_s": 0.0,
                "reverse_attempted": False,
                "reverse_selected": False,
                "forward_pass_count": 1,
                "model_cache_hit": bool(cache_hit),
                "model_device": str(device),
                "tensor_device": str(tensor1.device),
                "cuda_available": bool(torch.cuda.is_available()),
                "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
                "preferred_device": str(self._preferred_torch_device()),
            },
            "oracle": {"mode": "off", "evaluated": False, "decision": "not_used"},
        }
        return int(active_count), int(num_inliers), float(confidence), direction, diagnostics
