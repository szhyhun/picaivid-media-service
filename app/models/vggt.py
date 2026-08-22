"""Device-aware loader for the VGGT-1B-Commercial reconstruction model."""
from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import monotonic
from typing import Any, ContextManager

import torch
from safetensors.torch import load_file as safetensors_load_file

from app.core.config import settings
from app.pipeline.phase1_analyze.matcher_loaders import _ensure_local_file, _ensure_local_repo

logger = logging.getLogger(__name__)


def _synchronize_device(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _allocated_memory_mb(device: str) -> float | None:
    try:
        if device == "cuda":
            return torch.cuda.memory_allocated() / (1024 * 1024)
        if device == "mps":
            return torch.mps.current_allocated_memory() / (1024 * 1024)
    except (AttributeError, RuntimeError):
        pass
    return None


def _log_stage(stage: str, started_at: float, device: str, image_count: int) -> None:
    _synchronize_device(device)
    logger.info(
        "VGGT_STAGE_COMPLETE stage=%s images=%s elapsed_seconds=%.3f allocated_mb=%s",
        stage,
        image_count,
        monotonic() - started_at,
        round(_allocated_memory_mb(device), 1) if _allocated_memory_mb(device) is not None else "n/a",
    )


@dataclass
class _VGGTImports:
    VGGT: Any
    load_and_preprocess_images_square: Any
    pose_encoding_to_extri_intri: Any
    unproject_depth_map_to_point_map: Any


def _resolve_repo_dir() -> str:
    fallback = os.path.abspath(os.path.join(os.path.dirname(settings.MODEL_CACHE_DIR), "..", "third_party", "vggt"))
    return _ensure_local_repo(str(settings.VGGT_REPO_DIR or fallback), "VGGT_REPO_ARCHIVE_S3_URI")


def _resolve_checkpoint_path() -> str:
    repo_dir = _resolve_repo_dir()
    if settings.VGGT_MODEL_CHECKPOINT:
        return _ensure_local_file(str(settings.VGGT_MODEL_CHECKPOINT), "VGGT_MODEL_CHECKPOINT_S3_URI")
    for candidate in (
        os.path.abspath("vggt-commercial/vggt_1B_commercial.pt"),
        os.path.join(repo_dir, "checkpoints", "vggt_1B_commercial.pt"),
        os.path.join(repo_dir, "vggt_1B_commercial.pt"),
        os.path.join(repo_dir, "checkpoints", "VGGT-1B-Commercial", "model.pt"),
    ):
        if os.path.isfile(candidate):
            return candidate
    preferred_target = os.path.join(repo_dir, "checkpoints", "vggt_1B_commercial.pt")
    Path(preferred_target).parent.mkdir(parents=True, exist_ok=True)
    return _ensure_local_file(preferred_target, "VGGT_MODEL_CHECKPOINT_S3_URI")


@lru_cache(maxsize=1)
def _mps_tensor_smoke_test() -> bool:
    try:
        tensor = torch.ones((2, 2), device="mps")
        return tensor.device.type == "mps" and float(tensor.sum().cpu().item()) == 4.0
    except (RuntimeError, TypeError):
        return False


def _device() -> str:
    requested = str(settings.VGGT_DEVICE or "auto").lower()
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": bool(torch.backends.mps.is_available()) and _mps_tensor_smoke_test(),
        "cpu": True,
    }
    if requested not in {"auto", *available}:
        raise RuntimeError(f"Unsupported VGGT_DEVICE={requested!r}")
    if requested != "auto":
        if not available[requested]:
            raise RuntimeError(f"VGGT_DEVICE={requested} was requested but is unavailable")
        return requested
    return next(device for device in ("cuda", "mps", "cpu") if available[device])


def _dtype(device: str | None = None) -> torch.dtype:
    device = device or _device()
    requested = str(settings.VGGT_PRECISION or "auto").lower()
    options = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if requested != "auto":
        if requested not in options:
            raise RuntimeError(f"Unsupported VGGT_PRECISION={requested!r}")
        if device != "cuda" and options[requested] != torch.float32:
            raise RuntimeError(f"VGGT_PRECISION={requested} is unsupported on {device}; use float32")
        return options[requested]
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    return torch.float32


def _autocast(device: str, dtype: torch.dtype) -> ContextManager[None]:
    if device == "cuda" and dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


@lru_cache(maxsize=2)
def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_commit(repo_dir: str) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", repo_dir, "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return settings.VGGT_REPO_COMMIT


@lru_cache(maxsize=1)
def _load_imports() -> _VGGTImports:
    repo_dir = _resolve_repo_dir()
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images_square
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map
    except Exception as err:  # pragma: no cover - external runtime
        raise RuntimeError("Failed to import VGGT-1B-Commercial from the configured repository") from err
    return _VGGTImports(VGGT, load_and_preprocess_images_square, pose_encoding_to_extri_intri, unproject_depth_map_to_point_map)


@lru_cache(maxsize=1)
def _load_model() -> Any:
    imports = _load_imports()
    checkpoint_path = _resolve_checkpoint_path()
    model = imports.VGGT(img_size=int(settings.VGGT_IMAGE_SIZE))
    state_dict = safetensors_load_file(checkpoint_path, device="cpu") if checkpoint_path.endswith(".safetensors") else torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and isinstance(state_dict.get("state_dict"), dict):
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    device = _device()
    model.eval().to(device=device, dtype=_dtype(device))
    logger.info("Loaded VGGT-1B-Commercial checkpoint=%s device=%s", checkpoint_path, _device())
    return model


class VGGTModel:
    """Runs commercial VGGT and returns CPU tensors that are safe to persist."""

    def _ensure_loaded(self) -> None:
        _load_model()

    @property
    def device(self) -> str:
        return _device()

    @property
    def dtype(self) -> torch.dtype:
        return _dtype(self.device)

    def runtime_metadata(self) -> dict[str, Any]:
        checkpoint_path = _resolve_checkpoint_path()
        repo_dir = _resolve_repo_dir()
        return {
            "model": "VGGT-1B-Commercial",
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": _file_sha256(checkpoint_path),
            "repo_dir": repo_dir,
            "repo_commit": _repo_commit(repo_dir),
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
        }

    def load_and_preprocess_images(self, image_paths: list[str]) -> torch.Tensor:
        imports = _load_imports()
        images, _ = imports.load_and_preprocess_images_square(image_paths, target_size=int(settings.VGGT_IMAGE_SIZE))
        return images.to(self.device, dtype=self.dtype)

    def predict(self, image_paths: list[str]) -> dict[str, Any]:
        imports = _load_imports()
        model = _load_model()
        started_at = monotonic()
        logger.info(
            "VGGT_INFERENCE_START images=%s device=%s dtype=%s",
            len(image_paths),
            self.device,
            str(self.dtype).replace("torch.", ""),
        )
        stage_started_at = monotonic()
        images = self.load_and_preprocess_images(image_paths)[None]
        _log_stage("preprocess", stage_started_at, self.device, len(image_paths))
        with torch.inference_mode():
            stage_started_at = monotonic()
            with _autocast(self.device, self.dtype):
                aggregated_tokens_list, patch_start_idx = model.aggregator(images)
            _log_stage("aggregator", stage_started_at, self.device, len(image_paths))
            stage_started_at = monotonic()
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = imports.pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            _log_stage("camera_head", stage_started_at, self.device, len(image_paths))
            stage_started_at = monotonic()
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, patch_start_idx)
            _log_stage("depth_head", stage_started_at, self.device, len(image_paths))
            stage_started_at = monotonic()
            point_map, point_conf = model.point_head(aggregated_tokens_list, images, patch_start_idx)
            _log_stage("point_head", stage_started_at, self.device, len(image_paths))
            # The official helper returns NumPy world points even when tensors are supplied.
            stage_started_at = monotonic()
            unprojected = imports.unproject_depth_map_to_point_map(depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0))
            _log_stage("unproject", stage_started_at, self.device, len(image_paths))
        runtime = self.runtime_metadata()
        runtime.update({"image_count": len(image_paths), "runtime_seconds": round(monotonic() - started_at, 3)})
        logger.info(
            "VGGT_INFERENCE_COMPLETE images=%s runtime_seconds=%.3f",
            len(image_paths),
            monotonic() - started_at,
        )
        return {
            "extrinsic": extrinsic.squeeze(0).detach().cpu(),
            "intrinsic": intrinsic.squeeze(0).detach().cpu(),
            "depth_map": depth_map.squeeze(0).detach().cpu(),
            "depth_conf": depth_conf.squeeze(0).detach().cpu(),
            "point_map": point_map.squeeze(0).detach().cpu(),
            "point_conf": point_conf.squeeze(0).detach().cpu(),
            "point_map_unprojected": torch.as_tensor(unprojected).detach().cpu(),
            "runtime": runtime,
        }

    def predict_tracks(self, image_paths: list[str], query_points: torch.Tensor) -> dict[str, Any]:
        """Track a compact, preselected point set through one verified scene component."""
        model = _load_model()
        started_at = monotonic()
        logger.info(
            "VGGT_TRACK_START images=%s points=%s device=%s",
            len(image_paths),
            len(query_points),
            self.device,
        )
        images = self.load_and_preprocess_images(image_paths)[None]
        query_points = query_points.to(self.device, dtype=torch.float32)[None]
        with torch.inference_mode():
            with _autocast(self.device, self.dtype):
                aggregated_tokens_list, patch_start_idx = model.aggregator(images)
            track_list, visibility, confidence = model.track_head(
                aggregated_tokens_list,
                images,
                patch_start_idx,
                query_points=query_points,
            )
        _synchronize_device(self.device)
        logger.info(
            "VGGT_TRACK_COMPLETE images=%s points=%s elapsed_seconds=%.3f",
            len(image_paths),
            query_points.shape[1],
            monotonic() - started_at,
        )
        return {
            "track": track_list[-1].squeeze(0).detach().cpu(),
            "visibility": visibility.squeeze(0).detach().cpu(),
            "confidence": confidence.squeeze(0).detach().cpu(),
        }


vggt_model = VGGTModel()
