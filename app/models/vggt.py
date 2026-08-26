"""Device-aware VGGT-Omega reconstruction runtime."""
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
    allocated = _allocated_memory_mb(device)
    logger.info(
        "VGGT_OMEGA_STAGE_COMPLETE stage=%s images=%s elapsed_seconds=%.3f allocated_mb=%s",
        stage,
        image_count,
        monotonic() - started_at,
        round(allocated, 1) if allocated is not None else "n/a",
    )


@dataclass
class _OmegaImports:
    VGGTOmega: Any
    load_and_preprocess_images: Any
    encoding_to_camera: Any


def _resolve_repo_dir() -> str:
    fallback = os.path.abspath(os.path.join(os.path.dirname(settings.MODEL_CACHE_DIR), "..", "third_party", "vggt-omega"))
    return _ensure_local_repo(str(settings.VGGT_REPO_DIR or fallback), "VGGT_REPO_ARCHIVE_S3_URI")


def _resolve_checkpoint_path() -> str:
    repo_dir = _resolve_repo_dir()
    if settings.VGGT_MODEL_CHECKPOINT:
        return _ensure_local_file(str(settings.VGGT_MODEL_CHECKPOINT), "VGGT_MODEL_CHECKPOINT_S3_URI")
    cache_root = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--VGGT-Omega/snapshots")
    candidates = (
        os.path.join(cache_root, "f9f63d8da7155e6e28990236d4f5a78f394853b0", "vggt_omega_1b_512.pt"),
        os.path.join(repo_dir, "checkpoints", "vggt_omega_1b_512.pt"),
        os.path.join(repo_dir, "vggt_omega_1b_512.pt"),
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    preferred_target = os.path.join(repo_dir, "checkpoints", "vggt_omega_1b_512.pt")
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
def _load_imports() -> _OmegaImports:
    repo_dir = _resolve_repo_dir()
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    try:
        from vggt_omega.models import VGGTOmega
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        from vggt_omega.utils.pose_enc import encoding_to_camera
    except Exception as error:  # pragma: no cover - external runtime
        raise RuntimeError("Failed to import VGGT-Omega from the configured repository") from error
    return _OmegaImports(VGGTOmega, load_and_preprocess_images, encoding_to_camera)


@lru_cache(maxsize=1)
def _load_model() -> Any:
    imports = _load_imports()
    checkpoint_path = _resolve_checkpoint_path()
    model = imports.VGGTOmega()
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and isinstance(state_dict.get("state_dict"), dict):
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    device = _device()
    model.eval().to(device=device, dtype=_dtype(device))
    logger.info("Loaded VGGT-Omega checkpoint=%s device=%s", checkpoint_path, device)
    return model


def _unproject_depth(depth_map: torch.Tensor, extrinsic: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor:
    """Unproject Omega depth using its documented OpenCV world-to-camera pose."""
    depth = depth_map.squeeze(-1)
    frame_count, height, width = depth.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    x = x.expand(frame_count, -1, -1)
    y = y.expand(frame_count, -1, -1)
    fx = intrinsic[:, 0, 0, None, None]
    fy = intrinsic[:, 1, 1, None, None]
    cx = intrinsic[:, 0, 2, None, None]
    cy = intrinsic[:, 1, 2, None, None]
    camera_points = torch.stack(((x - cx) * depth / fx, (y - cy) * depth / fy, depth), dim=-1)
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return torch.einsum("nij,nhwj->nhwi", rotation.transpose(1, 2), camera_points - translation[:, None, None, :])


def _valid_pixel_masks(image_paths: list[str], padded_hw: tuple[int, int]) -> torch.Tensor:
    """True where a pixel came from the photo, False where preprocessing padded.

    Omega resizes each image to its own patch-aligned shape, then pads the batch to a
    common size with constant white and no mask (`utils/load_fn.py::_pad_images_to_common_size`).
    On a mixed-aspect listing that is 5.1-7.1% of every frame -- synthetic pixels that
    otherwise enter depth, points, overlap denominators and confidence. Thresholds must
    never be calibrated against them.

    Recomputes each image's pre-padding shape with Omega's own helpers so the mask matches
    the tensor exactly; it does not re-run or modify the model.
    """
    imports = _load_imports()
    repo_dir = _resolve_repo_dir()
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from vggt_omega.utils.load_fn import (  # noqa: PLC0415 - optional third-party path
        _balanced_target_shape,
        _crop_to_supported_aspect_ratio,
        _load_rgb_image,
        _max_size_target_shape,
    )

    mode = str(settings.VGGT_IMAGE_MODE).lower()
    resolution = int(settings.VGGT_IMAGE_SIZE)
    max_height, max_width = padded_hw
    masks = torch.zeros((len(image_paths), max_height, max_width), dtype=torch.bool)
    for index, image_path in enumerate(image_paths):
        image = _crop_to_supported_aspect_ratio(_load_rgb_image(image_path))
        width, height = image.size
        aspect_ratio = height / max(width, 1)
        if mode == "balanced":
            target_h, target_w = _balanced_target_shape(aspect_ratio, resolution, 16)
        else:
            target_h, target_w = _max_size_target_shape(aspect_ratio, resolution, 16)
        target_h = min(target_h, max_height)
        target_w = min(target_w, max_width)
        # Omega centres the padding, so mirror that placement exactly.
        top = (max_height - target_h) // 2
        left = (max_width - target_w) // 2
        masks[index, top:top + target_h, left:left + target_w] = True
    return masks


class VGGTModel:
    """Runs VGGT-Omega and normalizes its outputs for the scene planner."""

    supports_tracks = False

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
            "model": "VGGT-Omega-1B-512",
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": _file_sha256(checkpoint_path),
            "repo_dir": repo_dir,
            "repo_commit": _repo_commit(repo_dir),
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
            "image_mode": settings.VGGT_IMAGE_MODE,
            "track_head": "unavailable",
        }

    def load_and_preprocess_images(self, image_paths: list[str]) -> torch.Tensor:
        imports = _load_imports()
        mode = str(settings.VGGT_IMAGE_MODE).lower()
        if mode not in {"balanced", "max_size"}:
            raise RuntimeError("VGGT_IMAGE_MODE must be 'balanced' or 'max_size'")
        images = imports.load_and_preprocess_images(
            image_paths,
            mode=mode,
            image_resolution=int(settings.VGGT_IMAGE_SIZE),
        )
        return images.to(self.device, dtype=self.dtype)

    def predict(self, image_paths: list[str]) -> dict[str, Any]:
        imports = _load_imports()
        model = _load_model()
        started_at = monotonic()
        logger.info(
            "VGGT_OMEGA_INFERENCE_START images=%s device=%s dtype=%s",
            len(image_paths), self.device, str(self.dtype).replace("torch.", ""),
        )
        stage_started_at = monotonic()
        images = self.load_and_preprocess_images(image_paths)[None]
        valid_mask = _valid_pixel_masks(image_paths, tuple(images.shape[-2:]))
        _log_stage("preprocess", stage_started_at, self.device, len(image_paths))
        with torch.inference_mode():
            # Omega's public forward hardcodes CUDA autocast. Execute its heads
            # directly so MPS stays on supported float32 without mutating its repo.
            stage_started_at = monotonic()
            with _autocast(self.device, self.dtype):
                aggregated_tokens_list, patch_token_start = model.aggregator(images)
            _log_stage("aggregator", stage_started_at, self.device, len(image_paths))
            if aggregated_tokens_list[-1] is None:
                raise RuntimeError("VGGT-Omega did not produce final aggregator tokens")
            stage_started_at = monotonic()
            pose_encoding = model.camera_head(aggregated_tokens_list, patch_token_start=patch_token_start)
            extrinsic, intrinsic = imports.encoding_to_camera(pose_encoding, images.shape[-2:])
            _log_stage("camera_head", stage_started_at, self.device, len(image_paths))
            stage_started_at = monotonic()
            depth_map, depth_conf = model.dense_head(
                aggregated_tokens_list,
                images=images,
                patch_token_start=patch_token_start,
            )
            _log_stage("depth_head", stage_started_at, self.device, len(image_paths))
            stage_started_at = monotonic()
            unprojected = _unproject_depth(depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0))
            _log_stage("unproject", stage_started_at, self.device, len(image_paths))
        runtime = self.runtime_metadata()
        runtime.update({"image_count": len(image_paths), "runtime_seconds": round(monotonic() - started_at, 3)})
        logger.info("VGGT_OMEGA_INFERENCE_COMPLETE images=%s runtime_seconds=%.3f", len(image_paths), monotonic() - started_at)
        # Omega predicts depth confidence but not a separate point confidence.
        # The planner uses that same measured confidence for point selection.
        return {
            "extrinsic": extrinsic.squeeze(0).detach().cpu(),
            "intrinsic": intrinsic.squeeze(0).detach().cpu(),
            "depth_map": depth_map.squeeze(0).detach().cpu(),
            "depth_conf": depth_conf.squeeze(0).detach().cpu(),
            "point_map": unprojected.detach().cpu(),
            "point_conf": depth_conf.squeeze(0).detach().cpu(),
            "point_map_unprojected": unprojected.detach().cpu(),
            # True where a pixel is real image data; False where preprocessing padded.
            "valid_mask": valid_mask,
            "runtime": runtime,
        }

    def predict_tracks(self, image_paths: list[str], query_points: torch.Tensor) -> dict[str, Any]:
        raise NotImplementedError("VGGT-Omega-1B-512 does not expose a track head")


vggt_model = VGGTModel()
