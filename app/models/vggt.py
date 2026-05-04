"""Lazy VGGT runtime loader."""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch

from app.core.config import settings
from app.pipeline.phase1_analyze.matcher_loaders import _ensure_local_file, _ensure_local_repo

logger = logging.getLogger(__name__)


@dataclass
class _VGGTImports:
    VGGT: Any
    load_and_preprocess_images: Any
    pose_encoding_to_extri_intri: Any
    unproject_depth_map_to_point_map: Any


def _resolve_repo_dir() -> str:
    fallback = os.path.join(os.path.dirname(settings.MODEL_CACHE_DIR), "third_party", "vggt")
    return _ensure_local_repo(str(settings.VGGT_REPO_DIR or fallback), "VGGT_REPO_ARCHIVE_S3_URI")


def _resolve_checkpoint_path() -> str:
    repo_dir = _resolve_repo_dir()
    fallback = os.path.join(repo_dir, "checkpoints", "VGGT-1B-Commercial", "model.pt")
    return _ensure_local_file(str(settings.VGGT_MODEL_CHECKPOINT or fallback), "VGGT_MODEL_CHECKPOINT_S3_URI")


def _device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("VGGT phase-1 requires a CUDA GPU worker.")
    return "cuda"


def _dtype() -> torch.dtype:
    capability = torch.cuda.get_device_capability()[0]
    return torch.bfloat16 if capability >= 8 else torch.float16


@lru_cache(maxsize=1)
def _load_imports() -> _VGGTImports:
    repo_dir = _resolve_repo_dir()
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map
    except Exception as err:  # pragma: no cover - external runtime
        raise RuntimeError(
            "Failed to import VGGT runtime. Ensure the commercial VGGT repo and dependencies are installed."
        ) from err
    return _VGGTImports(
        VGGT=VGGT,
        load_and_preprocess_images=load_and_preprocess_images,
        pose_encoding_to_extri_intri=pose_encoding_to_extri_intri,
        unproject_depth_map_to_point_map=unproject_depth_map_to_point_map,
    )


@lru_cache(maxsize=1)
def _load_model() -> Any:
    imports = _load_imports()
    checkpoint_path = _resolve_checkpoint_path()
    model = imports.VGGT().to(_device())
    state_dict = torch.load(checkpoint_path, map_location=_device())
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Loaded VGGT checkpoint from %s", checkpoint_path)
    return model


class VGGTModel:
    def _ensure_loaded(self) -> None:
        _load_model()

    @property
    def device(self) -> str:
        return _device()

    @property
    def dtype(self) -> torch.dtype:
        return _dtype()

    def load_and_preprocess_images(self, image_paths: list[str]) -> Any:
        imports = _load_imports()
        return imports.load_and_preprocess_images(image_paths).to(self.device)

    def predict(self, image_paths: list[str]) -> dict[str, Any]:
        imports = _load_imports()
        model = _load_model()
        images = self.load_and_preprocess_images(image_paths)[None]
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                aggregated_tokens_list, ps_idx = model.aggregator(images)
                pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                extrinsic, intrinsic = imports.pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
                depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
                point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
                point_map_by_unprojection = imports.unproject_depth_map_to_point_map(
                    depth_map.squeeze(0),
                    extrinsic.squeeze(0),
                    intrinsic.squeeze(0),
                )

        return {
            "extrinsic": extrinsic.squeeze(0).detach().cpu(),
            "intrinsic": intrinsic.squeeze(0).detach().cpu(),
            "depth_map": depth_map.squeeze(0).detach().cpu(),
            "depth_conf": depth_conf.squeeze(0).detach().cpu(),
            "point_map": point_map.squeeze(0).detach().cpu(),
            "point_conf": point_conf.squeeze(0).detach().cpu(),
            "point_map_unprojected": point_map_by_unprojection.detach().cpu(),
        }


vggt_model = VGGTModel()
