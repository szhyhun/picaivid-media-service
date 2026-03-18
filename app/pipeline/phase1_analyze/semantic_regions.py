"""Semantic region extraction and per-photo caching for interior matching.

This stage is designed to guide downstream scoring without changing the LoFTR
matcher input in v1. If SAM/SAM2 is unavailable, it falls back to heuristic
regions so the rest of the pipeline remains stable.
"""
from __future__ import annotations

import importlib
import logging
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from app.core.config import settings
from app.models.openclip import openclip_model

logger = logging.getLogger(__name__)

ANCHOR_LABEL_BY_ROOM = {
    "bathroom": "vanity",
    "laundry room": "appliance_core",
    "bedroom": "bed",
    "kitchen": "island_or_counter",
    "living room": "seating_cluster",
    "family room": "seating_cluster",
    "dining room": "table",
    "office": "desk",
    "hallway": "passage_core",
    "entryway": "entry_core",
}

ANCHOR_REGION_BY_ROOM = {
    "bathroom": (0.32, 0.18, 0.98, 0.88),
    "laundry room": (0.18, 0.22, 0.88, 0.92),
    "bedroom": (0.14, 0.28, 0.86, 0.92),
    "kitchen": (0.14, 0.28, 0.90, 0.92),
    "living room": (0.14, 0.24, 0.88, 0.92),
    "dining room": (0.18, 0.28, 0.86, 0.92),
    "family room": (0.14, 0.24, 0.88, 0.92),
    "office": (0.18, 0.26, 0.86, 0.90),
    "hallway": (0.20, 0.14, 0.80, 0.92),
    "entryway": (0.18, 0.18, 0.82, 0.92),
}

BACKGROUND_LABELS = {"wall", "floor", "ceiling", "sky"}
WINDOW_LABELS = {"window", "glass"}
OBJECT_LABELS = {
    "bed",
    "table",
    "seating_cluster",
    "island_or_counter",
    "vanity",
    "appliance_core",
    "desk",
    "object",
}

CLIP_LABELS = [
    "bed",
    "table",
    "seating cluster",
    "kitchen island",
    "bathroom vanity",
    "washer and dryer",
    "desk",
    "window",
    "glass partition",
    "wall",
    "floor",
    "ceiling",
    "sky",
    "furniture object",
]
CLIP_PROMPT_TEMPLATES = (
    "a photo of a {label}",
    "an interior photo featuring a {label}",
)

_SAM_GENERATOR: Any = None
_SAM_BACKEND_NAME: str | None = None
_SAM_ATTEMPTED = False
_SAM_MAX_LONG_SIDE = 1024


def _choose_sam_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # SAM2 currently hits unsupported bicubic interpolation ops on MPS.
    # Use CPU on Apple Silicon so semantic extraction stays reliable.
    return "cpu"


def _resolve_sam2_config_name(config_value: str) -> str:
    config_path = Path(config_value).expanduser()
    if not config_path.is_absolute():
        return config_value
    if not config_path.exists():
        return config_value

    path_parts = list(config_path.parts)
    if "configs" in path_parts:
        idx = path_parts.index("configs")
        return "/".join(path_parts[idx:])
    return config_path.name


def _load_sam2_checkpoint(model: Any, ckpt_path: str) -> None:
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
    load_result = model.load_state_dict(state_dict)
    missing_keys = list(getattr(load_result, "missing_keys", []))
    unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
    if missing_keys or unexpected_keys:
        if missing_keys:
            logger.error("SAM2 checkpoint missing keys: %s", missing_keys)
        if unexpected_keys:
            logger.error("SAM2 checkpoint unexpected keys: %s", unexpected_keys)
        raise RuntimeError("SAM2 checkpoint load failed")


def _resize_image_for_sam(array: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = array.shape[:2]
    long_side = max(height, width)
    if long_side <= _SAM_MAX_LONG_SIDE:
        return array, (width, height)
    scale = float(_SAM_MAX_LONG_SIDE) / float(long_side)
    resized = cv2.resize(
        array,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, (width, height)


def _resize_mask_to_shape(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask.astype(bool)
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0


def _build_sam2_generator() -> tuple[Any | None, str | None]:
    checkpoint = settings.SAM2_CHECKPOINT
    config = settings.SAM2_CONFIG
    if not checkpoint or not config:
        return None, None

    import sam2  # noqa: F401
    from hydra import compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    automatic_mask_generator = importlib.import_module("sam2.automatic_mask_generator")

    config_name = _resolve_sam2_config_name(config)
    cfg = compose(config_name=config_name, overrides=[])
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_sam2_checkpoint(model, checkpoint)

    device = _choose_sam_device()
    model = model.to(device)
    model.eval()
    generator = automatic_mask_generator.SAM2AutomaticMaskGenerator(
        model,
        points_per_side=16,
        points_per_batch=32,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.90,
        min_mask_region_area=128,
        output_mode="binary_mask",
    )
    return generator, "sam2_clip"


def normalize_room_label(room_label: str | None) -> str:
    if not room_label:
        return "unknown"
    label = str(room_label).strip().lower()
    label = re.sub(r"\s+", " ", label)
    label = re.sub(r"\s+\d+$", "", label)
    return label or "unknown"


def _mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _bbox_to_norm(bbox: tuple[int, int, int, int] | None, width: int, height: int) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    w = max(width, 1)
    h = max(height, 1)
    return (
        float(np.clip(x0 / w, 0.0, 1.0)),
        float(np.clip(y0 / h, 0.0, 1.0)),
        float(np.clip(x1 / w, 0.0, 1.0)),
        float(np.clip(y1 / h, 0.0, 1.0)),
    )


def _bbox_polygon_norm(bbox: tuple[float, float, float, float] | None) -> list[tuple[float, float]]:
    if bbox is None:
        return []
    x0, y0, x1, y1 = bbox
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _crop_from_bbox(array: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    if bbox is None:
        return np.empty((0, 0, 3), dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    return array[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]


def _embedding_from_crop(array: np.ndarray) -> list[float]:
    if array.size == 0:
        return []
    embedding = openclip_model.get_embedding(Image.fromarray(array))
    return [float(v) for v in np.asarray(embedding, dtype=np.float32).flatten().tolist()]


def _cosine_similarity(vec_a: list[float] | None, vec_b: list[float] | None) -> float | None:
    if not vec_a or not vec_b:
        return None
    arr_a = np.asarray(vec_a, dtype=np.float32)
    arr_b = np.asarray(vec_b, dtype=np.float32)
    if arr_a.shape != arr_b.shape or arr_a.size == 0:
        return None
    denom = float(np.linalg.norm(arr_a) * np.linalg.norm(arr_b))
    if denom <= 1e-8:
        return None
    return float(np.clip(np.dot(arr_a, arr_b) / denom, -1.0, 1.0))


def _coarse_label_from_scores(scores: dict[str, float], normalized_room_label: str) -> tuple[str, str]:
    if not scores:
        return "object", "object"
    best_label = max(scores.items(), key=lambda item: item[1])[0]
    best_label = best_label.lower()
    if best_label in {"window", "glass partition"}:
        return ("glass" if "glass" in best_label else "window"), "window"
    if best_label in BACKGROUND_LABELS:
        return best_label, "background"
    if best_label == "kitchen island":
        return "island_or_counter", "anchor" if normalized_room_label == "kitchen" else "object"
    if best_label == "bathroom vanity":
        return "vanity", "anchor" if normalized_room_label == "bathroom" else "object"
    if best_label == "washer and dryer":
        return "appliance_core", "anchor" if normalized_room_label == "laundry room" else "object"
    if best_label == "seating cluster":
        return "seating_cluster", "anchor" if normalized_room_label in {"living room", "family room"} else "object"
    if best_label in {"bed", "table", "desk"}:
        anchor_target = ANCHOR_LABEL_BY_ROOM.get(normalized_room_label)
        return best_label, "anchor" if anchor_target == best_label else "object"
    if best_label == "furniture object":
        return "object", "object"
    return "object", "object"


def _window_component_boxes(array: np.ndarray) -> list[tuple[int, int, int, int]]:
    hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    mask = ((value >= 210) & (saturation >= 20)).astype(np.uint8) * 255
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    height, width = mask.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    total_area = float(height * width)
    for label_idx in range(1, num_labels):
        area = float(stats[label_idx, cv2.CC_STAT_AREA]) / max(total_area, 1.0)
        comp_w = float(stats[label_idx, cv2.CC_STAT_WIDTH]) / max(width, 1)
        comp_h = float(stats[label_idx, cv2.CC_STAT_HEIGHT]) / max(height, 1)
        if area < 0.012 or comp_w < 0.08 or comp_h < 0.06:
            continue
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        boxes.append((x, y, x + w, y + h))
    return boxes[:6]


def _rect_mask(height: int, width: int, bbox_norm: tuple[float, float, float, float]) -> np.ndarray:
    x0 = int(np.clip(round(bbox_norm[0] * width), 0, width - 1))
    y0 = int(np.clip(round(bbox_norm[1] * height), 0, height - 1))
    x1 = int(np.clip(round(bbox_norm[2] * width), x0 + 1, width))
    y1 = int(np.clip(round(bbox_norm[3] * height), y0 + 1, height))
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _build_fallback_regions(array: np.ndarray, normalized_room_label: str) -> list[dict[str, Any]]:
    height, width = array.shape[:2]
    regions: list[dict[str, Any]] = []
    anchor_label = ANCHOR_LABEL_BY_ROOM.get(normalized_room_label)
    anchor_region = ANCHOR_REGION_BY_ROOM.get(normalized_room_label)
    if anchor_label and anchor_region:
        mask = _rect_mask(height, width, anchor_region)
        bbox = _mask_to_bbox(mask)
        bbox_norm = _bbox_to_norm(bbox, width, height)
        regions.append(
            {
                "label": anchor_label,
                "region_type": "anchor",
                "score": 0.95,
                "mask": mask,
                "bbox": bbox,
                "bbox_norm": bbox_norm,
                "polygon_norm": _bbox_polygon_norm(bbox_norm),
                "embedding": _embedding_from_crop(_crop_from_bbox(array, bbox)),
                "area_ratio": float(mask.mean()),
            }
        )

    for bbox in _window_component_boxes(array):
        mask = np.zeros((height, width), dtype=bool)
        x0, y0, x1, y1 = bbox
        mask[y0:y1, x0:x1] = True
        bbox_norm = _bbox_to_norm(bbox, width, height)
        regions.append(
            {
                "label": "window",
                "region_type": "window",
                "score": 0.80,
                "mask": mask,
                "bbox": bbox,
                "bbox_norm": bbox_norm,
                "polygon_norm": _bbox_polygon_norm(bbox_norm),
                "embedding": [],
                "area_ratio": float(mask.mean()),
            }
        )

    for label, bbox_norm in (
        ("ceiling", (0.0, 0.0, 1.0, 0.18)),
        ("floor", (0.0, 0.82, 1.0, 1.0)),
        ("wall", (0.0, 0.0, 0.12, 1.0)),
        ("wall", (0.88, 0.0, 1.0, 1.0)),
    ):
        mask = _rect_mask(height, width, bbox_norm)
        bbox = _mask_to_bbox(mask)
        bbox_norm_out = _bbox_to_norm(bbox, width, height)
        regions.append(
            {
                "label": label,
                "region_type": "background",
                "score": 0.60,
                "mask": mask,
                "bbox": bbox,
                "bbox_norm": bbox_norm_out,
                "polygon_norm": _bbox_polygon_norm(bbox_norm_out),
                "embedding": [],
                "area_ratio": float(mask.mean()),
            }
        )

    return regions


def _ensure_sam_generator() -> tuple[Any | None, str | None]:
    global _SAM_GENERATOR, _SAM_BACKEND_NAME, _SAM_ATTEMPTED
    if _SAM_ATTEMPTED:
        return _SAM_GENERATOR, _SAM_BACKEND_NAME
    _SAM_ATTEMPTED = True
    if not settings.SEMANTIC_REGIONS_ENABLED:
        return None, None

    if settings.SAM2_REPO_DIR:
        repo_dir = Path(settings.SAM2_REPO_DIR).expanduser()
        if repo_dir.exists():
            repo_path = str(repo_dir)
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)

    try:
        if settings.SEMANTIC_REGIONS_BACKEND.lower().startswith("sam2"):
            generator, backend_name = _build_sam2_generator()
            if generator is not None:
                _SAM_GENERATOR = generator
                _SAM_BACKEND_NAME = backend_name
                return _SAM_GENERATOR, _SAM_BACKEND_NAME
    except Exception as err:
        logger.info("SAM2 unavailable, using heuristic semantic fallback: %s", err)

    try:
        segment_anything = importlib.import_module("segment_anything")
        sam_checkpoint = settings.SAM2_CHECKPOINT
        if sam_checkpoint:
            model = segment_anything.sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
            generator = segment_anything.SamAutomaticMaskGenerator(model)
            _SAM_GENERATOR = generator
            _SAM_BACKEND_NAME = "sam_clip"
            return _SAM_GENERATOR, _SAM_BACKEND_NAME
    except Exception:
        pass

    _SAM_GENERATOR = None
    _SAM_BACKEND_NAME = None
    return None, None


def _generate_sam_masks(array: np.ndarray) -> tuple[list[np.ndarray], str | None]:
    generator, backend_name = _ensure_sam_generator()
    if generator is None:
        return [], None
    try:
        work_array, original_shape = _resize_image_for_sam(array)
        generated = generator.generate(work_array)
        mask_list = [np.asarray(item.get("segmentation"), dtype=bool) for item in generated if isinstance(item, dict)]
        original_width, original_height = original_shape
        if work_array.shape[:2] != array.shape[:2]:
            mask_list = [
                _resize_mask_to_shape(mask, original_width, original_height)
                for mask in mask_list
            ]
        return mask_list[:16], backend_name
    except Exception as err:
        logger.info("SAM mask generation failed, using heuristic semantic fallback: %s", err)
        return [], None


@lru_cache(maxsize=1)
def _label_priors() -> dict[str, tuple[str, ...]]:
    return {
        "bedroom": ("bed", "furniture object"),
        "dining room": ("table", "furniture object"),
        "living room": ("seating cluster", "furniture object"),
        "family room": ("seating cluster", "furniture object"),
        "kitchen": ("kitchen island", "table", "furniture object"),
        "bathroom": ("bathroom vanity", "glass partition", "window"),
        "laundry room": ("washer and dryer", "furniture object"),
        "office": ("desk", "furniture object"),
    }


def _classify_mask_crop(crop: np.ndarray, normalized_room_label: str) -> tuple[str, str, float, list[float]]:
    embedding = _embedding_from_crop(crop)
    if crop.size == 0:
        return "object", "object", 0.0, embedding
    labels = list(CLIP_LABELS)
    priors = _label_priors().get(normalized_room_label, ())
    if priors:
        labels = list(dict.fromkeys([*priors, *labels]))
    scores = openclip_model.score_labels(Image.fromarray(crop), labels, prompt_templates=CLIP_PROMPT_TEMPLATES)
    label, region_type = _coarse_label_from_scores(scores, normalized_room_label)
    best_score = max(scores.values()) if scores else 0.0
    return label, region_type, float(best_score), embedding


def _finalize_semantic_result(
    array: np.ndarray,
    normalized_room_label: str,
    semantic_backend: str,
    semantic_regions_available: bool,
    regions: list[dict[str, Any]],
) -> dict[str, Any]:
    height, width = array.shape[:2]
    dilation_px = max(0, int(settings.SEMANTIC_MATCH_EDGE_DILATION_PX))

    anchor_label_target = ANCHOR_LABEL_BY_ROOM.get(normalized_room_label)
    anchor_candidates = [r for r in regions if r["region_type"] == "anchor"]
    if not anchor_candidates and anchor_label_target:
        anchor_candidates = [r for r in regions if r["label"] == anchor_label_target]
    anchor_region = max(anchor_candidates, key=lambda item: (item.get("score", 0.0), item.get("area_ratio", 0.0)), default=None)

    label_masks: dict[str, np.ndarray] = {}
    label_masks_dilated: dict[str, np.ndarray] = {}
    debug_regions: list[dict[str, Any]] = []
    object_embeddings: list[np.ndarray] = []
    object_labels: set[str] = set()
    ignored_mask = np.zeros((height, width), dtype=bool)
    window_mask = np.zeros((height, width), dtype=bool)
    object_mask = np.zeros((height, width), dtype=bool)
    anchor_mask = np.zeros((height, width), dtype=bool)

    kernel = None
    if dilation_px > 0:
        size = max(3, dilation_px * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    for idx, region in enumerate(regions):
        mask = np.asarray(region["mask"], dtype=bool)
        if mask.shape != (height, width):
            continue
        label = str(region["label"])
        region_type = str(region["region_type"])
        if label not in label_masks:
            label_masks[label] = np.zeros((height, width), dtype=bool)
        label_masks[label] |= mask
        if region_type == "background":
            ignored_mask |= mask
        elif region_type == "window":
            window_mask |= mask
        else:
            object_mask |= mask
            object_labels.add(label)
            emb = region.get("embedding") or []
            if emb:
                object_embeddings.append(np.asarray(emb, dtype=np.float32))
        if anchor_region is region:
            anchor_mask |= mask
        debug_regions.append(
            {
                "id": int(idx),
                "label": label,
                "region_type": region_type,
                "bbox": list(region.get("bbox_norm") or (0.0, 0.0, 0.0, 0.0)),
                "polygon": [[float(x), float(y)] for x, y in (region.get("polygon_norm") or [])],
                "score": float(region.get("score", 0.0) or 0.0),
                "area_ratio": float(region.get("area_ratio", 0.0) or 0.0),
            }
        )

    for label, mask in label_masks.items():
        if kernel is None:
            dilated = mask.copy()
        else:
            dilated = cv2.dilate(mask.astype(np.uint8) * 255, kernel, iterations=1) > 0
        label_masks_dilated[label] = dilated

    anchor_embedding = []
    anchor_bbox_norm = None
    anchor_label = None
    if anchor_region is not None:
        anchor_embedding = [float(v) for v in np.asarray(anchor_region.get("embedding") or [], dtype=np.float32).flatten().tolist()]
        anchor_bbox_norm = anchor_region.get("bbox_norm")
        anchor_label = str(anchor_region.get("label"))
    elif anchor_label_target:
        anchor_label = anchor_label_target

    object_embedding = []
    if object_embeddings:
        stacked = np.stack(object_embeddings, axis=0)
        mean_embedding = stacked.mean(axis=0)
        norm = float(np.linalg.norm(mean_embedding))
        if norm > 1e-8:
            mean_embedding /= norm
        object_embedding = [float(v) for v in mean_embedding.tolist()]

    return {
        "semantic_backend": semantic_backend,
        "semantic_regions_available": bool(semantic_regions_available),
        "normalized_room_label": normalized_room_label,
        "anchor_label": anchor_label,
        "anchor_bbox_norm": list(anchor_bbox_norm) if anchor_bbox_norm is not None else None,
        "anchor_embedding": anchor_embedding,
        "object_embedding": object_embedding,
        "object_labels": sorted(object_labels),
        "regions": debug_regions,
        "anchor_mask": anchor_mask,
        "object_mask": object_mask,
        "ignored_mask": ignored_mask,
        "window_mask": window_mask,
        "label_masks": label_masks,
        "label_masks_dilated": label_masks_dilated,
        "width": int(width),
        "height": int(height),
    }


def extract_semantic_regions(image: Image.Image, room_label: str | None) -> dict[str, Any]:
    """Extract semantic regions for one photo.

    Returns both serialized debug regions and in-memory masks used for
    per-match attribution.
    """
    array = np.asarray(image.convert("RGB"))
    normalized_room_label = normalize_room_label(room_label)
    if not settings.SEMANTIC_REGIONS_ENABLED:
        return _finalize_semantic_result(
            array=array,
            normalized_room_label=normalized_room_label,
            semantic_backend="heuristic_fallback",
            semantic_regions_available=False,
            regions=_build_fallback_regions(array, normalized_room_label),
        )

    mask_list, backend_name = _generate_sam_masks(array)
    if not mask_list:
        return _finalize_semantic_result(
            array=array,
            normalized_room_label=normalized_room_label,
            semantic_backend="heuristic_fallback",
            semantic_regions_available=False,
            regions=_build_fallback_regions(array, normalized_room_label),
        )

    regions: list[dict[str, Any]] = []
    height, width = array.shape[:2]
    for mask in mask_list:
        if mask.shape != (height, width):
            continue
        bbox = _mask_to_bbox(mask)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        area_ratio = float(mask.mean())
        if area_ratio < 0.002 or area_ratio > 0.75:
            continue
        crop = _crop_from_bbox(array, bbox)
        label, region_type, score, embedding = _classify_mask_crop(crop, normalized_room_label)
        bbox_norm = _bbox_to_norm(bbox, width, height)
        regions.append(
            {
                "label": label,
                "region_type": region_type,
                "score": score,
                "mask": mask,
                "bbox": bbox,
                "bbox_norm": bbox_norm,
                "polygon_norm": _bbox_polygon_norm(bbox_norm),
                "embedding": embedding,
                "area_ratio": area_ratio,
            }
        )

    if not regions:
        return _finalize_semantic_result(
            array=array,
            normalized_room_label=normalized_room_label,
            semantic_backend="heuristic_fallback",
            semantic_regions_available=False,
            regions=_build_fallback_regions(array, normalized_room_label),
        )

    return _finalize_semantic_result(
        array=array,
        normalized_room_label=normalized_room_label,
        semantic_backend=backend_name or "sam2_clip",
        semantic_regions_available=True,
        regions=regions,
    )


def get_semantic_regions_cached(
    photo_id: int,
    image: Image.Image,
    room_label: str | None,
    cache: dict[int, dict[str, Any]] | None,
) -> dict[str, Any]:
    if cache is None:
        return extract_semantic_regions(image, room_label)
    cached = cache.get(photo_id)
    if cached is not None:
        return cached
    regions = extract_semantic_regions(image, room_label)
    cache[photo_id] = regions
    return regions


def point_semantic_label(
    semantic_regions: dict[str, Any] | None,
    x_norm: float,
    y_norm: float,
) -> tuple[str | None, str | None]:
    if not isinstance(semantic_regions, dict):
        return None, None
    width = int(semantic_regions.get("width") or 0)
    height = int(semantic_regions.get("height") or 0)
    if width <= 0 or height <= 0:
        return None, None
    px = int(np.clip(round(float(x_norm) * width), 0, max(width - 1, 0)))
    py = int(np.clip(round(float(y_norm) * height), 0, max(height - 1, 0)))

    label_masks = semantic_regions.get("label_masks_dilated") or {}
    anchor_label = semantic_regions.get("anchor_label")
    if anchor_label and anchor_label in label_masks and bool(label_masks[anchor_label][py, px]):
        return str(anchor_label), "anchor"
    for label, mask in label_masks.items():
        if label == anchor_label:
            continue
        if bool(mask[py, px]):
            if label in WINDOW_LABELS:
                return str(label), "window"
            if label in BACKGROUND_LABELS:
                return str(label), "background"
            return str(label), "object"

    ignored_mask = semantic_regions.get("ignored_mask")
    if isinstance(ignored_mask, np.ndarray) and ignored_mask.shape == (height, width) and bool(ignored_mask[py, px]):
        return "background", "background"
    window_mask = semantic_regions.get("window_mask")
    if isinstance(window_mask, np.ndarray) and window_mask.shape == (height, width) and bool(window_mask[py, px]):
        return "window", "window"
    object_mask = semantic_regions.get("object_mask")
    if isinstance(object_mask, np.ndarray) and object_mask.shape == (height, width) and bool(object_mask[py, px]):
        return "object", "object"
    return None, None
