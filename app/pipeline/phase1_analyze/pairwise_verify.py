"""Stateless pairwise verification — the membership authority.

Runs VGGT-Omega on **exactly two photos** and returns raw geometric evidence.

Why exactly two: batch composition measurably flips verdicts. In an 8-photo mixed
batch a known different-room pair (positions 37/42) scored depth_ok 0.289 while a
known same-room pair (36/37) scored 0.156 — inverted. Confidence behaves the same
way: a bathroom trio alone gives conf 7.50, and adding two unrelated photos drops
the same frames to 1.59.

Why stateless: `verify()` depends only on the two images, the checkpoint and the
preprocessing. It never inspects graph connectivity, so results are cacheable,
order-independent and re-scorable offline under any threshold. Classification and
merge policy live outside this module.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

from app.core.config import settings
from app.models.vggt import PreparedImage, vggt_model

SAMPLE_POINTS = 4096
DEPTH_AGREEMENT = 0.15
EVIDENCE_SCHEMA_VERSION = "2026-08-25.1"


@dataclass
class DirectionEvidence:
    """Evidence for projecting the source frame into the target frame."""

    sampled: int = 0
    in_front: int = 0
    in_image: int = 0
    depth_agree: int = 0
    visible_fraction: float = 0.0
    depth_ok: float = 0.0
    median_relative_depth_error: float | None = None   # None, not NaN: NaN is not valid JSON
    conf_region: float = 0.0
    median_dx: float = 0.0     # image-space motion of agreeing points, for motion planning
    median_dy: float = 0.0


@dataclass
class RelativePose:
    """Pose of B relative to A, in A's frame.

    Stored so the component pose graph can be built from cached evidence. Without
    this the graph would need the model re-run, defeating the cache. `scale` is the
    run's median depth: each 2-photo run has its own arbitrary scale, so
    translations are only comparable after dividing by it.
    """

    rotation: list[list[float]] = field(default_factory=list)   # 3x3, B relative to A
    translation: list[float] = field(default_factory=list)      # 3, in run units
    scale: float = 0.0                                          # run median depth
    rot_deg: float = 0.0


@dataclass
class PairEvidence:
    """Raw, threshold-free evidence for one photo pair.

    `photo_a`/`photo_b` are canonical content hashes, not temporary file paths.
    This keeps cached evidence stable across re-ingestion and prevents directional
    evidence from being served swapped.
    """

    photo_a: str
    photo_b: str
    conf_pair: float = 0.0
    conf_frame_a: float = 0.0
    conf_frame_b: float = 0.0
    baseline: float = 0.0
    median_depth: float = 0.0
    bl_over_depth: float = 0.0
    rot_deg: float = 0.0
    valid_fraction_a: float = 1.0
    valid_fraction_b: float = 1.0
    relative_pose: RelativePose = field(default_factory=RelativePose)
    forward: DirectionEvidence = field(default_factory=DirectionEvidence)
    backward: DirectionEvidence = field(default_factory=DirectionEvidence)
    runtime_seconds: float = 0.0

    @property
    def depth_ok_min(self) -> float:
        return min(self.forward.depth_ok, self.backward.depth_ok)

    @property
    def depth_ok_max(self) -> float:
        return max(self.forward.depth_ok, self.backward.depth_ok)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["depth_ok_min"] = self.depth_ok_min
        payload["depth_ok_max"] = self.depth_ok_max
        return payload


def evidence_key(
    path_a: str,
    path_b: str,
    checkpoint_sha: str,
    image_mode: str,
    image_size: int,
    *,
    model_revision: str = "",
    precision: str = "",
    schema_version: str = EVIDENCE_SCHEMA_VERSION,
    image_digests: tuple[str, str] | None = None,
) -> str:
    """Content-addressed cache key.

    Keyed on image *content* so it survives re-ingestion (job_photos ids are
    reassigned every analysis run). Ordered so verify(a,b) and verify(b,a) share
    an entry. Classification thresholds are deliberately excluded. The evidence
    schema, model implementation and precision are included because they can
    change the measured evidence itself.
    """
    digests = sorted(image_digests or (file_sha256(path_a), file_sha256(path_b)))
    parts = [
        *digests,
        checkpoint_sha,
        model_revision,
        precision,
        image_mode,
        str(image_size),
        schema_version,
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def file_sha256(path: str) -> str:
    """Hash an image once at job scope, then pass the digest to pair helpers."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _direction_evidence(
    source: int,
    target: int,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    depth: np.ndarray,
    points: np.ndarray,
    conf: np.ndarray,
    masks: np.ndarray,
) -> DirectionEvidence:
    height, width = depth.shape[1:]
    source_points = points[source].reshape(-1, 3)
    source_confidence = conf[source].reshape(-1)
    valid_source = (
        masks[source].reshape(-1)
        & np.isfinite(source_points).all(axis=1)
        & np.isfinite(source_confidence)
    )
    candidates = np.flatnonzero(valid_source)
    if candidates.size == 0:
        return DirectionEvidence()

    # Uniform stride keeps sampling deterministic and unbiased across the frame.
    if candidates.size > SAMPLE_POINTS:
        candidates = candidates[np.linspace(0, candidates.size - 1, SAMPLE_POINTS, dtype=int)]

    sampled_points = source_points[candidates]
    sampled_conf = source_confidence[candidates]

    camera = sampled_points @ extrinsic[target][:3, :3].T + extrinsic[target][:3, 3]
    z = camera[:, 2]
    projected = camera @ intrinsic[target].T
    pixels = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-8)
    x = np.rint(pixels[:, 0]).astype(int)
    y = np.rint(pixels[:, 1]).astype(int)

    in_front = z > 1e-6
    inside = in_front & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not np.any(inside):
        return DirectionEvidence(sampled=int(candidates.size), in_front=int(in_front.sum()))

    # A projection landing on padded target pixels is not evidence of anything.
    target_valid = np.zeros_like(inside)
    target_valid[inside] = masks[target][y[inside], x[inside]]
    usable = inside & target_valid
    if not np.any(usable):
        return DirectionEvidence(
            sampled=int(candidates.size), in_front=int(in_front.sum()), in_image=int(inside.sum())
        )

    usable_indices = np.flatnonzero(usable)
    target_depth = depth[target][y[usable], x[usable]]
    source_depth = z[usable]
    target_conf = conf[target][y[usable], x[usable]].reshape(-1)
    finite = np.isfinite(target_depth) & np.isfinite(source_depth) & np.isfinite(target_conf)
    if not np.any(finite):
        return DirectionEvidence(
            sampled=int(candidates.size),
            in_front=int(in_front.sum()),
            in_image=int(inside.sum()),
        )
    usable_indices = usable_indices[finite]
    target_depth = target_depth[finite]
    source_depth = source_depth[finite]
    target_conf = target_conf[finite]
    relative_error = np.abs(source_depth - target_depth) / np.maximum(
        np.maximum(np.abs(source_depth), np.abs(target_depth)), 1e-6
    )
    agree = relative_error <= DEPTH_AGREEMENT

    # image-space motion of the depth-agreeing points, used later for motion direction
    source_y, source_x = np.divmod(candidates[usable_indices], width)
    usable_pixels = pixels[usable_indices]
    if np.any(agree):
        dx = float(np.median(usable_pixels[agree][:, 0] - source_x[agree]))
        dy = float(np.median(usable_pixels[agree][:, 1] - source_y[agree]))
    else:
        dx = dy = 0.0

    return DirectionEvidence(
        sampled=int(candidates.size),
        in_front=int(in_front.sum()),
        in_image=int(inside.sum()),
        depth_agree=int(agree.sum()),
        visible_fraction=float(finite.sum()) / float(candidates.size),
        depth_ok=float(agree.sum()) / float(candidates.size),
        median_relative_depth_error=float(np.median(relative_error)),
        conf_region=float(min(np.median(sampled_conf[usable_indices]), np.median(target_conf))),
        median_dx=dx,
        median_dy=dy,
    )


def canonical_order(path_a: str, path_b: str) -> tuple[str, str]:
    """Order a pair deterministically by image content.

    The cache key sorts content hashes, so verify(A,B) and verify(B,A) share an
    entry. Inference must therefore run in that same order, or a reversed request
    would receive forward/backward evidence swapped.
    """
    ordered = canonical_order_with_ids(path_a, 0, path_b, 1)
    return ordered[0][0], ordered[1][0]


def canonical_order_with_ids(
    path_a: str,
    photo_a_id: int,
    path_b: str,
    photo_b_id: int,
    *,
    digest_a: str | None = None,
    digest_b: str | None = None,
) -> tuple[tuple[str, int], tuple[str, int]]:
    """Keep external photo ids aligned with canonical directional evidence.

    Pair metrics such as overlap are symmetric, but relative pose and image-space
    movement are not. Any manifest attaching ids after `verify()` must use this
    same ordering or it can silently assign A-to-B motion to B-to-A photos.
    """
    pairs = [
        (digest_a or file_sha256(path_a), path_a, photo_a_id),
        (digest_b or file_sha256(path_b), path_b, photo_b_id),
    ]
    pairs.sort(key=lambda item: (item[0], item[1]))
    return (pairs[0][1], pairs[0][2]), (pairs[1][1], pairs[1][2])


def _finite_masked_values(values: np.ndarray, mask: np.ndarray, name: str) -> np.ndarray:
    selected = np.asarray(values)[mask].reshape(-1).astype(np.float64, copy=False)
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        raise RuntimeError(f"VGGT pair inference produced no finite {name} values")
    return selected


def verify(
    path_a: str,
    path_b: str,
    *,
    image_digests: tuple[str, str] | None = None,
    prepared_images: tuple[PreparedImage, PreparedImage] | None = None,
) -> PairEvidence:
    """Run one 2-photo reconstruction and return raw evidence (no thresholds).

    Evidence is returned in canonical (content-hash) order regardless of argument
    order, so results are identical and safely cacheable either way.
    """
    from time import monotonic

    digest_a, digest_b = image_digests or (file_sha256(path_a), file_sha256(path_b))
    ordered = [
        (digest_a, path_a, prepared_images[0] if prepared_images else None),
        (digest_b, path_b, prepared_images[1] if prepared_images else None),
    ]
    ordered.sort(key=lambda item: (item[0], item[1]))
    (digest_a, path_a, prepared_a), (digest_b, path_b, prepared_b) = ordered
    started = monotonic()
    predictions = (
        vggt_model.predict_prepared([prepared_a, prepared_b])
        if prepared_a is not None and prepared_b is not None
        else vggt_model.predict([path_a, path_b])
    )
    extrinsic = predictions["extrinsic"].numpy().astype(np.float64)
    intrinsic = predictions["intrinsic"].numpy().astype(np.float64)
    # Keep dense model outputs in their native float32. Projection promotes only
    # sampled points against float64 camera matrices, avoiding three full-frame
    # float64 copies per pair with no loss of model information.
    depth = predictions["depth_map"].numpy().squeeze(-1)
    points = predictions["point_map"].numpy()
    conf = predictions["depth_conf"].numpy()
    masks = predictions["valid_mask"].numpy()

    depth_a = _finite_masked_values(depth[0], masks[0], "depth for frame A")
    depth_b = _finite_masked_values(depth[1], masks[1], "depth for frame B")
    conf_a = _finite_masked_values(conf[0], masks[0], "confidence for frame A")
    conf_b = _finite_masked_values(conf[1], masks[1], "confidence for frame B")

    centre_a = -extrinsic[0][:3, :3].T @ extrinsic[0][:3, 3]
    centre_b = -extrinsic[1][:3, :3].T @ extrinsic[1][:3, 3]
    baseline = float(np.linalg.norm(centre_a - centre_b))
    median_depth = float(0.5 * (np.median(depth_a) + np.median(depth_b)))
    rotation = extrinsic[0][:3, :3] @ extrinsic[1][:3, :3].T
    rot_deg = math.degrees(math.acos(float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))))
    # Pose of B expressed in A's frame: what the pose graph consumes.
    rotation_ab = extrinsic[1][:3, :3] @ extrinsic[0][:3, :3].T
    translation_ab = extrinsic[1][:3, 3] - rotation_ab @ extrinsic[0][:3, 3]

    return PairEvidence(
        photo_a=digest_a,
        photo_b=digest_b,
        # Confidence over real pixels only; padded regions carry no information.
        conf_pair=float(np.median(np.concatenate([conf_a, conf_b]))),
        conf_frame_a=float(np.median(conf_a)),
        conf_frame_b=float(np.median(conf_b)),
        baseline=baseline,
        median_depth=median_depth,
        bl_over_depth=baseline / max(median_depth, 1e-12),
        rot_deg=rot_deg,
        relative_pose=RelativePose(
            rotation=[[float(v) for v in row] for row in rotation_ab],
            translation=[float(v) for v in translation_ab],
            scale=median_depth,
            rot_deg=rot_deg,
        ),
        valid_fraction_a=float(masks[0].mean()),
        valid_fraction_b=float(masks[1].mean()),
        forward=_direction_evidence(0, 1, extrinsic, intrinsic, depth, points, conf, masks),
        backward=_direction_evidence(1, 0, extrinsic, intrinsic, depth, points, conf, masks),
        runtime_seconds=monotonic() - started,
    )


def _read_cache_record(path: str) -> dict[str, Any] | None:
    try:
        with open(path) as handle:
            record = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(record, dict):
        return None
    if record.get("_evidence_schema_version") != EVIDENCE_SCHEMA_VERSION:
        return None
    return record


def _write_cache_record(path: str, record: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(path), delete=False) as handle:
            temporary = handle.name
            json.dump(record, handle, allow_nan=False)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def verify_with_cache(
    path_a: str,
    path_b: str,
    *,
    runtime: dict[str, Any] | None = None,
    cache_dir: str | None = None,
    image_digests: tuple[str, str] | None = None,
    prepared_loader: Callable[[str], PreparedImage] | None = None,
) -> tuple[dict[str, Any], bool, str]:
    """Return raw pair evidence, reusing content-addressed inference when possible.

    Returns ``(record, cache_hit, key)``. Thresholds stay outside the key so
    membership and pairing policy can be retuned without rerunning VGGT.
    """
    runtime = runtime or vggt_model.runtime_metadata()
    digest_a, digest_b = image_digests or (file_sha256(path_a), file_sha256(path_b))
    key = evidence_key(
        path_a,
        path_b,
        str(runtime["checkpoint_sha256"]),
        str(settings.VGGT_IMAGE_MODE).lower(),
        int(settings.VGGT_IMAGE_SIZE),
        model_revision=str(runtime.get("repo_commit") or ""),
        precision=str(runtime.get("dtype") or ""),
        image_digests=(digest_a, digest_b),
    )
    directory = os.path.abspath(cache_dir or settings.VGGT_PAIR_CACHE_DIR)
    cache_path = os.path.join(directory, f"{key}.json")
    cached = _read_cache_record(cache_path)
    if cached is not None:
        return dict(cached), True, key

    prepared = (
        (prepared_loader(path_a), prepared_loader(path_b))
        if prepared_loader is not None
        else None
    )
    record = verify(
        path_a,
        path_b,
        image_digests=(digest_a, digest_b),
        prepared_images=prepared,
    ).to_dict()
    record["_evidence_schema_version"] = EVIDENCE_SCHEMA_VERSION
    _write_cache_record(cache_path, record)
    return dict(record), False, key
