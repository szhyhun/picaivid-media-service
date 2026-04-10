"""Precision-first clustering pipeline with certified pairs and sequences."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List, Tuple

try:
    import torch as _torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

import numpy as np
from PIL import Image, ImageOps

from app.db.models import JobPhoto, PhotoSimilarity
from app.pipeline.phase1_analyze.mast3r_pipeline import run_mast3r_phase1

logger = logging.getLogger(__name__)
PRE_PIPELINE_DUPLICATE_SIMILARITY_THRESHOLD = 0.99
PRE_PIPELINE_DUPLICATE_MAX_GAP = 4
PRE_PIPELINE_DUPLICATE_PHASH_MAX_DISTANCE = 8
PRE_PIPELINE_UTILITY_ROOM_LABELS = {
    "garage",
    "storage",
    "storage room",
    "laundry room",
    "utility room",
    "mechanical room",
    "boiler room",
}


def _normalize_embedding(value: object) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    try:
        return [float(item) for item in value]
    except Exception:
        return None


def _load_job_photo_context(
    db_session,
    photo_ids: list[int],
    room_labels: list[str] | None,
) -> tuple[list[int], list[str], np.ndarray | None, list[float]]:
    if db_session is None or not photo_ids:
        fallback_positions = list(range(len(photo_ids)))
        fallback_rooms = room_labels or ["unknown"] * len(photo_ids)
        fallback_quality = [0.0] * len(photo_ids)
        return fallback_positions, fallback_rooms, None, fallback_quality

    rows = (
        db_session.query(JobPhoto)
        .filter(JobPhoto.id.in_(photo_ids))
        .all()
    )
    by_id = {int(row.id): row for row in rows}
    positions: list[int] = []
    resolved_rooms: list[str] = []
    embeddings: list[list[float] | None] = []
    quality_scores: list[float] = []
    for index, photo_id in enumerate(photo_ids):
        row = by_id.get(int(photo_id))
        positions.append(int(row.position or index) if row is not None else index)
        if room_labels is not None and index < len(room_labels):
            resolved_rooms.append(room_labels[index])
        elif row is not None:
            resolved_rooms.append(row.room_override or row.room_label or "unknown")
        else:
            resolved_rooms.append("unknown")
        embeddings.append(_normalize_embedding(row.embedding if row is not None else None))
        quality_scores.append(_photo_quality_score(row))

    if any(item is None for item in embeddings):
        return positions, resolved_rooms, None, quality_scores
    return positions, resolved_rooms, np.asarray(embeddings, dtype=np.float32), quality_scores


def _photo_quality_score(row: JobPhoto | None) -> float:
    if row is None:
        return 0.0
    final_score = float(row.final_score or 0.0)
    if final_score > 0.0:
        return final_score
    base_score = float(row.base_score or 0.0)
    sharpness = float(row.sharpness or 0.0)
    exposure = float(row.exposure_score or 0.0)
    composition = float(row.composition_score or 0.0)
    return 0.45 * base_score + 0.20 * sharpness + 0.20 * exposure + 0.15 * composition


def _prefilter_utility_rooms(
    image_list: list[object],
    photo_ids: list[int],
    positions: list[int],
    room_labels: list[str],
    embeddings: np.ndarray | None,
    quality_scores: list[float],
) -> tuple[list[object], list[int], list[int], list[str], np.ndarray | None, list[float], list[int]]:
    excluded_indices: list[int] = []
    excluded_photo_ids: list[int] = []
    for idx, room_label in enumerate(room_labels):
        normalized = (room_label or "").strip().lower().replace("_", " ")
        if normalized in PRE_PIPELINE_UTILITY_ROOM_LABELS:
            excluded_indices.append(idx)
            excluded_photo_ids.append(int(photo_ids[idx]))

    if not excluded_indices:
        return image_list, photo_ids, positions, room_labels, embeddings, quality_scores, []

    excluded_set = set(excluded_indices)
    keep_indices = [idx for idx in range(len(photo_ids)) if idx not in excluded_set]
    filtered_images = [image_list[idx] for idx in keep_indices]
    filtered_photo_ids = [int(photo_ids[idx]) for idx in keep_indices]
    filtered_positions = [int(positions[idx]) for idx in keep_indices]
    filtered_rooms = [room_labels[idx] for idx in keep_indices]
    filtered_quality = [float(quality_scores[idx]) for idx in keep_indices]
    filtered_embeddings = embeddings[keep_indices] if embeddings is not None else None
    logger.info(
        "Precision prefilter removed utility photos before retrieval: removed=%s kept=%s ids=%s",
        len(excluded_photo_ids),
        len(filtered_photo_ids),
        excluded_photo_ids,
    )
    return (
        filtered_images,
        filtered_photo_ids,
        filtered_positions,
        filtered_rooms,
        filtered_embeddings,
        filtered_quality,
        excluded_photo_ids,
    )


def _prefilter_obvious_duplicates(
    image_list: list[object],
    photo_ids: list[int],
    positions: list[int],
    room_labels: list[str],
    embeddings: np.ndarray | None,
    quality_scores: list[float],
    similarity_threshold: float = PRE_PIPELINE_DUPLICATE_SIMILARITY_THRESHOLD,
    max_gap: int = PRE_PIPELINE_DUPLICATE_MAX_GAP,
) -> tuple[list[object], list[int], list[int], list[str], np.ndarray | None, dict[int, int]]:
    if embeddings is None or len(photo_ids) <= 1:
        return image_list, photo_ids, positions, room_labels, embeddings, {}

    n = len(photo_ids)
    duplicate_of_map: dict[int, int] = {}
    removed_indices: set[int] = set()
    perceptual_hashes = [_compute_perceptual_hash(image) for image in image_list]

    for i in range(n):
        if i in removed_indices:
            continue
        room_i = (room_labels[i] or "").strip().lower()
        if not room_i or room_i == "unknown":
            continue
        for j in range(i + 1, min(n, i + int(max_gap) + 1)):
            if j in removed_indices:
                continue
            room_j = (room_labels[j] or "").strip().lower()
            if room_i != room_j or not room_j or room_j == "unknown":
                continue
            semantic_similarity = float(embeddings[i] @ embeddings[j])
            hash_distance = _hash_distance(perceptual_hashes[i], perceptual_hashes[j])
            is_duplicate_pair = semantic_similarity >= float(similarity_threshold)
            if hash_distance is not None and hash_distance <= PRE_PIPELINE_DUPLICATE_PHASH_MAX_DISTANCE:
                is_duplicate_pair = True
            if not is_duplicate_pair:
                continue
            canonical = max(
                (i, j),
                key=lambda idx: (
                    float(quality_scores[idx] if idx < len(quality_scores) else 0.0),
                    -int(positions[idx] if idx < len(positions) else idx),
                ),
            )
            duplicate = j if canonical == i else i
            removed_indices.add(duplicate)
            duplicate_of_map[int(photo_ids[duplicate])] = int(photo_ids[canonical])
            logger.info(
                "Precision pre-dedup duplicate %s -> %s (room=%s, sem=%.3f, hash_distance=%s, quality=%.3f<%.3f, gap=%s)",
                photo_ids[duplicate],
                photo_ids[canonical],
                room_labels[duplicate],
                semantic_similarity,
                "n/a" if hash_distance is None else hash_distance,
                float(quality_scores[duplicate] if duplicate < len(quality_scores) else 0.0),
                float(quality_scores[canonical] if canonical < len(quality_scores) else 0.0),
                abs(int(positions[duplicate]) - int(positions[canonical])),
            )

    if not duplicate_of_map:
        return image_list, photo_ids, positions, room_labels, embeddings, {}

    keep_indices = [idx for idx in range(n) if idx not in removed_indices]
    filtered_images = [image_list[idx] for idx in keep_indices]
    filtered_photo_ids = [int(photo_ids[idx]) for idx in keep_indices]
    filtered_positions = [int(positions[idx]) for idx in keep_indices]
    filtered_rooms = [room_labels[idx] for idx in keep_indices]
    filtered_embeddings = embeddings[keep_indices] if embeddings is not None else None
    logger.info(
        "Precision pre-dedup summary: %s -> %s photos (removed=%s threshold=%.2f max_gap=%s)",
        n,
        len(filtered_photo_ids),
        len(duplicate_of_map),
        float(similarity_threshold),
        int(max_gap),
    )
    return (
        filtered_images,
        filtered_photo_ids,
        filtered_positions,
        filtered_rooms,
        filtered_embeddings,
        duplicate_of_map,
    )


def _compute_perceptual_hash(image: object) -> int | None:
    pil_image: Image.Image | None = None
    if isinstance(image, Image.Image):
        pil_image = image
    elif isinstance(image, np.ndarray):
        try:
            pil_image = Image.fromarray(image)
        except Exception:
            return None
    if pil_image is None:
        return None
    try:
        grayscale = ImageOps.exif_transpose(pil_image).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = np.asarray(grayscale, dtype=np.int16)
        diff = pixels[:, 1:] > pixels[:, :-1]
        hash_value = 0
        for bit in diff.flatten():
            hash_value = (hash_value << 1) | int(bool(bit))
        return hash_value
    except Exception:
        return None


def _hash_distance(left: int | None, right: int | None) -> int | None:
    if left is None or right is None:
        return None
    return int((left ^ right).bit_count())


def _persist_embeddings(db_session, photo_ids: list[int], embeddings: np.ndarray) -> None:
    if db_session is None or embeddings is None or len(photo_ids) != len(embeddings):
        return
    rows = (
        db_session.query(JobPhoto)
        .filter(JobPhoto.id.in_(photo_ids))
        .all()
    )
    by_id = {int(row.id): row for row in rows}
    changed = False
    for photo_id, embedding in zip(photo_ids, embeddings):
        row = by_id.get(int(photo_id))
        if row is None:
            continue
        normalized = _normalize_embedding(row.embedding)
        if normalized is not None:
            continue
        row.embedding = [float(value) for value in embedding.tolist()]
        changed = True
    if changed:
        db_session.flush()


def _persist_pair_records(db_session, job_id: int, pair_records: list[dict[str, Any]]) -> None:
    if db_session is None or job_id is None:
        return
    db_session.query(PhotoSimilarity).filter(PhotoSimilarity.job_id == int(job_id)).delete(synchronize_session=False)
    if not pair_records:
        db_session.flush()
        return
    payloads = []
    allowed_columns = set(PhotoSimilarity.__table__.columns.keys())
    for record in pair_records:
        payload = {
            key: value
            for key, value in record.items()
            if key in allowed_columns and key not in {"raw_matches_payload", "inlier_matches_payload", "hard_reject"}
        }
        payload["job_id"] = int(job_id)
        payloads.append(payload)
    db_session.bulk_insert_mappings(PhotoSimilarity, payloads)
    db_session.flush()


def _build_clusters(
    photo_ids: list[int],
    pair_records: list[dict[str, Any]],
    positions_by_photo: dict[int, int],
    room_by_photo: dict[int, str],
) -> list[list[int]]:
    eligible = [
        record
        for record in pair_records
        if str(record.get("certification_status")) in {"strong", "usable"}
        and not bool(record.get("hard_reject"))
        and int(record.get("is_connected") or 0) == 1
    ]
    eligible.sort(
        key=lambda row: (
            float(row.get("pair_rank", 0.0) or 0.0),
            float(row.get("overlap_ratio", 0.0) or 0.0),
            float(row.get("combined_geometry_score", 0.0) or 0.0),
            float(row.get("reciprocal_match_count", 0.0) or 0.0),
            float(row.get("order_proximity", 0.0) or 0.0),
        ),
        reverse=True,
    )
    assigned: set[int] = set()
    clusters: list[list[int]] = []
    for record in eligible:
        left = int(record["photo_a_id"])
        right = int(record["photo_b_id"])
        if left in assigned or right in assigned:
            continue
        room_left = normalize_room_label(room_by_photo.get(left))
        room_right = normalize_room_label(room_by_photo.get(right))
        if not room_left or room_left == "unknown":
            continue
        if room_left != room_right:
            continue
        clusters.append(sorted([left, right], key=lambda photo_id: positions_by_photo.get(photo_id, 0)))
        assigned.add(left)
        assigned.add(right)

    for photo_id in photo_ids:
        if int(photo_id) not in assigned:
            clusters.append([int(photo_id)])

    clusters.sort(key=lambda cluster: positions_by_photo.get(cluster[0], 0))
    return clusters


def cluster_photos_precision_first(
    images: Iterable[object],
    photo_ids: list[int],
    db_session=None,
    job_id: int | None = None,
    room_labels: list[str] | None = None,
    return_metadata: bool = False,
) -> list[list[int]] | tuple[list[list[int]], dict[str, object]]:
    """Run the MASt3R-first graph pipeline with conservative 2-photo clustering."""
    image_list = list(images)
    if len(image_list) <= 1:
        clusters = [photo_ids] if photo_ids else []
        if return_metadata:
            return clusters, {"duplicate_of_map": {}, "duplicates_dropped": False, "transition_sequences": []}
        return clusters

    logger.info("MASt3R graph pipeline start: photos=%s job_id=%s", len(photo_ids), job_id)

    positions, resolved_room_labels, persisted_embeddings, quality_scores = _load_job_photo_context(
        db_session=db_session,
        photo_ids=photo_ids,
        room_labels=room_labels,
    )
    utility_excluded_photo_ids: list[int] = []
    image_list, photo_ids, positions, resolved_room_labels, persisted_embeddings, quality_scores, utility_excluded_photo_ids = _prefilter_utility_rooms(
        image_list=image_list,
        photo_ids=photo_ids,
        positions=positions,
        room_labels=resolved_room_labels,
        embeddings=persisted_embeddings,
        quality_scores=quality_scores,
    )
    pre_pipeline_duplicate_of_map: dict[int, int] = {}
    image_list, photo_ids, positions, resolved_room_labels, persisted_embeddings, pre_pipeline_duplicate_of_map = _prefilter_obvious_duplicates(
        image_list=image_list,
        photo_ids=photo_ids,
        positions=positions,
        room_labels=resolved_room_labels,
        embeddings=persisted_embeddings,
        quality_scores=quality_scores,
    )
    _persist_embeddings(db_session, photo_ids, persisted_embeddings) if persisted_embeddings is not None else None
    pipeline_started_at = time.perf_counter()
    final_clusters, pair_records, pose_rows, similarity, transition_sequences = run_mast3r_phase1(
        images=image_list,
        photo_ids=photo_ids,
        positions=positions,
        room_labels=resolved_room_labels,
        db_session=db_session,
        job_id=job_id,
    )
    _persist_pair_records(db_session, job_id, pair_records)
    logger.info(
        "MASt3R graph inference complete: photos=%s edges=%s poses=%s elapsed_ms=%.1f",
        len(photo_ids),
        len(pair_records),
        len(pose_rows),
        (time.perf_counter() - pipeline_started_at) * 1000.0,
    )

    logger.info(
        "MASt3R clustering complete: photos=%s edges=%s certified=%s clusters=%s sequences=%s",
        len(photo_ids),
        len(pair_records),
        sum(1 for record in pair_records if str(record.get("certification_status")) in {"strong", "usable"}),
        len(final_clusters),
        len(transition_sequences),
    )

    if return_metadata:
        return final_clusters, {
            "duplicate_of_map": pre_pipeline_duplicate_of_map,
            "duplicates_dropped": bool(pre_pipeline_duplicate_of_map),
            "utility_excluded_ids": utility_excluded_photo_ids,
            "pair_records": pair_records,
            "transition_sequences": transition_sequences,
            "similarity_matrix": similarity,
        }
    return final_clusters
