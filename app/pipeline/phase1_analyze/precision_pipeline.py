"""Precision-first clustering pipeline with certified pairs and sequences."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from app.db.models import JobPhoto, PhotoSimilarity
from app.pipeline.phase1_analyze.candidate_retrieval import build_candidate_pairs
from app.pipeline.phase1_analyze.pair_verification import verify_pair_precision_first
from app.pipeline.phase1_analyze.sequence_builder import build_transition_sequences

logger = logging.getLogger(__name__)


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
) -> tuple[list[int], list[str], np.ndarray | None]:
    if db_session is None or not photo_ids:
        fallback_positions = list(range(len(photo_ids)))
        fallback_rooms = room_labels or ["unknown"] * len(photo_ids)
        return fallback_positions, fallback_rooms, None

    rows = (
        db_session.query(JobPhoto)
        .filter(JobPhoto.id.in_(photo_ids))
        .all()
    )
    by_id = {int(row.id): row for row in rows}
    positions: list[int] = []
    resolved_rooms: list[str] = []
    embeddings: list[list[float] | None] = []
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

    if any(item is None for item in embeddings):
        return positions, resolved_rooms, None
    return positions, resolved_rooms, np.asarray(embeddings, dtype=np.float32)


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
    db_session.query(PhotoSimilarity).filter(PhotoSimilarity.job_id == int(job_id)).delete()
    if not pair_records:
        db_session.flush()
        return
    payloads = []
    for record in pair_records:
        payload = {key: value for key, value in record.items() if key not in {"raw_matches_payload", "inlier_matches_payload", "hard_reject"}}
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
            float(row.get("order_proximity", 0.0) or 0.0),
            float(row.get("dinov2_similarity", 0.0) or 0.0),
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
        room_left = (room_by_photo.get(left) or "").strip().lower()
        room_right = (room_by_photo.get(right) or "").strip().lower()
        if room_left and room_right and room_left != "unknown" and room_right != "unknown" and room_left != room_right:
            continue
        pair = sorted([left, right], key=lambda photo_id: positions_by_photo.get(photo_id, 0))
        clusters.append(pair)
        assigned.add(left)
        assigned.add(right)

    for photo_id in sorted(photo_ids, key=lambda value: positions_by_photo.get(value, 0)):
        if photo_id not in assigned:
            clusters.append([int(photo_id)])

    clusters.sort(key=lambda cluster: min(positions_by_photo.get(photo_id, 0) for photo_id in cluster))
    return clusters


def cluster_photos_precision_first(
    images: Iterable[object],
    photo_ids: list[int],
    db_session=None,
    job_id: int | None = None,
    room_labels: list[str] | None = None,
    return_metadata: bool = False,
) -> list[list[int]] | tuple[list[list[int]], dict[str, object]]:
    """Run the precision-first candidate -> verify -> certify pipeline."""
    image_list = list(images)
    if len(image_list) <= 1:
        clusters = [photo_ids] if photo_ids else []
        if return_metadata:
            return clusters, {"duplicate_of_map": {}, "duplicates_dropped": False, "transition_sequences": []}
        return clusters

    logger.info("Precision-first pipeline start: photos=%s job_id=%s", len(photo_ids), job_id)

    positions, resolved_room_labels, persisted_embeddings = _load_job_photo_context(
        db_session=db_session,
        photo_ids=photo_ids,
        room_labels=room_labels,
    )
    candidate_stage_started_at = time.perf_counter()
    embeddings, similarity, candidates = build_candidate_pairs(
        image_list,
        positions=positions,
        room_labels=resolved_room_labels,
        embeddings=persisted_embeddings,
    )
    logger.info(
        "Precision-first candidate retrieval complete: candidates=%s elapsed_ms=%.1f",
        len(candidates),
        (time.perf_counter() - candidate_stage_started_at) * 1000.0,
    )
    _persist_embeddings(db_session, photo_ids, embeddings)

    depth_cache: dict[int, np.ndarray] = {}
    preprocessed_cache: dict[int, dict[str, Any]] = {}
    pair_records: list[dict[str, Any]] = []
    total_candidates = len(candidates)
    for idx, candidate in enumerate(candidates, start=1):
        photo_a_id = int(photo_ids[candidate.left_idx])
        photo_b_id = int(photo_ids[candidate.right_idx])
        pair_started_at = time.perf_counter()
        logger.info(
            "Precision-first pair %s/%s: %s <-> %s source=%s dino=%.3f",
            idx,
            total_candidates,
            photo_a_id,
            photo_b_id,
            candidate.pair_source,
            float(candidate.dinov2_similarity),
        )
        result = verify_pair_precision_first(
            photo_a_id=photo_a_id,
            photo_b_id=photo_b_id,
            image_a=image_list[candidate.left_idx],
            image_b=image_list[candidate.right_idx],
            dinov2_similarity=float(candidate.dinov2_similarity),
            position_a=int(positions[candidate.left_idx]),
            position_b=int(positions[candidate.right_idx]),
            depth_cache=depth_cache,
            preprocessed_cache=preprocessed_cache,
        )
        result["pair_source"] = candidate.pair_source
        result["photo_a_id"] = min(photo_a_id, photo_b_id)
        result["photo_b_id"] = max(photo_a_id, photo_b_id)
        compact_result = {
            key: value
            for key, value in result.items()
            if key not in {"raw_matches_payload", "inlier_matches_payload"}
        }
        pair_records.append(compact_result)
        logger.info(
            "Precision-first result %s/%s: %s <-> %s status=%s rank=%.3f inliers=%s overlap=%.3f crossing=%.3f elapsed_ms=%.1f",
            idx,
            total_candidates,
            photo_a_id,
            photo_b_id,
            str(compact_result.get("certification_status") or "reject"),
            float(compact_result.get("pair_rank", 0.0) or 0.0),
            int(compact_result.get("f_inliers", 0) or 0),
            float(compact_result.get("overlap_ratio", 0.0) or 0.0),
            float(compact_result.get("crossing_penalty", 0.0) or 0.0),
            (time.perf_counter() - pair_started_at) * 1000.0,
        )

    _persist_pair_records(db_session, job_id, pair_records)

    positions_by_photo = {int(photo_ids[idx]): int(positions[idx]) for idx in range(len(photo_ids))}
    room_by_photo = {int(photo_ids[idx]): resolved_room_labels[idx] for idx in range(len(photo_ids))}
    final_clusters = _build_clusters(
        photo_ids=photo_ids,
        pair_records=pair_records,
        positions_by_photo=positions_by_photo,
        room_by_photo=room_by_photo,
    )
    cluster_by_photo: dict[int, int] = {}
    for cluster_index, cluster in enumerate(final_clusters):
        for photo_id in cluster:
            cluster_by_photo[int(photo_id)] = cluster_index

    transition_sequences = build_transition_sequences(
        photo_ids=[int(photo_id) for photo_id in photo_ids],
        pair_records=pair_records,
        room_labels=room_by_photo,
        cluster_by_photo=cluster_by_photo,
    )

    logger.info(
        "Precision-first clustering complete: photos=%s candidates=%s certified=%s clusters=%s sequences=%s",
        len(photo_ids),
        len(candidates),
        sum(1 for record in pair_records if str(record.get("certification_status")) in {"strong", "usable"}),
        len(final_clusters),
        len(transition_sequences),
    )

    if return_metadata:
        return final_clusters, {
            "duplicate_of_map": {},
            "duplicates_dropped": False,
            "pair_records": pair_records,
            "transition_sequences": transition_sequences,
            "similarity_matrix": similarity,
        }
    return final_clusters
