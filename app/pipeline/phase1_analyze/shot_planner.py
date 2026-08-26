"""Renderer-neutral editorial planning from verified Phase 1 scene evidence."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import AnalysisResult, Job, JobPhoto, PhotoRelation, RoomCluster


PLANNER_VERSION = "v2.3-typed-shot-score"


def build_and_persist_shot_plan(db: Session, job: Job, clusters: list[RoomCluster]) -> dict[str, Any]:
    """Persist one compact shot record per cluster and return the complete plan.

    AnalysisResult.debug_metrics is intentionally used as the persistence boundary:
    Phase 2 can consume this plan without taking a dependency on a renderer or a new
    database table.
    """
    # The session is created with autoflush=False, and plan_motion_for_cluster()
    # only db.add()s its AnalysisResult rows. Without an explicit flush the query
    # below returns none of them, `analyses` comes back empty, and the persistence
    # loop silently `continue`s on every cluster -- producing a job with no shots.
    db.flush()

    photos = db.query(JobPhoto).filter(JobPhoto.job_id == job.id).all()
    photos_by_cluster: dict[int, list[JobPhoto]] = defaultdict(list)
    photo_map = {int(photo.id): photo for photo in photos}
    for photo in photos:
        if photo.room_cluster_id is not None:
            photos_by_cluster[int(photo.room_cluster_id)].append(photo)
    for cluster_photos in photos_by_cluster.values():
        cluster_photos.sort(key=lambda photo: (photo.cluster_order or 0, photo.position or 0, photo.id))

    analyses = {
        int(analysis.room_cluster_id): analysis
        for analysis in db.query(AnalysisResult).filter(AnalysisResult.job_id == job.id).all()
        if analysis.room_cluster_id is not None
    }
    relations = {
        (min(int(row.photo_a_id), int(row.photo_b_id)), max(int(row.photo_a_id), int(row.photo_b_id))): row
        for row in db.query(PhotoRelation).filter(PhotoRelation.job_id == job.id).all()
    }

    candidates: list[tuple[tuple[Any, ...], RoomCluster, dict[str, Any]]] = []
    for cluster in clusters:
        cluster_photos = photos_by_cluster.get(int(cluster.id), [])
        if not cluster_photos:
            continue
        analysis = analyses.get(int(cluster.id))
        shot = _build_shot(cluster, cluster_photos, analysis, relations)
        candidates.append((_sort_key(cluster, cluster_photos, shot), cluster, shot))

    candidates = _order_component_blocks(candidates)
    previous: dict[str, Any] | None = None
    for index, (_, cluster, shot) in enumerate(candidates):
        cluster.sequence_order = index
        shot["order_index"] = index
        shot["transition_type"] = _transition_type(previous, shot, relations)
        if previous is not None:
            shot["transition_from_photo_id"] = previous["ordered_photo_ids"][-1]
            shot["previous_cluster_id"] = previous["cluster_id"]
        previous = shot
    ordered_shots = [shot for _, _, shot in candidates]
    for index, shot in enumerate(ordered_shots):
        shot["next_cluster_id"] = ordered_shots[index + 1]["cluster_id"] if index + 1 < len(ordered_shots) else None
        shot.setdefault("previous_cluster_id", None)
    _mark_redundant_neighbors(ordered_shots, relations)

    runtime = _runtime_from_clusters(clusters)
    plan = {
        "planner_version": PLANNER_VERSION,
        "job_id": int(job.id),
        "project_id": str(job.project_id),
        "runtime_provenance": runtime,
        "target_length_seconds": _target_length(job),
        "target_group_budget": _group_budget(_target_length(job)),
        "sequence_edges": _sequence_edges(ordered_shots, relations),
        "ordered_shots": ordered_shots,
    }

    for _, cluster, shot in candidates:
        analysis = analyses.get(int(cluster.id))
        if analysis is None:
            continue
        metrics = dict(analysis.debug_metrics or {})
        metrics["shot"] = shot
        # A complete plan on every row creates needless repeated JSON. Keep one
        # canonical copy on the first ordered shot and independent records elsewhere.
        metrics.pop("shot_plan", None)
        analysis.debug_metrics = metrics
    if candidates:
        first_analysis = analyses.get(int(candidates[0][1].id))
        if first_analysis is not None:
            metrics = dict(first_analysis.debug_metrics or {})
            metrics["shot_plan"] = plan
            first_analysis.debug_metrics = metrics

    db.flush()
    return plan


def _build_shot(
    cluster: RoomCluster,
    photos: list[JobPhoto],
    analysis: AnalysisResult | None,
    relations: dict[tuple[int, int], PhotoRelation] | None = None,
) -> dict[str, Any]:
    hero = _hero_photo(cluster, photos)
    role = _story_role(cluster, photos, hero)
    motion = analysis.recommended_motion if analysis is not None else cluster.recommended_motion
    duration = analysis.recommended_duration if analysis is not None else cluster.recommended_duration
    confidence = float(cluster.geometry_confidence or 0.0)
    # `sfm_eligible` is the motion-authorization decision. Do not apply the old
    # V1 same-scene threshold to V2's differently-scaled component diagnostic.
    multi_view = bool(cluster.sfm_eligible and len(photos) > 1)
    # A verified two-photo group is still two real photographs even when generated
    # camera motion between them is not authorized (V2 withholds interpolation
    # until transition ground truth exists). Reporting "single image move" for it
    # was simply wrong: the shot contains two images and cuts between them.
    verified_pair = bool(not multi_view and len(photos) > 1)
    connection_evidence = _connection_evidence(photos, relations or {})
    shot_score, shot_score_kind = _shot_score(connection_evidence, hero)
    return {
        "cluster_id": int(cluster.id),
        "scene_component_id": int(cluster.scene_component_id) if cluster.scene_component_id is not None else None,
        "room_id": (cluster.room_type or "scene").strip().lower(),
        "story_role": role,
        "ordered_photo_ids": [int(photo.id) for photo in photos],
        "hero_photo_id": int(hero.id),
        "shot_type": (
            "verified_multi_view" if multi_view
            else "verified_pair" if verified_pair
            else "single_image_move"
        ),
        "motion_intent": motion or "subtle_pan",
        "duration_seconds": float(duration or 3.0),
        "transition_type": "opening",
        "confidence": round(confidence, 4),
        "shot_score": round(shot_score, 4) if shot_score is not None else None,
        "shot_score_kind": shot_score_kind,
        "skip_recommended": False,
        "skip_reason": None,
        "duplicate_of_cluster_id": None,
        "evidence": {
            "photo_count": len(photos),
            "geometry_confidence": confidence,
            "cluster_overlap": float(cluster.overlap_score or 0.0),
            "motion_affordance": cluster.recommended_motion,
            "hard_editorial_role": _editorial_role(hero),
            "hero_quality": float(hero.final_score or 0.0),
            "requested_motion": ((analysis.debug_metrics or {}).get("requested_motion") if analysis is not None and isinstance(analysis.debug_metrics, dict) else None),
            "geometry_connections": connection_evidence,
            "sequence_mode": (
                "continuous_geometry"
                if connection_evidence and all(edge["continuity_type"] == "interpolation_safe" for edge in connection_evidence)
                else "matched_same_scene"
                if connection_evidence
                else "single_view"
            ),
        },
        "rejection_reasons": _rejection_reasons(multi_view, analysis, verified_pair),
    }


def _order_component_blocks(
    candidates: list[tuple[tuple[Any, ...], RoomCluster, dict[str, Any]]],
) -> list[tuple[tuple[Any, ...], RoomCluster, dict[str, Any]]]:
    blocks: dict[tuple[str, int], list[tuple[tuple[Any, ...], RoomCluster, dict[str, Any]]]] = defaultdict(list)
    for candidate in candidates:
        cluster = candidate[1]
        component_id = int(cluster.scene_component_id) if cluster.scene_component_id is not None else int(cluster.id)
        kind = "component" if cluster.scene_component_id is not None else "cluster"
        blocks[(kind, component_id)].append(candidate)
    ordered_blocks = sorted(blocks.values(), key=lambda block: min(candidate[0] for candidate in block))
    return [
        candidate
        for block in ordered_blocks
        for candidate in sorted(block, key=lambda item: (int(item[1].sequence_order or 0), int(item[1].id)))
    ]


def _relation_between_shots(
    left: dict[str, Any],
    right: dict[str, Any],
    relations: dict[tuple[int, int], PhotoRelation],
) -> PhotoRelation | None:
    candidates: list[PhotoRelation] = []
    for left_id in left["ordered_photo_ids"]:
        for right_id in right["ordered_photo_ids"]:
            relation = relations.get((min(int(left_id), int(right_id)), max(int(left_id), int(right_id))))
            if relation is not None:
                candidates.append(relation)
    if not candidates:
        return None
    return max(candidates, key=lambda relation: (float(relation.overlap_score or 0.0), float(relation.relation_confidence or 0.0)))


def _similar_angle_duplicate(relation: PhotoRelation | None) -> bool:
    if relation is None:
        return False
    if relation.continuity_type == "duplicate":
        return True
    transform = relation.relative_transform or {}
    rotation = float(transform.get("rotation_degrees", 180.0))
    baseline = float(transform.get("normalized_baseline", 1.0))
    return (
        float(relation.overlap_score or 0.0) >= 0.72
        and float(relation.relation_confidence or 0.0) >= 0.65
        and rotation <= 15.0
        and baseline <= 0.12
    )


def _mark_redundant_neighbors(
    shots: list[dict[str, Any]],
    relations: dict[tuple[int, int], PhotoRelation],
) -> None:
    for left, right in zip(shots, shots[1:]):
        relation = _relation_between_shots(left, right, relations)
        if not _similar_angle_duplicate(relation):
            continue
        protected = {"opening", "hero", "closing"}
        choices = [shot for shot in (left, right) if shot["evidence"].get("hard_editorial_role") not in protected]
        if not choices:
            continue
        skip = min(
            choices,
            key=lambda shot: (float(shot["evidence"].get("hero_quality") or 0.0), -int(shot["order_index"])),
        )
        keep = right if skip is left else left
        if keep.get("skip_recommended"):
            continue
        skip["skip_recommended"] = True
        skip["duplicate_of_cluster_id"] = int(keep["cluster_id"])
        skip["skip_reason"] = (
            "Adjacent shot covers the same reconstructed surfaces from a similar camera angle; "
            "keep it visible for review but omit it from the final render by default."
        )


def _sequence_edges(
    shots: list[dict[str, Any]],
    relations: dict[tuple[int, int], PhotoRelation],
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for left, right in zip(shots, shots[1:]):
        relation = _relation_between_shots(left, right, relations)
        edges.append({
            "from_cluster_id": int(left["cluster_id"]),
            "to_cluster_id": int(right["cluster_id"]),
            "transition_type": str(right["transition_type"]),
            "continuity_type": str(relation.continuity_type) if relation is not None else None,
            "relation_confidence": round(float(relation.relation_confidence or 0.0), 4) if relation is not None else None,
        })
    return edges


def _connection_evidence(
    photos: list[JobPhoto],
    relations: dict[tuple[int, int], PhotoRelation],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for left, right in zip(photos, photos[1:]):
        pair = (min(int(left.id), int(right.id)), max(int(left.id), int(right.id)))
        relation = relations.get(pair)
        if relation is None:
            continue
        metrics = relation.debug_metrics or {}
        evidence.append({
            "from_photo_id": int(left.id),
            "to_photo_id": int(right.id),
            "continuity_type": str(relation.continuity_type),
            "same_scene_confidence": round(float(relation.relation_confidence or 0.0), 4),
            "surface_overlap": round(float(relation.overlap_score or 0.0), 4),
            "reprojection_score": round(float(relation.reprojection_score or 0.0), 4),
            "image_motion": [round(float(relation.direction_dx or 0.0), 2), round(float(relation.direction_dy or 0.0), 2)],
            "pair_score": (
                round(float(metrics["pair_score"]), 4)
                if metrics.get("pair_score") is not None
                else None
            ),
            "depth_ok_forward": metrics.get("depth_ok_forward"),
            "depth_ok_backward": metrics.get("depth_ok_backward"),
            "rotation_degrees": metrics.get("rotation_degrees"),
            "conf_pair": metrics.get("conf_pair"),
            "bl_over_depth": metrics.get("bl_over_depth"),
        })
    return evidence


def _shot_score(
    connection_evidence: list[dict[str, Any]],
    hero: JobPhoto,
) -> tuple[float | None, str]:
    """Return an explicitly typed editorial score, never a fake confidence."""
    pair_scores = [
        float(edge["pair_score"])
        for edge in connection_evidence
        if edge.get("pair_score") is not None
    ]
    if pair_scores:
        return max(pair_scores), "pair_quality"
    quality = hero.final_score
    if quality is not None:
        return float(quality), "image_quality"
    return None, "unavailable"


def _hero_photo(cluster: RoomCluster, photos: list[JobPhoto]) -> JobPhoto:
    if cluster.hero_photo_id is not None:
        for photo in photos:
            if int(photo.id) == int(cluster.hero_photo_id):
                return photo
    explicit = [photo for photo in photos if _editorial_role(photo) == "hero"]
    return max(explicit or photos, key=lambda photo: (float(photo.final_score or 0.0), -(photo.position or 0), -int(photo.id)))


def _editorial_role(photo: JobPhoto) -> str:
    role = str((photo.manual_metadata or {}).get("editorial_role", "auto")).lower()
    return role if role in {"auto", "opening", "hero", "closing", "exclude"} else "auto"


def _rejection_reasons(
    multi_view: bool, analysis: AnalysisResult | None, verified_pair: bool = False
) -> list[str]:
    reason = None
    if analysis is not None and isinstance(analysis.debug_metrics, dict):
        reason = analysis.debug_metrics.get("motion_fallback_reason")
    reasons = [str(reason)] if reason else []
    if not multi_view:
        # Distinguish "the photos are not verified together" from "they are
        # verified together but generated motion between them is not authorized".
        # Reporting the first for both was misleading.
        reasons.insert(0, (
            "Verified same-room pair; generated camera motion between the two views "
            "is withheld until transition safety is labeled. Cut between them."
            if verified_pair else
            "Insufficient verified multi-view continuity; use a single-image move."
        ))
    return reasons


def _story_role(cluster: RoomCluster, photos: list[JobPhoto], hero: JobPhoto) -> str:
    roles = {_editorial_role(photo) for photo in photos}
    if "opening" in roles:
        return "opening"
    if "closing" in roles:
        return "closing"
    label = (cluster.room_type or "scene").lower()
    if "drone" in label:
        return "drone_opener"
    if any(token in label for token in ("front", "exterior", "facade")):
        return "front_exterior"
    if any(token in label for token in ("entry", "foyer", "hall", "stair")):
        return "approach_entry"
    if any(token in label for token in ("living", "kitchen", "dining", "great room")):
        return "social_hero" if _editorial_role(hero) == "hero" else "social_room"
    if any(token in label for token in ("bed", "bath", "closet")):
        return "private_room"
    if any(token in label for token in ("patio", "deck", "pool", "yard", "garden", "outdoor")):
        return "outdoor_payoff"
    return "detail_or_service"


def _sort_key(cluster: RoomCluster, photos: list[JobPhoto], shot: dict[str, Any]) -> tuple[Any, ...]:
    role = shot["story_role"]
    story_order = {
        "opening": 0,
        "drone_opener": 1,
        "front_exterior": 2,
        "approach_entry": 3,
        "social_hero": 4,
        "social_room": 5,
        "private_room": 6,
        "detail_or_service": 7,
        "outdoor_payoff": 8,
        "closing": 99,
    }
    hero_bonus = 0 if _editorial_role(_hero_photo(cluster, photos)) == "hero" else 1
    upload_position = min(int(photo.position or 0) for photo in photos)
    return (story_order[role], hero_bonus, -float(shot["confidence"]), upload_position, int(cluster.id))


def _transition_type(previous: dict[str, Any] | None, current: dict[str, Any], relations: dict[tuple[int, int], PhotoRelation]) -> str:
    if previous is None:
        return "opening"
    pair = (min(previous["ordered_photo_ids"][-1], current["ordered_photo_ids"][0]), max(previous["ordered_photo_ids"][-1], current["ordered_photo_ids"][0]))
    relation = relations.get(pair)
    if relation is not None and relation.continuity_type == "interpolation_safe" and relation.is_connected:
        return "interpolate"
    if relation is not None and relation.is_bridge_edge:
        return "doorway_cut"
    return "editorial_cut"


def _runtime_from_clusters(clusters: list[RoomCluster]) -> dict[str, Any]:
    for cluster in clusters:
        component = cluster.scene_component
        if component and isinstance(component.debug_metrics, dict):
            runtime = component.debug_metrics.get("runtime")
            if isinstance(runtime, dict):
                return runtime
    return {}


def _target_length(job: Job) -> float:
    try:
        return float(job.target_length or 45)
    except (TypeError, ValueError):
        return 45.0


def _group_budget(length: float) -> tuple[int, int]:
    if length <= 30:
        return (6, 8)
    if length <= 45:
        return (9, 12)
    return (12, 16)
