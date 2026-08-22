"""Renderer-neutral editorial planning from verified Phase 1 scene evidence."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import AnalysisResult, Job, JobPhoto, PhotoRelation, RoomCluster


PLANNER_VERSION = "v2.0"


def build_and_persist_shot_plan(db: Session, job: Job, clusters: list[RoomCluster]) -> dict[str, Any]:
    """Persist one compact shot record per cluster and return the complete plan.

    AnalysisResult.debug_metrics is intentionally used as the persistence boundary:
    Phase 2 can consume this plan without taking a dependency on a renderer or a new
    database table.
    """
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
        shot = _build_shot(cluster, cluster_photos, analysis)
        candidates.append((_sort_key(cluster, cluster_photos, shot), cluster, shot))

    candidates.sort(key=lambda candidate: candidate[0])
    previous: dict[str, Any] | None = None
    for index, (_, cluster, shot) in enumerate(candidates):
        cluster.sequence_order = index
        shot["order_index"] = index
        shot["transition_type"] = _transition_type(previous, shot, relations)
        if previous is not None:
            shot["transition_from_photo_id"] = previous["ordered_photo_ids"][-1]
        previous = shot

    runtime = _runtime_from_clusters(clusters)
    plan = {
        "planner_version": PLANNER_VERSION,
        "job_id": int(job.id),
        "project_id": str(job.project_id),
        "runtime_provenance": runtime,
        "target_length_seconds": _target_length(job),
        "target_group_budget": _group_budget(_target_length(job)),
        "ordered_shots": [shot for _, _, shot in candidates],
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


def _build_shot(cluster: RoomCluster, photos: list[JobPhoto], analysis: AnalysisResult | None) -> dict[str, Any]:
    hero = _hero_photo(cluster, photos)
    role = _story_role(cluster, photos, hero)
    motion = analysis.recommended_motion if analysis is not None else cluster.recommended_motion
    duration = analysis.recommended_duration if analysis is not None else cluster.recommended_duration
    confidence = float(cluster.geometry_confidence or 0.0)
    multi_view = bool(cluster.sfm_eligible and len(photos) > 1 and confidence >= 0.56)
    return {
        "cluster_id": int(cluster.id),
        "scene_component_id": int(cluster.scene_component_id) if cluster.scene_component_id is not None else None,
        "room_id": (cluster.room_type or "scene").strip().lower(),
        "story_role": role,
        "ordered_photo_ids": [int(photo.id) for photo in photos],
        "hero_photo_id": int(hero.id),
        "shot_type": "verified_multi_view" if multi_view else "single_image_move",
        "motion_intent": motion or "subtle_pan",
        "duration_seconds": float(duration or 3.0),
        "transition_type": "opening",
        "confidence": round(confidence, 4),
        "evidence": {
            "photo_count": len(photos),
            "geometry_confidence": confidence,
            "cluster_overlap": float(cluster.overlap_score or 0.0),
            "motion_affordance": cluster.recommended_motion,
            "hard_editorial_role": _editorial_role(hero),
            "requested_motion": ((analysis.debug_metrics or {}).get("requested_motion") if analysis is not None and isinstance(analysis.debug_metrics, dict) else None),
        },
        "rejection_reasons": _rejection_reasons(multi_view, analysis),
    }


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


def _rejection_reasons(multi_view: bool, analysis: AnalysisResult | None) -> list[str]:
    reason = None
    if analysis is not None and isinstance(analysis.debug_metrics, dict):
        reason = analysis.debug_metrics.get("motion_fallback_reason")
    reasons = [str(reason)] if reason else []
    if not multi_view:
        reasons.insert(0, "Insufficient verified multi-view continuity; use a single-image move.")
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
