"""Scene-component clustering using VGGT geometry outputs."""
from __future__ import annotations

import logging
from typing import Any
from collections import Counter
from typing import Dict, List, Optional

from PIL import Image
from sqlalchemy.orm import Session

from app.db.models import (
    Job,
    JobPhoto,
    RoomCluster,
    SceneComponent,
    SceneComponentMembership,
    PhotoSceneGeometry,
    PhotoRelation,
)
from app.pipeline.phase1_analyze import pairing
from app.pipeline.phase1_analyze.vggt_pipeline import (
    SceneComponentResult,
    PhotoGeometryResult,
    PhotoRelationResult,
    run_vggt_scene_pipeline,
)

logger = logging.getLogger(__name__)


def cluster_photos_by_room(
    db: Session,
    job: Job,
    photos: List[JobPhoto],
    s3_client=None,
    use_overlap_detection: bool = True,
    preloaded_images: Optional[Dict[int, Image.Image]] = None,
) -> List[RoomCluster]:
    del use_overlap_detection  # geometry-first pipeline ignores the old flag
    active_photos = sorted(
        [
            photo
            for photo in photos
            if not photo.exclude
            and not photo.is_duplicate
            and (photo.manual_metadata or {}).get("editorial_role") != "exclude"
        ],
        key=lambda photo: photo.position or 0,
    )
    logger.info("Clustering %s photos for job %s with VGGT scene graph", len(active_photos), job.id)
    if not active_photos:
        return []

    images: list[Image.Image] = []
    photo_ids: list[int] = []
    room_labels: list[str] = []
    positions: list[int] = []
    photo_map: dict[int, JobPhoto] = {}
    quality_scores: dict[int, float] = {}
    editorial_roles: dict[int, str] = {}
    embeddings: Dict[int, list] = {}
    for photo in active_photos:
        image = preloaded_images.get(photo.id) if preloaded_images is not None else None
        if image is None:
            image = s3_client.download_image(photo.s3_uri)
            if preloaded_images is not None:
                preloaded_images[photo.id] = image
        images.append(image)
        photo_ids.append(int(photo.id))
        room_labels.append(photo.room_override or photo.room_label or "")
        positions.append(int(photo.position or 0))
        photo_map[int(photo.id)] = photo
        quality_scores[int(photo.id)] = float(photo.final_score or 0.0)
        if photo.embedding:
            embeddings[int(photo.id)] = photo.embedding
        editorial_roles[int(photo.id)] = str((photo.manual_metadata or {}).get("editorial_role", "auto"))
        photo.room_cluster_id = None
        photo.cluster_order = None

    if preloaded_images is not None:
        # Every semantic and quality consumer has finished. Drop cache ownership
        # now so excluded photos are released immediately; active photos remain
        # owned by `images` until the pipeline writes and closes them once.
        active_ids = set(photo_ids)
        for photo_id, cached_image in preloaded_images.items():
            if int(photo_id) not in active_ids:
                cached_image.close()
        preloaded_images.clear()

    try:
        geometries, relations, components = run_vggt_scene_pipeline(
            images=images,
            photo_ids=photo_ids,
            room_labels=room_labels,
            positions=positions,
            job_id=int(job.id),
            quality_scores=quality_scores,
            editorial_roles=editorial_roles,
            embeddings=embeddings,
        )
    finally:
        for image in images:
            image.close()
        images.clear()

    scene_row_by_key = _replace_scene_state(
        db=db,
        job_id=int(job.id),
        geometries=geometries,
        relations=relations,
        components=components,
        photo_map=photo_map,
    )
    clusters = _materialize_room_clusters(
        db=db,
        job=job,
        components=components,
        relations=relations,
        photo_map=photo_map,
        scene_row_by_key=scene_row_by_key,
    )
    logger.info("Created %s room clusters from %s VGGT scene components", len(clusters), len(components))
    return clusters


def _replace_scene_state(
    db: Session,
    job_id: int,
    geometries: list[PhotoGeometryResult],
    relations: list[PhotoRelationResult],
    components: list[SceneComponentResult],
    photo_map: dict[int, JobPhoto],
) -> dict[str, SceneComponent]:
    db.query(SceneComponentMembership).filter(SceneComponentMembership.job_id == job_id).delete(synchronize_session=False)
    db.query(PhotoSceneGeometry).filter(PhotoSceneGeometry.job_id == job_id).delete(synchronize_session=False)
    db.query(PhotoRelation).filter(PhotoRelation.job_id == job_id).delete(synchronize_session=False)
    db.query(SceneComponent).filter(SceneComponent.job_id == job_id).delete(synchronize_session=False)
    db.query(RoomCluster).filter(RoomCluster.job_id == job_id).delete(synchronize_session=False)
    db.flush()

    scene_row_by_key: dict[str, SceneComponent] = {}
    role_by_photo: dict[int, str] = {}
    relations_by_pair = {
        (min(int(relation.photo_a_id), int(relation.photo_b_id)), max(int(relation.photo_a_id), int(relation.photo_b_id))): relation
        for relation in relations
    }
    for component in components:
        row = SceneComponent(
            job_id=job_id,
            scene_type=component.scene_type,
            component_key=component.component_key,
            photo_count=len(component.photo_ids),
            geometry_confidence=component.geometry_confidence,
            connectivity_confidence=component.connectivity_confidence,
            track_coverage=None,
            avg_reprojection_error=component.avg_reprojection_error,
            hero_photo_id=component.hero_photo_id,
            depth_range=None,
            motion_affordance=component.motion_affordance,
            debug_metrics=component.debug_metrics,
        )
        db.add(row)
        scene_row_by_key[component.component_key] = row
    # Allocate all component ids in one database round trip.
    db.flush()

    component_id_by_photo: dict[int, int] = {}
    for component in components:
        row = scene_row_by_key[component.component_key]
        for index, photo_id in enumerate(component.ordered_photo_ids):
            component_id_by_photo[int(photo_id)] = int(row.id)
            role = "support"
            if photo_id == component.hero_photo_id:
                role = "hero"
            elif index == 0 or index == len(component.ordered_photo_ids) - 1:
                role = "endpoint"
            for other_id in component.photo_ids:
                if other_id == photo_id:
                    continue
                relation = relations_by_pair.get((min(int(photo_id), int(other_id)), max(int(photo_id), int(other_id))))
                if relation is not None and relation.is_bridge_edge:
                    role = "bridge"
                    break
            role_by_photo[int(photo_id)] = role
            db.add(
                SceneComponentMembership(
                    job_id=job_id,
                    photo_id=int(photo_id),
                    scene_component_id=int(row.id),
                    order_index=index,
                    photo_role=role,
                    is_primary=True,
                )
            )

    for geometry in geometries:
        photo = photo_map.get(int(geometry.photo_id))
        if photo is not None:
            # V2 does not run a whole-listing depth pass. Clear legacy values so a
            # re-analysis cannot silently reuse stale V1 metrics.
            photo.depth_variance = None
            photo.depth_layers = None
        component_id = component_id_by_photo.get(int(geometry.photo_id))
        db.add(
            PhotoSceneGeometry(
                job_id=job_id,
                photo_id=int(geometry.photo_id),
                scene_component_id=component_id,
                pose_confidence=geometry.pose_confidence,
                depth_confidence=geometry.depth_confidence,
                point_confidence=None,
                visibility_score=geometry.visibility_score,
                reprojection_error=geometry.reprojection_error,
                camera_extrinsic=geometry.camera_extrinsic,
                camera_intrinsic=None,
                camera_center=geometry.camera_center,
                view_direction=geometry.view_direction,
                depth_artifact_uri=None,
                point_map_artifact_uri=None,
                local_metrics={**geometry.local_metrics, "photo_role": role_by_photo.get(int(geometry.photo_id), "support")},
            )
        )

    for relation in relations:
        left_component = component_id_by_photo.get(int(relation.photo_a_id))
        right_component = component_id_by_photo.get(int(relation.photo_b_id))
        component_id = left_component if left_component == right_component else None
        db.add(
            PhotoRelation(
                job_id=job_id,
                photo_a_id=int(relation.photo_a_id),
                photo_b_id=int(relation.photo_b_id),
                scene_component_id=component_id,
                overlap_score=relation.overlap_score,
                track_support=None,
                reprojection_score=relation.reprojection_score,
                relation_confidence=relation.relation_confidence,
                baseline_distance=relation.baseline_distance,
                relative_transform=relation.relative_transform,
                direction_dx=relation.direction_dx,
                direction_dy=relation.direction_dy,
                continuity_type=relation.continuity_type,
                is_bridge_edge=relation.is_bridge_edge,
                is_connected=relation.is_connected,
                debug_metrics=relation.debug_metrics,
            )
        )

    db.flush()
    return scene_row_by_key


def _materialize_room_clusters(
    db: Session,
    job: Job,
    components: list[SceneComponentResult],
    relations: list[PhotoRelationResult],
    photo_map: dict[int, JobPhoto],
    scene_row_by_key: dict[str, SceneComponent] | None = None,
) -> list[RoomCluster]:
    if scene_row_by_key is None:
        scene_rows = db.query(SceneComponent).filter(SceneComponent.job_id == int(job.id)).all()
        scene_row_by_key = {row.component_key: row for row in scene_rows}
    relation_by_pair = {
        (min(int(relation.photo_a_id), int(relation.photo_b_id)), max(int(relation.photo_a_id), int(relation.photo_b_id))): relation
        for relation in relations
    }
    component_key_by_photo = {
        int(photo_id): component.component_key
        for component in components
        for photo_id in component.photo_ids
    }
    relations_by_component: dict[str, dict[tuple[int, int], PhotoRelationResult]] = {}
    for pair, relation in relation_by_pair.items():
        component_key = component_key_by_photo.get(pair[0])
        if component_key is not None and component_key == component_key_by_photo.get(pair[1]):
            relations_by_component.setdefault(component_key, {})[pair] = relation
    clusters: list[RoomCluster] = []
    photo_assignments: list[tuple[RoomCluster, list[JobPhoto]]] = []
    sequence_order = 0
    for component in components:
        scene_row = scene_row_by_key[component.component_key]
        ordered_photos = [photo_map[int(photo_id)] for photo_id in component.ordered_photo_ids if int(photo_id) in photo_map]
        component_room = _majority_room_label(ordered_photos) if ordered_photos else None
        for cluster_photos in _derive_render_groups(
            ordered_photos,
            component.scene_type,
            relations_by_component.get(component.component_key),
            component_room,
        ):
            if not cluster_photos:
                continue
            room_type = _majority_room_label(cluster_photos)
            geometry_confidence = component.geometry_confidence
            confidence_tier = _confidence_tier(geometry_confidence, component.motion_affordance, len(cluster_photos))
            overlap_score = _cluster_overlap_score(cluster_photos, relation_by_pair)
            cluster = RoomCluster(
                job_id=int(job.id),
                scene_component_id=int(scene_row.id),
                room_type=room_type,
                confidence_tier=confidence_tier,
                sfm_eligible=component.motion_affordance in {"parallax", "multi_view"} and len(cluster_photos) >= 2,
                image_count=len(cluster_photos),
                overlap_score=overlap_score,
                depth_variance=None,
                geometry_confidence=geometry_confidence,
                sequence_order=sequence_order,
                recommended_motion=component.motion_affordance,
            )
            db.add(cluster)
            clusters.append(cluster)
            photo_assignments.append((cluster, cluster_photos))
            sequence_order += 1
    # Allocate all cluster ids in one round trip, then attach their photos.
    db.flush()
    for cluster, cluster_photos in photo_assignments:
        for photo_order, photo in enumerate(cluster_photos):
            photo.room_cluster_id = int(cluster.id)
            photo.cluster_order = photo_order
    return clusters


def _derive_render_groups(
    ordered_photos: list[JobPhoto],
    scene_type: str,
    relation_by_pair: dict[tuple[int, int], Any] | None = None,
    room_type: str | None = None,
) -> list[list[JobPhoto]]:
    """Split one verified component into the shots that reach the film.

    Previously this chunked the ordered photo list into adjacent pairs, which
    ignored the pairing scores entirely: photos 1-2 and 3-4 were paired because
    they were next to each other, not because they made a good two-shot.

    Now it uses `pairing.select_for_room`, which prefers complementary viewpoints
    (30-120 degrees apart), suppresses near-duplicates and honours the owner's
    per-room caps. Measured against the owner's labeled pairs: 77% of rooms receive
    at least one pair they chose, 67% have it ranked first.

    Falls back to the old adjacent chunking only when no pair evidence is
    available (historical jobs), so old analyses still render.
    """
    if not ordered_photos:
        return []
    if len(ordered_photos) == 1:
        return [ordered_photos]

    del scene_type
    groups: list[list[JobPhoto]] = []

    # Explicit editorial roles are never paired away.
    pinned: list[JobPhoto] = []
    candidates: list[JobPhoto] = []
    for photo in ordered_photos:
        role = str((photo.manual_metadata or {}).get("editorial_role", "auto")).lower()
        (pinned if role in {"opening", "hero", "closing"} else candidates).append(photo)

    evidence: list[dict] = []
    # Historical jobs (and unit fixtures) carry no ids; those take the fallback path.
    photo_by_id = {int(photo.id): photo for photo in ordered_photos if getattr(photo, "id", None) is not None}
    if relation_by_pair and photo_by_id:
        candidate_ids = {int(photo.id) for photo in candidates if getattr(photo, "id", None) is not None}
        for (left, right), relation in relation_by_pair.items():
            if left not in candidate_ids or right not in candidate_ids:
                continue
            metrics = relation.debug_metrics or {}
            if "pair_score" not in metrics:
                continue        # historical V1 relation; no pairing evidence
            evidence.append({
                "photo_a_id": left,
                "photo_b_id": right,
                "depth_ok_min": float(min(metrics.get("depth_ok_forward", 0.0),
                                          metrics.get("depth_ok_backward", 0.0))),
                "rot_deg": float(metrics.get("rotation_degrees", 0.0)),
                "conf_pair": float(metrics.get("conf_pair", 0.0)),
                "bl_over_depth": float(metrics.get("bl_over_depth", 0.0)),
            })

    if evidence:
        chosen, unpaired = pairing.select_for_room(evidence, room_type)
        for pair in chosen:
            left, right = photo_by_id.get(pair.photo_a), photo_by_id.get(pair.photo_b)
            if left is not None and right is not None:
                groups.append([left, right])
        for photo_id in unpaired:
            photo = photo_by_id.get(photo_id)
            if photo is not None:
                groups.append([photo])
        groups.extend([photo] for photo in pinned)
    else:
        # No pair evidence (historical job or fixture): reproduce the previous
        # behaviour exactly -- adjacent chunking in component order, with an
        # explicit editorial role acting as a separator so a hero is never
        # merged with its neighbours.
        current: list[JobPhoto] = []
        for photo in ordered_photos:
            role = str((photo.manual_metadata or {}).get("editorial_role", "auto")).lower()
            if role in {"opening", "hero", "closing"}:
                if current:
                    groups.append(current)
                    current = []
                groups.append([photo])
                continue
            if current and len(current) >= 2:
                groups.append(current)
                current = []
            current.append(photo)
        if current:
            groups.append(current)
        return groups

    if photo_by_id:
        # Preserve component ordering: sort groups by their earliest member.
        order = {id(photo): index for index, photo in enumerate(ordered_photos)}
        groups.sort(key=lambda group: min(order.get(id(photo), 0) for photo in group))
    return groups


def _majority_room_label(photos: list[JobPhoto]) -> str:
    labels = [
        (photo.room_override or photo.room_label or "scene").strip()
        for photo in photos
        if (photo.room_override or photo.room_label or "").strip()
    ]
    if not labels:
        return "scene"
    return Counter(labels).most_common(1)[0][0]


def _confidence_tier(geometry_confidence: float, motion_affordance: str, photo_count: int) -> str:
    if motion_affordance == "multi_view" and geometry_confidence >= 0.72 and photo_count >= 3:
        return "high"
    if motion_affordance in {"parallax", "reveal"} and geometry_confidence >= 0.56:
        return "medium"
    return "low"


def _cluster_overlap_score(photos: list[JobPhoto], relations_by_pair: dict[tuple[int, int], PhotoRelationResult]) -> float:
    scores: list[float] = []
    for index in range(len(photos) - 1):
        pair = (min(int(photos[index].id), int(photos[index + 1].id)), max(int(photos[index].id), int(photos[index + 1].id)))
        relation = relations_by_pair.get(pair)
        if relation is not None:
            scores.append(float(relation.overlap_score))
    return float(sum(scores) / len(scores)) if scores else 0.0
