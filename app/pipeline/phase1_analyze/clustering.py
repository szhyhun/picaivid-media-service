"""Scene-component clustering using VGGT geometry outputs."""
from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Optional

from PIL import Image, ImageOps
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
    active_photos = sorted([photo for photo in photos if not photo.exclude], key=lambda photo: photo.position or 0)
    logger.info("Clustering %s photos for job %s with VGGT scene graph", len(active_photos), job.id)
    if not active_photos:
        return []

    images: list[Image.Image] = []
    photo_ids: list[int] = []
    room_labels: list[str] = []
    positions: list[int] = []
    photo_map: dict[int, JobPhoto] = {}
    for photo in active_photos:
        image = preloaded_images.get(photo.id) if preloaded_images is not None else None
        if image is None:
            image = s3_client.download_image(photo.s3_uri)
            if preloaded_images is not None:
                preloaded_images[photo.id] = image
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass
        images.append(image)
        photo_ids.append(int(photo.id))
        room_labels.append(photo.room_override or photo.room_label or "")
        positions.append(int(photo.position or 0))
        photo_map[int(photo.id)] = photo
        photo.room_cluster_id = None
        photo.cluster_order = None

    geometries, relations, components = run_vggt_scene_pipeline(
        images=images,
        photo_ids=photo_ids,
        room_labels=room_labels,
        positions=positions,
        job_id=int(job.id),
        s3_client=s3_client,
    )

    _replace_scene_state(
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
    )
    db.commit()
    logger.info("Created %s room clusters from %s VGGT scene components", len(clusters), len(components))
    return clusters


def _replace_scene_state(
    db: Session,
    job_id: int,
    geometries: list[PhotoGeometryResult],
    relations: list[PhotoRelationResult],
    components: list[SceneComponentResult],
    photo_map: dict[int, JobPhoto],
) -> None:
    db.query(SceneComponentMembership).filter(SceneComponentMembership.job_id == job_id).delete(synchronize_session=False)
    db.query(PhotoSceneGeometry).filter(PhotoSceneGeometry.job_id == job_id).delete(synchronize_session=False)
    db.query(PhotoRelation).filter(PhotoRelation.job_id == job_id).delete(synchronize_session=False)
    db.query(SceneComponent).filter(SceneComponent.job_id == job_id).delete(synchronize_session=False)
    db.query(RoomCluster).filter(RoomCluster.job_id == job_id).delete(synchronize_session=False)
    db.flush()

    component_id_by_key: dict[str, int] = {}
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
            track_coverage=component.track_coverage,
            avg_reprojection_error=component.avg_reprojection_error,
            hero_photo_id=component.hero_photo_id,
            depth_range=component.depth_range,
            motion_affordance=component.motion_affordance,
            debug_metrics=component.debug_metrics,
        )
        db.add(row)
        db.flush()
        component_id_by_key[component.component_key] = int(row.id)
        for index, photo_id in enumerate(component.ordered_photo_ids):
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
        component_id = next(
            (
                component_id_by_key[component.component_key]
                for component in components
                if geometry.photo_id in component.photo_ids
            ),
            None,
        )
        db.add(
            PhotoSceneGeometry(
                job_id=job_id,
                photo_id=int(geometry.photo_id),
                scene_component_id=component_id,
                pose_confidence=geometry.pose_confidence,
                depth_confidence=geometry.depth_confidence,
                point_confidence=geometry.point_confidence,
                visibility_score=geometry.visibility_score,
                reprojection_error=geometry.reprojection_error,
                camera_extrinsic=geometry.camera_extrinsic,
                camera_intrinsic=geometry.camera_intrinsic,
                camera_center=geometry.camera_center,
                view_direction=geometry.view_direction,
                depth_artifact_uri=geometry.depth_artifact_uri,
                point_map_artifact_uri=geometry.point_map_artifact_uri,
                local_metrics={**geometry.local_metrics, "photo_role": role_by_photo.get(int(geometry.photo_id), "support")},
            )
        )

    for relation in relations:
        component_id = None
        for component in components:
            if relation.photo_a_id in component.photo_ids and relation.photo_b_id in component.photo_ids:
                component_id = component_id_by_key[component.component_key]
                break
        db.add(
            PhotoRelation(
                job_id=job_id,
                photo_a_id=int(relation.photo_a_id),
                photo_b_id=int(relation.photo_b_id),
                scene_component_id=component_id,
                overlap_score=relation.overlap_score,
                track_support=relation.track_support,
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


def _materialize_room_clusters(
    db: Session,
    job: Job,
    components: list[SceneComponentResult],
    relations: list[PhotoRelationResult],
    photo_map: dict[int, JobPhoto],
) -> list[RoomCluster]:
    scene_rows = db.query(SceneComponent).filter(SceneComponent.job_id == int(job.id)).all()
    scene_row_by_key = {row.component_key: row for row in scene_rows}
    relation_by_pair = {
        (min(int(relation.photo_a_id), int(relation.photo_b_id)), max(int(relation.photo_a_id), int(relation.photo_b_id))): relation
        for relation in relations
    }
    clusters: list[RoomCluster] = []
    sequence_order = 0
    for component in components:
        scene_row = scene_row_by_key[component.component_key]
        ordered_photos = [photo_map[int(photo_id)] for photo_id in component.ordered_photo_ids if int(photo_id) in photo_map]
        for cluster_photos in _derive_render_groups(ordered_photos, component.scene_type):
            if not cluster_photos:
                continue
            room_type = _majority_room_label(cluster_photos)
            geometry_confidence = component.geometry_confidence
            confidence_tier = _confidence_tier(geometry_confidence, component.motion_affordance, len(cluster_photos))
            overlap_score = _cluster_overlap_score(cluster_photos, relation_by_pair)
            depth_variance = _cluster_depth_variance(cluster_photos)
            cluster = RoomCluster(
                job_id=int(job.id),
                scene_component_id=int(scene_row.id),
                room_type=room_type,
                confidence_tier=confidence_tier,
                sfm_eligible=component.motion_affordance in {"parallax", "multi_view"} and len(cluster_photos) >= 2,
                image_count=len(cluster_photos),
                overlap_score=overlap_score,
                depth_variance=depth_variance,
                geometry_confidence=geometry_confidence,
                sequence_order=sequence_order,
                recommended_motion=component.motion_affordance,
            )
            db.add(cluster)
            db.flush()
            for photo_order, photo in enumerate(cluster_photos):
                photo.room_cluster_id = int(cluster.id)
                photo.cluster_order = photo_order
            clusters.append(cluster)
            sequence_order += 1
    return clusters


def _derive_render_groups(ordered_photos: list[JobPhoto], scene_type: str) -> list[list[JobPhoto]]:
    if len(ordered_photos) <= 2:
        return [ordered_photos]
    if scene_type in {"exterior", "drone"}:
        return [ordered_photos]
    groups: list[list[JobPhoto]] = []
    current_group: list[JobPhoto] = []
    current_label: str | None = None
    for photo in ordered_photos:
        label = (photo.room_override or photo.room_label or "").strip().lower()
        if current_group and current_label and label and label != current_label and len(current_group) >= 2:
            groups.append(current_group)
            current_group = []
        current_group.append(photo)
        current_label = label or current_label
    if current_group:
        groups.append(current_group)
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


def _cluster_depth_variance(photos: list[JobPhoto]) -> float:
    values = [float(photo.depth_variance) for photo in photos if photo.depth_variance is not None]
    return float(sum(values) / len(values)) if values else 0.0
