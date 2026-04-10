"""MASt3R graph pipeline for phase-1 clustering and debug."""
from __future__ import annotations

import logging
import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

import numpy as np
from PIL import Image

from app.core.config import settings
from app.db.models import PhotoPoseAlignment
from app.pipeline.phase1_analyze.candidate_retrieval import normalize_room_label, rooms_soft_compatible
from app.pipeline.phase1_analyze.sequence_builder import build_transition_sequences
from app.pipeline.phase1_analyze.matcher_loaders import _ensure_local_file, _ensure_local_repo

logger = logging.getLogger(__name__)

MAST3R_ENGINE_NAME = "mast3r_graph"


@dataclass
class _MASt3RImports:
    AsymmetricMASt3R: Any
    fast_reciprocal_NNs: Any
    inference: Any
    load_images: Any
    Retriever: Any
    make_pairs: Any
    sparse_global_alignment: Any


@dataclass
class _MASt3RAssets:
    repo_dir: str
    model_checkpoint: str
    retrieval_checkpoint: str
    retrieval_codebook: str


class _UnionFind:
    def __init__(self, items: Iterable[int]) -> None:
        self.parent = {int(item): int(item) for item in items}

    def find(self, item: int) -> int:
        item = int(item)
        parent = self.parent.setdefault(item, item)
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _order_proximity(position_a: int, position_b: int) -> float:
    gap = abs(int(position_a) - int(position_b))
    return float(max(0.0, 1.0 - gap / 12.0))


def _certification_status(pair_rank: float) -> str:
    if pair_rank >= 0.62:
        return "strong"
    if pair_rank >= 0.40:
        return "usable"
    return "reject"


@lru_cache(maxsize=1)
def _require_cuda_device() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "MASt3R main pipeline requires a CUDA GPU worker. "
            "Start media-service with WORKER_TYPE=gpu on a CUDA host."
        )
    return "cuda"


def _resolve_assets() -> _MASt3RAssets:
    repo_dir = _ensure_local_repo(
        str(settings.MAST3R_REPO_DIR or os.path.join(os.path.dirname(settings.MODEL_CACHE_DIR), "third_party", "mast3r")),
        "MAST3R_REPO_ARCHIVE_S3_URI",
    )
    model_checkpoint = _ensure_local_file(
        str(settings.MAST3R_MODEL_CHECKPOINT or os.path.join(repo_dir, "checkpoints", "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")),
        "MAST3R_MODEL_CHECKPOINT_S3_URI",
    )
    retrieval_checkpoint = _ensure_local_file(
        str(settings.MAST3R_RETRIEVAL_CHECKPOINT or os.path.join(repo_dir, "checkpoints", "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth")),
        "MAST3R_RETRIEVAL_CHECKPOINT_S3_URI",
    )
    retrieval_codebook = _ensure_local_file(
        str(settings.MAST3R_RETRIEVAL_CODEBOOK or os.path.join(repo_dir, "checkpoints", "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl")),
        "MAST3R_RETRIEVAL_CODEBOOK_S3_URI",
    )
    if os.path.dirname(retrieval_checkpoint) != os.path.dirname(retrieval_codebook):
        raise RuntimeError(
            "MASt3R retrieval checkpoint and codebook must live in the same directory. "
            f"checkpoint={retrieval_checkpoint} codebook={retrieval_codebook}"
        )
    return _MASt3RAssets(
        repo_dir=repo_dir,
        model_checkpoint=model_checkpoint,
        retrieval_checkpoint=retrieval_checkpoint,
        retrieval_codebook=retrieval_codebook,
    )


def _file_signature(path: str) -> tuple[str, int, int]:
    stat = os.stat(path)
    return (os.path.abspath(path), int(stat.st_mtime_ns), int(stat.st_size))


@lru_cache(maxsize=4)
def _load_imports_cached(repo_dir: str) -> _MASt3RImports:
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    dust3r_dir = os.path.join(repo_dir, "dust3r")
    if dust3r_dir not in sys.path:
        sys.path.insert(0, dust3r_dir)
    try:
        from mast3r.model import AsymmetricMASt3R
        from mast3r.fast_nn import fast_reciprocal_NNs
        import mast3r.utils.path_to_dust3r  # noqa: F401
        from dust3r.inference import inference
        from dust3r.utils.image import load_images
        from mast3r.retrieval.processor import Retriever
        from mast3r.image_pairs import make_pairs
        from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    except Exception as err:  # pragma: no cover - depends on external environment
        raise RuntimeError(
            "Failed to import MASt3R runtime dependencies. Install MASt3R, DUSt3R, asmk, faiss, and required extras first. "
            f"Original error: {err}"
        ) from err
    return _MASt3RImports(
        AsymmetricMASt3R=AsymmetricMASt3R,
        fast_reciprocal_NNs=fast_reciprocal_NNs,
        inference=inference,
        load_images=load_images,
        Retriever=Retriever,
        make_pairs=make_pairs,
        sparse_global_alignment=sparse_global_alignment,
    )


def _load_imports() -> _MASt3RImports:
    assets = _resolve_assets()
    return _load_imports_cached(os.path.abspath(assets.repo_dir))


@lru_cache(maxsize=4)
def _load_model_cached(repo_dir: str, checkpoint_path: str, checkpoint_signature: tuple[str, int, int], device: str) -> Any:
    imports = _load_imports_cached(os.path.abspath(repo_dir))
    started_at = time.perf_counter()
    model = imports.AsymmetricMASt3R.from_pretrained(checkpoint_path).to(device)
    model.eval()
    logger.info(
        "Loaded MASt3R model checkpoint=%s device=%s elapsed_ms=%.1f",
        checkpoint_path,
        device,
        (time.perf_counter() - started_at) * 1000.0,
    )
    return model


def _load_model() -> Any:
    assets = _resolve_assets()
    device = _require_cuda_device()
    return _load_model_cached(
        assets.repo_dir,
        assets.model_checkpoint,
        _file_signature(assets.model_checkpoint),
        device,
    )


@lru_cache(maxsize=4)
def _load_retriever_cached(
    repo_dir: str,
    model_checkpoint_path: str,
    retrieval_checkpoint: str,
    retrieval_signature: tuple[str, int, int],
    retrieval_codebook_signature: tuple[str, int, int],
    model_checkpoint_signature: tuple[str, int, int],
    device: str,
) -> Any:
    imports = _load_imports_cached(os.path.abspath(repo_dir))
    started_at = time.perf_counter()
    retriever = imports.Retriever(
        retrieval_checkpoint,
        backbone=_load_model_cached(repo_dir, model_checkpoint_path, model_checkpoint_signature, device),
        device=device,
    )
    logger.info(
        "Loaded MASt3R retriever checkpoint=%s elapsed_ms=%.1f",
        retrieval_checkpoint,
        (time.perf_counter() - started_at) * 1000.0,
    )
    return retriever


def _load_retriever() -> Any:
    assets = _resolve_assets()
    device = _require_cuda_device()
    return _load_retriever_cached(
        assets.repo_dir,
        assets.model_checkpoint,
        assets.retrieval_checkpoint,
        _file_signature(assets.retrieval_checkpoint),
        _file_signature(assets.retrieval_codebook),
        _file_signature(assets.model_checkpoint),
        device,
    )


def warmup_mast3r() -> None:
    _load_model()
    _load_retriever()


def reset_mast3r_runtime_caches() -> None:
    _load_imports_cached.cache_clear()
    _load_model_cached.cache_clear()
    _load_retriever_cached.cache_clear()


def _save_images_to_temp(image_list: list[object], photo_ids: list[int]) -> tuple[str, list[str]]:
    tmpdir = tempfile.mkdtemp(prefix="mast3r_images_")
    filelist: list[str] = []
    for idx, (image, photo_id) in enumerate(zip(image_list, photo_ids)):
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            raise RuntimeError(f"Unsupported image type for MASt3R: {type(image)!r}")
        path = os.path.join(tmpdir, f"{idx:03d}_{int(photo_id)}.png")
        pil_image.convert("RGB").save(path)
        filelist.append(path)
    return tmpdir, filelist


def _overlap_ratio_from_matches(matches0: np.ndarray, matches1: np.ndarray, width0: int, height0: int, width1: int, height1: int) -> float:
    if len(matches0) < 3 or len(matches1) < 3:
        return 0.0

    def _hull_area(points: np.ndarray) -> float:
        try:
            import cv2
            hull = cv2.convexHull(points.astype(np.float32))
            return float(cv2.contourArea(hull))
        except Exception:
            xs = points[:, 0]
            ys = points[:, 1]
            return float(max(0.0, (xs.max() - xs.min()) * (ys.max() - ys.min())))

    area0 = _hull_area(matches0)
    area1 = _hull_area(matches1)
    norm0 = area0 / max(float(width0 * height0), 1.0)
    norm1 = area1 / max(float(width1 * height1), 1.0)
    return float(max(0.0, min(norm0, norm1)))


def _pairwise_mast3r_metrics(
    path_a: str,
    path_b: str,
    device: str,
    image_a_record: dict[str, Any] | None = None,
    image_b_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    imports = _load_imports()
    model = _load_model()
    if image_a_record is not None and image_b_record is not None:
        images = [dict(image_a_record), dict(image_b_record)]
    else:
        images = imports.load_images([path_a, path_b], size=int(settings.MAST3R_IMAGE_SIZE), verbose=False)
    started_at = time.perf_counter()
    output = imports.inference([tuple(images)], model, device, batch_size=1, verbose=False)
    inference_seconds = time.perf_counter() - started_at

    view1 = output["view1"]
    view2 = output["view2"]
    pred1 = output["pred1"]
    pred2 = output["pred2"]

    desc1 = pred1["desc"].squeeze(0).detach()
    desc2 = pred2["desc"].squeeze(0).detach()
    matches_im0, matches_im1 = imports.fast_reciprocal_NNs(
        desc1,
        desc2,
        subsample_or_initxy1=8,
        device=device,
        dist="dot",
        block_size=2 ** 13,
    )

    matches_im0 = np.asarray(matches_im0, dtype=np.int32)
    matches_im1 = np.asarray(matches_im1, dtype=np.int32)
    H0, W0 = [int(v) for v in view1["true_shape"][0]]
    H1, W1 = [int(v) for v in view2["true_shape"][0]]
    valid_matches = (
        (matches_im0[:, 0] >= 3)
        & (matches_im0[:, 0] < W0 - 3)
        & (matches_im0[:, 1] >= 3)
        & (matches_im0[:, 1] < H0 - 3)
        & (matches_im1[:, 0] >= 3)
        & (matches_im1[:, 0] < W1 - 3)
        & (matches_im1[:, 1] >= 3)
        & (matches_im1[:, 1] < H1 - 3)
    ) if len(matches_im0) else np.zeros((0,), dtype=bool)
    matches_im0 = matches_im0[valid_matches]
    matches_im1 = matches_im1[valid_matches]

    raw_match_count = int(len(matches_im0))
    overlap_ratio = _overlap_ratio_from_matches(matches_im0, matches_im1, W0, H0, W1, H1)
    if raw_match_count == 0:
        return {
            "raw_matches": [],
            "reciprocal_match_count": 0,
            "pointmap_consistency": 0.0,
            "alignment_residual": None,
            "reprojection_error": None,
            "overlap_ratio": overlap_ratio,
            "combined_geometry_score": 0.0,
            "direction_dx": 0.0,
            "direction_dy": 0.0,
            "timing": {"time_mast3r_inference_s": float(inference_seconds)},
        }

    pts3d_1 = pred1["pts3d"].squeeze(0).detach().cpu().numpy()
    pts3d_2 = pred2["pts3d_in_other_view"].squeeze(0).detach().cpu().numpy()
    pts1 = pts3d_1[matches_im0[:, 1], matches_im0[:, 0]]
    pts2 = pts3d_2[matches_im1[:, 1], matches_im1[:, 0]]
    finite_mask = np.isfinite(pts1).all(axis=1) & np.isfinite(pts2).all(axis=1)
    matches_im0 = matches_im0[finite_mask]
    matches_im1 = matches_im1[finite_mask]
    pts1 = pts1[finite_mask]
    pts2 = pts2[finite_mask]

    if len(matches_im0) == 0:
        return {
            "raw_matches": [],
            "reciprocal_match_count": 0,
            "pointmap_consistency": 0.0,
            "alignment_residual": None,
            "reprojection_error": None,
            "overlap_ratio": overlap_ratio,
            "combined_geometry_score": 0.0,
            "direction_dx": 0.0,
            "direction_dy": 0.0,
            "timing": {"time_mast3r_inference_s": float(inference_seconds)},
        }

    diffs = np.linalg.norm(pts1 - pts2, axis=1)
    median_residual = float(np.median(diffs)) if len(diffs) else None
    pointmap_consistency = float(math.exp(-median_residual)) if median_residual is not None else 0.0
    mean_dx = float(np.mean(matches_im1[:, 0] - matches_im0[:, 0])) / max(float(W0), 1.0)
    mean_dy = float(np.mean(matches_im1[:, 1] - matches_im0[:, 1])) / max(float(H0), 1.0)
    geometry_score = float(
        0.55 * np.clip(raw_match_count / 256.0, 0.0, 1.0)
        + 0.45 * np.clip(pointmap_consistency, 0.0, 1.0)
    )
    raw_matches_payload = [
        {
            "x0": float(x0),
            "y0": float(y0),
            "x1": float(x1),
            "y1": float(y1),
            "dx": float(x1 - x0),
            "dy": float(y1 - y0),
        }
        for (x0, y0), (x1, y1) in zip(matches_im0.tolist(), matches_im1.tolist())
    ]
    return {
        "raw_matches": raw_matches_payload,
        "reciprocal_match_count": int(len(matches_im0)),
        "pointmap_consistency": pointmap_consistency,
        "alignment_residual": median_residual,
        "reprojection_error": median_residual,
        "overlap_ratio": overlap_ratio,
        "combined_geometry_score": geometry_score,
        "direction_dx": mean_dx,
        "direction_dy": mean_dy,
        "timing": {"time_mast3r_inference_s": float(inference_seconds)},
    }


def _component_map(num_items: int, edges: list[tuple[int, int]]) -> dict[int, int]:
    uf = _UnionFind(range(num_items))
    for left, right in edges:
        uf.union(left, right)
    roots = {idx: uf.find(idx) for idx in range(num_items)}
    ordered_roots = {root: component_id for component_id, root in enumerate(sorted(set(roots.values())))}
    return {idx: ordered_roots[root] for idx, root in roots.items()}


def _compute_parallax_score(camera_center_a: np.ndarray | None, camera_center_b: np.ndarray | None, sparse_pts: np.ndarray | None) -> float:
    if camera_center_a is None or camera_center_b is None:
        return 0.0
    baseline = float(np.linalg.norm(camera_center_a - camera_center_b))
    if sparse_pts is None or len(sparse_pts) == 0:
        return float(np.clip(baseline, 0.0, 1.0))
    scene_scale = float(np.median(np.linalg.norm(sparse_pts - sparse_pts.mean(axis=0), axis=1))) if len(sparse_pts) else 0.0
    if scene_scale <= 1e-6:
        return float(np.clip(baseline, 0.0, 1.0))
    return float(np.clip(baseline / (scene_scale * 2.5), 0.0, 1.0))


def _graph_edge_score(*, retrieval_score: float, reciprocal_match_count: int, pointmap_consistency: float, parallax_score: float, order_proximity: float, room_bonus: float) -> float:
    match_score = float(np.clip(reciprocal_match_count / 256.0, 0.0, 1.0))
    return float(
        0.26 * float(np.clip(retrieval_score, 0.0, 1.0))
        + 0.26 * match_score
        + 0.22 * float(np.clip(pointmap_consistency, 0.0, 1.0))
        + 0.18 * float(np.clip(parallax_score, 0.0, 1.0))
        + 0.05 * float(np.clip(order_proximity, 0.0, 1.0))
        + 0.03 * room_bonus
    )


def _room_bonus(room_a: str, room_b: str) -> float:
    normalized_a = normalize_room_label(room_a)
    normalized_b = normalize_room_label(room_b)
    if normalized_a and normalized_a == normalized_b:
        return 1.0
    if rooms_soft_compatible(room_a, room_b):
        return 0.5
    return 0.0


def _edge_status(record: dict[str, Any]) -> tuple[str, str | None]:
    if float(record.get("retrieval_score") or 0.0) < float(settings.MAST3R_MIN_RETRIEVAL_SCORE):
        return "reject", "low_retrieval_score"
    if int(record.get("reciprocal_match_count") or 0) < int(settings.MAST3R_MIN_RECIPROCAL_MATCHES):
        return "reject", "low_reciprocal_matches"
    if float(record.get("pointmap_consistency") or 0.0) < float(settings.MAST3R_MIN_POINTMAP_CONSISTENCY):
        return "reject", "low_pointmap_consistency"
    if float(record.get("parallax_score") or 0.0) < float(settings.MAST3R_MIN_PARALLAX_SCORE):
        return "reject", "low_parallax"
    score = float(record.get("graph_edge_score") or 0.0)
    status = _certification_status(score)
    if status == "reject":
        return "reject", "low_pair_rank"
    return status, None


def _choose_final_clusters(photo_ids: list[int], pair_records: list[dict[str, Any]], positions_by_photo: dict[int, int]) -> list[list[int]]:
    eligible = [
        record for record in pair_records
        if str(record.get("certification_status")) in {"strong", "usable"}
        and float(record.get("graph_edge_score") or 0.0) >= float(settings.MAST3R_MIN_GRAPH_EDGE_SCORE)
    ]
    eligible.sort(
        key=lambda row: (
            float(row.get("graph_edge_score") or 0.0),
            float(row.get("parallax_score") or 0.0),
            int(row.get("reciprocal_match_count") or 0),
            float(row.get("retrieval_score") or 0.0),
            float(row.get("order_proximity") or 0.0),
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
        clusters.append(sorted([left, right], key=lambda photo_id: positions_by_photo.get(photo_id, 0)))
        assigned.add(left)
        assigned.add(right)
        record["is_connected"] = 1
    for record in pair_records:
        if "is_connected" not in record:
            record["is_connected"] = 0
    for photo_id in photo_ids:
        if int(photo_id) not in assigned:
            clusters.append([int(photo_id)])
    clusters.sort(key=lambda cluster: positions_by_photo.get(cluster[0], 0))
    return clusters


def _persist_pose_rows(db_session, job_id: int, rows: list[dict[str, Any]]) -> None:
    if db_session is None or job_id is None:
        return
    db_session.query(PhotoPoseAlignment).filter(PhotoPoseAlignment.job_id == int(job_id)).delete(synchronize_session=False)
    if rows:
        db_session.bulk_insert_mappings(PhotoPoseAlignment, rows)
    db_session.flush()


def run_mast3r_phase1(
    images: list[object],
    photo_ids: list[int],
    positions: list[int],
    room_labels: list[str],
    db_session=None,
    job_id: int | None = None,
) -> tuple[list[list[int]], list[dict[str, Any]], list[dict[str, Any]], np.ndarray, list[dict[str, Any]]]:
    imports = _load_imports()
    device = _require_cuda_device()
    model = _load_model()
    retriever = _load_retriever()
    temp_dir, filelist = _save_images_to_temp(images, photo_ids)
    try:
        retrieval_started_at = time.perf_counter()
        sim_matrix = np.asarray(retriever(filelist), dtype=np.float32)
        retrieval_seconds = time.perf_counter() - retrieval_started_at
        logger.info(
            "MASt3R retrieval complete: photos=%s elapsed_ms=%.1f",
            len(photo_ids),
            retrieval_seconds * 1000.0,
        )

        imgs_loaded = imports.load_images(filelist, size=int(settings.MAST3R_IMAGE_SIZE), verbose=False)
        loaded_by_index = {int(idx): imgs_loaded[idx] for idx in range(len(imgs_loaded))}
        scene_graph = f"retrieval-{int(settings.MAST3R_SCENE_GRAPH_ANCHORS)}-{int(settings.MAST3R_SCENE_GRAPH_K)}"
        graph_pairs = imports.make_pairs(imgs_loaded, scene_graph=scene_graph, prefilter=None, symmetrize=True, sim_mat=sim_matrix)
        unique_edges: set[tuple[int, int]] = set()
        for view_a, view_b in graph_pairs:
            left_idx = int(view_a["idx"])
            right_idx = int(view_b["idx"])
            if left_idx == right_idx:
                continue
            unique_edges.add((min(left_idx, right_idx), max(left_idx, right_idx)))
        component_lookup = _component_map(len(photo_ids), sorted(unique_edges))

        pose_rows: list[dict[str, Any]] = []
        component_pose_lookup: dict[tuple[int, int], dict[str, Any]] = {}
        component_sparse_pts: dict[int, np.ndarray] = {}
        for component_id in sorted(set(component_lookup.values())):
            component_indices = [idx for idx, cid in component_lookup.items() if cid == component_id]
            if len(component_indices) <= 1:
                single_idx = component_indices[0]
                pose_rows.append(
                    {
                        "job_id": int(job_id) if job_id is not None else None,
                        "photo_id": int(photo_ids[single_idx]),
                        "graph_component_id": int(component_id),
                        "pose_confidence": 0.0,
                        "reprojection_error": None,
                        "focal_length": None,
                        "principal_point": None,
                        "camera_center": None,
                        "camera_pose": None,
                    }
                )
                continue
            component_paths = [filelist[idx] for idx in component_indices]
            component_imgs = [dict(loaded_by_index[idx]) for idx in component_indices]
            component_sim = sim_matrix[np.ix_(component_indices, component_indices)]
            component_pairs = imports.make_pairs(
                component_imgs,
                scene_graph=scene_graph,
                prefilter=None,
                symmetrize=True,
                sim_mat=component_sim,
            )
            cache_dir = tempfile.mkdtemp(prefix=f"mast3r_component_{component_id}_")
            try:
                scene = imports.sparse_global_alignment(
                    component_paths,
                    component_pairs,
                    cache_dir,
                    model,
                    lr1=float(settings.MAST3R_LR1),
                    niter1=int(settings.MAST3R_NITER1),
                    lr2=float(settings.MAST3R_LR2),
                    niter2=int(settings.MAST3R_NITER2),
                    device=device,
                    matching_conf_thr=float(settings.MAST3R_MATCHING_CONFIDENCE_THRESHOLD),
                    shared_intrinsics=bool(settings.MAST3R_SHARED_INTRINSICS),
                )
                poses = scene.get_im_poses().detach().cpu().numpy()
                focals = scene.get_focals().detach().cpu().numpy()
                pps = scene.get_principal_points().detach().cpu().numpy()
                sparse_pts = scene.get_sparse_pts3d()
                component_sparse_pts[int(component_id)] = np.concatenate(
                    [pts.detach().cpu().numpy() for pts in sparse_pts if hasattr(pts, "detach")],
                    axis=0,
                ) if sparse_pts else np.empty((0, 3), dtype=np.float32)
                for local_idx, global_idx in enumerate(component_indices):
                    pose = poses[local_idx]
                    component_pose_lookup[(int(component_id), int(global_idx))] = {
                        "camera_center": np.asarray(pose[:3, 3], dtype=np.float32),
                        "camera_pose": pose,
                        "focal_length": float(focals[local_idx]) if len(focals) > local_idx else None,
                        "principal_point": [float(v) for v in pps[local_idx].tolist()] if len(pps) > local_idx else None,
                    }
                    pose_rows.append(
                        {
                            "job_id": int(job_id) if job_id is not None else None,
                            "photo_id": int(photo_ids[global_idx]),
                            "graph_component_id": int(component_id),
                            "pose_confidence": 1.0,
                            "reprojection_error": None,
                            "focal_length": float(focals[local_idx]) if len(focals) > local_idx else None,
                            "principal_point": [float(v) for v in pps[local_idx].tolist()] if len(pps) > local_idx else None,
                            "camera_center": [float(v) for v in pose[:3, 3].tolist()],
                            "camera_pose": pose.tolist(),
                        }
                    )
            finally:
                shutil.rmtree(cache_dir, ignore_errors=True)

        pair_records: list[dict[str, Any]] = []
        positions_by_photo = {int(photo_ids[idx]): int(positions[idx]) for idx in range(len(photo_ids))}
        room_by_photo = {int(photo_ids[idx]): room_labels[idx] for idx in range(len(photo_ids))}
        for left_idx, right_idx in sorted(unique_edges):
            photo_a_id = int(photo_ids[left_idx])
            photo_b_id = int(photo_ids[right_idx])
            metrics = _pairwise_mast3r_metrics(
                filelist[left_idx],
                filelist[right_idx],
                device=device,
                image_a_record=loaded_by_index.get(int(left_idx)),
                image_b_record=loaded_by_index.get(int(right_idx)),
            )
            component_id = int(component_lookup[left_idx])
            pose_a = component_pose_lookup.get((component_id, int(left_idx)))
            pose_b = component_pose_lookup.get((component_id, int(right_idx)))
            sparse_pts = component_sparse_pts.get(component_id)
            parallax_score = _compute_parallax_score(
                None if pose_a is None else np.asarray(pose_a.get("camera_center")),
                None if pose_b is None else np.asarray(pose_b.get("camera_center")),
                sparse_pts,
            )
            retrieval_score = float(sim_matrix[left_idx, right_idx])
            order_proximity = _order_proximity(int(positions[left_idx]), int(positions[right_idx]))
            room_bonus = _room_bonus(room_labels[left_idx], room_labels[right_idx])
            graph_edge_score = _graph_edge_score(
                retrieval_score=retrieval_score,
                reciprocal_match_count=int(metrics["reciprocal_match_count"]),
                pointmap_consistency=float(metrics["pointmap_consistency"]),
                parallax_score=parallax_score,
                order_proximity=order_proximity,
                room_bonus=room_bonus,
            )
            record = {
                "photo_a_id": min(photo_a_id, photo_b_id),
                "photo_b_id": max(photo_a_id, photo_b_id),
                "pair_source": "mast3r_retrieval_graph",
                "match_engine": MAST3R_ENGINE_NAME,
                "retrieval_score": retrieval_score,
                "reciprocal_match_count": int(metrics["reciprocal_match_count"]),
                "pointmap_consistency": float(metrics["pointmap_consistency"]),
                "alignment_residual": metrics["alignment_residual"],
                "reprojection_error": metrics["reprojection_error"],
                "parallax_score": parallax_score,
                "graph_component_id": component_id,
                "graph_edge_score": graph_edge_score,
                "overlap_ratio": float(metrics["overlap_ratio"]),
                "combined_geometry_score": float(metrics["combined_geometry_score"]),
                "order_proximity": order_proximity,
                "pair_rank": graph_edge_score,
                "direction_dx": float(metrics["direction_dx"]),
                "direction_dy": float(metrics["direction_dy"]),
                "raw_matches_payload": metrics["raw_matches"],
                "inlier_matches_payload": metrics["raw_matches"],
                "timing": metrics["timing"],
            }
            status, rejection_reason = _edge_status(record)
            record["certification_status"] = status
            record["rejection_reason"] = rejection_reason
            record["is_connected"] = 0
            pair_records.append(record)

        final_clusters = _choose_final_clusters(
            photo_ids=[int(photo_id) for photo_id in photo_ids],
            pair_records=pair_records,
            positions_by_photo=positions_by_photo,
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
        _persist_pose_rows(db_session, job_id, pose_rows)
        return final_clusters, pair_records, pose_rows, sim_matrix, transition_sequences
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def debug_pair_mast3r(left_image: object, right_image: object) -> tuple[int, int, float, tuple[float, float], dict[str, Any]]:
    device = _require_cuda_device()
    temp_dir, filelist = _save_images_to_temp([left_image, right_image], [0, 1])
    try:
        metrics = _pairwise_mast3r_metrics(filelist[0], filelist[1], device=device)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    reciprocal = int(metrics["reciprocal_match_count"])
    score = float(
        0.45 * np.clip(reciprocal / 256.0, 0.0, 1.0)
        + 0.35 * float(metrics["pointmap_consistency"])
        + 0.20 * float(metrics["overlap_ratio"])
    )
    diagnostics = {
        "matcher": MAST3R_ENGINE_NAME,
        "checkpoint": _resolve_assets().model_checkpoint,
        "geometry_model": "mast3r_pointmap_consistency",
        "raw_correspondence_count": reciprocal,
        "raw_matches": metrics["raw_matches"],
        "inlier_matches": metrics["raw_matches"],
        "threshold_trials": [],
        "timing": {
            "time_pair_total_s": float(metrics["timing"].get("time_mast3r_inference_s", 0.0) or 0.0),
            "time_mast3r_inference_s": float(metrics["timing"].get("time_mast3r_inference_s", 0.0) or 0.0),
        },
        "reciprocal_match_count": reciprocal,
        "pointmap_consistency": float(metrics["pointmap_consistency"]),
        "alignment_residual": metrics["alignment_residual"],
        "reprojection_error": metrics["reprojection_error"],
        "overlap_ratio": float(metrics["overlap_ratio"]),
        "combined_geometry_score": float(metrics["combined_geometry_score"]),
        "native_matching_scores": {
            "retrieval_score": None,
            "pointmap_consistency": float(metrics["pointmap_consistency"]),
            "overlap_ratio": float(metrics["overlap_ratio"]),
            "parallax_score": None,
            "graph_edge_score": score,
        },
        "strict_gate": {},
    }
    return reciprocal, reciprocal, score, (float(metrics["direction_dx"]), float(metrics["direction_dy"])), diagnostics
