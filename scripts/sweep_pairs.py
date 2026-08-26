"""Calibration sweep: verify every nominated pair on the labeled listings.

Produces raw, threshold-free evidence so precision/recall curves can be computed
offline without re-running inference. This is the prerequisite for choosing any
threshold (plan section 6.3).

    python -m scripts.sweep_pairs [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
from sqlalchemy import text

import torch

from app.db.session import SessionLocal
from app.models.vggt import vggt_model
from app.pipeline.phase1_analyze.candidate_pairs import nominate, split_tiers
from app.pipeline.phase1_analyze.pairwise_verify import (
    EVIDENCE_SCHEMA_VERSION,
    canonical_order_with_ids,
    evidence_key,
    verify,
)
from app.core.config import settings
from app.services.storage.s3_client import s3_client

OUT = os.path.join(os.path.dirname(__file__), "..", "tmp_sweep")
CACHE = os.path.join(OUT, "cache")
# Sustained MPS inference degrades badly without periodic release: the first run
# went 0.85 -> 4.3 s/pair over ~350 calls on a cold machine with 12 GB free, so it
# was allocator growth rather than thermal or system memory pressure.
RELEASE_EVERY = 25


def _release_accelerator() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _cached(key: str) -> dict | None:
    path = os.path.join(CACHE, f"{key}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            record = json.load(handle)
        if not isinstance(record, dict):
            return None
        if record.get("_evidence_schema_version") != EVIDENCE_SCHEMA_VERSION:
            return None
        return record
    except (json.JSONDecodeError, OSError):
        return None   # a torn write from an interrupted run; just recompute


def _write_json_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # atomic: a kill mid-write must not leave a half file that looks cached
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(path), delete=False) as handle:
            temporary = handle.name
            json.dump(payload, handle, allow_nan=False)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _store(key: str, record: dict) -> None:
    _write_json_atomic(os.path.join(CACHE, f"{key}.json"), record)


def _download(db, job_id: int, directory: str) -> dict[int, str]:
    rows = db.execute(
        text("SELECT id, s3_uri FROM job_photos WHERE job_id = :j ORDER BY position"), {"j": job_id}
    ).fetchall()
    paths: dict[int, str] = {}
    for row in rows:
        key = row.s3_uri[5:].split("/", 1)[1] if row.s3_uri.startswith("s3://") else row.s3_uri
        target = os.path.join(directory, f"{row.id}.jpg")
        if not os.path.exists(target):
            s3_client.client.download_file(settings.S3_BUCKET, key, target)
        paths[int(row.id)] = target
    return paths


def sweep(project_id: str, limit: int | None = None, tier: str = "primary") -> dict:
    db = SessionLocal()
    try:
        truth = db.execute(
            text("SELECT room_instances, listing_slug FROM scene_truth_sets "
                 "WHERE project_id = :p AND status = 'complete'"), {"p": project_id}
        ).fetchone()
        job_id = db.execute(
            text("SELECT id FROM jobs WHERE project_id = :p ORDER BY created_at DESC LIMIT 1"),
            {"p": project_id},
        ).scalar()
        rows = db.execute(
            text("SELECT id, rails_photo_id, position, room_label, embedding "
                 "FROM job_photos WHERE job_id = :j"), {"j": job_id}
        ).fetchall()
        geo = db.execute(
            text("SELECT photo_id, camera_center, view_direction FROM photo_scene_geometry WHERE job_id = :j"),
            {"j": job_id},
        ).fetchall()
        rels = db.execute(
            text("SELECT photo_a_id a, photo_b_id b, debug_metrics->>'same_scene_score' s "
                 "FROM photo_relations WHERE job_id = :j"), {"j": job_id}
        ).fetchall()

        photos = [{"id": r.id, "position": r.position, "room_label": r.room_label} for r in rows]
        embeddings = {int(r.id): r.embedding for r in rows if r.embedding}
        centers = {int(g.photo_id): np.array(g.camera_center, dtype=float) for g in geo if g.camera_center}
        views = {int(g.photo_id): np.array(g.view_direction, dtype=float) for g in geo if g.view_direction}
        scores = {(int(r.a), int(r.b)): float(r.s) for r in rels if r.s is not None}

        candidates = nominate(photos, embeddings=embeddings, global_scores=scores,
                              centers=centers, views=views)
        primary, fallback = split_tiers(candidates)
        candidates = primary if tier == "primary" else fallback if tier == "fallback" else candidates
        if limit:
            candidates = candidates[:limit]

        # ground truth by room instance, keyed through the stable rails id
        id_by_rails = {r.rails_photo_id: int(r.id) for r in rows}
        room_of: dict[int, str] = {}
        for entry in truth.room_instances or []:
            for rails_id in entry.get("photo_keys", []):
                if rails_id in id_by_rails:
                    room_of[id_by_rails[rails_id]] = entry["instance"]

        slug = truth.listing_slug or project_id[:8]
        directory = os.path.join(tempfile.gettempdir(), f"sweep_{job_id}")
        os.makedirs(directory, exist_ok=True)
        paths = _download(db, job_id, directory)
    finally:
        db.close()

    runtime = vggt_model.runtime_metadata()
    checkpoint_sha = str(runtime["checkpoint_sha256"])
    model_revision = str(runtime.get("repo_commit") or "")
    precision = str(runtime.get("dtype") or "")
    mode, size = str(settings.VGGT_IMAGE_MODE).lower(), int(settings.VGGT_IMAGE_SIZE)
    print(f"\n=== {slug}: {len(photos)} photos, {len(candidates)} {tier} pairs ===", flush=True)
    results = []
    started = time.monotonic()
    computed = hits = 0
    for index, pair in enumerate(candidates, 1):
        a, b = pair.key
        (path_a, evidence_a), (path_b, evidence_b) = canonical_order_with_ids(
            paths[a], a, paths[b], b
        )
        key = evidence_key(
            path_a,
            path_b,
            checkpoint_sha,
            mode,
            size,
            model_revision=model_revision,
            precision=precision,
        )
        record = _cached(key)
        if record is None:
            evidence = verify(path_a, path_b)
            record = evidence.to_dict()
            record["_evidence_schema_version"] = EVIDENCE_SCHEMA_VERSION
            _store(key, record)          # persisted per pair: an interrupt loses one run
            computed += 1
            if computed % RELEASE_EVERY == 0:
                _release_accelerator()
        else:
            hits += 1
        record = dict(record)
        record.update({
            # Directional pose/motion in `record` is canonical A -> B, so the
            # attached ids must use the same content-hash order.
            "photo_a_id": evidence_a, "photo_b_id": evidence_b,
            "sources": sorted(pair.sources), "tier": pair.tier,
            "room_a": room_of.get(evidence_a), "room_b": room_of.get(evidence_b),
            "same_room": (
                room_of.get(evidence_a) is not None
                and room_of.get(evidence_a) == room_of.get(evidence_b)
            ),
        })
        results.append(record)
        if index % 25 == 0 or index == len(candidates):
            rate = (time.monotonic() - started) / computed if computed else 0.0
            remaining = len(candidates) - index
            print(f"  {index}/{len(candidates)}  {rate:.2f}s/computed  cache_hits={hits}  "
                  f"eta {rate*remaining/60:.1f}m", flush=True)
    if computed:
        _release_accelerator()

    # A partial run must never overwrite a completed sweep of the same listing.
    suffix = f".partial-{len(results)}" if limit else ""
    out_path = os.path.join(OUT, f"{slug}.{tier}{suffix}.json")
    _write_json_atomic(
        out_path,
        {
            "listing": slug,
            "project_id": project_id,
            "job_id": job_id,
            "tier": tier,
            "photos": len(photos),
            "pairs": results,
        },
    )
    positives = sum(1 for r in results if r["same_room"])
    print(f"  wrote {out_path}: {len(results)} pairs ({positives} same-room, "
          f"{len(results)-positives} different-room), {(time.monotonic()-started)/60:.1f} min", flush=True)
    return {"listing": slug, "pairs": len(results), "positives": positives}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tier", choices=["primary", "fallback", "all"], default="primary")
    args = parser.parse_args()
    db = SessionLocal()
    try:
        project_ids = [r.project_id for r in db.execute(
            text("SELECT project_id FROM scene_truth_sets WHERE status='complete' ORDER BY updated_at")
        ).fetchall()]
    finally:
        db.close()
    for project_id in project_ids:
        sweep(project_id, args.limit, args.tier)
    print("\nSWEEP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
