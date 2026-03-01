#!/usr/bin/env python3
"""Export/compare clustering baselines by filename.

Purpose:
- Snapshot a "good enough" clustering baseline from a known project/job.
- Compare future runs against that baseline using filename-level grouping.

This script is designed for regression guardrails where cluster IDs are unstable
between runs, but photo filenames are stable.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402


@dataclass
class PhotoRow:
    photo_id: int
    filename: str
    room_cluster_id: int | None
    cluster_order: int | None
    room_label: str | None
    room_type: str | None


def _normalize_filename(name: str) -> str:
    return name.strip().lower()


def _resolve_job_id(conn, project_id: str | None, job_id: int | None) -> int:
    if job_id is not None:
        exists = conn.execute(
            text("SELECT 1 FROM jobs WHERE id = :job_id"),
            {"job_id": int(job_id)},
        ).fetchone()
        if not exists:
            raise ValueError(f"Job {job_id} not found")
        return int(job_id)

    if not project_id:
        raise ValueError("Provide --project-id or --job-id")

    row = conn.execute(
        text(
            """
            SELECT id
            FROM jobs
            WHERE project_id = :project_id
            ORDER BY created_at DESC
            LIMIT 1
            """
        ),
        {"project_id": project_id},
    ).fetchone()
    if not row:
        raise ValueError(f"No jobs found for project {project_id}")
    return int(row.id)


def _load_job_photos(conn, job_id: int) -> List[PhotoRow]:
    rows = conn.execute(
        text(
            """
            SELECT
                jp.id AS photo_id,
                jp.filename AS filename,
                jp.room_cluster_id AS room_cluster_id,
                jp.cluster_order AS cluster_order,
                jp.room_label AS room_label,
                rc.room_type AS room_type
            FROM job_photos jp
            LEFT JOIN room_clusters rc
              ON rc.id = jp.room_cluster_id
            WHERE jp.job_id = :job_id
              AND COALESCE(jp.exclude, false) = false
            ORDER BY jp.cluster_order ASC NULLS LAST, jp.id ASC
            """
        ),
        {"job_id": int(job_id)},
    ).fetchall()

    photos = []
    for r in rows:
        filename = (r.filename or "").strip()
        if not filename:
            filename = f"photo-{int(r.photo_id)}"
        photos.append(
            PhotoRow(
                photo_id=int(r.photo_id),
                filename=filename,
                room_cluster_id=int(r.room_cluster_id) if r.room_cluster_id is not None else None,
                cluster_order=int(r.cluster_order) if r.cluster_order is not None else None,
                room_label=r.room_label,
                room_type=r.room_type,
            )
        )
    return photos


def _build_clusters(photos: Iterable[PhotoRow]) -> List[dict]:
    groups: Dict[str, List[PhotoRow]] = defaultdict(list)
    for p in photos:
        key = (
            str(p.room_cluster_id)
            if p.room_cluster_id is not None
            else f"unassigned-{p.photo_id}"
        )
        groups[key].append(p)

    output = []
    for key, plist in groups.items():
        ordered = sorted(
            plist,
            key=lambda x: (
                x.cluster_order is None,
                x.cluster_order if x.cluster_order is not None else math.inf,
                x.photo_id,
            ),
        )
        output.append(
            {
                "cluster_key": key,
                "room_type": ordered[0].room_type,
                "room_labels": sorted({p.room_label for p in ordered if p.room_label}),
                "photo_ids": [p.photo_id for p in ordered],
                "photo_filenames": [p.filename for p in ordered],
                "normalized_filenames": [_normalize_filename(p.filename) for p in ordered],
                "size": len(ordered),
            }
        )

    output.sort(
        key=lambda c: (
            c["size"] <= 1,    # show non-singletons first
            -c["size"],
            c["cluster_key"],
        )
    )
    return output


def _cluster_map_from_baseline(baseline: dict) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for idx, cluster in enumerate(baseline["clusters"]):
        label = f"baseline_{idx}"
        for name in cluster["normalized_filenames"]:
            mapping[name] = label
    return mapping


def _cluster_map_from_current(photos: Iterable[PhotoRow]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for p in photos:
        label = (
            str(p.room_cluster_id)
            if p.room_cluster_id is not None
            else f"unassigned-{p.photo_id}"
        )
        mapping[_normalize_filename(p.filename)] = label
    return mapping


def _pairwise_metrics(
    baseline_map: Dict[str, str],
    current_map: Dict[str, str],
) -> Tuple[int, int, int, float, float, float]:
    names = sorted(set(baseline_map.keys()) & set(current_map.keys()))
    tp = fp = fn = 0
    for a, b in combinations(names, 2):
        same_baseline = baseline_map[a] == baseline_map[b]
        same_current = current_map[a] == current_map[b]
        if same_current and same_baseline:
            tp += 1
        elif same_current and not same_baseline:
            fp += 1
        elif (not same_current) and same_baseline:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return tp, fp, fn, precision, recall, f1


def cmd_export(args: argparse.Namespace) -> int:
    engine = create_engine(settings.DATABASE_URL)
    with engine.connect() as conn:
        job_id = _resolve_job_id(conn, args.project_id, args.job_id)
        job = conn.execute(
            text("SELECT id, project_id, status, created_at FROM jobs WHERE id = :job_id"),
            {"job_id": job_id},
        ).fetchone()
        photos = _load_job_photos(conn, job_id)

    clusters = _build_clusters(photos)
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_id": job.project_id,
        "job_id": int(job.id),
        "job_status": job.status,
        "job_created_at": job.created_at.isoformat() if job.created_at else None,
        "photo_count": len(photos),
        "cluster_count": len(clusters),
        "clusters": clusters,
    }

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Exported baseline: {out_path}")
    print(f"job_id={job_id}, photos={len(photos)}, clusters={len(clusters)}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    baseline_path = Path(args.baseline).resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    engine = create_engine(settings.DATABASE_URL)
    with engine.connect() as conn:
        job_id = _resolve_job_id(conn, args.project_id, args.job_id)
        job = conn.execute(
            text("SELECT id, project_id, status FROM jobs WHERE id = :job_id"),
            {"job_id": job_id},
        ).fetchone()
        photos = _load_job_photos(conn, job_id)

    baseline_map = _cluster_map_from_baseline(baseline)
    current_map = _cluster_map_from_current(photos)

    common = sorted(set(baseline_map.keys()) & set(current_map.keys()))
    missing_from_current = sorted(set(baseline_map.keys()) - set(current_map.keys()))
    extra_in_current = sorted(set(current_map.keys()) - set(baseline_map.keys()))

    tp, fp, fn, precision, recall, f1 = _pairwise_metrics(baseline_map, current_map)
    print(f"Baseline file : {baseline_path}")
    print(f"Baseline job  : {baseline.get('job_id')} ({baseline.get('project_id')})")
    print(f"Current job   : {job.id} ({job.project_id}) status={job.status}")
    print(f"Common photos : {len(common)}")
    print(f"Missing photos: {len(missing_from_current)}")
    print(f"Extra photos  : {len(extra_in_current)}")
    print()
    print("Pairwise co-cluster metrics (filename-level):")
    print(f"  TP={tp} FP={fp} FN={fn}")
    print(f"  Precision={precision:.4f}")
    print(f"  Recall   ={recall:.4f}")
    print(f"  F1       ={f1:.4f}")

    if missing_from_current:
        print("\nMissing filenames (first 20):")
        for n in missing_from_current[:20]:
            print(f"  - {n}")

    if extra_in_current:
        print("\nExtra filenames (first 20):")
        for n in extra_in_current[:20]:
            print(f"  - {n}")

    # Emit split/merge hints for quick debugging.
    baseline_cluster_to_files: Dict[str, set[str]] = defaultdict(set)
    for fname, c in baseline_map.items():
        baseline_cluster_to_files[c].add(fname)

    current_cluster_to_files: Dict[str, set[str]] = defaultdict(set)
    for fname, c in current_map.items():
        if fname in baseline_map:
            current_cluster_to_files[c].add(fname)

    noisy_current = []
    for cid, files in current_cluster_to_files.items():
        baseline_ids = {baseline_map[f] for f in files}
        if len(baseline_ids) > 1:
            noisy_current.append((cid, len(files), len(baseline_ids)))
    noisy_current.sort(key=lambda x: (-x[1], -x[2], x[0]))
    if noisy_current:
        print("\nMerged clusters (current cluster contains multiple baseline groups):")
        for cid, n_files, n_groups in noisy_current[:20]:
            print(f"  - current_cluster={cid} files={n_files} baseline_groups={n_groups}")

    split_baseline = []
    for bid, files in baseline_cluster_to_files.items():
        current_ids = {current_map[f] for f in files if f in current_map}
        if len(current_ids) > 1:
            split_baseline.append((bid, len(files), len(current_ids)))
    split_baseline.sort(key=lambda x: (-x[1], -x[2], x[0]))
    if split_baseline:
        print("\nSplit baseline groups (one baseline cluster split across current clusters):")
        for bid, n_files, n_groups in split_baseline[:20]:
            print(f"  - baseline_cluster={bid} files={n_files} current_clusters={n_groups}")

    if args.fail_on_f1_below is not None and f1 < float(args.fail_on_f1_below):
        print(f"\nFAIL: F1 {f1:.4f} < threshold {float(args.fail_on_f1_below):.4f}")
        return 2

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cluster baseline export/compare by filename")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_export = sub.add_parser("export", help="Export baseline JSON from project/job")
    p_export.add_argument("--project-id", type=str, default=None, help="Project UUID (latest job)")
    p_export.add_argument("--job-id", type=int, default=None, help="Explicit job id")
    p_export.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output baseline JSON path",
    )
    p_export.set_defaults(func=cmd_export)

    p_compare = sub.add_parser("compare", help="Compare project/job against baseline JSON")
    p_compare.add_argument("--baseline", type=str, required=True, help="Baseline JSON path")
    p_compare.add_argument("--project-id", type=str, default=None, help="Project UUID (latest job)")
    p_compare.add_argument("--job-id", type=int, default=None, help="Explicit job id")
    p_compare.add_argument(
        "--fail-on-f1-below",
        type=float,
        default=None,
        help="Exit 2 when pairwise F1 is below this threshold",
    )
    p_compare.set_defaults(func=cmd_compare)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
