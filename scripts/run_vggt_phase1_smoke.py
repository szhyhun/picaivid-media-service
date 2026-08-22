#!/usr/bin/env python3
"""Run one real VGGT Phase 1 analysis against owned listing images.

Examples:
  ./venv/bin/python scripts/run_vggt_phase1_smoke.py --images ~/Listings/example --count 4
  ./venv/bin/python scripts/run_vggt_phase1_smoke.py --images ~/Listings/example --count 50 --output /tmp/shot-plan.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

# Allow `python scripts/run_vggt_phase1_smoke.py` without requiring PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline.phase1_analyze.vggt_pipeline import run_vggt_scene_pipeline


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the real VGGT-1B-Commercial Phase 1 pipeline.")
    parser.add_argument("--images", required=True, type=Path, help="Directory containing owned listing photos")
    parser.add_argument("--count", required=True, type=int, help="Exact number of ordered photos to analyze")
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    args = parser.parse_args()

    paths = sorted(path for path in args.images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if len(paths) < args.count:
        parser.error(f"{args.images} contains {len(paths)} supported images; --count {args.count} was requested")
    paths = paths[:args.count]
    images = [Image.open(path).convert("RGB") for path in paths]
    photo_ids = list(range(1, len(paths) + 1))
    labels = [""] * len(paths)

    geometries, relations, components = run_vggt_scene_pipeline(
        images=images,
        photo_ids=photo_ids,
        room_labels=labels,
        positions=list(range(len(paths))),
        job_id=0,
    )
    if not geometries or any(geometry.local_metrics.get("runtime", {}).get("model") != "VGGT-1B-Commercial" for geometry in geometries):
        raise RuntimeError("VGGT did not produce real commercial-model geometry")
    result = {
        "photo_count": len(paths),
        "runtime": geometries[0].local_metrics.get("runtime", {}),
        "geometry_count": len(geometries),
        "relation_count": len(relations),
        "component_count": len(components),
        "components": [component.__dict__ for component in components],
        "relations": [relation.__dict__ for relation in relations],
    }
    serialized = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
