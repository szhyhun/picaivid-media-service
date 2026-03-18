#!/usr/bin/env python3
"""Direct matcher probe for one JobPhoto pair.

Uses the same media-service DB and S3 paths, saves the exact preprocessed
grayscale inputs, and compares raw / confidence-filtered / geometry-inlier
counts across selected matcher variants outside the API wrapper.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from app.db.models.photo import JobPhoto
from app.db.session import get_db_context
from app.pipeline.phase1_analyze.learned_matching import (
    DEFAULT_LOFTR_INPUT_SIZE,
    LOFTR_NATIVE_CONFIDENCE_THRESHOLD,
    _build_native_tensor,
    _estimate_geometric_inliers,
    _extract_loftr_points_and_scores,
    _get_native_preprocessed_entry,
    _matching_score_summary,
    _release_device_cache,
)
from app.pipeline.phase1_analyze.matcher_loaders import (
    load_loftr_checkpoint,
    load_zju_loftr_debug_variant,
)
from app.services.storage.s3_client import s3_client

SUPPORTED_MATCHERS = (
    "loftr_kornia_indoor_native",
    "loftr_zju_indoor_ds_debug",
    "loftr_zju_indoor_ot_debug",
)


def _save_gray_image(gray: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(gray * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(array, mode="L").save(destination)


def _load_matcher(name: str) -> tuple[Any, dict[str, str]]:
    if name == "loftr_kornia_indoor_native":
        return load_loftr_checkpoint("indoor"), {"checkpoint": "kornia:indoor"}
    if name == "loftr_zju_indoor_ds_debug":
        matcher, meta = load_zju_loftr_debug_variant("indoor_ds")
        return matcher, meta
    if name == "loftr_zju_indoor_ot_debug":
        matcher, meta = load_zju_loftr_debug_variant("indoor_ot")
        return matcher, meta
    raise ValueError(f"Unsupported matcher: {name}")


def _run_matcher(
    matcher_name: str,
    prep_left: dict[str, Any],
    prep_right: dict[str, Any],
    confidence_threshold: float,
    device_name: str,
) -> dict[str, Any]:
    matcher, meta = _load_matcher(matcher_name)
    if device_name != "auto":
        matcher = matcher.to(torch.device(device_name))
        matcher.eval()
    device = next(matcher.parameters()).device

    tensor_left = _build_native_tensor(prep_left["gray_resized"], device=device)
    tensor_right = _build_native_tensor(prep_right["gray_resized"], device=device)

    raw_output = None
    correspondences: dict[str, Any] | None = None
    batch = {"image0": tensor_left, "image1": tensor_right}
    with torch.inference_mode():
        raw_output = matcher(batch)
        correspondences = raw_output if isinstance(raw_output, dict) else batch

    points0, points1, scores = _extract_loftr_points_and_scores(correspondences)
    raw_count = int(len(points0))
    conf_mask = scores >= float(confidence_threshold) if scores.size > 0 else np.zeros((0,), dtype=bool)
    conf_points0 = points0[conf_mask] if raw_count > 0 else np.empty((0, 2), dtype=np.float32)
    conf_points1 = points1[conf_mask] if raw_count > 0 else np.empty((0, 2), dtype=np.float32)
    conf_scores = scores[conf_mask] if raw_count > 0 else np.empty((0,), dtype=np.float32)
    conf_count = int(len(conf_points0))

    inlier_mask = None
    geometry_model = "none"
    inlier_count = 0
    if conf_count >= 8:
        inlier_mask, geometry_model = _estimate_geometric_inliers(conf_points0, conf_points1)
        if inlier_mask is not None:
            inlier_count = int(inlier_mask.sum())

    try:
        del correspondences
    except Exception:
        pass
    del batch
    del tensor_left
    del tensor_right
    try:
        del raw_output
    except Exception:
        pass
    _release_device_cache(device=device)

    return {
        "matcher": matcher_name,
        "device": str(device),
        "checkpoint_meta": meta,
        "loftr_output_count": raw_count,
        "confidence_filtered_count": conf_count,
        "geometry_inlier_count": inlier_count,
        "geometry_model": geometry_model,
        "raw_score_summary": _matching_score_summary(scores),
        "confidence_score_summary": _matching_score_summary(conf_scores),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-photo-id", type=int, required=True)
    parser.add_argument("--right-photo-id", type=int, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=LOFTR_NATIVE_CONFIDENCE_THRESHOLD)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument(
        "--matcher",
        action="append",
        choices=SUPPORTED_MATCHERS,
        help="Matcher(s) to run. Defaults to all supported LoFTR variants for this probe.",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/loftr_pair_debug",
        help="Directory for saved preprocessed images.",
    )
    args = parser.parse_args()

    with get_db_context() as db:
        left_photo = db.query(JobPhoto).filter(JobPhoto.id == int(args.left_photo_id)).first()
        right_photo = db.query(JobPhoto).filter(JobPhoto.id == int(args.right_photo_id)).first()

    if left_photo is None or right_photo is None:
        raise SystemExit("One or both JobPhoto ids were not found.")

    left_image = s3_client.download_image(left_photo.s3_uri)
    right_image = s3_client.download_image(right_photo.s3_uri)

    target_long_side = max(64, int(max(DEFAULT_LOFTR_INPUT_SIZE)))
    native_prep_left = _get_native_preprocessed_entry(left_image, target_long_side=target_long_side)
    native_prep_right = _get_native_preprocessed_entry(right_image, target_long_side=target_long_side)
    output_dir = Path(args.output_dir) / f"{left_photo.id}_{right_photo.id}"
    _save_gray_image(native_prep_left["gray_resized"], output_dir / "left_preprocessed.png")
    _save_gray_image(native_prep_right["gray_resized"], output_dir / "right_preprocessed.png")

    print(f"left_photo_id={left_photo.id} s3_uri={left_photo.s3_uri}")
    print(f"right_photo_id={right_photo.id} s3_uri={right_photo.s3_uri}")
    print(f"confidence_threshold={args.confidence_threshold}")
    print(f"original_left_size={left_image.size} original_right_size={right_image.size}")
    print(
        "preprocessed_left="
        f"{native_prep_left['meta']['content_w']}x{native_prep_left['meta']['content_h']} pad=({native_prep_left['meta']['pad_w']},{native_prep_left['meta']['pad_h']})"
    )
    print(
        "preprocessed_right="
        f"{native_prep_right['meta']['content_w']}x{native_prep_right['meta']['content_h']} pad=({native_prep_right['meta']['pad_w']},{native_prep_right['meta']['pad_h']})"
    )
    print(f"saved_preprocessed_dir={output_dir}")

    matcher_names = args.matcher or list(SUPPORTED_MATCHERS)
    for matcher_name in matcher_names:
        print("---")
        print(f"matcher={matcher_name}")
        try:
            result = _run_matcher(
                matcher_name=matcher_name,
                prep_left=native_prep_left,
                prep_right=native_prep_right,
                confidence_threshold=float(args.confidence_threshold),
                device_name=args.device,
            )
        except Exception as exc:
            print(f"status=unavailable")
            print(f"error={exc}")
            continue
        print(f"status=ok")
        print(f"device={result['device']}")
        print(f"checkpoint_meta={result['checkpoint_meta']}")
        print(f"loftr_output_count={result['loftr_output_count']}")
        print(f"confidence_filtered_count={result['confidence_filtered_count']}")
        print(f"geometry_inlier_count={result['geometry_inlier_count']}")
        print(f"geometry_model={result['geometry_model']}")
        print(f"raw_score_summary={result['raw_score_summary']}")
        print(f"confidence_score_summary={result['confidence_score_summary']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
