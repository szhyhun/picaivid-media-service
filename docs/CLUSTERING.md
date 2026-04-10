# Clustering and Geometry

## Goal

Build room-consistent clusters and transition-safe ordering from listing photos.

## Current Matching Path

- Production matcher: `mast3r_graph`
- MASt3R retrieval builds the graph of candidate photo pairs.
- MASt3R pair inference and sparse global alignment produce edge quality and component state.
- Final user-facing clusters are capped at two photos; same-component extras are debug suggestions only.

## Edge Acceptance (High Level)

Cluster edges require strict geometry checks:
- minimum reciprocal matches
- pointmap consistency
- alignment/reprojection quality
- parallax score
- graph edge score gate
- geometry quality score gate

This intentionally rejects many semantic-only or weak-planar links.

## Strictness Policy (No Safety Fallbacks)

For clustering and pair ranking, "safe fallback" behavior is disallowed because it creates wrong pair choices.

- If a strict metric is required and missing (`combined_score`, `overlap_ratio`, `geometric_inliers`), reject the edge for ranking.
- Do not substitute alternate metrics when strict metrics are missing.
- Do not silently downgrade to weaker ranking logic.
- Emit explicit reason logs for rejected edges/links.

Rule: be precise or fail with detailed reason.

## Key Outputs Persisted

For each checked pair (when available):
- retrieval score
- reciprocal match count
- pointmap consistency
- alignment residual / reprojection quality
- parallax score
- graph edge score
- direction vector (`dx`, `dy`)
- component and pose diagnostics

## Pair Debug Usage

Use `/projects/:id/pair-debug` in React to inspect:
- raw MASt3R correspondences
- pointmap/graph metrics and transition recommendation
- score components and matcher timing

## Runtime Performance Signals

Use `pair_debug_timing` logs to validate backend and latency:

- `model_device` / `tensor_device`
- `cuda_available`
- `preferred_device`
- `model_mast3r_inference_ms`

For AWS deployment, production target is CUDA (`model_device=cuda`).
If CUDA is unavailable, MASt3R phase 1 should fail fast instead of falling back to CPU.

## Regression Guard

Baseline tools live in `scripts/baselines/`.
Use them when tuning thresholds or matcher behavior to prevent silent drift.
