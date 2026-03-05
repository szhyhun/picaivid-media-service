# Clustering and Geometry

## Goal

Build room-consistent clusters and transition-safe ordering from listing photos.

## Current Matching Path

- Matcher: `loftr_kornia_indoor_native`
- Pair debug and production both run through native LoFTR diagnostics path
- Reverse retry is enabled for weak forward results to reduce order sensitivity (`A->B` vs `B->A`)

## Edge Acceptance (High Level)

Cluster edges require strict geometry checks:
- minimum matches/inliers
- allowed geometric model (`fundamental_*`)
- native confidence distribution thresholds
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
- semantic similarity
- geometric matches/inliers/score
- direction vector (`dx`, `dy`)
- segment overlap scores
- oracle/diagnostic metadata

## Pair Debug Usage

Use `/projects/:id/pair-debug` in React to inspect:
- raw/inlier correspondences
- segment metrics and transition recommendation
- score components and matcher timing

## Runtime Performance Signals

Use `pair_debug_timing` logs to validate backend and latency:

- `model_device` / `tensor_device`
- `cuda_available`
- `preferred_device`
- `model_loftr_ms` (dominant inference cost)

For AWS deployment, production target is CUDA (`model_device=cuda`).
If it shows CPU, pair-debug and geometry checks will be significantly slower.

## Regression Guard

Baseline tools live in `scripts/baselines/`.
Use them when tuning thresholds or matcher behavior to prevent silent drift.
