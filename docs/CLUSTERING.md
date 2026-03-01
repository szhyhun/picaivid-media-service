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

## Key Outputs Persisted

For each checked pair (when available):
- semantic similarity
- geometric matches/inliers/score
- direction vector (`dx`, `dy`)
- segment overlap scores
- oracle/diagnostic metadata

## Pair Debug Usage

Use `/projects/:id/pair-debug` in React to inspect:
- raw/filtered/inlier correspondences
- segment metrics and transition recommendation
- score components and filter tables

## Regression Guard

Baseline tools live in `scripts/baselines/`.
Use them when tuning thresholds or matcher behavior to prevent silent drift.
