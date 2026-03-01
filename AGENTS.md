# Media Service Agent Guide

## Scope

This repository owns compute-heavy media logic only:
- Photo analysis and clustering
- Geometric matching and transition scoring
- Clip generation and render orchestration
- Worker execution of media jobs

It does **not** own product/business logic, auth, billing, or user-facing API contracts beyond media-specific endpoints.

## Current Production Direction

- Primary matcher path is `loftr_kornia_indoor_native`.
- Cluster membership is geometry-first and strict-gated.
- Pair debug is matcher-limited (Kornia native path).
- Rails remains the orchestration and user-facing system of record.

If you change these assumptions, update:
- `docs/CLUSTERING.md`
- `README.md`
- any API/request schema touched by the change

## Code Areas

- API server: `app/main.py`
- Worker entrypoint: `app/worker.py`
- Pipeline orchestration: `app/pipeline/orchestrator.py`
- Phase 1 clustering/matching: `app/pipeline/phase1_analyze/`
- Phase 2 render logic: `app/pipeline/phase2_render/`
- DB models/migrations: `app/db/`, `alembic/`

## Engineering Rules

- Keep controllers/endpoints thin; push logic into pipeline/services.
- Prefer explicit diagnostics over hidden heuristics.
- Do not add semantic fallback shortcuts that bypass geometry gates without documenting why.
- When changing scoring thresholds, add/update regression checks in `scripts/`.
- Keep logs actionable: include pair IDs, model/matcher, and key gate reasons.

## Validation Before Commit

- Python compile check:
  - `./venv/bin/python -m py_compile app/main.py app/pipeline/phase1_analyze/learned_matching.py`
- Run targeted tests/scripts for changed area.
- If clustering behavior changed, compare against baseline:
  - see `scripts/baselines/README.md`

## Documentation Policy

- Keep docs short and operational.
- Remove stale plans/status docs instead of accumulating versions.
- Prefer one canonical doc per concern.
