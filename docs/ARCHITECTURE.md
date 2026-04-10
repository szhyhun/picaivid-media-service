# Media Service Architecture

## What This Service Owns

- Photo-level analysis and geometry matching
- Cluster construction and ordering metadata
- Clip generation and rendering phases
- Background worker execution for media jobs

## What It Does Not Own

- Product/business workflows
- User auth and authorization
- Billing and subscription rules
- Frontend state/UI contracts

Those are owned by `picaivid-rails` and `picaivid-react`.

## Runtime Components

- FastAPI app (`app/main.py`)
- SQS worker (`app/worker.py`)
- Pipeline orchestrator (`app/pipeline/orchestrator.py`)
- Postgres models (`app/db/models/*`)
- S3/MinIO client (`app/services/storage/s3_client.py`)

## Pipeline Shape

1. Phase 1 analyze
   - MASt3R retrieval graph + geometry matching
   - Room clusters + pair diagnostics
2. Phase 2 render
   - Clip generation and media outputs
3. Persist outputs and metrics for Rails/UI consumption

## Geometry/Clustering Notes

- Production matcher default: `mast3r_graph`
- Edge acceptance is strict-gated by geometry quality and model constraints
- Pair debug is intended for diagnosing pair-level failures, not changing business flow

For details see `docs/CLUSTERING.md`.
