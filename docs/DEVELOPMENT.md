# Development Workflow

## Prerequisites

- Python 3.11+
- Virtualenv created at `venv/`
- Local infra running from `picaivid-rails/docker-compose.yml`

## Start Local Stack

From `picaivid-rails`:

```bash
docker-compose up -d
```

Expected services:
- Postgres on `localhost:5432`
- MinIO on `localhost:9000`
- LocalStack (SQS) on `localhost:4566`

## Run Media Service

API:

```bash
cd picaivid-media-service
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

Worker:

```bash
cd picaivid-media-service
source venv/bin/activate
python -m app.worker
```

## Migrations

```bash
cd picaivid-media-service
source venv/bin/activate
alembic upgrade head
```

## Common Checks

```bash
./venv/bin/python -m py_compile app/main.py app/pipeline/phase1_analyze/mast3r_pipeline.py
```

If clustering logic changed, run baseline comparison from `scripts/baselines/README.md`.

## Pair-Debug Performance Check

When testing pair-debug latency, use `pair_debug_timing` logs from media-service.
Key fields:

- `model_device`, `tensor_device`
- `cuda_available`, `preferred_device`
- `model_mast3r_inference_ms`

MASt3R phase 1 and live pair debug require CUDA. On AWS, target must be `model_device=cuda`.
