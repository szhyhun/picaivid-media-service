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
./venv/bin/python -m py_compile app/main.py app/pipeline/phase1_analyze/learned_matching.py
```

If clustering logic changed, run baseline comparison from `scripts/baselines/README.md`.

## Pair-Debug Performance Check

When testing pair-debug latency, use `pair_debug_timing` logs from media-service.
Key fields:

- `model_loftr_ms` (inference time)
- `model_device`, `tensor_device`
- `cuda_available`, `preferred_device`

On local Mac, CPU or MPS may be used.
On AWS production, target must be CUDA (`model_device=cuda`).
