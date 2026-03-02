# Picaivid Media Service

FastAPI + worker service for photo analysis, clustering, transition scoring, and video clip generation.

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Worker (separate terminal):

```bash
source venv/bin/activate
python -m app.worker
```

## Deployment Note (LoFTR Geometry)

For cloud deployment, LoFTR must run on CUDA to keep pair-level geometry fast.
Validate with logs:

- `Loaded LoFTR matcher (indoor) on cuda`
- `pair_debug_timing ... model_device=cuda ... preferred_device=cuda`

## Local Dependencies

This service expects local infrastructure from `picaivid-rails/docker-compose.yml`:
- PostgreSQL
- LocalStack (SQS)
- MinIO (S3-compatible storage)

## Endpoints

- Health: `http://localhost:8000/health`
- OpenAPI docs: `http://localhost:8000/docs`

## Core Docs

- `AGENTS.md` - Coding and ownership rules for this repo
- `docs/DEVELOPMENT.md` - Day-to-day local workflow
- `docs/ARCHITECTURE.md` - Runtime architecture and ownership boundaries
- `docs/CLUSTERING.md` - Current clustering and geometry behavior
- `docs/AWS_SETUP.md` - Minimal AWS deployment setup
- `scripts/baselines/README.md` - Clustering baseline regression checks
