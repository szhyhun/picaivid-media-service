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

For reproducible environments (recommended for staging/AWS), install from lock file:

```bash
pip install -r requirements.lock.txt
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

## Debug Matcher Dependencies

- `roma_v2_debug` requires `romatch` (already included in `requirements.txt` and `requirements.lock.txt`).
- Optional RoMa device override:
  - `ROMA_DEBUG_DEVICE=auto|cpu|mps|cuda`
  - default `auto`: uses `cuda` when available; on macOS `mps` defaults to `cpu` for stability
- Optional MatchFormer device override:
  - `MATCHFORMER_DEBUG_DEVICE=auto|cpu|mps|cuda`
  - default `auto`: uses `cuda` when available; on macOS `mps` defaults to `cpu` for stability
- `matchformer_indoor_debug` / `matchformer_outdoor_debug` require:
  - `MATCHFORMER_REPO_DIR`
  - `MATCHFORMER_INDOOR_CKPT`
  - `MATCHFORMER_OUTDOOR_CKPT`

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
- `docs/AWS_DEPLOYMENT.md` - Full AWS runbook (all apps, CD, Spot cost control)
- `scripts/aws/` - GPU instance start/stop/status cost-control scripts
- `scripts/aws/bootstrap-ec2.sh` - EC2 bootstrap for media-service GPU host
- `.github/workflows/deploy.yml` - GitHub Actions OIDC + SSM deploy workflow
- `deploy/systemd/` - Media API/worker service unit templates
- `deploy/env/` - Example env files for media API/worker systemd services
- `scripts/baselines/README.md` - Clustering baseline regression checks
