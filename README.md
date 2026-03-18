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

- `loftr_kornia_indoor_native` uses the Kornia checkpoint cached by Torch locally:
  - `~/.cache/torch/hub/checkpoints/loftr_indoor.ckpt`
  - `~/.cache/torch/hub/checkpoints/loftr_indoor_ds_new.ckpt`
  - if these cache files are deleted, Kornia will re-download them on first use
- `loftr_zju_indoor_ds_debug` / `loftr_zju_indoor_ot_debug` require:
  - `LOFTR_ZJU_REPO_DIR`
  - `LOFTR_ZJU_INDOOR_DS_CKPT`
  - `LOFTR_ZJU_INDOOR_OT_CKPT`
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

### AWS Asset Hydration Rule

Do not depend on workstation-specific absolute paths in cloud environments.

On AWS rollout, all external matcher repos, checkpoints, and Hugging Face model caches must be hydrated from S3 into local disk before they are used. Do not rely on live internet downloads in production.

For semantic/depth models, the service is now expected to load from the local Hugging Face cache only. In particular:

- DINOv3 semantic model:
  - `facebook/dinov3-vitb16-pretrain-lvd1689m`
- MiDaS / DPT depth model
- any other Hugging Face-hosted model used by the service

These must exist under the configured model cache directory before startup:

- `MODEL_CACHE_DIR` (defaults to `./ml_models`)
- optional DINOv3 S3 hydrate source:
  - `DINO_V3_CACHE_ARCHIVE_S3_URI`
- optional last-resort remote fallback:
  - `DINO_V3_ALLOW_REMOTE_FALLBACK=true|false`

Recommended pattern for AWS:

- pre-build the Hugging Face cache locally once
- upload the relevant cached model directories to S3
- hydrate them onto the instance under `MODEL_CACHE_DIR` during bootstrap
- start API/worker only after the cache is present

For external geometry matchers, the loader supports S3 hydration directly:

- repo archives:
  - `LOFTR_ZJU_REPO_ARCHIVE_S3_URI`
  - `MATCHFORMER_REPO_ARCHIVE_S3_URI`
- checkpoint objects:
  - `LOFTR_ZJU_*_CKPT_S3_URI`
  - `MATCHFORMER_*_CKPT_S3_URI`

Behavior:

- if the configured local repo/checkpoint path exists, the loader uses it
- if the local path is missing and the matching `*_S3_URI` is set, the loader downloads/extracts the asset from S3 into the configured local path
- if neither local asset nor S3 fallback exists, matcher loading fails fast with a clear error
- if the DINOv3 local cache is missing and `DINO_V3_CACHE_ARCHIVE_S3_URI` is set, the service hydrates the cache from S3 into `MODEL_CACHE_DIR`
- if the DINOv3 local cache is missing and S3 hydration is not configured, the service may fall back to remote model download only when `DINO_V3_ALLOW_REMOTE_FALLBACK=true`
- for AWS, prefer `local -> S3 hydrate -> load locally`; keep URL fallback as recovery only, not the normal path

For semantic region extraction, SAM2 should follow the same policy:

- local repo:
  - `SAM2_REPO_DIR`
- local checkpoint:
  - `SAM2_CHECKPOINT`
- local config:
  - `SAM2_CONFIG`
- semantic stage controls:
  - `SEMANTIC_REGIONS_ENABLED`
  - `SEMANTIC_REGIONS_BACKEND`
  - `SEMANTIC_MATCH_EDGE_DILATION_PX`
  - `SEMANTIC_REMOTE_FALLBACK`

Local development layout used here:

- source repo:
  - `/Users/serhiizhyhun/Desktop/projects/picaivid/third_party/sam2`
- checkpoint:
  - `/Users/serhiizhyhun/Desktop/projects/picaivid/picaivid-media-service/ml_models/sam2/sam2.1_hiera_small.pt`
- config:
  - `/Users/serhiizhyhun/Desktop/projects/picaivid/third_party/sam2/sam2/configs/sam2.1/sam2.1_hiera_s.yaml`

For AWS, store the SAM2 repo archive and checkpoint in S3/artifact storage, hydrate them to local disk at boot, and load only from local paths during runtime.

For production bootstrapping, prefer setting the `*_S3_URI` env vars and stable local target paths under the instance model cache directory.

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
