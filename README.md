# Picaivid Media Service

FastAPI + worker service for photo analysis, clustering, transition scoring, and video clip generation.

## Current Runtime Behavior

### Legacy semantic embedding backend

Phase 1 no longer uses DINO as the matching/reconstruction engine. The active phase-1 path is MASt3R-only.

DINO remains documented here only as a temporary auxiliary analyzer/debug artifact until the legacy analyzer cleanup is complete:

- local artifact path: `/Users/serhiizhyhun/Desktop/projects/picaivid/third_party/dinov3-vitb16-pretrain-lvd1689m`
- expected artifact: `facebook/dinov3-vitb16-pretrain-lvd1689m`
- the snapshot must contain the real `model.safetensors`, not a Git LFS pointer file
- if the local artifact is missing, fix artifact hydration; do not use live model download

### Semantic region backend

- Semantic regions use `SAM2 + OpenCLIP` when SAM2 is available locally.
- If SAM2 is unavailable or fails, the pipeline falls back to heuristic anchor/window/background regions.
- On Apple Silicon, SAM2 runs on `CPU` intentionally because the required path is not reliable on `MPS`.
- SAM2 is expensive. Regions are cached per photo per run, but full-project analysis still slows down materially on a laptop.
- The semantic labeler now uses room-specific allowed/forbidden labels plus bbox sanity checks, not raw CLIP prompts alone.
- Current semantic labels include:
  - anchor/furniture: `bed`, `table`, `chair`, `seating_cluster`, `sofa`, `island_or_counter`, `vanity`, `desk`, `appliance_core`, `fireplace`
  - bathroom: `toilet`, `shower`, `bath`, `mirror`
  - secondary objects: `plant`, `cabinet`, `artwork`, `rug`, `tv`, `lighting_fixture`, `shelving`, `counter_stool`, `door`
  - openings/background: `window`, `glass_door`, `wall`, `floor`, `ceiling`, `sky`
  - outdoor areas: `patio`, `deck`, `balcony`

### Phase 1 clustering policy

Phase 1 is now a **MASt3R-only** graph pipeline.

Current behavior:

- obvious duplicates are removed **before** MASt3R retrieval
- utility rooms are removed **before** MASt3R retrieval
- MASt3R retrieval builds the scene graph
- MASt3R pair inference + sparse global alignment produce edge quality and component state
- final user-facing clusters are capped at **2 photos**
- unmatched photos remain singleton clusters
- additional same-component photos are exposed only as **debug suggestions**

This is deliberate: the system is optimizing for strongest parallax pairs, not large connected components.

Current phase-1 flow:

```text
images
-> duplicate filter
-> utility-room filter
-> MASt3R retrieval graph
-> MASt3R pair inference
-> sparse global alignment
-> edge scoring
-> best disjoint pair selection
-> final 2-photo clusters + singleton leftovers
```

Main pair score inputs:

- retrieval overlap strength
- reciprocal match count
- pointmap consistency
- alignment residual / reprojection quality
- parallax score
- upload-order proximity

CLIP/SAM fine labels are no longer part of acceptance logic in the MASt3R path.

## Pair Debug

Pair debug is now MASt3R-native.

It exposes:

- match engine
- retrieval score
- reciprocal match count
- pointmap consistency
- alignment residual
- reprojection error
- parallax score
- graph edge score
- graph component id when a stored row exists
- MASt3R timing

Cluster debug also exposes:

- final selected pair
- same-component suggested photos
- per-suggestion score
- exclusion reason:
  - `lower than chosen edge`
  - `already consumed by a stronger pair`
  - `failed pair safety threshold`

Semantic overlays remain available for auxiliary inspection, but the active pair-debug matcher path is MASt3R.

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

## Deployment Note (MASt3R)

Phase 1 now requires a **CUDA GPU**. CPU workers should not run phase 1.
Validate with logs:

- `Loaded MASt3R model checkpoint=... device=cuda`
- `Loaded MASt3R retriever checkpoint=...`
- `pair_debug_timing ... model_device=cuda ... preferred_device=cuda`

Startup warmup behavior:

- API and GPU worker startup now warm **MASt3R only**
- legacy semantic/depth models (`DINO`, `OpenCLIP`, `MiDaS`) are no longer preloaded during normal startup
- those legacy models are now lazy-loaded only if an auxiliary legacy/debug path explicitly touches them
- this avoids paying old startup cost and avoids carrying stale legacy runtime state into the MASt3R main path

Required MASt3R local artifacts:

- `MAST3R_REPO_DIR`
- `MAST3R_MODEL_CHECKPOINT`
- `MAST3R_RETRIEVAL_CHECKPOINT`
- `MAST3R_RETRIEVAL_CODEBOOK`

Recommended local layout:

- repo:
  - `/Users/serhiizhyhun/Desktop/projects/picaivid/third_party/mast3r`
- checkpoints under that repo:
  - `checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`
  - `checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth`
  - `checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl`

### AWS Asset Hydration Rule

Do not depend on workstation-specific absolute paths in cloud environments.

On AWS rollout, all MASt3R artifacts and model caches must be hydrated from S3 into local disk before they are used. Do not rely on live internet downloads in staging or production.

The current staging artifact layout is:

- `s3://picaivid-staging-media/artifacts/mast3r`
- `s3://picaivid-staging-media/artifacts/dinov3-vitb16-pretrain-lvd1689m` temporarily, until DINO analyzer dependencies are fully removed
- `s3://picaivid-staging-media/artifacts/sam2` only if SAM2 remains enabled

Upload and hydrate the DINO snapshot only while legacy auxiliary/debug paths still need it. It is not a MASt3R fallback and it is not an active phase-1 matcher.

For MASt3R, the loader supports S3 hydration directly:

- repo archive:
  - `MAST3R_REPO_ARCHIVE_S3_URI`
- checkpoint objects:
  - `MAST3R_MODEL_CHECKPOINT_S3_URI`
  - `MAST3R_RETRIEVAL_CHECKPOINT_S3_URI`
  - `MAST3R_RETRIEVAL_CODEBOOK_S3_URI`

Behavior:

- if the configured local repo/checkpoint path exists, the loader uses it
- if the local path is missing and the matching `*_S3_URI` is set, the loader downloads/extracts the asset from S3 into the configured local path
- if neither local asset nor S3 fallback exists, MASt3R loading fails fast with a clear error
- if an auxiliary legacy DINO path is touched and the local DINO artifact is missing, the service fails that path instead of downloading from the internet
- for AWS, use `local -> S3 hydrate -> load locally`; live URL fallback is not a runtime path

For semantic region extraction, SAM2 should follow the same policy:

- local repo:
  - `SAM2_REPO_DIR`
- local checkpoint:
  - `SAM2_CHECKPOINT`
- local config:
  - `SAM2_CONFIG`

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
