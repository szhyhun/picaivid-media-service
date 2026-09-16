# Picaivid Media Service

FastAPI + worker service for geometry-first photo analysis and clip generation.

## Current Direction

Phase 1 is now **VGGT-first**.

VGGT is the core intelligence layer for this product. The primary product remains cinematic video generation from listing photos. Any future `tour_3d` or splat-based interactive experience is a secondary track that should build on the VGGT-centered pipeline rather than replace it.

The service reconstructs project geometry from the full photo set, derives scene components, orders photos within each component, and feeds geometry-backed render clusters into phase 2. Pairwise relations still exist, but only as derived support data.

Primary outputs:

- per-photo camera pose and intrinsics
- per-photo depth and point-map artifacts
- scene components with ordered photos
- derived photo relations
- geometry-driven motion decisions

## Local Run

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Worker:

```bash
WORKER_TYPE=gpu WORKER_PHASES=1 ./venv/bin/python -m app.worker
```

### Runware credentials

For local testing, put the key in the untracked `picaivid-media-service/.env`:

```dotenv
RUNWARE_API_KEY=your-runware-key
RUNWARE_MODEL=luma:ray@3.2
RUNWARE_DURATION_SECONDS=5
RUNWARE_RESOLUTION=720p
RUNWARE_HDR=true
```

Never put this key in the React `.env.local`, source control, or a prompt. On
AWS, set the same values in `/etc/picaivid/media-worker.env` (and
`media-api.env` if the media API handles generation), then restart the service.
The client rejects any model other than `luma:ray@3.2` and rejects invalid HDR,
resolution, or duration combinations before submitting a paid task.

The full current manual startup order is documented in the workspace
[`LOCAL_RUN.md`](../LOCAL_RUN.md).

For reproducible installs:

```bash
pip install -r requirements.lock.txt
```

## Runtime Notes

- Phase 1 selects CUDA, then Apple Silicon MPS, then CPU. CUDA is used for the on-demand worker; MPS is supported for local review.
- Dense geometry artifacts belong in S3, not Postgres.
- Synthetic geometry is disabled. Every analysis requires the pinned VGGT-Omega repository and external 512 checkpoint.
- The Omega checkpoint is never committed. Workers hydrate the repository and checkpoint from configured local or S3 artifacts.

## Main Endpoints

- `GET /health`
- `GET /api/projects/{project_id}/clips`
- `GET /api/projects/{project_id}/scenes/debug`
- `GET /api/projects/{project_id}/shot_plan`
- `GET /api/projects/{project_id}/timeline`
- `POST /api/projects/{project_id}/timeline/generate`
- `PUT /api/projects/{project_id}/timeline/music`
- `POST /api/projects/{project_id}/timeline/assemble`
- `POST /api/projects/{project_id}/relations/debug`
- `POST /internal/jobs`

## Core Docs

- [docs/ARCHITECTURE.md](/Users/serhiizhyhun/Desktop/projects/picaivid/picaivid-media-service/docs/ARCHITECTURE.md)
- [docs/CLUSTERING.md](/Users/serhiizhyhun/Desktop/projects/picaivid/picaivid-media-service/docs/CLUSTERING.md)
- [docs/DEVELOPMENT.md](/Users/serhiizhyhun/Desktop/projects/picaivid/picaivid-media-service/docs/DEVELOPMENT.md)
- [docs/GOLDEN_REVIEW.md](/Users/serhiizhyhun/Desktop/projects/picaivid/picaivid-media-service/docs/GOLDEN_REVIEW.md)
- [docs/AWS_SETUP.md](/Users/serhiizhyhun/Desktop/projects/picaivid/picaivid-media-service/docs/AWS_SETUP.md)
- [docs/AWS_DEPLOYMENT.md](/Users/serhiizhyhun/Desktop/projects/picaivid/picaivid-media-service/docs/AWS_DEPLOYMENT.md)

## Reference

- PlayCanvas SuperSplat and related splat tooling are reference material for a possible later `tour_3d` side project, not the main product direction:
  - [playcanvas/supersplat](https://github.com/playcanvas/supersplat)
  - [playcanvas/model-viewer](https://github.com/playcanvas/model-viewer)
