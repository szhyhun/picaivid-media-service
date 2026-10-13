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
uvicorn app.main:app --host 127.0.0.1 --reload --port 8000
```

The HTTP process is an internal Rails-facing API, not a browser API. Production
workers are separate SQS processes: `python -m app.workers.gpu` for VGGT and
GPU clip work, and `python -m app.workers.cpu` for Runware and FFmpeg. Neither
worker has an HTTP listener.

Worker:

```bash
WORKER_TYPE=gpu WORKER_PHASES=1 ./venv/bin/python -m app.workers.gpu
```

The CPU worker uses the same codebase with `WORKER_TYPE=cpu` and consumes
`SQS_CPU_QUEUE_URL` for timeline generation and FFmpeg assembly. GPU and CPU
workers are separate entrypoints and deployments; they share only the common
database, storage, queue, and webhook modules. Neither worker is a
browser-facing app.

### Internal service and Runware credentials

Rails is the only caller of the optional media API. For local testing, put the
internal service keys and the Runware key in the untracked
`picaivid-media-service/.env`:

```dotenv
MEDIA_SERVICE_API_KEY=dev-secret-key
MEDIA_SERVICE_WEBHOOK_KEY=dev-webhook-key
RUNWARE_API_KEY=your-runware-key
RUNWARE_MODEL=luma:ray@3.2
RUNWARE_DURATION_SECONDS=5
RUNWARE_RESOLUTION=720p
RUNWARE_HDR=true
RUNWARE_SUBMIT_CONCURRENCY=4
RUNWARE_MAX_WAIT_SECONDS=1800
```

Never put the Runware key in the React `.env.local`, source control, an SQS
message, or a prompt. On AWS, put `MEDIA_SERVICE_API_KEY` in
`/etc/picaivid/media-api.env`, and put `MEDIA_SERVICE_WEBHOOK_KEY` plus
`RUNWARE_API_KEY` in `/etc/picaivid/media-cpu-worker.env`. The API does not
perform paid generation; the CPU worker consumes Rails SQS commands and calls
Runware.
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
- `GET /api/projects/{project_id}/timelines`
- `GET/PUT /api/projects/{project_id}/timeline?timeline_id=...`
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
