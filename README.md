# Picaivid Media Service

FastAPI + worker service for geometry-first photo analysis and clip generation.

## Current Direction

Phase 1 is now **VGGT-first**.

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
source venv/bin/activate
python -m app.worker
```

For reproducible installs:

```bash
pip install -r requirements.lock.txt
```

## Runtime Notes

- GPU phase 1 requires CUDA.
- Dense geometry artifacts belong in S3, not Postgres.
- Local development may use the synthetic VGGT fallback when the repo/checkpoint are not present.
- Production and AWS should use the hydrated VGGT commercial checkpoint only.

## Main Endpoints

- `GET /health`
- `GET /api/projects/{project_id}/clips`
- `GET /api/projects/{project_id}/scenes/debug`
- `POST /api/projects/{project_id}/relations/debug`
- `POST /internal/jobs`

## Core Docs

- [docs/ARCHITECTURE.md](/Users/serhiizhyhun/Desktop/projects/picaivid/picaivid-media-service/docs/ARCHITECTURE.md)
- [docs/CLUSTERING.md](/Users/serhiizhyhun/Desktop/projects/picaivid/picaivid-media-service/docs/CLUSTERING.md)
- [docs/DEVELOPMENT.md](/Users/serhiizhyhun/Desktop/projects/picaivid/picaivid-media-service/docs/DEVELOPMENT.md)
- [docs/AWS_SETUP.md](/Users/serhiizhyhun/Desktop/projects/picaivid/picaivid-media-service/docs/AWS_SETUP.md)
- [docs/AWS_DEPLOYMENT.md](/Users/serhiizhyhun/Desktop/projects/picaivid/picaivid-media-service/docs/AWS_DEPLOYMENT.md)
