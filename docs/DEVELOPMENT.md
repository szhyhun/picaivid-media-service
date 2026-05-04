# Development

## Local workflow

```bash
cd picaivid-media-service
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

## Useful checks

```bash
./venv/bin/python -m py_compile app/main.py app/pipeline/phase1_analyze/clustering.py app/pipeline/phase1_analyze/vggt_pipeline.py app/models/vggt.py
git diff --check
```

## What to verify after phase-1 changes

- scene components are stable
- ordered photo traversal is sensible
- bridge/outlier roles are reasonable
- motion decisions match geometry confidence
- `/api/projects/:id/scenes/debug` and `/api/projects/:id/relations/debug` stay in sync with the React UI
