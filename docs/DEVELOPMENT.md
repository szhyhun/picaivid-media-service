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

## Apple Silicon VGGT validation

VGGT-Omega-1B-512 is the reconstruction model. Keep its checkpoint outside the repository and set
`VGGT_MODEL_CHECKPOINT` to its absolute path; it must never be committed. The official repository
is pinned at `../third_party/vggt-omega` (currently
`282ec70363edeff59424bf43731658092fba3d37`).

```bash
./venv/bin/python - <<'PY'
import torch
assert torch.backends.mps.is_available(), 'Install an Apple Silicon PyTorch build first'
x = torch.ones((2, 2), device='mps')
assert x.device.type == 'mps' and x.sum().cpu().item() == 4
print('MPS tensor smoke test passed')
PY
./venv/bin/python -m unittest discover -s tests
```

Run real 4, 12, and approximately 50 photo listings only with owned images. Capture the
scene-debug and shot-plan JSON for golden review; synthetic geometry is disabled outside unit tests.
Use [`GOLDEN_REVIEW.md`](GOLDEN_REVIEW.md) and `scripts/run_vggt_phase1_smoke.py` for the exact
local commands and release gates.

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
- `/api/projects/:id/scenes/debug`, `/api/projects/:id/relations/debug`, and `/api/projects/:id/shot_plan` stay in sync with the React UI
