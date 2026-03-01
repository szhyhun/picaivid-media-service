# Clustering Baselines

Use this folder to store filename-level baseline clusterings and detect drift.

Current baseline:
- `project_7dc060b2-f2ef-483b-87b2-bc4f2ccb4273_baseline.json`

## Export

```bash
cd /Users/serhiizhyhun/Desktop/projects/picaivid/picaivid-media-service
./venv/bin/python scripts/cluster_baseline_guard.py export \
  --project-id 7dc060b2-f2ef-483b-87b2-bc4f2ccb4273 \
  --output scripts/baselines/project_7dc060b2-f2ef-483b-87b2-bc4f2ccb4273_baseline.json
```

## Compare

```bash
./venv/bin/python scripts/cluster_baseline_guard.py compare \
  --baseline scripts/baselines/project_7dc060b2-f2ef-483b-87b2-bc4f2ccb4273_baseline.json \
  --project-id <project_uuid>
```

## Guardrail (CI/local)

```bash
./venv/bin/python scripts/cluster_baseline_guard.py compare \
  --baseline scripts/baselines/project_7dc060b2-f2ef-483b-87b2-bc4f2ccb4273_baseline.json \
  --project-id <project_uuid> \
  --fail-on-f1-below 0.92
```

## Oracle Modes

Set `KORNIA_ORACLE_MODE` before running worker:
- `off` (baseline)
- `shadow` (observe only)
- `gate` (enforced)
