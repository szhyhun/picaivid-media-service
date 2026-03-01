# Clustering Baselines

This folder stores filename-level clustering baselines used for regression checks.

Current baseline:
- `project_7dc060b2-f2ef-483b-87b2-bc4f2ccb4273_baseline.json`
- Source job: `56`

## Export a baseline

```bash
cd /Users/serhiizhyhun/Desktop/projects/picaivid/picaivid-media-service
./venv/bin/python scripts/cluster_baseline_guard.py export \
  --project-id 7dc060b2-f2ef-483b-87b2-bc4f2ccb4273 \
  --output scripts/baselines/project_7dc060b2-f2ef-483b-87b2-bc4f2ccb4273_baseline.json
```

## Compare a new run against baseline

```bash
cd /Users/serhiizhyhun/Desktop/projects/picaivid/picaivid-media-service
./venv/bin/python scripts/cluster_baseline_guard.py compare \
  --baseline scripts/baselines/project_7dc060b2-f2ef-483b-87b2-bc4f2ccb4273_baseline.json \
  --project-id <new_project_uuid>
```

## Use as guardrail (non-zero exit when drift is high)

```bash
cd /Users/serhiizhyhun/Desktop/projects/picaivid/picaivid-media-service
./venv/bin/python scripts/cluster_baseline_guard.py compare \
  --baseline scripts/baselines/project_7dc060b2-f2ef-483b-87b2-bc4f2ccb4273_baseline.json \
  --project-id <new_project_uuid> \
  --fail-on-f1-below 0.92
```

## Kornia Oracle A/B modes

Set in `picaivid-media-service/.env` (or shell env) before running API/worker:

```bash
# Baseline (default)
KORNIA_ORACLE_MODE=off

# A-mode: observe only (records oracle mode in pair_source as |koS)
KORNIA_ORACLE_MODE=shadow

# B-mode: enforce oracle gate for geometric edges (pair_source |koG / |koR)
KORNIA_ORACLE_MODE=gate

# Optional thresholds
KORNIA_ORACLE_MIN_OVERLAP_RATIO=0.08
KORNIA_ORACLE_MIN_SIDE_OVERLAP=0.06
KORNIA_ORACLE_MIN_INLIER_RATIO=0.20
KORNIA_ORACLE_INLIER_THRESHOLD_PX=2.0
```

Recommended A/B loop:

```bash
# 1) Run one job with KORNIA_ORACLE_MODE=off
# 2) Run same photo set with KORNIA_ORACLE_MODE=shadow
# 3) Run same photo set with KORNIA_ORACLE_MODE=gate
# 4) Compare each run against baseline by project/job
./venv/bin/python scripts/cluster_baseline_guard.py compare \
  --baseline scripts/baselines/project_7dc060b2-f2ef-483b-87b2-bc4f2ccb4273_baseline.json \
  --project-id <project_uuid>
```
