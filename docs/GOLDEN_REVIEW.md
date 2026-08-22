# Phase 1 Golden Review

Use owned listing photos only. Golden media and generated JSON are local review assets and are not
committed to this repository.

Create six listings with these coverage profiles:

1. interior-heavy
2. exterior plus drone
3. luxury with outdoor amenities
4. multi-floor home
5. repeated rooms and near-duplicates
6. sparse or no-overlap photos

For each listing, run the model on 4, 12, and approximately 50 photos when available:

```bash
./venv/bin/python scripts/run_vggt_phase1_smoke.py --images ~/Listings/<listing> --count 4 --output /tmp/<listing>-4.json
./venv/bin/python scripts/run_vggt_phase1_smoke.py --images ~/Listings/<listing> --count 12 --output /tmp/<listing>-12.json
./venv/bin/python scripts/run_vggt_phase1_smoke.py --images ~/Listings/<listing> --count 50 --output /tmp/<listing>-50.json
```

Before enabling the on-demand GPU worker, review the JSON and scene-debug API against these gates:

- at least 95% interpolation precision
- at least 85% valid-transition recall
- 100% compliance with explicit opening, hero, closing, and exclude roles
- no unrelated-room interpolation
- repeat runs produce identical ordered plans
- an editor approves at least five of six complete story plans

Record any failed relation using its saved geometry, RoMa, and track evidence. Do not compensate
with synthetic geometry or hand-edit a result silently.
