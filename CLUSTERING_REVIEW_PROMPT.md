# Clustering Pipeline Review — AI Agent Prompt

## Mission

You are reviewing and fixing the photo clustering pipeline for a real estate video generation service.
The pipeline takes listing photos (e.g., 50 photos of a house), groups them into clusters of **max 3 photos** representing the same camera position/angle, and orders them for smooth video transitions.

Goal: Make clustering **100% consistent** — no false merges between different rooms/locations, no missed merges for photos that clearly belong together, no clusters exceeding 3 photos, and correct photo ordering within every cluster.

---

## Project Setup

### Directory Structure

```
picaivid-media-service/
├── app/
│   ├── main.py                    # FastAPI app (API server)
│   ├── worker.py                  # SQS worker entry point
│   ├── pipeline/
│   │   └── phase1_analyze/
│   │       ├── learned_matching.py  ← MAIN CLUSTERING FILE (1,400+ lines)
│   │       └── clustering.py        ← Clustering orchestrator
│   ├── db/
│   │   └── models/
│   │       ├── photo.py             # JobPhoto model
│   │       ├── photo_similarity.py  # PhotoSimilarity model
│   │       ├── clip.py              # Clip model (= cluster result)
│   │       └── room_cluster.py      # RoomCluster model
│   └── services/sqs/consumer.py    # SQS consumer
├── scripts/
│   ├── test_cross_cluster_merge.py  ← CLUSTERING TESTS
│   └── analyze_photo_pairs.py       ← Pair analysis tool
└── MANUAL_TESTING.md                ← Full setup guide
```

### Start Infrastructure (Docker)

All external services run via Docker Compose from the Rails repo:

```bash
cd ../picaivid-rails
docker-compose up -d

# Verify:
docker-compose ps
# Expected: postgres, minio, localstack all "Up (healthy)"

# Check SQS queue exists:
aws --endpoint-url=http://localhost:4566 --region us-east-1 sqs list-queues
# If missing, create it:
aws --endpoint-url=http://localhost:4566 --region us-east-1 sqs create-queue --queue-name picaivid-jobs
```

### Activate Python Environment

```bash
cd picaivid-media-service
source venv/bin/activate
```

### Run API Server (auto-reloads on code change)

```bash
uvicorn app.main:app --reload --port 8000
```

### Run SQS Worker (must manually restart after code changes)

```bash
python -m app.worker

# Restart worker after code changes:
kill $(pgrep -f "app.worker")
python -m app.worker
```

---

## Running Clustering Tests

```bash
source venv/bin/activate

# Run all test sets
python scripts/test_cross_cluster_merge.py --test-set all

# Run specific test set
python scripts/test_cross_cluster_merge.py --test-set 1   # Photo IDs 569-705
python scripts/test_cross_cluster_merge.py --test-set 2   # Photo IDs 861-912
python scripts/test_cross_cluster_merge.py --test-set 3   # Photo IDs 805-815

# List jobs in DB (check which photo IDs are available)
python scripts/test_cross_cluster_merge.py --list-jobs
```

Tests validate:
- **Same cluster**: Expected photos that should be grouped together
- **Different clusters**: Photos that must NOT be merged (different rooms/patios)
- **Ordering**: Which photo should be at the start (endpoint) of a cluster
- **Sequence**: Photos appear consecutively in the right order

---

## Querying the Database

The media service uses its own PostgreSQL database (not shared with Rails).

### Quick DB inspection (paste into terminal)

```bash
source venv/bin/activate
python - <<'EOF'
from app.db.session import SessionLocal
from app.db.models import Job, JobPhoto, PhotoSimilarity, Clip

db = SessionLocal()

# Latest job
job = db.query(Job).order_by(Job.created_at.desc()).first()
print(f"Job {job.id}: status={job.status}, project={job.project_id}")
print()

# All clusters (clips)
clips = db.query(Clip).filter(Clip.job_id == job.id).all()
print(f"=== {len(clips)} Clusters ===")
for clip in clips:
    print(f"  Clip {clip.id}: photos={clip.source_photo_ids}")
print()

# All photos with room labels
photos = db.query(JobPhoto).filter(JobPhoto.job_id == job.id).order_by(JobPhoto.position).all()
print(f"=== {len(photos)} Photos ===")
for p in photos:
    print(f"  [{p.position:2d}] Photo {p.id}: room={p.room_label}, score={p.final_score:.2f}")

db.close()
EOF
```

### Inspect similarity between specific photos

```bash
python - <<'EOF'
from app.db.session import SessionLocal
from app.db.models import PhotoSimilarity

db = SessionLocal()
PHOTO_IDS = [1426, 1427, 1428]  # ← Replace with photos you're investigating

for i in PHOTO_IDS:
    for j in PHOTO_IDS:
        if i >= j:
            continue
        sim = db.query(PhotoSimilarity).filter(
            PhotoSimilarity.photo_a_id == i,
            PhotoSimilarity.photo_b_id == j
        ).first()
        if sim:
            print(f"{i} <-> {j}: dinov2={sim.dinov2_similarity:.3f}, "
                  f"inliers={sim.geometric_inliers}, source={sim.pair_source}, "
                  f"connected={sim.is_connected}")
        else:
            print(f"{i} <-> {j}: NOT COMPUTED (pair wasn't evaluated)")
db.close()
EOF
```

### Direct PostgreSQL (via Docker)

```bash
docker exec -it picaivid-rails-postgres-1 psql -U postgres -d picaivid_development

-- Latest job
SELECT id, status, project_id FROM jobs ORDER BY created_at DESC LIMIT 1;

-- All clusters for job 33
SELECT id, source_photo_ids FROM clips WHERE job_id = 33;

-- Photo room labels
SELECT id, position, room_label, final_score FROM job_photos WHERE job_id = 33 ORDER BY position;

-- Similarity pairs
SELECT photo_a_id, photo_b_id, dinov2_similarity, geometric_inliers, is_connected
FROM photo_similarities WHERE job_id = 33 ORDER BY photo_a_id;
```

---

## Architecture: How Clustering Works

File: `app/pipeline/phase1_analyze/learned_matching.py`

### 3-Stage Pipeline

**Stage 1 — Semantic (DINOv2 embeddings):**
- Each photo → 768-dim feature vector
- Top-K most similar photos become candidate pairs (K=4 per photo)
- Also adds photos within ±2 upload positions (temporal window) as candidate pairs

**Stage 2a — Temporal pairs (adjacent photos):**
- For photos within ±2 positions of each other
- If semantic similarity ≥ 0.88 → connect (trust semantic, still run geometry for direction)
- If semantic similarity ≥ 0.60 → run geometric verification, connect if ≥ 15 inliers
- If adjacent (gap=1) and same room label → trust even with ≥ 0.65 semantic
- If different room labels and adjacent → need ≥ 15 inliers (not 30)
- If different room labels and non-adjacent → skip entirely

**Stage 2b — Non-temporal pairs (from DINOv2 top-K):**
- Different room labels → skip (no cross-room merging)
- Same room, position gap ≥ 3 → need ≥ 25 inliers
- Same room, position gap < 3 → need ≥ 15 inliers

**Connected Components:**
- Build graph from connected pairs
- Each connected component = one raw cluster

**Ordering within each cluster:**
- Find spatial endpoints (photos that "face outward")
- Build direction-consistent chain from endpoint
- Unconnectable photos become isolated mini-clusters

**Deduplication + Split:**
- Remove consecutive same-angle shots (semantic similarity ≥ 0.94)
- If cluster > 3 photos after dedup → split at weakest transition point
- Result: multiple clusters of ≤ 3 photos each

### Key Constants

```python
MIN_INLIERS_FOR_OVERLAP = 15          # Base threshold to connect two photos
MIN_INLIERS_CROSS_ROOM = 30           # Non-adjacent cross-room pairs
MIN_INLIERS_CROSS_ROOM_ADJACENT = 15  # Adjacent cross-room (ML labels often wrong)
POSITION_GAP_THRESHOLD = 3            # Photos this far apart need stronger evidence
MIN_INLIERS_FAR_APART = 25            # Required inliers when gap ≥ 3
TEMPORAL_WINDOW = 2                   # Check ±2 positions as candidate pairs
TEMPORAL_SEMANTIC_THRESHOLD = 0.88    # Trust semantic alone (no geometric needed)
NEIGHBOR_TRUST_THRESHOLD = 0.65       # Trust immediate neighbor even with weak geometry
DUPLICATE_SIMILARITY_THRESHOLD = 0.94 # Mark as same-angle duplicate
MAX_CLUSTER_SIZE = 3                  # Hard max photos per cluster
```

---

## Known Clustering Issues

These are real production failures from recent jobs. Fix them all.

### Issue 1: Cluster exceeds 3 photos
**Observed:** Clip with photos `[1415, 1416, 1417, 1418]` — 4 photos in one cluster.
**Expected:** Split into 2 clusters of 2 (or cluster of 3 + singleton).
**Why it happened:** `deduplicate_and_split_cluster()` was recently changed to return `List[List[int]]` but old code path may not call it correctly.
**Where to look:** `cluster_photos_graph_based()` in `learned_matching.py` — check that `deduplicate_and_split_cluster()` return value is used (not a single list).

### Issue 2: Adjacent same-room photos not connected (ML mislabeling)
**Observed:** Photos `1426` (living room), `1427` (dining room), `1428` (living room) — positions 25, 26, 27.
- 1426 ↔ 1428 are connected (24 inliers, same room label) → Clip [1426, 1428]
- 1427 is isolated → Clip [1427]
**Reality:** All three are actually dining room photos. The ML room classifier mislabeled 1426 and 1428.
**Root cause:** Cross-room pairs between adjacent photos needed 30 inliers — 1426↔1427 had only 8, 1427↔1428 had only 16.
**Fix:** `MIN_INLIERS_CROSS_ROOM_ADJACENT = 15` (already added to constants). Verify it's being used in Stage 2a for adjacent cross-room pairs.

### Issue 3: Different physical locations with same room label incorrectly merged
**Observed:** Photos `1447` (patio, position 46) and `1452` (patio, position 51) — 5 positions apart — clustered together.
**Reality:** These are from two completely different outdoor patios. They shouldn't be in the same cluster.
**Data:** dinov2=0.492, inliers=17, connected=1 (source: dinov2_topk)
**Root cause:** Both labeled "patio", 17 inliers ≥ 15 (old threshold), position gap = 5.
**Fix:** `MIN_INLIERS_FAR_APART = 25` and `POSITION_GAP_THRESHOLD = 3` (already added). Verify Stage 2b applies this threshold.

### Issue 4: Photo ordering wrong (start/middle/end positions confused)
**Observed:** Photos within a cluster appear in wrong spatial order — video would show camera "jumping" instead of panning smoothly.
**Root cause:** The endpoint detection (which photo to start from) or direction-consistency check may fail.
**Where to look:** `order_cluster_for_transitions()` in `learned_matching.py` — the `endpoint_scores` computation and greedy pathfinding.

### Issue 5: TypeError crash (fixed, verify)
**Was:** `order_cluster_for_transitions()` early returns (len ≤ 1, len == 2) returned `List[int]` but function now returns `Tuple[List[int], List[int]]`.
**Fix applied:** Early returns changed to `return ([...], [])`.
**Verify:** Lines ~353-357 in `learned_matching.py`.

---

## What "Consistent Clustering" Means

A correctly clustered listing should satisfy all of the following:

1. **Max 3 photos per cluster** — Hard rule. Never 4+.
2. **No cross-room merges** — Kitchen photos never with patio, bedroom never with living room. Exception: adjacent photos where ML label might be wrong (allow if 15+ geometric inliers).
3. **No different-location merges** — Two different patios, two different bedrooms → separate clusters, even if ML labels match.
4. **Smooth ordering** — Photos within a cluster ordered so each consecutive pair has clear geometric overlap (camera panning, not jumping).
5. **Direction consistency** — Camera motion should be in one direction within a cluster (left→right or right→left, not both).
6. **Correct endpoint detection** — Cluster should start from the spatial "edge" photo, not from the middle.
7. **No duplicate photos** — If two photos are nearly identical (same angle, same position), keep only one.
8. **Isolated photos are OK** — A single-photo cluster is valid if that photo doesn't match anything else well enough.

---

## Investigation Workflow

When you see a clustering problem, follow this workflow:

### Step 1: Identify the photos in question
```python
# Find which cluster each photo ended up in, and its room label
for pid in [1426, 1427, 1428]:
    photo = db.query(JobPhoto).filter(JobPhoto.id == pid).first()
    print(f"Photo {pid}: room={photo.room_label}, position={photo.position}")
```

### Step 2: Check similarity scores between them
```python
# Check if they were even evaluated as a pair
sim = db.query(PhotoSimilarity).filter(
    PhotoSimilarity.photo_a_id == min(pid_a, pid_b),
    PhotoSimilarity.photo_b_id == max(pid_a, pid_b)
).first()
print(f"dinov2={sim.dinov2_similarity:.3f}, inliers={sim.geometric_inliers}, connected={sim.is_connected}")
```

### Step 3: Trace the decision in the algorithm
- Was the pair even considered? Check `pair_source` (temporal vs dinov2_topk)
- What was the threshold applied? Check constants against the pair's position gap and room labels
- Was the threshold met? Compare `geometric_inliers` to the threshold that should apply

### Step 4: Fix the threshold or logic, restart worker, re-run job

---

## Files to Modify

- **`app/pipeline/phase1_analyze/learned_matching.py`** — All clustering logic, thresholds, ordering
- **`app/pipeline/phase1_analyze/clustering.py`** — Orchestrator (usually don't need to change)
- **`scripts/test_cross_cluster_merge.py`** — Add new test cases for any new issues you find

## Files to NOT modify
- Database models (unless adding a column for debugging)
- API endpoints
- SQS consumer
- Phase 2/3/4 pipeline

---

## Expected Test Results

After all fixes, `python scripts/test_cross_cluster_merge.py --test-set all` should pass with 0 failures.

Key things to verify manually (not in automated tests yet):
- Job 33: Clip [1415, 1416, 1417, 1418] should be split into 2 clusters
- Job 33: Photos 1426, 1427, 1428 should be in the same cluster
- Job 33: Photos 1447 and 1452 should be in different clusters
