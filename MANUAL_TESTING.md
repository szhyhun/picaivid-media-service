# Phase 1 Manual Testing Guide

This guide walks through setting up and testing the Phase 1 pipeline locally on macOS.

## Prerequisites

- Python 3.11+ (installed via `brew install python@3.11`)
- Docker (for PostgreSQL, MinIO, and LocalStack)
- AWS CLI (`brew install awscli`)

## Step 1: Start Infrastructure

```bash
# From the Rails directory
cd picaivid-rails

# Start Postgres, MinIO (S3), and LocalStack (SQS)
docker-compose up -d

# Verify services are running
docker-compose ps
```

Expected output:
```
NAME                        STATUS
picaivid-rails-postgres-1   Up
picaivid-rails-minio-1      Up (healthy)
picaivid-rails-localstack-1 Up (healthy)
```

## Step 2: Verify LocalStack SQS

```bash
# Check if queue was created (note: --region is required)
aws --endpoint-url=http://localhost:4566 --region us-east-1 sqs list-queues
```

If no queues are listed, create the queue manually:
```bash
aws --endpoint-url=http://localhost:4566 --region us-east-1 sqs create-queue --queue-name picaivid-jobs
```

Verify:
```bash
aws --endpoint-url=http://localhost:4566 --region us-east-1 sqs list-queues
# Should show: http://sqs.us-east-1.localhost.localstack.cloud:4566/000000000000/picaivid-jobs
```

## Step 3: Set Up Python Environment

```bash
cd picaivid-media-service

# Activate virtual environment
source venv/bin/activate

# Install dependencies (first time only)
pip install -r requirements.txt
```

## Step 4: Run Database Migrations

**Important:** Make sure you've activated the virtual environment first!

```bash
# Activate venv if not already active
source venv/bin/activate

# Apply migrations
alembic upgrade head
```

Verify tables were created:
```bash
# Using Docker's postgres container
docker exec -it picaivid-rails-postgres-1 psql -U postgres -d picaivid_development -c "\dt"
```

Expected tables: `jobs`, `job_photos`, `room_clusters`, `analysis_results`, `clips`, `timelines`, `timeline_clips`

## Step 5: Set Up Rails

```bash
cd picaivid-rails

# Install new gem
bundle install

# Run Rails migrations if needed
rails db:migrate
```

## Step 6: Create Test Data in Rails

```bash
rails console
```

```ruby
# Create a user and project with photos
user = User.create!(email: "test@example.com", password: "password123")
project = Project.create!(user: user, name: "Test Property", status: :photos_uploaded)

# Create photo records
3.times do |i|
  Photo.create!(
    project: project,
    filename: "room_#{i}.jpg",
    s3_object_key: "photos/test_room_#{i}.jpg",
    status: :ready,
    position: i
  )
end

puts "Created project: #{project.id}"
```

## Step 7: Upload Test Images to MinIO

Access MinIO console at http://localhost:9001 (login: minioadmin / minioadmin)

Or via CLI:
```bash
# Install MinIO client
brew install minio/stable/mc

# Configure
mc alias set local http://localhost:9000 minioadmin minioadmin

# Create bucket and upload
mc mb local/picaivid-dev
mc cp /path/to/image1.jpg local/picaivid-dev/photos/test_room_0.jpg
mc cp /path/to/image2.jpg local/picaivid-dev/photos/test_room_1.jpg
mc cp /path/to/image3.jpg local/picaivid-dev/photos/test_room_2.jpg
```

## Step 8: Start Media Service API

```bash
cd picaivid-media-service
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

## Step 9: Test Phase 1 via API

```bash
curl -X POST http://localhost:8000/internal/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "YOUR_PROJECT_UUID",
    "template_type": "standard",
    "target_length": 60.0
  }'
```

## Step 10: Verify Results

```bash
# Check job_photos
docker exec -it picaivid-rails-postgres-1 psql -U postgres -d picaivid_development -c \
  "SELECT id, room_label, final_score, depth_variance FROM job_photos WHERE job_id = 1"

# Check room_clusters
docker exec -it picaivid-rails-postgres-1 psql -U postgres -d picaivid_development -c \
  "SELECT id, room_type, confidence_tier, recommended_motion FROM room_clusters WHERE job_id = 1"

# Check analysis_results
docker exec -it picaivid-rails-postgres-1 psql -U postgres -d picaivid_development -c \
  "SELECT id, tier, recommended_motion, model_recommendation FROM analysis_results WHERE job_id = 1"
```

## Step 11: Test via SQS

```bash
# Send message to SQS
aws --endpoint-url=http://localhost:4566 --region us-east-1 sqs send-message \
  --queue-url http://sqs.us-east-1.localhost.localstack.cloud:4566/000000000000/picaivid-jobs \
  --message-body '{"action":"run","project_id":"YOUR_PROJECT_UUID"}'

# Start worker
cd picaivid-media-service
source venv/bin/activate
python -m app.worker
```

## Step 12: Test via Rails

```ruby
# In Rails console
project = Project.last
VideoGenerationJob.perform_now(project.id)
```

## Troubleshooting

### ML Models Not Loading

First run downloads models (~2GB):
```bash
ls -la ml_models/
```

### Database Connection Issues

```bash
# Check Docker postgres is running
docker-compose ps

# Test connection via Docker
docker exec -it picaivid-rails-postgres-1 psql -U postgres -d picaivid_development -c "SELECT 1"
```

### S3/MinIO Issues

```bash
curl http://localhost:9000/minio/health/live
mc ls local/picaivid-dev
```

### SQS Issues

```bash
# Check LocalStack health
curl http://localhost:4566/_localstack/health

# List queues (region required)
aws --endpoint-url=http://localhost:4566 --region us-east-1 sqs list-queues

# Create queue if missing
aws --endpoint-url=http://localhost:4566 --region us-east-1 sqs create-queue --queue-name picaivid-jobs
```

### Alembic Migration Issues

Make sure virtual environment is activated:
```bash
cd picaivid-media-service
source venv/bin/activate
alembic upgrade head
```

## What Phase 1 Does

1. Reads photos from Rails DB (read-only)
2. Creates JobPhoto records in Python DB
3. Computes OpenCLIP embeddings
4. Classifies room types
5. Analyzes depth using MiDaS
6. Computes quality scores
7. Clusters photos by room
8. Plans motion strategy per cluster

---

## Cluster Debugging

### Query Clusters and Photos for a Job

```bash
source venv/bin/activate
python - <<'EOF'
from app.db.session import SessionLocal
from app.db.models import Job, JobPhoto, PhotoSimilarity, Clip

db = SessionLocal()
job = db.query(Job).order_by(Job.created_at.desc()).first()
print(f"Latest Job: {job.id}, status={job.status}")

# Show all clips (clusters) with their photos
clips = db.query(Clip).filter(Clip.job_id == job.id).all()
for clip in clips:
    print(f"  Clip {clip.id}: photos={clip.source_photo_ids}")

# Show all photos with room labels
photos = db.query(JobPhoto).filter(JobPhoto.job_id == job.id).order_by(JobPhoto.position).all()
for p in photos:
    print(f"  Photo {p.id}: pos={p.position}, room={p.room_label}, score={p.final_score:.2f}")
db.close()
EOF
```

### Inspect Similarity Between Specific Photos

```bash
python - <<'EOF'
from app.db.session import SessionLocal
from app.db.models import PhotoSimilarity

db = SessionLocal()
PHOTO_IDS = [1426, 1427, 1428]  # Replace with your photo IDs

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
            print(f"{i} <-> {j}: NOT COMPUTED")
db.close()
EOF
```

### Debug Why Photos Are/Aren't Clustered Together

```bash
python - <<'EOF'
from app.db.session import SessionLocal
from app.db.models import Job, JobPhoto, PhotoSimilarity, Clip

db = SessionLocal()
job = db.query(Job).order_by(Job.created_at.desc()).first()

# Find which clip each photo ended up in
PROBLEM_PHOTOS = [1415, 1416, 1417, 1418]  # Replace with your photo IDs

for pid in PROBLEM_PHOTOS:
    photo = db.query(JobPhoto).filter(JobPhoto.id == pid).first()
    print(f"Photo {pid}: room={photo.room_label}, position={photo.position}")

clips = db.query(Clip).filter(Clip.job_id == job.id).all()
for clip in clips:
    if clip.source_photo_ids and any(p in PROBLEM_PHOTOS for p in clip.source_photo_ids):
        print(f"  -> Clip {clip.id}: {clip.source_photo_ids}")
db.close()
EOF
```

### Run Clustering Tests

```bash
cd /path/to/picaivid-media-service
source venv/bin/activate

# Run all test sets
python scripts/test_cross_cluster_merge.py --test-set all

# Run a specific test set
python scripts/test_cross_cluster_merge.py --test-set 1
python scripts/test_cross_cluster_merge.py --test-set 2
python scripts/test_cross_cluster_merge.py --test-set 3

# List available jobs in DB (to see which photo IDs exist)
python scripts/test_cross_cluster_merge.py --list-jobs
```

### Analyze Photo Pair Matching (Threshold Tuning)

```bash
python scripts/analyze_photo_pairs.py
```

Computes DINOv2 + geometric similarity for pairs and compares to current thresholds.

---

## Worker Management

### Start the SQS Worker

```bash
cd picaivid-media-service
source venv/bin/activate
python -m app.worker
```

### Restart Worker (After Code Changes)

The worker does **not** auto-reload. After modifying clustering code, you must restart:

```bash
# Find and kill the old worker
kill $(pgrep -f "app.worker")

# Start fresh
source venv/bin/activate
python -m app.worker
```

### API Server (Auto-Reloads)

```bash
uvicorn app.main:app --reload --port 8000
```

The API server uses `--reload` so it auto-restarts on Python file changes.

---

## Key Clustering Constants (learned_matching.py)

| Constant | Value | Purpose |
|----------|-------|---------|
| `MIN_INLIERS_FOR_OVERLAP` | 15 | Min geometric inliers to connect two photos |
| `MIN_INLIERS_CROSS_ROOM` | 30 | Min inliers for non-adjacent cross-room pairs |
| `MIN_INLIERS_CROSS_ROOM_ADJACENT` | 15 | Min inliers for adjacent cross-room pairs (ML labels often wrong) |
| `MIN_INLIERS_FAR_APART` | 25 | Min inliers when position gap ≥ 3 (prevents same-room-label false matches) |
| `POSITION_GAP_THRESHOLD` | 3 | Position gap above which to require stronger evidence |
| `TEMPORAL_WINDOW` | 2 | Check photos within ±2 positions as potential same-cluster |
| `TEMPORAL_SEMANTIC_THRESHOLD` | 0.88 | Trust semantic similarity alone if ≥ 0.88 (no geometric needed) |
| `NEIGHBOR_TRUST_THRESHOLD` | 0.65 | Trust immediate neighbor even without full geometric verification |
| `DUPLICATE_SIMILARITY_THRESHOLD` | 0.94 | Mark as duplicate if consecutive photos are ≥ 94% similar |
| `MAX_CLUSTER_SIZE` | 3 | Maximum photos per cluster |
