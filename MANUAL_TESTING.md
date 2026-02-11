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
