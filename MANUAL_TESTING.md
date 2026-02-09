# Phase 1 Manual Testing Guide

This guide walks through setting up and testing the Phase 1 pipeline locally on macOS.

## Prerequisites

- Python 3.11+ (installed via `brew install python@3.11`)
- PostgreSQL 14+
- Docker (for MinIO and LocalStack)

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
NAME                  STATUS
picaivid-postgres     Up
picaivid-minio        Up (healthy)
picaivid-localstack   Up (healthy)
```

## Step 2: Verify LocalStack SQS

```bash
# Check SQS queue was created
aws --endpoint-url=http://localhost:4566 sqs list-queues
```

## Step 3: Set Up Python Environment

```bash
cd picaivid-media-service

# Activate virtual environment (already created)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 4: Run Database Migrations

```bash
# Generate initial migration
alembic revision --autogenerate -m "Create Phase 1 tables"

# Apply migration
alembic upgrade head
```

Verify tables were created:
```bash
psql -h localhost -U postgres -d picaivid_development -c "\dt"
```

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
psql -h localhost -U postgres -d picaivid_development -c \
  "SELECT id, room_label, final_score, depth_variance FROM job_photos WHERE job_id = 1"

# Check room_clusters
psql -h localhost -U postgres -d picaivid_development -c \
  "SELECT id, room_type, confidence_tier, recommended_motion FROM room_clusters WHERE job_id = 1"

# Check analysis_results
psql -h localhost -U postgres -d picaivid_development -c \
  "SELECT id, tier, recommended_motion, model_recommendation FROM analysis_results WHERE job_id = 1"
```

## Step 11: Test via SQS

```bash
# Send message to SQS
aws --endpoint-url=http://localhost:4566 sqs send-message \
  --queue-url http://sqs.us-east-1.localhost.localstack.cloud:4566/000000000000/picaivid-jobs \
  --message-body '{"action":"run","project_id":"YOUR_PROJECT_UUID"}'

# Start worker
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
docker-compose ps postgres
psql -h localhost -U postgres -d picaivid_development -c "SELECT 1"
```

### S3/MinIO Issues
```bash
curl http://localhost:9000/minio/health/live
mc ls local/picaivid-dev
```

### SQS Issues
```bash
curl http://localhost:4566/_localstack/health
aws --endpoint-url=http://localhost:4566 sqs list-queues
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
