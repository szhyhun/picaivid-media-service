# Media Service Initial Structure

This document describes the folder structure, configuration files, and setup instructions for the Picaivid media processing service. This structure aligns with the phased pipeline architecture defined in OVERVIEW.md.

## Architecture Overview

The service implements a 4-phase pipeline:
- **Phase 1**: Analyze And Plan (OpenCLIP, MiDaS, optional COLMAP)
- **Phase 2**: Render Clips (LTX-2 for depth-aware motion)
- **Phase 3**: Timeline And Beat Sync (librosa for beat detection)
- **Phase 4**: Final Assembly (ffmpeg)

Postgres is the system of record. S3 stores only binary artifacts.

## Folder Structure

```
media_service/
├── app/
│   ├── api/                     # FastAPI routes
│   │   ├── __init__.py
│   │   ├── deps.py             # Dependencies (auth, db session)
│   │   ├── health.py           # Health check endpoints
│   │   └── jobs.py             # Job control endpoints
│   ├── core/                    # Core configuration
│   │   ├── __init__.py
│   │   ├── config.py           # Settings (env vars)
│   │   ├── security.py         # API key auth
│   │   └── logging.py          # Logging config
│   ├── db/                      # Database
│   │   ├── __init__.py
│   │   ├── base.py             # Base class
│   │   ├── session.py          # DB session
│   │   └── models/             # SQLAlchemy models
│   │       ├── __init__.py
│   │       ├── job.py          # jobs table
│   │       ├── photo.py        # photos table
│   │       ├── room_cluster.py # room_clusters table
│   │       ├── analysis.py     # analysis_results table
│   │       ├── clip.py         # clips table
│   │       ├── timeline.py     # timeline table
│   │       ├── timeline_clip.py # timeline_clips table
│   │       └── edit.py         # edits table
│   ├── schemas/                 # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── job.py
│   │   ├── photo.py
│   │   ├── clip.py
│   │   ├── timeline.py
│   │   └── common.py
│   ├── pipeline/                # Phased pipeline implementation
│   │   ├── __init__.py
│   │   ├── orchestrator.py     # Phase orchestration and state machine
│   │   ├── phase1_analyze/     # Phase 1: Analyze And Plan
│   │   │   ├── __init__.py
│   │   │   ├── analyzer.py     # Main analysis coordinator
│   │   │   ├── clustering.py   # Room clustering with OpenCLIP
│   │   │   ├── depth.py        # MiDaS depth estimation
│   │   │   ├── scoring.py      # Photo quality scoring
│   │   │   └── motion_planner.py # Motion strategy decisions
│   │   ├── phase2_render/      # Phase 2: Render Clips
│   │   │   ├── __init__.py
│   │   │   ├── renderer.py     # Main render coordinator
│   │   │   ├── ltx_generator.py # LTX-2 motion generation
│   │   │   ├── validator.py    # Depth validation and downgrade
│   │   │   └── motion_types.py # Motion type definitions
│   │   ├── phase3_timeline/    # Phase 3: Timeline And Beat Sync
│   │   │   ├── __init__.py
│   │   │   ├── builder.py      # Timeline construction
│   │   │   ├── beat_sync.py    # librosa beat detection
│   │   │   └── template.py     # Template engine
│   │   └── phase4_assembly/    # Phase 4: Final Assembly
│   │       ├── __init__.py
│   │       ├── assembler.py    # Main assembly coordinator
│   │       ├── ffmpeg_wrapper.py # ffmpeg filter complex
│   │       └── audio_mixer.py  # Audio mixing and normalization
│   ├── services/                # Shared services
│   │   ├── __init__.py
│   │   ├── storage/
│   │   │   ├── __init__.py
│   │   │   └── s3_client.py
│   │   ├── sqs/
│   │   │   ├── __init__.py
│   │   │   ├── consumer.py     # SQS message consumer
│   │   │   └── producer.py     # SQS message producer
│   │   └── spot/
│   │       ├── __init__.py
│   │       └── interruption.py # Spot interruption handler (SIGTERM)
│   ├── models/                  # ML model wrappers
│   │   ├── __init__.py
│   │   ├── openclip.py         # OpenCLIP embeddings
│   │   ├── midas.py            # MiDaS depth estimation
│   │   ├── ltx.py              # LTX-2 video generation
│   │   └── colmap.py           # Optional COLMAP wrapper
│   ├── utils/                   # Utilities
│   │   ├── __init__.py
│   │   ├── image.py
│   │   ├── video.py
│   │   ├── audio.py
│   │   └── file.py
│   ├── __init__.py
│   ├── main.py                  # FastAPI app
│   └── worker.py                # SQS worker (phased execution)
├── alembic/                     # Database migrations
│   ├── versions/
│   ├── env.py
│   └── script.py.mako
├── ml_models/                   # ML model files (gitignored)
│   ├── openclip/
│   │   └── .gitkeep
│   ├── midas/
│   │   └── .gitkeep
│   ├── ltx/
│   │   └── .gitkeep
│   └── colmap/
│       └── .gitkeep
├── tests/
│   ├── unit/
│   │   ├── test_phase1_analyze.py
│   │   ├── test_phase2_render.py
│   │   ├── test_phase3_timeline.py
│   │   ├── test_phase4_assembly.py
│   │   ├── test_orchestrator.py
│   │   └── test_spot_interruption.py
│   ├── integration/
│   │   ├── test_api.py
│   │   ├── test_pipeline.py
│   │   └── test_sqs.py
│   ├── fixtures/
│   │   └── sample_images/
│   ├── conftest.py
│   └── __init__.py
├── scripts/
│   ├── download_models.py      # Download ML models
│   ├── test_phase.py           # Test individual phases
│   └── benchmark.py            # Performance benchmarks
├── .env.example                 # Environment variables template
├── .gitignore
├── .dockerignore
├── Dockerfile
├── Dockerfile.worker
├── alembic.ini                  # Alembic config
├── pyproject.toml              # Project metadata
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
├── pytest.ini
├── README.md
├── OVERVIEW.md                  # Master design document
└── INITIAL_STRUCTURE.md        # This file
```

## Key Configuration Files

### requirements.txt

```txt
# Core
fastapi==0.109.0
uvicorn[standard]==0.27.0

# Database
sqlalchemy==2.0.25
alembic==1.13.1
psycopg2-binary==2.9.9

# ML & CV - Core
torch==2.1.2
torchvision==0.16.2
opencv-python==4.9.0.80
pillow==10.2.0
numpy==1.26.3
scikit-image==0.22.0

# Phase 1: Analysis Models
open-clip-torch==2.24.0      # OpenCLIP for embeddings and clustering
timm==0.9.12                 # Required for MiDaS

# Phase 2: Motion Generation
# LTX-2 installed from source or HuggingFace
diffusers==0.25.0            # For LTX-2 inference
accelerate==0.25.0           # GPU acceleration
safetensors==0.4.1           # Model loading

# Phase 3: Beat Sync
librosa==0.10.1              # Beat detection and audio analysis
soundfile==0.12.1            # Audio file I/O

# Phase 4: Assembly
ffmpeg-python==0.2.0         # ffmpeg wrapper

# AWS
boto3==1.34.20               # S3 and SQS

# HTTP Client
httpx==0.26.0

# Utilities
python-multipart==0.0.6
python-dotenv==1.0.0
pydantic==2.5.3
pydantic-settings==2.1.0

# Clustering
scikit-learn==1.4.0          # For room clustering
```

### requirements-dev.txt

```txt
-r requirements.txt

# Testing
pytest==7.4.4
pytest-asyncio==0.23.3
pytest-cov==4.1.0
pytest-mock==3.12.0

# Code Quality
black==23.12.1
flake8==7.0.0
mypy==1.8.0
isort==5.13.2
pre-commit==3.6.0

# Debugging
ipython==8.20.0
```

### pyproject.toml

```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "media-service"
version = "0.1.0"
description = "AI and media processing service for Virtual Listing Studio"
requires-python = ">=3.11"

[tool.black]
line-length = 100
target-version = ['py311']
exclude = '''
/(
    \.git
  | \.venv
  | \.tox
  | build
  | dist
  | alembic
)/
'''

[tool.isort]
profile = "black"
line_length = 100
skip_gitignore = true

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --cov=app --cov-report=html --cov-report=term"
```

### app/main.py

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import setup_logging
from app.api import health, jobs

# Setup logging
setup_logging()

# Create FastAPI app
app = FastAPI(
    title="Picaivid Media Service",
    description="Phased video pipeline for real estate media",
    version="0.1.0",
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
)

# CORS (internal only)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(jobs.router, prefix="/internal/jobs", tags=["jobs"])

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    # Preload ML models if GPU available
    pass

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    pass
```

### app/worker.py

```python
"""
SQS Worker for phased pipeline execution.

Worker types:
- CPU: Runs Phase 1, Phase 3, Phase 4 (analysis, timeline, assembly)
- GPU: Runs Phase 2 only (video generation on Spot instances)

Rails sends: { job_id, action: "run", start_phase: optional }
Worker reads job state from Postgres and executes phases.
"""
import logging
import signal
from app.core.config import settings
from app.core.logging import setup_logging
from app.services.sqs.consumer import SQSConsumer
from app.services.spot.interruption import SpotInterruptionHandler
from app.pipeline.orchestrator import PipelineOrchestrator
from app.db.session import get_db

setup_logging()
logger = logging.getLogger(__name__)


# Phases each worker type handles
CPU_PHASES = [1, 3, 4]  # Analyze, Timeline, Assembly
GPU_PHASES = [2]         # Render clips


def process_message(message: dict) -> None:
    """Process a single SQS message."""
    job_id = message.get("job_id")
    action = message.get("action", "run")
    start_phase = message.get("start_phase")

    if action != "run":
        logger.warning(f"Unknown action: {action}")
        return

    allowed_phases = GPU_PHASES if settings.WORKER_TYPE == "gpu" else CPU_PHASES

    with get_db() as db:
        orchestrator = PipelineOrchestrator(db)
        orchestrator.execute(
            job_id,
            start_phase=start_phase,
            allowed_phases=allowed_phases,
        )


def main():
    """Main worker entry point."""
    logger.info(f"Starting {settings.WORKER_TYPE.upper()} worker...")

    # GPU workers: handle Spot interruption
    if settings.WORKER_TYPE == "gpu":
        handler = SpotInterruptionHandler()
        signal.signal(signal.SIGTERM, handler.handle_sigterm)

    consumer = SQSConsumer(
        queue_url=settings.SQS_QUEUE_URL,
        handler=process_message,
        visibility_timeout=3600,  # 1 hour
    )

    consumer.start()


if __name__ == "__main__":
    main()
```

### app/core/config.py

```python
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # API
    API_KEY: str
    BACKEND_URL: str
    CORS_ORIGINS: List[str] = ["http://localhost:3001"]

    # Database (Postgres is system of record)
    DATABASE_URL: str

    # AWS
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_REGION: str = "us-east-1"

    # S3 (binary artifact storage only)
    S3_BUCKET: str | None = None
    LOCAL_STORAGE_PATH: str = "./tmp/media"  # For local dev

    # SQS (job orchestration)
    SQS_QUEUE_URL: str | None = None

    # ML Models
    MODEL_CACHE_DIR: str = "./ml_models"
    TORCH_HOME: str = "./ml_models/torch"
    HUGGINGFACE_HUB_CACHE: str = "./ml_models/huggingface"

    # Phase 1: Analysis
    OPENCLIP_MODEL: str = "ViT-B-32"
    MIDAS_MODEL: str = "DPT_Large"

    # Phase 2: Rendering
    LTX_MODEL_PATH: str | None = None  # Path to LTX-2 weights
    LTX_ENABLED: bool = False  # Disable for CPU-only dev

    # Phase 3: Timeline
    DEFAULT_BPM: int = 120
    ENABLE_BEAT_SYNC: bool = True

    # Phase 4: Assembly
    FFMPEG_PATH: str = "/usr/local/bin/ffmpeg"
    OUTPUT_RESOLUTION: str = "1920x1080"
    OUTPUT_FPS: int = 30

    # Worker Type
    WORKER_TYPE: str = "cpu"  # "cpu" or "gpu"

    # Processing
    GPU_ENABLED: bool = False
    MAX_WORKERS: int = 4

    # Spot Instance Handling
    SPOT_TERMINATION_ENDPOINT: str = "http://169.254.169.254/latest/meta-data/spot/termination-time"

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

### .env.example

```bash
# Environment
ENVIRONMENT=development
DEBUG=true

# API
API_KEY=dev-secret-key-change-in-production
BACKEND_URL=http://localhost:3001
CORS_ORIGINS=http://localhost:3001

# Database (Postgres is system of record)
DATABASE_URL=postgresql://localhost/picaivid_dev

# AWS (optional for local dev)
# AWS_ACCESS_KEY_ID=your-access-key
# AWS_SECRET_ACCESS_KEY=your-secret-key
# AWS_REGION=us-east-1

# S3 (use local storage for dev)
# S3_BUCKET=picaivid-dev
LOCAL_STORAGE_PATH=./tmp/media

# SQS (optional for local dev - use direct API calls)
# SQS_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/123456789/picaivid-jobs

# ML Models
MODEL_CACHE_DIR=./ml_models
TORCH_HOME=./ml_models/torch
HUGGINGFACE_HUB_CACHE=./ml_models/huggingface

# Phase 1: Analysis
OPENCLIP_MODEL=ViT-B-32
MIDAS_MODEL=DPT_Large

# Worker Type (cpu for local dev, gpu for Spot instances)
WORKER_TYPE=cpu

# Phase 2: Rendering (GPU workers only)
LTX_ENABLED=false
# LTX_MODEL_PATH=/path/to/ltx-2-weights

# Phase 3: Timeline
DEFAULT_BPM=120
ENABLE_BEAT_SYNC=true

# Phase 4: Assembly
FFMPEG_PATH=/usr/local/bin/ffmpeg
OUTPUT_RESOLUTION=1920x1080
OUTPUT_FPS=30

# Processing
MAX_WORKERS=4
GPU_ENABLED=false

# Logging
LOG_LEVEL=DEBUG
```

### Database Models (app/db/models/)

The database schema follows OVERVIEW.md. Postgres is the system of record.

```python
# app/db/models/job.py
from sqlalchemy import Column, Integer, String, Float, Boolean
from app.db.base import Base

class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True)
    listing_id = Column(Integer, nullable=False)
    status = Column(String(50), default="pending")
    current_phase = Column(Integer, default=0)
    template_type = Column(String(50))
    target_length = Column(Float)
    music_uri = Column(String(500))
    # bpm and beat_offset are detected in Phase 3, not from Rails
    bpm = Column(Integer, nullable=True)
    beat_offset = Column(Float, nullable=True)
    enable_beat_sync = Column(Boolean, default=True)


# app/db/models/photo.py
from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, JSON
from app.db.base import Base

class Photo(Base):
    __tablename__ = "photos"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    s3_uri = Column(String(500), nullable=False)
    room_label = Column(String(100))
    room_override = Column(String(100))
    exclude = Column(Boolean, default=False)
    manual_metadata = Column(JSON)
    sharpness = Column(Float)
    exposure_score = Column(Float)
    composition_score = Column(Float)
    base_score = Column(Float)
    final_score = Column(Float)


# app/db/models/room_cluster.py
from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey
from app.db.base import Base

class RoomCluster(Base):
    __tablename__ = "room_clusters"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    room_type = Column(String(100))
    confidence_tier = Column(String(20))  # low, medium, high
    sfm_eligible = Column(Boolean, default=False)
    image_count = Column(Integer)
    overlap_score = Column(Float)
    depth_variance = Column(Float)


# app/db/models/analysis.py
from sqlalchemy import Column, Integer, String, Float, ForeignKey, JSON
from app.db.base import Base

class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    room_cluster_id = Column(Integer, ForeignKey("room_clusters.id"))
    hero_photo_id = Column(Integer, ForeignKey("photos.id"))
    recommended_motion = Column(String(50))
    allowed_motion_types = Column(JSON)  # Array of allowed motions
    recommended_duration = Column(Float)
    tier = Column(String(20))
    model_recommendation = Column(String(100))
    debug_metrics = Column(JSON)


# app/db/models/clip.py
from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, JSON
from app.db.base import Base

class Clip(Base):
    __tablename__ = "clips"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    room_cluster_id = Column(Integer, ForeignKey("room_clusters.id"))
    source_photo_ids = Column(JSON)  # Array of photo IDs
    motion_type = Column(String(50))
    model_used = Column(String(100))
    is_3d = Column(Boolean, default=False)
    duration = Column(Float)
    s3_uri = Column(String(500))
    validation_score = Column(Float)
    status = Column(String(50), default="pending")


# app/db/models/timeline.py
from sqlalchemy import Column, Integer, String, Float, ForeignKey, JSON
from app.db.base import Base

class Timeline(Base):
    __tablename__ = "timeline"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    version = Column(Integer, default=1)
    status = Column(String(50), default="draft")
    beat_grid = Column(JSON)
    total_duration = Column(Float)


# app/db/models/timeline_clip.py
from sqlalchemy import Column, Integer, String, Float, ForeignKey
from app.db.base import Base

class TimelineClip(Base):
    __tablename__ = "timeline_clips"

    id = Column(Integer, primary_key=True)
    timeline_id = Column(Integer, ForeignKey("timeline.id"), nullable=False)
    clip_id = Column(Integer, ForeignKey("clips.id"), nullable=False)
    order_index = Column(Integer, nullable=False)
    in_time = Column(Float)
    out_time = Column(Float)
    transition_type = Column(String(50))
    audio_policy = Column(String(50))


# app/db/models/edit.py
from sqlalchemy import Column, Integer, String, ForeignKey, JSON, DateTime
from sqlalchemy.sql import func
from app.db.base import Base

class Edit(Base):
    __tablename__ = "edits"

    id = Column(Integer, primary_key=True)
    timeline_id = Column(Integer, ForeignKey("timeline.id"), nullable=False)
    user_id = Column(Integer)
    edit_type = Column(String(50))
    payload = Column(JSON)
    created_at = Column(DateTime, server_default=func.now())
```

### Running ML Models Locally

All ML models can be run locally with Python (CPU or GPU).

```bash
# Install dependencies
pip install -r requirements.txt

# Models are auto-downloaded from HuggingFace Hub on first use
# They are cached in MODEL_CACHE_DIR (default: ./ml_models)
```

```python
# Example: Test OpenCLIP embeddings locally
from app.models.openclip import OpenCLIPModel

model = OpenCLIPModel()  # Auto-downloads on first use
embedding = model.get_embedding("path/to/image.jpg")
print(f"Embedding shape: {embedding.shape}")

# Example: Test MiDaS depth estimation locally
from app.models.midas import MiDaSModel

midas = MiDaSModel()  # Auto-downloads on first use
depth_map = midas.estimate_depth("path/to/image.jpg")
print(f"Depth variance: {depth_map.var()}")

# Example: Test LTX-2 motion (requires GPU)
from app.models.ltx import LTXModel

ltx = LTXModel()  # Requires GPU_ENABLED=true and LTX_ENABLED=true
clip = ltx.generate_motion("path/to/image.jpg", motion_type="push_in")
```

### Dockerfile

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Dockerfile.worker.cpu

CPU worker for Phase 1, Phase 3, Phase 4 tasks:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (CPU only)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# CPU worker mode
ENV WORKER_TYPE=cpu

# Run SQS worker
CMD ["python", "-m", "app.worker"]
```

### Dockerfile.worker.gpu

GPU worker for Phase 2 video generation (runs on Spot instances):

```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies with CUDA support
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Copy application
COPY . .

# GPU worker mode
ENV WORKER_TYPE=gpu
ENV GPU_ENABLED=true

# Spot interruption handling
STOPSIGNAL SIGTERM

# Run SQS worker
CMD ["python", "-m", "app.worker"]
```

## Compute Environment

| Environment | Device | Use Case |
|------------|--------|----------|
| Local (macOS) | Apple Silicon (CPU) | Phase 1 analysis, timeline, debugging |
| AWS ECS (CPU) | CPU instances | Orchestration, Phase 1, Phase 3, Phase 4 |
| AWS ECS (GPU) | Spot g5.xlarge | Phase 2 video generation (LTX-2) |

Local machines run CPU tasks only. GPU video generation runs on remote Spot instances.

## How to Run Locally

### Prerequisites

- Python 3.11+
- PostgreSQL 14+
- FFmpeg 6+
- macOS with Apple Silicon (M1/M2/M3)

### Installation

```bash
# Navigate to media_service directory
cd media_service

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Copy environment variables
cp .env.example .env

# Edit .env with your settings

# Create database
createdb picaivid_dev

# Run migrations
alembic upgrade head
```

### Running the Service

#### API Server

```bash
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

#### CPU Worker (Local)

Runs Phase 1, Phase 3, Phase 4 tasks:

```bash
source venv/bin/activate
python -m app.worker
```

#### GPU Worker (Remote Only)

GPU workers run on AWS Spot instances only. For local testing, simulate GPU outputs or use a dev Spot instance.

The API will be available at http://localhost:8000

## How to Test

### Run All Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=app --cov-report=html
open htmlcov/index.html
```

### Run Specific Tests

```bash
# Unit tests
pytest tests/unit

# Integration tests
pytest tests/integration

# Specific file
pytest tests/unit/test_classifier.py

# Specific test
pytest tests/unit/test_classifier.py::test_classify_interior
```

### Code Quality

```bash
# Format code
black app tests

# Sort imports
isort app tests

# Lint
flake8 app tests

# Type check
mypy app
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Environment Variables

### Required

- `API_KEY`: API key for authentication
- `BACKEND_URL`: URL of backend service
- `DATABASE_URL`: PostgreSQL connection string

### AWS

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `S3_BUCKET`: S3 bucket for artifacts
- `SQS_QUEUE_URL`: SQS queue for job orchestration

### Storage (Local Dev)

- `LOCAL_STORAGE_PATH=./tmp/media`

### Worker

- `WORKER_TYPE`: `cpu` or `gpu`
- `GPU_ENABLED`: Enable GPU processing (GPU workers only)
- `MAX_WORKERS`: Number of worker processes
- `FFMPEG_PATH`: Path to FFmpeg binary

## Database Migrations

### Create Migration

```bash
alembic revision --autogenerate -m "Description"
```

### Apply Migrations

```bash
alembic upgrade head
```

### Rollback

```bash
alembic downgrade -1
```

### Check Status

```bash
alembic current
alembic history
```

## Common Tasks

### Download ML Models

```bash
python scripts/download_models.py
```

### Test Classification

```bash
python scripts/test_classification.py path/to/image.jpg
```

### Benchmark Performance

```bash
python scripts/benchmark.py
```

### Python Shell

```bash
# IPython shell with app context
ipython

from app.services.classification import PhotoClassifier
classifier = PhotoClassifier()
```

## Debugging

### Use Debugger

```python
import pdb; pdb.set_trace()  # Set breakpoint
# or
breakpoint()  # Python 3.7+
```

### Check Logs

```bash
# API logs
tail -f logs/api.log

# Worker logs
tail -f logs/worker.log
```

### SQS Monitoring

```bash
# Check queue depth (via AWS CLI)
aws sqs get-queue-attributes \
  --queue-url $SQS_QUEUE_URL \
  --attribute-names ApproximateNumberOfMessages

# Check job status in Postgres
psql -d picaivid_dev -c "SELECT id, status, current_phase FROM jobs ORDER BY id DESC LIMIT 10"
```

## Common Issues

### FFmpeg Not Found

```bash
# Install FFmpeg
brew install ffmpeg  # macOS
sudo apt install ffmpeg  # Ubuntu

# Verify
ffmpeg -version
```

### GPU Out of Memory (Remote Workers)

- Reduce batch size in clip render
- Clear GPU cache: `torch.cuda.empty_cache()`
- Check g5.xlarge has sufficient VRAM (24GB)

### Model Download Fails

```bash
# Models auto-download from HuggingFace Hub
# Check model cache
ls -lh ml_models/

# Verify HuggingFace cache
ls -lh $HUGGINGFACE_HUB_CACHE
```

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python path
python -c "import sys; print(sys.path)"
```

## Production Deployment

### Build Docker Images

```bash
# API
docker build -t picaivid-api .

# CPU Worker (Phase 1, 3, 4)
docker build -f Dockerfile.worker.cpu -t picaivid-worker-cpu .

# GPU Worker (Phase 2 - Spot instances)
docker build -f Dockerfile.worker.gpu -t picaivid-worker-gpu .
```

### Deploy to AWS ECS

```bash
# Push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REGISTRY
docker push $ECR_REGISTRY/picaivid-api
docker push $ECR_REGISTRY/picaivid-worker-cpu
docker push $ECR_REGISTRY/picaivid-worker-gpu
```

### Environment Variables

CPU Worker:
```bash
ENVIRONMENT=production
WORKER_TYPE=cpu
DATABASE_URL=<postgres-url>
SQS_QUEUE_URL=<sqs-url>
```

GPU Worker (Spot):
```bash
ENVIRONMENT=production
WORKER_TYPE=gpu
GPU_ENABLED=true
DATABASE_URL=<postgres-url>
SQS_QUEUE_URL=<sqs-url>
```

## Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

### Metrics

```bash
curl http://localhost:8000/internal/metrics
```

### GPU Cost Monitoring

Track GPU usage and Spot instance costs:

```bash
# Check running GPU workers
aws ecs list-tasks --cluster picaivid-gpu --service gpu-worker

# Check Spot instance status
aws ec2 describe-spot-instance-requests --filters "Name=state,Values=active"
```

## Security Checklist

- [ ] Use secure API key in production
- [ ] Validate all file inputs
- [ ] Sanitize file paths
- [ ] Limit file sizes
- [ ] Use HTTPS in production
- [ ] Secure model files
- [ ] Don't log sensitive data

## Performance Tips

- Run Phase 1 analysis on CPU workers (cost efficient)
- Use Spot GPU instances for Phase 2 video generation
- Chunk GPU work into single-clip tasks for Spot tolerance
- Cache model weights on EBS volumes for faster startup
- Monitor GPU hours and Spot interruption rates
- Scale GPU from zero when no work exists

## Next Steps

1. Review [OVERVIEW.md](./OVERVIEW.md) for architecture and design principles
2. Set up your local development environment (CPU only)
3. Run Phase 1 analysis locally
4. Set up a dev Spot GPU instance for Phase 2 testing
5. Start with Phase 1 implementation
