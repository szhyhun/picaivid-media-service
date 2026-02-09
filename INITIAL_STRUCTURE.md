# Media Service Initial Structure

This document describes the folder structure, configuration files, and setup instructions for the Virtual Listing Studio media processing service.

## Folder Structure

```
media_service/
├── app/
│   ├── api/                     # FastAPI routes
│   │   ├── __init__.py
│   │   ├── deps.py             # Dependencies (auth, etc.)
│   │   ├── health.py           # Health check endpoints
│   │   ├── photos.py           # Photo processing endpoints
│   │   ├── staging.py          # Staging endpoints
│   │   └── videos.py           # Video rendering endpoints
│   ├── core/                    # Core configuration
│   │   ├── __init__.py
│   │   ├── config.py           # Settings (env vars)
│   │   ├── security.py         # API key auth
│   │   └── logging.py          # Logging config
│   ├── db/                      # Database
│   │   ├── __init__.py
│   │   ├── base.py             # Base class
│   │   ├── session.py          # DB session
│   │   └── models.py           # SQLAlchemy models
│   ├── schemas/                 # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── photo.py
│   │   ├── staging.py
│   │   ├── video.py
│   │   └── common.py
│   ├── services/                # Business logic
│   │   ├── __init__.py
│   │   ├── classification/
│   │   │   ├── __init__.py
│   │   │   ├── classifier.py
│   │   │   ├── models.py
│   │   │   └── preprocessor.py
│   │   ├── staging/
│   │   │   ├── __init__.py
│   │   │   ├── pipeline.py
│   │   │   ├── segmentation.py
│   │   │   ├── inpainting.py
│   │   │   └── design_kit.py
│   │   ├── video/
│   │   │   ├── __init__.py
│   │   │   ├── renderer.py
│   │   │   ├── clip_generator.py
│   │   │   ├── audio_sync.py
│   │   │   └── ffmpeg_wrapper.py
│   │   └── storage/
│   │       ├── __init__.py
│   │       └── s3_client.py
│   ├── tasks/                   # Celery tasks
│   │   ├── __init__.py
│   │   ├── classification.py
│   │   ├── staging.py
│   │   ├── video.py
│   │   └── callbacks.py
│   ├── utils/                   # Utilities
│   │   ├── __init__.py
│   │   ├── image.py
│   │   ├── video.py
│   │   ├── audio.py
│   │   └── file.py
│   ├── __init__.py
│   ├── main.py                  # FastAPI app
│   └── worker.py                # Celery worker
├── alembic/                     # Database migrations
│   ├── versions/
│   ├── env.py
│   └── script.py.mako
├── models/                      # ML model files (gitignored)
│   ├── classification/
│   │   └── .gitkeep
│   ├── staging/
│   │   └── .gitkeep
│   └── object_detection/
│       └── .gitkeep
├── tests/
│   ├── unit/
│   │   ├── test_classifier.py
│   │   ├── test_staging.py
│   │   └── test_video.py
│   ├── integration/
│   │   ├── test_api.py
│   │   └── test_tasks.py
│   ├── fixtures/
│   │   └── sample_images/
│   ├── conftest.py
│   └── __init__.py
├── scripts/
│   ├── download_models.py      # Download ML models
│   ├── test_classification.py  # Test classification
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
├── AGENTS.md
├── IMPLEMENTATION_PLAN.md
└── INITIAL_STRUCTURE.md        # This file
```

## Key Configuration Files

### requirements.txt

```txt
# Core
fastapi==0.109.0
uvicorn[standard]==0.27.0
celery==5.3.4
redis==5.0.1

# Database
sqlalchemy==2.0.25
alembic==1.13.1
psycopg2-binary==2.9.9

# ML & CV
torch==2.1.2
torchvision==0.16.2
opencv-python==4.9.0.80
pillow==10.2.0
numpy==1.26.3
scikit-image==0.22.0

# Media Processing
ffmpeg-python==0.2.0
librosa==0.10.1
pydub==0.25.1

# ML Models
transformers==4.36.2
timm==0.9.12
ultralytics==8.1.0
onnxruntime==1.16.3

# AWS
boto3==1.34.20

# HTTP Client
httpx==0.26.0

# Utilities
python-multipart==0.0.6
python-dotenv==1.0.0
pydantic==2.5.3
pydantic-settings==2.1.0
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
from app.api import health, photos, staging, videos

# Setup logging
setup_logging()

# Create FastAPI app
app = FastAPI(
    title="Virtual Listing Studio Media Service",
    description="AI and media processing service",
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
app.include_router(photos.router, prefix="/internal/photos", tags=["photos"])
app.include_router(staging.router, prefix="/internal/staging", tags=["staging"])
app.include_router(videos.router, prefix="/internal/video", tags=["videos"])

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    pass

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    pass
```

### app/worker.py

```python
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "media_service",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.tasks.classification",
        "app.tasks.staging",
        "app.tasks.video",
    ]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour
    task_soft_time_limit=3300,  # 55 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
)

if __name__ == "__main__":
    celery_app.start()
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

    # Database
    DATABASE_URL: str

    # Redis
    REDIS_URL: str

    # Celery
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str

    # Storage
    STORAGE_PROVIDER: str = "s3"  # s3 or local
    LOCAL_STORAGE_PATH: str = "./tmp/media"
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_REGION: str = "us-east-1"
    AWS_BUCKET: str | None = None

    # ML Models
    MODEL_CACHE_DIR: str = "./models"
    TORCH_HOME: str = "./models/torch"
    HUGGINGFACE_HUB_CACHE: str = "./models/huggingface"

    # Processing
    MAX_WORKERS: int = 4
    GPU_ENABLED: bool = False
    FFMPEG_PATH: str = "/usr/bin/ffmpeg"

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

# Database
DATABASE_URL=postgresql://localhost/media_service_dev

# Redis
REDIS_URL=redis://localhost:6379/1

# Celery
CELERY_BROKER_URL=redis://localhost:6379/2
CELERY_RESULT_BACKEND=redis://localhost:6379/3

# Storage (local for development)
STORAGE_PROVIDER=local
LOCAL_STORAGE_PATH=./tmp/media

# Or use S3
# STORAGE_PROVIDER=s3
# AWS_ACCESS_KEY_ID=your-access-key
# AWS_SECRET_ACCESS_KEY=your-secret-key
# AWS_REGION=us-east-1
# AWS_BUCKET=virtual-listing-studio-dev

# ML Models
MODEL_CACHE_DIR=./models
TORCH_HOME=./models/torch
HUGGINGFACE_HUB_CACHE=./models/huggingface

# Processing
MAX_WORKERS=4
GPU_ENABLED=false
FFMPEG_PATH=/usr/local/bin/ffmpeg

# Logging
LOG_LEVEL=DEBUG
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

# Download models
RUN python scripts/download_models.py

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Dockerfile.worker

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

# Download models
RUN python scripts/download_models.py

# Run Celery worker
CMD ["celery", "-A", "app.worker", "worker", "--loglevel=info"]
```

## How to Run Locally

### Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Redis 7+
- FFmpeg 6+

### Installation

```bash
# Navigate to media_service directory
cd media_service

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Copy environment variables
cp .env.example .env

# Edit .env with your settings

# Create database
createdb media_service_dev

# Run migrations
alembic upgrade head

# Download ML models
python scripts/download_models.py
```

### Running the Service

#### API Server

```bash
# Activate virtual environment
source venv/bin/activate

# Run with uvicorn
uvicorn app.main:app --reload --port 8000

# Or with hot reload
uvicorn app.main:app --reload --port 8000 --log-level debug
```

#### Celery Worker

In a separate terminal:

```bash
# Activate virtual environment
source venv/bin/activate

# Run Celery worker
celery -A app.worker worker --loglevel=info

# With auto-reload (development)
watchmedo auto-restart --directory=app --pattern=*.py --recursive -- celery -A app.worker worker --loglevel=info
```

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
- `REDIS_URL`: Redis connection string
- `CELERY_BROKER_URL`: Celery broker URL
- `CELERY_RESULT_BACKEND`: Celery result backend URL

### Storage (Choose one)

**Local**:
- `STORAGE_PROVIDER=local`
- `LOCAL_STORAGE_PATH=./tmp/media`

**S3**:
- `STORAGE_PROVIDER=s3`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `AWS_BUCKET`

### Optional

- `GPU_ENABLED`: Enable GPU processing
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

### Celery Monitoring

```bash
# Check queue
celery -A app.worker inspect active

# Check workers
celery -A app.worker inspect stats

# Purge queue
celery -A app.worker purge
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

### CUDA Out of Memory

- Reduce batch size
- Clear GPU cache: `torch.cuda.empty_cache()`
- Use CPU: `GPU_ENABLED=false`

### Model Download Fails

```bash
# Manually download models
python scripts/download_models.py

# Check model cache
ls -lh models/
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
docker build -t media-service-api .

# Worker
docker build -f Dockerfile.worker -t media-service-worker .
```

### Run with Docker Compose

```bash
docker-compose up -d
```

### Environment Variables

Set production environment variables:

```bash
ENVIRONMENT=production
DEBUG=false
API_KEY=<secure-key>
GPU_ENABLED=true
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

### Celery Monitoring

Use Flower:

```bash
pip install flower
celery -A app.worker flower --port=5555
```

Visit http://localhost:5555

## Security Checklist

- [ ] Use secure API key in production
- [ ] Validate all file inputs
- [ ] Sanitize file paths
- [ ] Limit file sizes
- [ ] Use HTTPS in production
- [ ] Secure model files
- [ ] Don't log sensitive data

## Performance Tips

- Use GPU for ML inference
- Batch processing when possible
- Cache model outputs
- Use ONNX for faster inference
- Monitor GPU memory
- Use smaller models in development

## Next Steps

1. Review [AGENTS.md](./AGENTS.md) for contribution guidelines
2. Check [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) for feature roadmap
3. Set up your development environment
4. Download ML models
5. Run the service locally
6. Start with a simple feature or bug fix
