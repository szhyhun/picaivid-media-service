# Media Service - Agent Guidelines

## Purpose

The media service is the **AI, computer vision, and media processing engine** for Virtual Listing Studio. It owns all machine learning, image/video processing, and computationally intensive operations.

This is a production-grade processing service, not a demo or prototype.

## Responsibilities

### Core Processing Tasks
- **Photo Classification**: Detect room types, interior/exterior, photo quality
- **Virtual Staging**: Apply ML-based staging to photos
- **Design Kit Enforcement**: Apply consistent design styles across photos
- **Image Analysis**: Detect objects, colors, lighting conditions
- **Video Generation**: Create videos from photos with transitions and effects
- **Audio Analysis**: Beat detection, BPM analysis, audio synchronization
- **Speed Ramp Logic**: Calculate Ken Burns effects and motion
- **FFmpeg Orchestration**: Encode, transcode, render videos
- **Clip Generation**: Create individual clips from photos
- **Quality Checks**: Validate output quality and consistency

### Technical Responsibilities
- **ML Model Serving**: Run PyTorch, TensorFlow models
- **GPU Workload Management**: Optimize GPU utilization
- **Async Task Processing**: Celery task execution
- **Result Caching**: Cache intermediate results for regeneration
- **Progress Reporting**: Report progress back to backend
- **Error Handling**: Graceful degradation, retries
- **Resource Management**: Memory, GPU, CPU monitoring

## NOT Responsible For

### Explicitly Forbidden
- **No User Management**: No authentication, no user accounts
- **No Business Logic**: No pricing, no permissions, no validation beyond technical requirements
- **No Template Ownership**: Backend owns template metadata; media service only executes
- **No Direct Frontend Communication**: Only talk to backend
- **No Billing**: No credit tracking, no subscription logic
- **No UI**: No web interface (except maybe health check)

### Keep It Focused
- Don't store business data (users, listings, etc.)
- Don't implement authorization (trust backend API key)
- Don't make business decisions (backend orchestrates)
- Don't expose public API (internal only)

## Technology Stack

### Core Framework
- **Python 3.11+**
- **FastAPI** for REST API
- **Celery** for async task processing
- **Redis** for task queue and result backend

### ML & CV Libraries
- **PyTorch** for deep learning
- **OpenCV** for computer vision
- **Pillow (PIL)** for image manipulation
- **scikit-image** for image processing
- **numpy** for numerical operations

### Media Processing
- **FFmpeg** (via ffmpeg-python or subprocess)
- **librosa** for audio analysis
- **pydub** for audio manipulation

### Models & Tools
- **transformers** (Hugging Face) for pre-trained models
- **timm** for computer vision models
- **ultralytics** (YOLO) for object detection
- **onnxruntime** for optimized inference

### Database
- **PostgreSQL** (for operational state only, not business data)
- **SQLAlchemy** as ORM
- **Alembic** for migrations

### Development Tools
- **pytest** for testing
- **black** for code formatting
- **flake8** for linting
- **mypy** for type checking
- **pre-commit** for git hooks

## Coding Standards

### File Organization
```
media_service/
├── app/
│   ├── api/                    # FastAPI routes
│   │   ├── __init__.py
│   │   ├── health.py
│   │   ├── photos.py
│   │   ├── staging.py
│   │   └── videos.py
│   ├── core/                   # Core configuration
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── security.py
│   │   └── logging.py
│   ├── models/                 # Database models
│   │   ├── __init__.py
│   │   ├── job.py
│   │   └── result.py
│   ├── schemas/                # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── photo.py
│   │   ├── staging.py
│   │   └── video.py
│   ├── services/               # Business logic
│   │   ├── classification/
│   │   │   ├── __init__.py
│   │   │   ├── classifier.py
│   │   │   └── models.py
│   │   ├── staging/
│   │   │   ├── __init__.py
│   │   │   ├── pipeline.py
│   │   │   └── design_kit.py
│   │   ├── video/
│   │   │   ├── __init__.py
│   │   │   ├── renderer.py
│   │   │   ├── clip_generator.py
│   │   │   └── audio_sync.py
│   │   └── storage/
│   │       ├── __init__.py
│   │       └── s3_client.py
│   ├── tasks/                  # Celery tasks
│   │   ├── __init__.py
│   │   ├── classification.py
│   │   ├── staging.py
│   │   └── video.py
│   ├── utils/                  # Utilities
│   │   ├── __init__.py
│   │   ├── image.py
│   │   ├── video.py
│   │   └── audio.py
│   ├── worker.py               # Celery worker
│   └── main.py                 # FastAPI app
├── models/                     # ML model files (gitignored)
│   ├── classification/
│   ├── staging/
│   └── object_detection/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── alembic/                    # Database migrations
│   ├── versions/
│   └── env.py
├── scripts/
│   └── download_models.py
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── alembic.ini
├── pytest.ini
├── pyproject.toml
├── README.md
├── AGENTS.md
├── IMPLEMENTATION_PLAN.md
└── INITIAL_STRUCTURE.md
```

### Naming Conventions
- **Files/Modules**: snake_case (`photo_classifier.py`)
- **Classes**: PascalCase (`PhotoClassifier`)
- **Functions**: snake_case (`classify_photo()`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_IMAGE_SIZE`)
- **Private**: Leading underscore (`_internal_method()`)

### API Endpoint Pattern
```python
from fastapi import APIRouter, HTTPException, Depends
from app.core.security import verify_api_key
from app.schemas.photo import ClassificationRequest, ClassificationResponse
from app.tasks.classification import classify_photos_task

router = APIRouter()

@router.post("/internal/photos/classify", response_model=ClassificationResponse)
async def classify_photos(
    request: ClassificationRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Classify photos asynchronously.

    This endpoint is internal and requires API key authentication.
    """
    task = classify_photos_task.delay(
        photo_ids=request.photo_ids,
        callback_url=request.callback_url
    )

    return ClassificationResponse(
        task_id=task.id,
        status="pending"
    )
```

### Service Pattern
```python
from typing import List, Dict
import torch
from PIL import Image
from app.services.storage import S3Client

class PhotoClassifier:
    """Classify real estate photos using ML models."""

    def __init__(self):
        self.model = self._load_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.storage = S3Client()

    def classify(self, photo_url: str) -> Dict:
        """
        Classify a single photo.

        Args:
            photo_url: URL of photo to classify

        Returns:
            Classification results with room type, quality score, etc.
        """
        # Download image
        image = self.storage.download_image(photo_url)

        # Preprocess
        tensor = self._preprocess(image)

        # Inference
        with torch.no_grad():
            output = self.model(tensor.to(self.device))

        # Postprocess
        results = self._postprocess(output)

        return results

    def _load_model(self):
        """Load pre-trained classification model."""
        model_path = "models/classification/resnet50_rooms.pth"
        model = torch.load(model_path, map_location="cpu")
        model.eval()
        return model

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model."""
        # Resize, normalize, etc.
        pass

    def _postprocess(self, output: torch.Tensor) -> Dict:
        """Convert model output to classification results."""
        pass
```

### Celery Task Pattern
```python
from celery import Task
from app.worker import celery_app
from app.services.classification import PhotoClassifier
import requests

class CallbackTask(Task):
    """Base task with callback support."""

    def on_success(self, retval, task_id, args, kwargs):
        """Send success callback."""
        callback_url = kwargs.get('callback_url')
        if callback_url:
            requests.post(callback_url, json={
                'task_id': task_id,
                'status': 'completed',
                'result': retval
            }, headers={'X-API-Key': settings.API_KEY})

@celery_app.task(
    base=CallbackTask,
    bind=True,
    max_retries=3,
    default_retry_delay=60
)
def classify_photos_task(self, photo_ids: List[str], callback_url: str):
    """
    Classify multiple photos.

    Args:
        photo_ids: List of photo IDs to classify
        callback_url: URL to call when complete
    """
    try:
        classifier = PhotoClassifier()
        results = []

        for photo_id in photo_ids:
            result = classifier.classify(photo_id)
            results.append({
                'photo_id': photo_id,
                **result
            })

        return {
            'photo_ids': photo_ids,
            'results': results
        }

    except Exception as exc:
        # Retry with exponential backoff
        raise self.retry(exc=exc)
```

### Error Handling Pattern
```python
from fastapi import HTTPException
from app.core.logging import logger

class ProcessingError(Exception):
    """Base exception for processing errors."""
    pass

class ModelLoadError(ProcessingError):
    """Model failed to load."""
    pass

class InferenceError(ProcessingError):
    """Model inference failed."""
    pass

# In service:
try:
    result = self.model.predict(image)
except torch.cuda.OutOfMemoryError as e:
    logger.error(f"GPU OOM: {e}")
    raise InferenceError("GPU out of memory. Try reducing batch size.")
except Exception as e:
    logger.exception(f"Unexpected error: {e}")
    raise ProcessingError(f"Classification failed: {str(e)}")
```

### Testing Pattern
```python
import pytest
from PIL import Image
from app.services.classification import PhotoClassifier

@pytest.fixture
def classifier():
    return PhotoClassifier()

@pytest.fixture
def sample_image():
    # Create test image
    return Image.new('RGB', (800, 600), color='white')

def test_classify_interior(classifier, sample_image):
    """Test classification of interior photo."""
    result = classifier.classify(sample_image)

    assert 'room_type' in result
    assert 'is_interior' in result
    assert result['is_interior'] is True
    assert result['confidence'] > 0.5

def test_classify_handles_invalid_image(classifier):
    """Test handling of invalid image."""
    with pytest.raises(ValueError):
        classifier.classify(None)
```

## Architectural Principles

### 1. Stateless API
- API endpoints don't maintain state
- All state in database or Redis
- Horizontal scaling friendly

### 2. Async Everything
- Long operations via Celery
- Never block HTTP request
- Return task ID immediately

### 3. Resource Aware
- Monitor GPU memory
- Queue depth awareness
- Graceful degradation

### 4. Idempotent Tasks
- Tasks can be retried safely
- Use task IDs to prevent duplicates
- Check if work already done

### 5. Caching Strategy
- Cache ML model outputs
- Cache intermediate results
- Enable fast regeneration

### 6. Fail Fast
- Validate inputs early
- Check resources before processing
- Clear error messages

## Communication with Backend

### Receiving Requests
```python
# Backend calls media service
POST /internal/photos/classify
Headers: { "X-API-Key": "secret" }
{
  "listing_id": "uuid",
  "photo_ids": ["id1", "id2"],
  "callback_url": "https://backend/webhooks/classification-complete"
}

# Response:
202 Accepted
{
  "task_id": "celery-task-id",
  "status": "pending",
  "estimated_duration": 120
}
```

### Sending Callbacks
```python
# Media service posts to backend webhook
POST https://backend/webhooks/classification-complete
Headers: {
  "X-API-Key": "secret",
  "X-Idempotency-Key": "task-id-123"
}
{
  "listing_id": "uuid",
  "results": [
    {
      "photo_id": "id1",
      "room_type": "living_room",
      "is_interior": true,
      "quality_score": 0.92,
      "classification": {...}
    }
  ]
}
```

### Progress Updates
```python
# Send progress during long operations
POST https://backend/webhooks/video-progress
{
  "video_id": "uuid",
  "progress_percent": 42,
  "stage": "rendering",
  "estimated_time_remaining": 180
}
```

## Types of Changes Allowed

### ✅ Always Allowed
- New ML models
- New processing pipelines
- Performance optimizations
- Bug fixes in processing logic
- New API endpoints (coordinate with backend)
- Image/video algorithm improvements
- Quality check enhancements

### ⚠️ Requires Review
- API contract changes
- Callback payload changes
- Major model architecture changes
- Resource allocation changes
- Task retry logic changes

### ❌ Never Allowed
- User authentication/authorization logic
- Business rules (pricing, permissions)
- Direct frontend communication
- Storing business data
- Template metadata ownership

## ML Model Management

### Model Versioning
```python
# models/classification/
# - v1.0/resnet50_rooms.pth
# - v1.1/resnet50_rooms.pth
# - v2.0/efficientnet_rooms.pth

# In code:
MODEL_VERSION = os.getenv("CLASSIFICATION_MODEL_VERSION", "v2.0")
model_path = f"models/classification/{MODEL_VERSION}/resnet50_rooms.pth"
```

### Model Download
```python
# scripts/download_models.py
def download_models():
    """Download ML models from S3 or Hugging Face."""
    models = [
        ("classification", "resnet50_rooms.pth"),
        ("staging", "segmentation_model.pth"),
        ("object_detection", "yolov8n.pt"),
    ]

    for category, filename in models:
        download_from_s3(f"models/{category}/{filename}")
```

### Model Caching
- Cache models in memory (singleton pattern)
- Warm up models on worker start
- Implement model pooling for high concurrency

## FFmpeg Usage

### Video Rendering
```python
import ffmpeg

def render_video(clips: List[str], output_path: str, audio_path: str):
    """
    Render video from clips with audio.

    Args:
        clips: List of clip file paths
        output_path: Output video path
        audio_path: Audio track path
    """
    # Create concat file
    concat_file = create_concat_file(clips)

    # FFmpeg command
    stream = ffmpeg.input(concat_file, format='concat', safe=0)
    audio = ffmpeg.input(audio_path)

    stream = ffmpeg.output(
        stream,
        audio,
        output_path,
        vcodec='libx264',
        acodec='aac',
        preset='medium',
        crf=23,
        **{'b:v': '5M', 'b:a': '192k'}
    )

    # Run with progress monitoring
    stream.run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
```

### Progress Monitoring
```python
def run_ffmpeg_with_progress(cmd, callback):
    """Run FFmpeg and report progress."""
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

    for line in process.stderr:
        if "time=" in line:
            time_str = line.split("time=")[1].split()[0]
            progress = parse_time_to_percent(time_str, total_duration)
            callback(progress)
```

## Performance Guidelines

### GPU Usage
```python
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Batch processing
def process_batch(images: List[Image.Image]):
    batch = torch.stack([preprocess(img) for img in images])
    with torch.no_grad():
        outputs = model(batch.to(device))
    return outputs

# Clear GPU memory
torch.cuda.empty_cache()
```

### Memory Management
```python
# Use context managers
with torch.no_grad():
    # Disable gradient computation for inference
    output = model(input)

# Process in chunks
def process_large_batch(items, chunk_size=10):
    for i in range(0, len(items), chunk_size):
        chunk = items[i:i+chunk_size]
        yield process_batch(chunk)
```

### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def classify_photo_cached(photo_hash: str) -> Dict:
    """Cached classification to avoid reprocessing."""
    return classify_photo(photo_hash)
```

## Testing Requirements

- minimal testing only, save tokens as much as possible!
<!-- ### Unit Tests -->
<!-- - All service classes -->
<!-- - Utility functions -->
<!-- - Data preprocessing/postprocessing -->

<!-- ### Integration Tests -->
<!-- - API endpoints -->
<!-- - Celery tasks -->
<!-- - Callback webhooks -->

<!-- ### Model Tests -->
<!-- - Model loading -->
<!-- - Inference accuracy (on test set) -->
<!-- - Performance benchmarks -->

## Security Considerations

- API key authentication for all endpoints
- Validate all file inputs (size, format)
- Sanitize file paths
- Rate limit endpoints
- Don't log sensitive data
- Secure model files

## Monitoring Requirements

- GPU utilization
- Memory usage
- Task queue depth
- Task duration (p50, p95, p99)
- Error rates
- Model inference time

## Documentation Requirements

- Docstrings for all public functions
- Type hints for all functions
- README for each major service
- Update this AGENTS.md when responsibilities change

## Getting Help

When contributing, always:
1. Read this document first
2. Check existing services for patterns
3. Write tests for all code
4. Use type hints
5. Profile performance-critical code
6. Ask before adding large dependencies
