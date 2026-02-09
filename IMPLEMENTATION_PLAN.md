# Media Service Implementation Plan

This document outlines the phased implementation plan for the Virtual Listing Studio media processing service.

## Phase 1: MVP (Weeks 1-4)

Goal: Basic ML and media processing to support frontend/backend MVP.

### 1.1 Project Setup & Infrastructure

**Tasks**:
- Initialize Python project structure
- Set up FastAPI application
- Configure Celery with Redis
- Set up PostgreSQL for operational state
- Configure Alembic for migrations
- Set up pytest for testing
- Configure black, flake8, mypy

**Deliverables**:
- Working FastAPI application
- Celery worker configuration
- Database setup
- Test framework

**Database Tables**:
```python
# jobs
- id (uuid, primary key)
- type (string) # classification, staging, video
- status (enum: pending, processing, completed, failed)
- input_data (jsonb)
- result (jsonb)
- error (text)
- started_at (datetime)
- completed_at (datetime)
- created_at, updated_at
```

---

### 1.2 Core API Framework

**Tasks**:
- Health check endpoint
- API key authentication
- Request/response schemas (Pydantic)
- Error handling middleware
- Logging configuration
- CORS configuration (internal only)

**Deliverables**:
- Basic API structure
- Authentication
- Health monitoring

**API Endpoints**:
- `GET /health` (public)
- `GET /internal/status` (authenticated)

---

### 1.3 Storage Integration

**Tasks**:
- S3 client implementation
- Download images from S3
- Upload processed images to S3
- Temporary file management
- File cleanup utilities

**Deliverables**:
- S3 integration
- File download/upload
- Temp file handling

**Utilities**:
```python
class S3Client:
    def download_image(url: str) -> Image
    def upload_image(image: Image, key: str) -> str
    def download_file(url: str, local_path: str)
    def upload_file(local_path: str, key: str) -> str
```

---

### 1.4 Photo Classification

**Tasks**:
- Download/integrate pre-trained image classification model
- Implement photo classifier service
- Room type detection (living room, bedroom, kitchen, etc.)
- Interior/exterior detection
- Quality score calculation
- Classification API endpoint
- Classification Celery task
- Callback webhook to backend

**Deliverables**:
- Working photo classification
- API endpoint
- Async task processing

**ML Models**:
- ResNet50 or EfficientNet for room classification
- Binary classifier for interior/exterior

**API Endpoints**:
- `POST /internal/photos/classify`

**Celery Tasks**:
- `classify_photos_task`

**Callback**:
- `POST {BACKEND_URL}/webhooks/classification-complete`

---

### 1.5 Basic Video Generation

**Tasks**:
- FFmpeg integration
- Simple slideshow generation
- Static transition between photos
- Add audio track
- Video rendering task
- Progress reporting
- Callback webhook to backend

**Deliverables**:
- Basic video generation
- Audio sync
- Progress tracking

**API Endpoints**:
- `POST /internal/video/render`

**Celery Tasks**:
- `render_video_task`

**Callbacks**:
- `POST {BACKEND_URL}/webhooks/video-progress`
- `POST {BACKEND_URL}/webhooks/video-complete`

---

### 1.6 Error Handling & Retry Logic

**Tasks**:
- Implement retry logic with exponential backoff
- GPU error handling (OOM, etc.)
- Network error handling
- Task failure notifications
- Idempotency for callbacks

**Deliverables**:
- Robust error handling
- Automatic retries
- Failure notifications

---

### 1.7 Basic Monitoring

**Tasks**:
- Logging configuration
- Task duration tracking
- Error rate tracking
- Queue depth monitoring
- Simple metrics endpoint

**Deliverables**:
- Operational logging
- Basic metrics

**API Endpoints**:
- `GET /internal/metrics`

---

### MVP Phase 1 Deliverables Summary

**Core Features**:
- Photo classification (room type, quality)
- Basic video generation (slideshow with music)
- S3 integration
- Async task processing
- Error handling & retries
- Progress reporting

**Key Metrics**:
- Photo classification: < 5s per photo
- Video generation: < 2min for 20 photos
- Error rate: < 5%

---

## Phase 2: Core Product (Weeks 5-10)

Goal: Advanced ML features and production-ready processing.

### 2.1 Virtual Staging Pipeline

**Tasks**:
- Image segmentation model (segment walls, floors, furniture)
- Furniture removal model
- Texture application
- Design kit integration
- Staging quality check
- Virtual staging API endpoint
- Staging Celery task

**Deliverables**:
- Full virtual staging pipeline
- Design kit support
- Quality validation

**ML Models**:
- Semantic segmentation (DeepLabV3, Mask R-CNN)
- Inpainting model for furniture removal
- Style transfer model (optional)

**API Endpoints**:
- `POST /internal/staging/process`
- `GET /internal/design-kits/:id`

**Celery Tasks**:
- `stage_photos_task`

**Callbacks**:
- `POST {BACKEND_URL}/webhooks/staging-complete`

---

### 2.2 Advanced Video Features

**Tasks**:
- Ken Burns effect (zoom/pan)
- Speed ramp calculations
- Dynamic transitions
- Beat-synced cuts
- Audio analysis (BPM detection)
- Template-based rendering
- Custom branding overlay

**Deliverables**:
- Professional video generation
- Audio sync with beats
- Template support
- Branding integration

**Dependencies**:
- `librosa` for audio analysis
- FFmpeg complex filters

---

### 2.3 Audio Processing

**Tasks**:
- BPM detection
- Beat detection
- Audio waveform generation
- Music trimming/looping
- Audio normalization
- Audio mixing

**Deliverables**:
- Audio analysis service
- Beat-synced video cuts
- Waveform visualization

**API Endpoints**:
- `POST /internal/audio/analyze`

---

### 2.4 Object Detection

**Tasks**:
- YOLO or similar object detection model
- Detect key features (pool, fireplace, view)
- Generate photo tags
- Smart cropping based on objects
- Focus point calculation

**Deliverables**:
- Object detection service
- Feature tagging
- Smart cropping

**ML Models**:
- YOLOv8 or similar

---

### 2.5 Quality Enhancement

**Tasks**:
- Image upscaling (ESRGAN or similar)
- Color correction
- Brightness/contrast adjustment
- Noise reduction
- HDR processing (optional)

**Deliverables**:
- Image enhancement pipeline
- Quality improvement tools

**ML Models**:
- Super-resolution model (Real-ESRGAN)

---

### 2.6 Clip Generation & Caching

**Tasks**:
- Generate individual clips for each photo
- Cache clips for regeneration
- Smart cache invalidation
- Clip reuse logic
- Cache storage management

**Deliverables**:
- Efficient clip caching
- Fast regeneration

**Database Tables**:
```python
# cached_clips
- id (uuid)
- photo_id (uuid)
- config_hash (string) # hash of render config
- file_path (string)
- duration (integer)
- created_at
- accessed_at
- expires_at
```

---

### 2.7 Video Timeline Logic

**Tasks**:
- Calculate optimal clip durations based on music
- Transition timing
- Beat alignment
- Speed ramp curves
- Pacing algorithms

**Deliverables**:
- Timeline calculation service
- Smart pacing

---

### 2.8 Design Kit System

**Tasks**:
- Design kit storage
- Asset management
- Consistency checks across photos
- Style application
- Color palette extraction

**Deliverables**:
- Design kit management
- Consistent styling

---

### 2.9 Batch Processing

**Tasks**:
- Parallel photo processing
- GPU batch inference
- Efficient resource utilization
- Priority queue management

**Deliverables**:
- Optimized batch processing
- Better GPU utilization

---

### 2.10 Performance Optimization

**Tasks**:
- Model quantization
- ONNX runtime integration
- GPU memory optimization
- Task scheduling optimization
- Worker auto-scaling logic

**Deliverables**:
- 2x faster processing
- Better resource efficiency

---

### Core Product Phase 2 Deliverables Summary

**Enhanced Features**:
- Virtual staging
- Advanced video effects
- Audio analysis & sync
- Object detection
- Quality enhancement
- Clip caching
- Design kit system

**Key Metrics**:
- Photo classification: < 2s per photo
- Virtual staging: < 10s per photo
- Video generation: < 1min for 20 photos
- 95%+ accuracy on classification

---

## Phase 3: Scale & Optimization (Weeks 11-16)

Goal: Enterprise-scale processing with advanced features.

### 3.1 Multi-GPU Support

**Tasks**:
- Distributed inference
- GPU pooling
- Load balancing across GPUs
- GPU affinity for tasks

**Deliverables**:
- Multi-GPU processing
- Horizontal scaling

---

### 3.2 Advanced ML Models

**Tasks**:
- Fine-tune models on real estate data
- Custom room classification model
- Better staging models
- Photo quality prediction
- Scene understanding

**Deliverables**:
- Custom ML models
- Better accuracy
- Real estate-specific features

---

### 3.3 Real-Time Processing

**Tasks**:
- Low-latency classification
- Streaming video preview
- Progressive enhancement
- Incremental rendering

**Deliverables**:
- Near real-time processing
- Preview generation

---

### 3.4 Advanced Caching

**Tasks**:
- Distributed caching (Redis cluster)
- Model result caching
- Intermediate result caching
- Cache warming strategies

**Deliverables**:
- Comprehensive caching
- 10x faster regeneration

---

### 3.5 Quality Assurance Automation

**Tasks**:
- Automated quality checks
- Anomaly detection
- Output validation
- Consistency verification
- A/B testing framework

**Deliverables**:
- Automated QA
- Quality metrics

---

### 3.6 Advanced Video Features

**Tasks**:
- 3D transitions
- Parallax effects
- Advanced color grading
- Motion tracking
- Depth-based effects

**Deliverables**:
- Cinematic video effects
- Professional-grade output

---

### 3.7 Model Management System

**Tasks**:
- Model versioning
- A/B testing models
- Model performance tracking
- Canary deployments
- Rollback capabilities

**Deliverables**:
- Production model management
- Safe model updates

---

### 3.8 Resource Optimization

**Tasks**:
- Dynamic worker scaling
- Cost optimization (spot instances)
- Resource prediction
- Intelligent job scheduling
- Power management

**Deliverables**:
- 50% cost reduction
- Better resource utilization

---

### 3.9 Advanced Monitoring

**Tasks**:
- Distributed tracing
- APM integration
- Custom metrics
- Alerting system
- Performance dashboards

**Deliverables**:
- Production monitoring
- Proactive alerts

---

### 3.10 ML Pipeline Automation

**Tasks**:
- Automated model training
- Data collection pipeline
- Model evaluation
- Continuous learning
- Feedback loop

**Deliverables**:
- MLOps pipeline
- Continuous improvement

---

### Scale & Optimization Phase 3 Deliverables Summary

**Enterprise Features**:
- Multi-GPU processing
- Custom ML models
- Real-time processing
- Advanced caching
- Automated QA
- Advanced video effects
- Model management
- Resource optimization

**Key Metrics**:
- Photo classification: < 1s per photo
- Virtual staging: < 5s per photo
- Video generation: < 30s for 20 photos
- 99%+ uptime
- 99%+ accuracy

---

## ML Models Summary

### Phase 1 (MVP)
- **Room Classification**: ResNet50/EfficientNet
- **Basic Quality**: Simple CNN

### Phase 2 (Core Product)
- **Segmentation**: DeepLabV3/Mask R-CNN
- **Inpainting**: LaMa or similar
- **Object Detection**: YOLOv8
- **Super Resolution**: Real-ESRGAN
- **Audio Analysis**: librosa + custom

### Phase 3 (Scale)
- **Custom Room Classifier**: Fine-tuned on real estate
- **Custom Staging Model**: Trained on design kits
- **Quality Prediction**: Custom regression model
- **Scene Understanding**: Multi-task model

---

## Processing Pipeline Examples

### Photo Classification Pipeline
```
1. Download image from S3
2. Preprocess (resize, normalize)
3. Run classification model
4. Postprocess results
5. Extract features
6. Calculate quality score
7. Upload results
8. Callback to backend
```

### Virtual Staging Pipeline
```
1. Download original photo
2. Run segmentation model
3. Detect furniture/walls/floors
4. Remove existing furniture (inpainting)
5. Load design kit assets
6. Apply textures and objects
7. Run consistency check
8. Upscale if needed
9. Upload staged photo
10. Callback to backend
```

### Video Generation Pipeline
```
1. Download all photos
2. Download music track
3. Analyze audio (BPM, beats)
4. Calculate timeline (durations, transitions)
5. Generate clips for each photo:
   - Apply Ken Burns effect
   - Add transitions
   - Apply branding
6. Render audio waveform
7. Concatenate clips
8. Mix with audio
9. Encode final video (FFmpeg)
10. Upload to S3
11. Callback to backend
```

---

## Celery Task Summary

- `classify_photos_task`
- `stage_photos_task`
- `render_video_task`
- `analyze_audio_task`
- `generate_clip_task`
- `enhance_photo_task`
- `detect_objects_task`

---

## API Endpoints Summary

### Internal (Backend only)
- `POST /internal/photos/classify`
- `POST /internal/photos/enhance`
- `POST /internal/staging/process`
- `POST /internal/video/render`
- `POST /internal/audio/analyze`
- `GET /internal/design-kits/:id`
- `GET /internal/metrics`
- `GET /internal/status`

### Public (Health checks)
- `GET /health`

---

## External Dependencies

### ML Libraries
- PyTorch
- transformers (Hugging Face)
- timm
- ultralytics (YOLO)
- opencv-python
- scikit-image
- pillow

### Media Processing
- ffmpeg-python
- librosa
- pydub

### Infrastructure
- FastAPI
- Celery
- Redis
- PostgreSQL
- boto3 (AWS SDK)

---

## Hardware Requirements

### Development
- CPU: 4+ cores
- RAM: 16GB
- GPU: Optional (CUDA-capable)
- Storage: 50GB

### Production (per worker)
- CPU: 8+ cores
- RAM: 32GB
- GPU: NVIDIA T4 or better (16GB+ VRAM)
- Storage: 100GB SSD

### Scaling
- Classification: CPU workers
- Staging: GPU workers
- Video: Mixed (CPU for FFmpeg, GPU for effects)

---

## Success Metrics

### Phase 1 (MVP)
- Classification: < 5s per photo
- Video: < 2min for 20 photos
- 90%+ classification accuracy

### Phase 2 (Core Product)
- Classification: < 2s per photo
- Staging: < 10s per photo
- Video: < 1min for 20 photos
- 95%+ accuracy

### Phase 3 (Scale)
- Classification: < 1s per photo
- Staging: < 5s per photo
- Video: < 30s for 20 photos
- 99%+ accuracy
- 99.9% uptime

---

## Cost Optimization

### Phase 1
- Use spot instances for workers
- Cache model outputs
- Batch processing

### Phase 2
- Model quantization
- ONNX runtime
- Intelligent caching

### Phase 3
- Auto-scaling
- Resource prediction
- Reserved instances for base load
- Spot for burst capacity

---

## Testing Strategy

### Unit Tests
- All service classes
- Utility functions
- Data processing

### Integration Tests
- API endpoints
- Celery tasks
- S3 integration

### Performance Tests
- Model inference time
- Video generation time
- Memory usage
- GPU utilization

### Accuracy Tests
- Classification accuracy on test set
- Staging quality metrics
- Video quality metrics

---

## Monitoring Metrics

- Task queue depth
- Task processing time (p50, p95, p99)
- GPU utilization
- Memory usage
- Error rates
- Model inference time
- Video generation time
- Callback success rate

---

## Next Steps

1. Review plan with team
2. Prioritize features
3. Set up infrastructure
4. Begin Phase 1 implementation
5. Establish model training pipeline
