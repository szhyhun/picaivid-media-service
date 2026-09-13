"""Application configuration loaded from environment variables."""
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # API
    API_KEY: str = "dev-api-key"
    CORS_ORIGINS: List[str] = ["http://localhost:3001", "http://localhost:3003"]

    # Database (Postgres is system of record)
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/picaivid_development"

    # AWS S3 (set credentials/endpoints explicitly for local MinIO; omit for AWS IAM roles)
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    S3_BUCKET: str = "picaivid-dev"
    S3_ENDPOINT: str | None = None

    # AWS SQS (set endpoint explicitly for local LocalStack; omit for AWS)
    SQS_ENDPOINT: str | None = None
    SQS_QUEUE_URL: str = "http://sqs.us-east-1.localhost.localstack.cloud:4566/000000000000/picaivid-jobs"

    # Rails webhook URL for status updates
    RAILS_WEBHOOK_URL: str | None = "http://localhost:3000"

    # Worker Type
    WORKER_TYPE: str = "cpu"
    WORKER_PHASES: str | None = None

    # ML Models
    MODEL_CACHE_DIR: str = "./ml_models"
    ANALYSIS_MATCH_ENGINE: str = "vggt_scene_graph"
    # VGGT-Omega is the private reconstruction runtime. Its repository and
    # checkpoint are supplied outside source control on each worker.
    VGGT_REPO_DIR: str | None = None
    VGGT_REPO_COMMIT: str | None = None
    VGGT_REPO_ARCHIVE_S3_URI: str | None = None
    VGGT_MODEL_CHECKPOINT: str | None = None
    VGGT_MODEL_CHECKPOINT_S3_URI: str | None = None
    VGGT_DEVICE: str = "auto"
    VGGT_PRECISION: str = "auto"
    VGGT_IMAGE_SIZE: int = 512
    VGGT_IMAGE_MODE: str = "balanced"
    VGGT_IMAGE_SAVE_WORKERS: int = 4
    # Stage profiling synchronizes the accelerator after every Omega head and is
    # intentionally off in normal runs because those barriers serialize GPU work.
    VGGT_PROFILE_STAGES: bool = False
    # MPS allocator growth was measured to degrade sustained pair inference. CUDA
    # keeps its caching allocator warm and is released only at the end of a job.
    VGGT_MPS_CACHE_RELEASE_EVERY: int = 25
    # Raw threshold-free pair evidence. Shared with the calibration sweep so a
    # re-analysis can reuse model inference and only rerun graph policy.
    VGGT_PAIR_CACHE_DIR: str = "./tmp_sweep/cache"
    OPENCLIP_MODEL: str = "ViT-B-32"
    OPENCLIP_PRETRAINED: str = "openai"
    OPENCLIP_BATCH_SIZE: int = 16
    # Object boxes for cinematic close-ups. This model is loaded only by the
    # review/detail path, not by the VGGT scene-graph worker.
    DETAIL_DETECTOR_MODEL: str = "IDEA-Research/grounding-dino-tiny"
    DETAIL_DETECTOR_MIN_SCORE: float = 0.35
    DETAIL_DETECTOR_BATCH_SIZE: int = 4

    # Video generation (Runware -> Vidu Q3 Turbo).
    #
    # Every shot buys RUNWARE_DURATION_SECONDS of source and the editor cuts a
    # beat-length slice out of the middle. Buying 3s costs a little more than
    # the ~1.2s we show, and it is what makes retiming and trimming free: any
    # tempo above 40 BPM fits inside it, so changing the music never
    # regenerates a clip.
    #
    # RESOLUTION defaults to the cheapest tier on purpose. Generation costs real
    # money per shot, so raising it is an explicit act, not a default anyone
    # inherits by forgetting.
    RUNWARE_API_KEY: str | None = None
    RUNWARE_API_URL: str = "https://api.runware.ai/v1"
    RUNWARE_MODEL: str = "vidu:4@2"
    RUNWARE_DURATION_SECONDS: int = 3
    RUNWARE_RESOLUTION: str = "540p"          # 720p for test, 1080p for production
    # `url` passes a presigned S3 URL. `data_uri` sends the image with the
    # request. `auto` uses URLs in production but switches to data URIs for
    # a private local MinIO endpoint that Runware cannot reach.
    RUNWARE_INPUT_MODE: str = "auto"
    # Runware publishes no hard rate limit but queues dynamically; their own
    # guidance is 2-4 concurrent for best throughput.
    RUNWARE_MAX_CONCURRENCY: int = 4
    RUNWARE_POLL_INTERVAL_SECONDS: float = 5.0
    RUNWARE_MAX_POLL_ATTEMPTS: int = 240
    RUNWARE_SUBMIT_URL_EXPIRY_SECONDS: int = 21600   # 6h: a queued job fetches late

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


settings = Settings()
