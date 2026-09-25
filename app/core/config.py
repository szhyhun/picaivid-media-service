"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # API
    # Internal Rails-to-media authentication. The worker does not expose an
    # HTTP API, but the optional media API accepts calls from Rails only.
    API_KEY: str | None = None
    MEDIA_SERVICE_API_KEY: str | None = None
    MEDIA_SERVICE_WEBHOOK_KEY: str | None = None

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
    # CPU-only timeline/assembly commands use a separate queue when supplied;
    # the fallback keeps older local stacks and deployments compatible.
    SQS_CPU_QUEUE_URL: str | None = None

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

    # Video generation (Runware -> Luma Ray3.2).
    #
    # Every shot buys five seconds of source and the editor cuts a beat-length
    # slice out of the middle. Ray3.2 requires five seconds when frameImages
    # are supplied, and the extra headroom keeps retiming and trimming free.
    # HDR is intentionally on for the first real output. Ray3.2 limits HDR to
    # 720p or 1080p, so invalid combinations fail before a paid task is queued.
    RUNWARE_API_KEY: str | None = None
    RUNWARE_API_URL: str = "https://api.runware.ai/v1"
    RUNWARE_MODEL: str = "luma:ray@3.2"
    RUNWARE_DURATION_SECONDS: int = 5
    RUNWARE_RESOLUTION: str = "720p"
    RUNWARE_HDR: bool = True
    RUNWARE_EXR_EXPORT: bool = False
    RUNWARE_OUTPUT_QUALITY: int = 95
    RUNWARE_INCLUDE_COST: bool = False
    # `url` passes a presigned S3 URL. `data_uri` sends the image with the
    # request. `auto` uses URLs in production but switches to data URIs for
    # a private local MinIO endpoint that Runware cannot reach.
    RUNWARE_INPUT_MODE: str = "auto"
    # Runware publishes no hard rate limit but queues dynamically; their own
    # guidance is 2-4 concurrent for best throughput.
    RUNWARE_MAX_CONCURRENCY: int = 4
    RUNWARE_POLL_INTERVAL_SECONDS: float = 5.0
    # A provider task may take 1-20 minutes. The wall-clock limit is the real
    # safety valve; attempts remain a second guard if the poll interval changes.
    RUNWARE_MAX_WAIT_SECONDS: int = 1800
    RUNWARE_MAX_POLL_ATTEMPTS: int = 360
    RUNWARE_SUBMIT_URL_EXPIRY_SECONDS: int = 21600   # 6h: a queued job fetches late

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


settings = Settings()
