"""Application configuration loaded from environment variables."""
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # API
    API_KEY: str = "dev-api-key"
    CORS_ORIGINS: List[str] = ["http://localhost:3001"]

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

    # ML Models
    MODEL_CACHE_DIR: str = "./ml_models"
    ANALYSIS_MATCH_ENGINE: str = "vggt_scene_graph"
    VGGT_REPO_DIR: str | None = None
    VGGT_REPO_ARCHIVE_S3_URI: str | None = None
    VGGT_MODEL_CHECKPOINT: str | None = None
    VGGT_MODEL_CHECKPOINT_S3_URI: str | None = None
    VGGT_IMAGE_SIZE: int = 518
    VGGT_WINDOW_SIZE: int = 30
    VGGT_WINDOW_OVERLAP: int = 6
    VGGT_RELATION_SCORE_THRESHOLD: float = 0.50
    VGGT_BRIDGE_SCORE_THRESHOLD: float = 0.42
    VGGT_TRACK_POINTS_PER_IMAGE: int = 256
    VGGT_USE_BUNDLE_ADJUSTMENT_EXPORT: bool = False
    OPENCLIP_MODEL: str = "ViT-B-32"
    OPENCLIP_PRETRAINED: str = "openai"

    # Clustering behavior
    DELETE_OBVIOUS_DUPLICATES: bool = True
    GEOMETRY_ONLY_CLUSTER_MEMBERSHIP: bool = True
    SCENE_RELATION_POSITION_GAP_WEIGHT: float = 0.08
    SCENE_RELATION_ROOM_LABEL_WEIGHT: float = 0.12
    SCENE_INTERIOR_CONFIDENCE_THRESHOLD: float = 0.60
    SCENE_EXTERIOR_CONFIDENCE_THRESHOLD: float = 0.52

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


settings = Settings()
