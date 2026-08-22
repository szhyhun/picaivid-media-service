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
    WORKER_PHASES: str | None = None

    # ML Models
    MODEL_CACHE_DIR: str = "./ml_models"
    ANALYSIS_MATCH_ENGINE: str = "vggt_scene_graph"
    VGGT_REPO_DIR: str | None = None
    VGGT_REPO_COMMIT: str | None = None
    VGGT_REPO_ARCHIVE_S3_URI: str | None = None
    VGGT_MODEL_CHECKPOINT: str | None = None
    VGGT_MODEL_CHECKPOINT_S3_URI: str | None = None
    VGGT_DEVICE: str = "auto"
    VGGT_PRECISION: str = "auto"
    VGGT_IMAGE_SIZE: int = 518
    VGGT_RETRY_WINDOW_SIZE: int = 16
    VGGT_RETRY_WINDOW_OVERLAP: int = 6
    VGGT_FINAL_WINDOW_SIZE: int = 10
    VGGT_FINAL_WINDOW_OVERLAP: int = 4
    VGGT_TRACK_POINTS_PER_IMAGE: int = 64
    VGGT_TRACK_GROUP_MAX: int = 8
    VGGT_MAX_GEOMETRY_POINTS_PER_PAIR: int = 4096
    VGGT_ROMA_NEIGHBORS_PER_PHOTO: int = 6
    ALLOW_SYNTHETIC_GEOMETRY: bool = False
    VGGT_RELATION_SCORE_THRESHOLD: float = 0.50
    VGGT_BRIDGE_SCORE_THRESHOLD: float = 0.42
    VGGT_USE_BUNDLE_ADJUSTMENT_EXPORT: bool = False
    VGGT_INTERIOR_EXTERIOR_PENALTY: float = 0.18
    VGGT_INTERIOR_DRONE_PENALTY: float = 0.24
    VGGT_EXTERIOR_DRONE_PENALTY: float = 0.12
    VGGT_CROSS_DOMAIN_CONFIDENCE_BONUS: float = 0.08
    VGGT_BRIDGE_POSITION_GAP_MAX: int = 3
    VGGT_OUTLIER_CONFIDENCE_THRESHOLD: float = 0.44
    VGGT_MIXED_COMPONENT_SPLIT_THRESHOLD: float = 0.56
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
