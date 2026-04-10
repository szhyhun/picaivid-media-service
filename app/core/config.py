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
    ANALYSIS_MATCH_ENGINE: str = "mast3r_graph"
    MAST3R_REPO_DIR: str | None = None
    MAST3R_REPO_ARCHIVE_S3_URI: str | None = None
    MAST3R_MODEL_CHECKPOINT: str | None = None
    MAST3R_MODEL_CHECKPOINT_S3_URI: str | None = None
    MAST3R_RETRIEVAL_CHECKPOINT: str | None = None
    MAST3R_RETRIEVAL_CHECKPOINT_S3_URI: str | None = None
    MAST3R_RETRIEVAL_CODEBOOK: str | None = None
    MAST3R_RETRIEVAL_CODEBOOK_S3_URI: str | None = None
    MAST3R_IMAGE_SIZE: int = 512
    MAST3R_SCENE_GRAPH_ANCHORS: int = 20
    MAST3R_SCENE_GRAPH_K: int = 10
    MAST3R_MATCHING_CONFIDENCE_THRESHOLD: float = 5.0
    MAST3R_LR1: float = 0.07
    MAST3R_NITER1: int = 300
    MAST3R_LR2: float = 0.01
    MAST3R_NITER2: int = 300
    MAST3R_SHARED_INTRINSICS: bool = False
    MAST3R_MIN_RETRIEVAL_SCORE: float = 0.15
    MAST3R_MIN_RECIPROCAL_MATCHES: int = 24
    MAST3R_MIN_POINTMAP_CONSISTENCY: float = 0.25
    MAST3R_MIN_PARALLAX_SCORE: float = 0.015
    MAST3R_MIN_GRAPH_EDGE_SCORE: float = 0.34
    MAST3R_SUGGESTION_MIN_EDGE_SCORE: float = 0.24
    OPENCLIP_MODEL: str = "ViT-B-32"
    OPENCLIP_PRETRAINED: str = "openai"

    # Clustering behavior
    DELETE_OBVIOUS_DUPLICATES: bool = True
    REQUIRE_GEOMETRIC_TRANSITIONS: bool = True
    REQUIRE_DIRECTION_FOR_TRANSITIONS: bool = False
    HARD_TRANSITION_MIN_SIDE_OVERLAP: float = 0.06
    HARD_TRANSITION_MIN_CENTER_OVERLAP: float = 0.09
    HARD_TRANSITION_MIN_OVERLAP_RATIO: float = 0.08
    GEOMETRIC_SCORE_WEIGHT: float = 0.90
    SEMANTIC_SCORE_WEIGHT: float = 0.10
    GEOMETRY_ONLY_CLUSTER_MEMBERSHIP: bool = True
    STRICT_SEMANTIC_COMPONENT_CONNECTIVITY: bool = True
    COMPONENT_SEMANTIC_ADJ_MIN: float = 0.78
    COMPONENT_SEMANTIC_DIST2_MIN: float = 0.82
    COMPONENT_SEMANTIC_RECOVERY_MIN: float = 0.88
    COMPONENT_FRONT_RECOVERY_MIN: float = 0.80
    COMPONENT_SEMANTIC_MAX_GAP: int = 2
    COMPONENT_SAME_LABEL_ADJ_MIN: float = 0.70
    COMPONENT_SAME_LABEL_DIST2_MIN: float = 0.74
    COMPONENT_AMBIGUOUS_SAME_LABEL_MIN: float = 0.82
    COMPONENT_CROSS_LABEL_ADJ_MIN: float = 0.84
    COMPONENT_CROSS_LABEL_DIST2_MIN: float = 0.88

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


settings = Settings()
