# Build stage for model download (can be cached as separate layer)
FROM python:3.11-slim as model-downloader

WORKDIR /app

# Install dependencies for model download
RUN pip install --no-cache-dir \
    transformers>=4.36.0 \
    torch==2.13.0 \
    torchvision==0.28.0 \
    open-clip-torch>=2.24.0 \
    huggingface_hub

# Copy download script
COPY scripts/download_models.py scripts/

# Download models during build (cached in Docker layer)
RUN python scripts/download_models.py --cache-dir /models

# ============================================
# Production image
# ============================================
FROM python:3.11-slim as production

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pre-downloaded models from builder stage
COPY --from=model-downloader /models /app/ml_models

# Copy application code
COPY app/ app/
COPY alembic/ alembic/
COPY alembic.ini .
COPY scripts/ scripts/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_CACHE_DIR=/app/ml_models
ENV HF_HOME=/app/ml_models/huggingface
ENV TRANSFORMERS_CACHE=/app/ml_models/huggingface
ENV HF_HUB_OFFLINE=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
