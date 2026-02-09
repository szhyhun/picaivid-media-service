from fastapi import FastAPI
from datetime import datetime

app = FastAPI(
    title="Virtual Listing Studio Media Service",
    description="AI and media processing service",
    version="0.1.0",
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "media_service",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Virtual Listing Studio Media Service",
        "version": "0.1.0"
    }
