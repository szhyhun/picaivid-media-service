# Virtual Listing Studio - Media Service

Python FastAPI service for AI and media processing.

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env

# Run development server
uvicorn app.main:app --reload --port 8000
```

Visit http://localhost:8000

## Documentation

- [AGENTS.md](./AGENTS.md) - Agent contribution guidelines
- [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) - Implementation roadmap
- [INITIAL_STRUCTURE.md](./INITIAL_STRUCTURE.md) - Project structure guide

## Health Check

http://localhost:8000/health

## API Documentation

http://localhost:8000/docs (when running)
