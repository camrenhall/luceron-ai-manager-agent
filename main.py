"""
Luceron AI Manager Agent - Production Implementation
Application entry point for the intelligent orchestration layer.
"""
import uvicorn

from src.core.app import create_app
from src.config.settings import settings

# Create the FastAPI application
app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.MANAGER_PORT,
        log_level="info"
    )