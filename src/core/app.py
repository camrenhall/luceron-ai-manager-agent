"""
FastAPI application factory and lifecycle management.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config.settings import settings
from ..core.http_client import init_http_client, close_http_client
from ..middleware.security import security_middleware
from ..routes import health, chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    # Startup
    await init_http_client()
    
    yield
    
    # Shutdown
    await close_http_client()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="Luceron AI Manager Agent",
        description="Central orchestration layer for the Luceron AI eDiscovery Platform",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Security middleware
    app.middleware("http")(security_middleware)
    
    # Include routers
    app.include_router(health.router)
    app.include_router(chat.router)
    
    return app