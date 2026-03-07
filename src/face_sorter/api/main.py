"""
FastAPI application main entry point.

This module initializes the FastAPI application with CORS, static file serving,
and route registration for the Face Sorter web UI.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from face_sorter.api.routes import router as api_router
from face_sorter.config import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for startup and shutdown events.

    Args:
        app: FastAPI application instance

    Yields:
        None
    """
    # Startup
    print("Starting Face Sorter Web API...")
    yield
    # Shutdown
    print("Shutting down Face Sorter Web API...")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Face Sorter API",
    description="Web API for face recognition and sorting",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes BEFORE static file mounts to avoid shadowing
app.include_router(api_router, prefix="/api")

# Mount static files for frontend
static_dir = Path(__file__).parent.parent / "web" / "frontend" / "dist"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="frontend")
else:
    print(f"Warning: Frontend static directory not found: {static_dir}")

# Serve images from cache directory
cache_dir = Path(settings.cache_dir)
if cache_dir.exists():
    app.mount("/images", StaticFiles(directory=str(cache_dir)), name="images")
else:
    print(f"Warning: Cache directory not found: {cache_dir}")


@app.get("/health")
async def health_check() -> dict[str, str]:
    """
    Health check endpoint.

    Returns:
        Dictionary with health status
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "face_sorter.api.main:app",
        host=settings.ui_host,
        port=settings.ui_port,
        reload=settings.ui_reload,
        log_level=settings.ui_log_level,
    )