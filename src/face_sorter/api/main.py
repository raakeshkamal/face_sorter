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
from face_sorter.database.session_repository import SessionRepository
from face_sorter.services.task_tracker import task_tracker

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

    # Initialize database indexes
    try:
        session_repo = SessionRepository()
        await session_repo.create_indexes()
        print("Database indexes created successfully.")
    except Exception as e:
        print(f"Warning: Failed to create database indexes: {e}")

    # Clean up orphaned sessions (marked RUNNING but no task in memory)
    try:
        session_repo = SessionRepository()
        from face_sorter.models.session import SessionStatus
        orphaned_sessions = await session_repo.get_all_sessions(status=SessionStatus.RUNNING)
        if orphaned_sessions:
            print(f"Found {len(orphaned_sessions)} orphaned sessions marked as RUNNING")
            for session in orphaned_sessions:
                # Check if task exists in memory
                if not task_tracker.get_active_task(session.task_id):
                    print(f"Deleting orphaned session {session.task_id} (no task in memory)")
                    await session_repo.delete_session(session.task_id)
            print("Orphaned session cleanup completed.")
    except Exception as e:
        print(f"Warning: Failed to cleanup orphaned sessions: {e}")

    # Start periodic task cleanup
    task_tracker.start_cleanup_task()
    print("Periodic task cleanup started.")

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

import urllib.parse
from fastapi.responses import FileResponse
from fastapi import HTTPException
import logging
main_logger = logging.getLogger(__name__)

@app.get("/images/{filename:path}")
async def serve_image(filename: str):
    decoded_filename = urllib.parse.unquote(filename)
    
    # Try to determine the cache directory from the most recent session
    try:
        session_repo = SessionRepository()
        # Get most recent sessions
        sessions = await session_repo.get_all_sessions(limit=5)
        
        # Sort by started_at descending (SessionRepository might not do it)
        sessions.sort(key=lambda s: s.started_at, reverse=True)
        
        possible_cache_dirs = []
        for session in sessions:
            if session.source_dir:
                src_path = Path(session.source_dir).resolve()
                derived_cache = src_path.parent / ".cache"
                if derived_cache.exists():
                    possible_cache_dirs.append(derived_cache)
        
        # Add default cache dir as fallback
        possible_cache_dirs.append(Path(settings.cache_dir))
        
        # Try to find the file in possible cache directories
        for cache_path in possible_cache_dirs:
            file_path = cache_path / decoded_filename
            if file_path.exists() and file_path.is_file():
                return FileResponse(file_path)
                
        # If not found in any cache, try if filename is an absolute path
        if Path(decoded_filename).exists() and Path(decoded_filename).is_file():
            return FileResponse(Path(decoded_filename))
            
        main_logger.error(f"Image not found: {decoded_filename}. Checked {len(possible_cache_dirs)} cache locations.")
        raise HTTPException(status_code=404, detail="Image not found")
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        main_logger.error(f"Error serving image {decoded_filename}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Mount static files for frontend (AFTER images mount)
static_dir = Path(__file__).parent.parent / "web" / "frontend" / "dist"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="frontend")
else:
    print(f"Warning: Frontend static directory not found: {static_dir}")


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