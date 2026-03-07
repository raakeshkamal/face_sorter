"""
API route definitions for Face Sorter.

This module registers all API route modules with the FastAPI application.
"""

from fastapi import APIRouter

# Import route modules
from face_sorter.api.routes.stats import router as stats_router
from face_sorter.api.routes.images import router as images_router
from face_sorter.api.routes.classes import router as classes_router
from face_sorter.api.routes.operations import router as operations_router

# Create main router
router = APIRouter()

# Register route modules
router.include_router(stats_router, prefix="/stats")
router.include_router(images_router, prefix="/images")
router.include_router(classes_router, prefix="/classes")
router.include_router(operations_router, prefix="/operations")

# Health check
@router.get("/health")
async def health() -> dict[str, str]:
    """
    Health check endpoint.

    Returns:
        Dictionary with health status
    """
    return {"status": "healthy"}