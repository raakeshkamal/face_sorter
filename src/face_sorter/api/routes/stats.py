"""
Statistics endpoints for Face Sorter API.

This module provides endpoints for retrieving system statistics and overview data.
"""

from fastapi import APIRouter

from face_sorter.database.repositories import ClassRepository, ClusterRepository, FaceRepository

router = APIRouter()


@router.get("/overview")
async def get_overview() -> dict[str, int]:
    """
    Get system overview statistics.

    Returns:
        Dictionary with total counts for images, faces, classes, and clusters.
    """
    face_repo = FaceRepository()
    class_repo = ClassRepository()
    cluster_repo = ClusterRepository()

    # Get counts from all collections
    face_count = await face_repo.count_faces()
    class_count = await class_repo.count_classes()
    cluster_count = await cluster_repo.count_clusters()

    return {
        "total_faces": face_count,
        "total_classes": class_count,
        "total_clusters": cluster_count,
        "total_images": face_count,  # Approximation, assuming one face per image
    }