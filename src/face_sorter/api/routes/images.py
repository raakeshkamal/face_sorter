"""
Image browsing endpoints for Face Sorter API.

This module provides endpoints for browsing and retrieving face images with metadata.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from face_sorter.database.repositories import FaceRepository

router = APIRouter()


class ImageResponse(BaseModel):
    """Response model for a single image."""

    idx: int
    filename: str
    path: str
    cache_url: Optional[str] = None
    bbox: list[int]
    det_score: float
    age: Optional[int] = None
    gender: Optional[int] = None

    class Config:
        """Pydantic config for ImageResponse."""
        from_attributes = True


@router.get("", response_model=list[ImageResponse])
async def get_images(
    skip: int = Query(0, ge=0, description="Number of images to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of images to return"),
    cluster_id: Optional[int] = Query(None, description="Filter by cluster ID"),
) -> list[ImageResponse]:
    """
    Get paginated list of face images.

    Args:
        skip: Number of images to skip (pagination).
        limit: Number of images to return.
        cluster_id: Optional cluster ID to filter by.

    Returns:
        List of image responses with metadata.
    """
    face_repo = FaceRepository()

    # Build projection to reduce data transfer
    projection = {
        "idx": 1,
        "item": 1,
        "path": 1,
        "cache_url": 1,
        "bbox": 1,
        "det_score": 1,
        "age": 1,
        "gender": 1,
    }

    # Build query if cluster_id is provided
    query = {"cluster": cluster_id} if cluster_id is not None else {}

    faces = await face_repo.get_faces_paginated(
        query=query, projection=projection, skip=skip, limit=limit
    )

    return [ImageResponse(**face) for face in faces]


@router.get("/{image_id}", response_model=ImageResponse)
async def get_image(image_id: int) -> ImageResponse:
    """
    Get a single image by ID.

    Args:
        image_id: Index of the face image.

    Returns:
        Image response with metadata.

    Raises:
        HTTPException: If image not found.
    """
    face_repo = FaceRepository()
    face = await face_repo.get_face_by_idx(image_id)

    if face is None:
        raise HTTPException(status_code=404, detail=f"Image with ID {image_id} not found")

    return ImageResponse(**face)


@router.get("/unsorted", response_model=list[ImageResponse])
async def get_unsorted_images(
    skip: int = Query(0, ge=0, description="Number of images to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of images to return"),
) -> list[ImageResponse]:
    """
    Get unsorted face images (not assigned to any class).

    Args:
        skip: Number of images to skip (pagination).
        limit: Number of images to return.

    Returns:
        List of unsorted image responses with metadata.
    """
    face_repo = FaceRepository()

    # Build projection to reduce data transfer
    projection = {
        "idx": 1,
        "item": 1,
        "path": 1,
        "cache_url": 1,
        "bbox": 1,
        "det_score": 1,
        "age": 1,
        "gender": 1,
    }

    # Query for faces without a class assignment
    query = {"class": {"$exists": False}}

    faces = await face_repo.get_faces_paginated(
        query=query, projection=projection, skip=skip, limit=limit
    )

    return [ImageResponse(**face) for face in faces]