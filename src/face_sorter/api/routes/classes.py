"""
Class management endpoints for Face Sorter API.

This module provides endpoints for creating, viewing, and deleting face classes.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from face_sorter.database.repositories import ClassRepository, ClusterRepository, FaceRepository
from face_sorter.api.routes.images import ImageResponse

router = APIRouter()


class CreateClassRequest(BaseModel):
    """Request model for creating a new class."""

    class_name: str
    cluster_id: int


class CreateClassWithEmbeddingRequest(BaseModel):
    """Request model for creating a new class with explicit embedding."""

    class_name: str
    embedding: list[float]


class ClassResponse(BaseModel):
    """Response model for a class."""

    class_name: str

    class Config:
        """Pydantic config for ClassResponse."""

        from_attributes = True


class ClassSummary(BaseModel):
    """Summary response for a class with preview faces."""

    class_name: str
    face_count: int
    preview_faces: list[ImageResponse] = []


@router.get("", response_model=list[ClassResponse])
async def get_classes() -> list[ClassResponse]:
    """
    Get all face classes.

    Returns:
        List of class names.
    """
    class_repo = ClassRepository()
    class_names = await class_repo.get_all_class_names()
    return [ClassResponse(class_name=name) for name in class_names]


@router.get("/summary", response_model=list[ClassSummary])
async def get_class_summaries() -> list[ClassSummary]:
    """
    Get summaries for all face classes.

    Returns:
        List of class summaries including counts and preview faces.
    """
    class_repo = ClassRepository()
    face_repo = FaceRepository()

    class_names = await class_repo.get_all_class_names()

    result = []
    for name in class_names:
        count = await face_repo.count_faces_in_class(name)
        # Get up to 4 preview faces for each class
        faces = await face_repo.get_faces_paginated(query={"class": name}, limit=4)
        preview_faces = [ImageResponse(**face) for face in faces]

        result.append(ClassSummary(class_name=name, face_count=count, preview_faces=preview_faces))

    # Sort classes by face count descending
    result.sort(key=lambda x: x.face_count, reverse=True)
    return result


@router.post("", response_model=ClassResponse)
async def create_class(
    request: CreateClassRequest | CreateClassWithEmbeddingRequest,
) -> ClassResponse:
    """
    Create a new face class.

    Args:
        request: Either CreateClassRequest (with cluster_id) or CreateClassWithEmbeddingRequest (with explicit embedding)

    Returns:
        Created class response.

    Raises:
        HTTPException: If class already exists or cluster not found.
    """
    class_repo = ClassRepository()

    # Check if class already exists
    existing_classes = await class_repo.get_all_class_names()
    if request.class_name in existing_classes:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Class '{request.class_name}' already exists",
        )

    # Handle different request types
    if isinstance(request, CreateClassRequest):
        # Fetch embedding from cluster
        cluster_repo = ClusterRepository()
        face_repo = FaceRepository()
        cluster = await cluster_repo.get_cluster(request.cluster_id)
        if not cluster:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Cluster {request.cluster_id} not found",
            )

        centroid = cluster.get("centroid", [])
        if not centroid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cluster {request.cluster_id} has no centroid data",
            )

        await class_repo.insert_class(request.class_name, centroid)

        # Update cluster with assigned class name
        await cluster_repo.update_cluster_class(request.cluster_id, request.class_name)

        # Update all faces in the cluster with the class name
        indices = cluster.get("indices", [])
        if indices:
            await face_repo.update_faces_class(indices, request.class_name)
    else:
        # Create with explicit embedding
        await class_repo.insert_class(request.class_name, request.embedding)

    return ClassResponse(class_name=request.class_name)


@router.delete("/{class_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_class(class_name: str) -> None:
    """
    Delete a face class.

    Args:
        class_name: Name of the class to delete.

    Raises:
        HTTPException: If class not found.
    """
    class_repo = ClassRepository()

    # Check if class exists
    existing_classes = await class_repo.get_all_class_names()
    if class_name not in existing_classes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Class '{class_name}' not found",
        )

    await class_repo.delete_class(class_name)
