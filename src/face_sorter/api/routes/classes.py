"""
Class management endpoints for Face Sorter API.

This module provides endpoints for creating, viewing, and deleting face classes.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from face_sorter.database.repositories import ClassRepository, ClusterRepository

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