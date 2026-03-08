"""
Cluster management endpoints for Face Sorter API.

This module provides endpoints for retrieving and managing face clusters.
"""

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from face_sorter.database.repositories import ClusterRepository, FaceRepository
from face_sorter.api.routes.images import ImageResponse

router = APIRouter()


class ClusterResponse(BaseModel):
    """Response model for a cluster."""

    cluster_id: int
    cluster_name: int
    size: int
    preview_faces: list[ImageResponse] = []

    class Config:
        """Pydantic config for ClusterResponse."""
        from_attributes = True


@router.get("", response_model=list[ClusterResponse])
async def get_clusters(
    limit: int = Query(10, ge=1, le=100, description="Number of clusters to return"),
) -> list[ClusterResponse]:
    """
    Get all face clusters.

    Returns:
        List of cluster summaries.
    """
    cluster_repo = ClusterRepository()
    face_repo = FaceRepository()
    
    clusters = await cluster_repo.get_all_clusters()
    
    # Sort by size (indices length) descending
    clusters.sort(key=lambda x: len(x.get("indices", [])), reverse=True)
    
    # Take only requested limit
    clusters = clusters[:limit]
    
    result = []
    for cluster in clusters:
        # Get up to 4 preview faces for each cluster
        indices = cluster.get("indices", [])[:4]
        preview_faces = []
        for idx in indices:
            face = await face_repo.get_face_by_idx(idx)
            if face:
                preview_faces.append(ImageResponse(**face))
                
        result.append(ClusterResponse(
            cluster_id=cluster["cluster_id"],
            cluster_name=cluster["cluster_name"],
            size=len(cluster.get("indices", [])),
            preview_faces=preview_faces
        ))
        
    return result


@router.get("/{cluster_id}/images", response_model=list[ImageResponse])
async def get_cluster_images(
    cluster_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
) -> list[ImageResponse]:
    """
    Get all images in a specific cluster.

    Args:
        cluster_id: ID of the cluster.
        skip: Pagination skip.
        limit: Pagination limit.

    Returns:
        List of image responses.
    """
    face_repo = FaceRepository()
    
    # Use the cluster field we added to the Face model
    query = {"cluster": cluster_id}
    
    faces = await face_repo.get_faces_paginated(
        query=query, skip=skip, limit=limit
    )
    
    return [ImageResponse(**face) for face in faces]
