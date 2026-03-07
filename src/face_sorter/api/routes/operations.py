"""
Operation endpoints for Face Sorter API.

This module provides endpoints for triggering and managing long-running operations
like training, cleaning, deduping, and sorting with real-time progress updates.
"""

import asyncio
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, status, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel

from face_sorter.api.websocket.manager import connection_manager
from face_sorter.services.clean import clean_dataset
from face_sorter.services.training import train

router = APIRouter()


class TrainRequest(BaseModel):
    """Request model for training operation."""

    source_dir: Optional[str] = None
    noface_dir: Optional[str] = None
    broken_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    duplicates_dir: Optional[str] = None


class CleanRequest(BaseModel):
    """Request model for cleaning operation."""

    source_dir: Optional[str] = None
    output_dir: Optional[str] = None
    broken_dir: Optional[str] = None
    batch_size: Optional[int] = None
    img_prefix: Optional[str] = None
    quality: Optional[int] = None
    recursive: Optional[bool] = None
    start_index: Optional[int] = None


class OperationResponse(BaseModel):
    """Response model for operation start."""

    task_id: str
    operation: str
    status: str


@router.post("/train", response_model=OperationResponse)
async def start_train(request: TrainRequest) -> OperationResponse:
    """
    Start training operation in background.

    Args:
        request: Training request with directories and options.

    Returns:
        Operation response with task_id for progress tracking.
    """
    task_id = str(uuid.uuid4())

    def progress_handler(current: int, total: int, status: str, current_item: str) -> None:
        """Handle progress updates during training."""
        asyncio.create_task(
            connection_manager.send_progress("training", task_id, current, total, status, current_item)
        )

    # Start training in background
    asyncio.create_task(
        train(
            source_dir=request.source_dir,
            noface_dir=request.noface_dir,
            broken_dir=request.broken_dir,
            cache_dir=request.cache_dir,
            duplicates_dir=request.duplicates_dir,
            progress_callback=progress_handler,
        )
    )

    # Send operation started message
    await connection_manager.send_progress("training", task_id, 0, 0, "Started", "")

    return OperationResponse(
        task_id=task_id,
        operation="training",
        status="started",
    )


@router.post("/clean", response_model=OperationResponse)
async def start_clean(request: CleanRequest) -> OperationResponse:
    """
    Start cleaning operation in background.

    Args:
        request: Cleaning request with directories and options.

    Returns:
        Operation response with task_id for progress tracking.
    """
    task_id = str(uuid.uuid4())

    def progress_handler(current: int, total: int, status: str, current_item: str) -> None:
        """Handle progress updates during cleaning."""
        asyncio.create_task(
            connection_manager.send_progress("cleaning", task_id, current, total, status, current_item)
        )

    # Start cleaning in background
    asyncio.create_task(
        clean_dataset(
            source_dir=request.source_dir,
            output_dir=request.output_dir,
            broken_dir=request.broken_dir,
            batch_size=request.batch_size,
            img_prefix=request.img_prefix,
            quality=request.quality,
            recursive=request.recursive,
            start_index=request.start_index,
            progress_callback=progress_handler,
        )
    )

    # Send operation started message
    await connection_manager.send_progress("cleaning", task_id, 0, 0, "Started", "")

    return OperationResponse(
        task_id=task_id,
        operation="cleaning",
        status="started",
    )


@router.websocket("/ws/{operation_type}/{task_id}")
async def websocket_endpoint(websocket: WebSocket, operation_type: str, task_id: str) -> None:
    """
    WebSocket endpoint for real-time progress updates.

    Args:
        websocket: WebSocket connection.
        operation_type: Type of operation (training, cleaning, deduping, sorting).
        task_id: Unique task identifier.
    """
    await connection_manager.connect(websocket, operation_type, task_id)

    try:
        # Keep connection alive and receive any messages
        while True:
            data = await websocket.receive_text()
            # Could handle client messages here if needed
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket, operation_type, task_id)
    except Exception as e:
        connection_manager.disconnect(websocket, operation_type, task_id)
        print(f"WebSocket error: {e}")