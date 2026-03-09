"""
Operation endpoints for Face Sorter API.

This module provides endpoints for triggering and managing long-running operations
like training, cleaning, deduping, and sorting with real-time progress updates.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, status, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel

from face_sorter.api.websocket.manager import connection_manager
from face_sorter.database import SessionRepository
from face_sorter.models.session import SessionStatus, SessionProgress, TrainingSession, TrainingCancelledError, CleaningCancelledError
from face_sorter.services.clean import clean_dataset
from face_sorter.services.task_tracker import task_tracker
from face_sorter.services.training import train
from face_sorter.services.sorting import sort

logger = logging.getLogger(__name__)
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


class SortRequest(BaseModel):
    """Request model for sorting operation."""

    source_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    max_results: Optional[int] = 10
    min_samples: Optional[int] = None
    min_cluster_size: Optional[int] = None


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
    source_dir = request.source_dir or ""

    # Create session in database
    session_repo = SessionRepository()
    session = TrainingSession(
        task_id=task_id,
        operation_type="training",
        status=SessionStatus.PENDING,
        source_dir=source_dir,
        config=request.model_dump(),
        progress=SessionProgress(0, 0, "Initializing", "").to_dict(),
        started_at=datetime.now(timezone.utc),
    )
    await session_repo.create_session(session)

    # Create cancellation event
    cancellation_event = asyncio.Event()

    async def run_training_with_session():
        """Run training with session tracking."""
        session_repo = SessionRepository()
        try:
            # Update session to RUNNING
            await session_repo.update_session(
                task_id,
                {"status": SessionStatus.RUNNING.value},
            )

            def progress_handler(current: int, total: int, status_text: str, current_item: str, image_data: dict | None = None) -> None:
                """Handle progress updates during training."""
                asyncio.create_task(
                    connection_manager.send_progress("training", task_id, current, total, status_text, current_item, image_data)
                )
                # Also update session progress
                asyncio.create_task(
                    session_repo.update_session(
                        task_id,
                        {
                            "progress": SessionProgress(current, total, status_text, current_item).to_dict(),
                        },
                    )
                )

            # Run training
            result = await train(
                source_dir=request.source_dir,
                noface_dir=request.noface_dir,
                broken_dir=request.broken_dir,
                cache_dir=request.cache_dir,
                duplicates_dir=request.duplicates_dir,
                progress_callback=progress_handler,
                cancellation_event=cancellation_event,
            )

            # Update session to COMPLETED
            await session_repo.update_session(
                task_id,
                {
                    "status": SessionStatus.COMPLETED.value,
                    "completed_at": datetime.now(timezone.utc),
                    "progress": SessionProgress(result.processed, result.total, "Complete", "").to_dict(),
                },
            )

            # Send completion message via WebSocket
            await connection_manager.send_progress("training", task_id, result.processed, result.total, "Complete", "")

        except TrainingCancelledError:
            # Update session to CANCELLED
            await session_repo.update_session(
                task_id,
                {
                    "status": SessionStatus.CANCELLED.value,
                    "completed_at": datetime.now(timezone.utc),
                },
            )
            logger.info(f"Training session {task_id} was cancelled")
            await connection_manager.send_progress("training", task_id, 0, 0, "Cancelled", "")

        except Exception as e:
            # Update session to FAILED
            await session_repo.update_session(
                task_id,
                {
                    "status": SessionStatus.FAILED.value,
                    "completed_at": datetime.now(timezone.utc),
                    "error": str(e),
                },
            )
            logger.error(f"Training session {task_id} failed: {e}")
            await connection_manager.send_progress("training", task_id, 0, 0, "Failed", str(e))

        finally:
            # Unregister task
            task_tracker.unregister_task(task_id)

    # Create and register the task
    task = asyncio.create_task(run_training_with_session())
    task_tracker.register_task(task_id, task)

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
    source_dir = request.source_dir or ""

    # Create session in database
    session_repo = SessionRepository()
    session = TrainingSession(
        task_id=task_id,
        operation_type="cleaning",
        status=SessionStatus.PENDING,
        source_dir=source_dir,
        config=request.model_dump(),
        progress=SessionProgress(0, 0, "Initializing", "").to_dict(),
        started_at=datetime.now(timezone.utc),
    )
    await session_repo.create_session(session)

    # Create cancellation event
    cancellation_event = asyncio.Event()

    async def run_cleaning_with_session():
        """Run cleaning with session tracking."""
        session_repo = SessionRepository()
        try:
            await session_repo.update_session(
                task_id,
                {"status": SessionStatus.RUNNING.value},
            )

            def progress_handler(
                current: int, total: int, status: str, current_item: str, current_data: Optional[dict] = None
            ) -> None:
                """Handle progress updates during cleaning."""
                asyncio.create_task(
                    connection_manager.send_progress(
                        "cleaning", task_id, current, total, status, current_item, current_data
                    )
                )
                # Also update session progress
                asyncio.create_task(
                    session_repo.update_session(
                        task_id,
                        {
                            "progress": SessionProgress(current, total, status, current_item).to_dict(),
                        },
                    )
                )

            result = await clean_dataset(
                source_dir=request.source_dir,
                output_dir=request.output_dir,
                broken_dir=request.broken_dir,
                batch_size=request.batch_size,
                img_prefix=request.img_prefix,
                quality=request.quality,
                recursive=request.recursive,
                start_index=request.start_index,
                progress_callback=progress_handler,
                cancellation_event=cancellation_event,
            )

            await session_repo.update_session(
                task_id,
                {
                    "status": SessionStatus.COMPLETED.value,
                    "completed_at": datetime.now(timezone.utc),
                    "progress": SessionProgress(result.processed, result.processed, "Complete", "").to_dict(),
                },
            )

            await connection_manager.send_progress("cleaning", task_id, result.processed, result.processed, "Complete", "")

        except asyncio.CancelledError:
            await session_repo.update_session(
                task_id,
                {
                    "status": SessionStatus.CANCELLED.value,
                    "completed_at": datetime.now(timezone.utc),
                },
            )
            logger.info(f"Cleaning session {task_id} was cancelled")
            await connection_manager.send_progress("cleaning", task_id, 0, 0, "Cancelled", "")

        except Exception as e:
            await session_repo.update_session(
                task_id,
                {
                    "status": SessionStatus.FAILED.value,
                    "completed_at": datetime.now(timezone.utc),
                    "error": str(e),
                },
            )
            logger.error(f"Cleaning session {task_id} failed: {e}")
            await connection_manager.send_progress("cleaning", task_id, 0, 0, "Failed", str(e))

        finally:
            task_tracker.unregister_task(task_id)

    # Create and register the task
    task = asyncio.create_task(run_cleaning_with_session())
    task_tracker.register_task(task_id, task)

    return OperationResponse(
        task_id=task_id,
        operation="cleaning",
        status="started",
    )


@router.post("/sort", response_model=OperationResponse)
async def start_sort(request: SortRequest) -> OperationResponse:
    """
    Start sorting operation in background.

    Args:
        request: Sorting request with options.

    Returns:
        Operation response with task_id for progress tracking.
    """
    task_id = str(uuid.uuid4())
    source_dir = request.source_dir or ""

    # Create session in database
    session_repo = SessionRepository()
    session = TrainingSession(
        task_id=task_id,
        operation_type="sorting",
        status=SessionStatus.PENDING,
        source_dir=source_dir,
        config=request.model_dump(),
        progress=SessionProgress(0, 100, "Initializing", "").to_dict(),
        started_at=datetime.now(timezone.utc),
    )
    await session_repo.create_session(session)

    # Create cancellation event
    cancellation_event = asyncio.Event()

    async def run_sorting_with_session():
        """Run sorting with session tracking."""
        session_repo = SessionRepository()
        try:
            await session_repo.update_session(
                task_id,
                {"status": SessionStatus.RUNNING.value},
            )

            def progress_handler(
                current: int, total: int, status: str, current_item: str, current_data: Optional[dict] = None
            ) -> None:
                """Handle progress updates during sorting."""
                asyncio.create_task(
                    connection_manager.send_progress(
                        "sorting", task_id, current, total, status, current_item, current_data
                    )
                )
                asyncio.create_task(
                    session_repo.update_session(
                        task_id,
                        {
                            "progress": SessionProgress(current, total, status, current_item).to_dict(),
                        },
                    )
                )

            try:
                await sort(
                    cache_dir=request.cache_dir,
                    source_dir=request.source_dir,
                    max_results=request.max_results or 10,
                    min_samples=request.min_samples,
                    min_cluster_size=request.min_cluster_size,
                    progress_callback=progress_handler,
                    cancellation_event=cancellation_event,
                )

                await session_repo.update_session(
                    task_id,
                    {
                        "status": SessionStatus.COMPLETED.value,
                        "completed_at": datetime.now(timezone.utc),
                        "progress": SessionProgress(100, 100, "Complete", "").to_dict(),
                    },
                )

                await connection_manager.send_progress("sorting", task_id, 100, 100, "Complete", "")
            except Exception as sort_error:
                logger.error(f"Error during sort operation: {sort_error}", exc_info=True)
                raise sort_error

        except asyncio.CancelledError:
            await session_repo.update_session(
                task_id,
                {
                    "status": SessionStatus.CANCELLED.value,
                    "completed_at": datetime.now(timezone.utc),
                },
            )
            await connection_manager.send_progress("sorting", task_id, 0, 0, "Cancelled", "")

        except Exception as e:
            await session_repo.update_session(
                task_id,
                {
                    "status": SessionStatus.FAILED.value,
                    "completed_at": datetime.now(timezone.utc),
                    "error": str(e),
                },
            )
            logger.error(f"Sorting session {task_id} failed: {e}")
            await connection_manager.send_progress("sorting", task_id, 0, 0, "Failed", str(e))

        finally:
            task_tracker.unregister_task(task_id)

    # Create and register the task
    task = asyncio.create_task(run_sorting_with_session())
    task_tracker.register_task(task_id, task)

    return OperationResponse(
        task_id=task_id,
        operation="sorting",
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
    # Validate that the session exists and is RUNNING before accepting connection
    session_repo = SessionRepository()
    session = await session_repo.get_session(task_id)

    if session is None:
        logger.warning(f"WebSocket connection rejected: Session {task_id} not found")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Session not found")
        return

    if session.status != SessionStatus.RUNNING:
        logger.warning(f"WebSocket connection rejected: Session {task_id} is not running (status: {session.status})")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=f"Session is not running (status: {session.status})")
        return

    await connection_manager.connect(websocket, operation_type, task_id)

    try:
        # Keep connection alive and receive any messages
        while True:
            await websocket.receive_text()
            # Could handle client messages here if needed
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket, operation_type, task_id)
    except Exception as e:
        connection_manager.disconnect(websocket, operation_type, task_id)
        print(f"WebSocket error: {e}")


@router.get("/sessions/active")
async def get_active_sessions() -> list[dict]:
    """
    Get all active (running) training sessions.

    Returns:
        List of active session dictionaries.
    """
    session_repo = SessionRepository()
    sessions = await session_repo.get_all_sessions(status=SessionStatus.RUNNING)
    return [session.to_dict() for session in sessions]


@router.get("/sessions/{task_id}")
async def get_session(task_id: str) -> dict:
    """
    Get a specific session by task_id.

    Args:
        task_id: Unique identifier for the session.

    Returns:
        Session dictionary.

    Raises:
        HTTPException: If session not found.
    """
    session_repo = SessionRepository()
    session = await session_repo.get_session(task_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {task_id} not found")
    return session.to_dict()


@router.post("/sessions/{task_id}/cancel", response_model=OperationResponse)
async def cancel_session(task_id: str) -> OperationResponse:
    """
    Cancel an ongoing training or cleaning session.

    Args:
        task_id: Unique identifier for the session.

    Returns:
        Operation response confirming cancellation.

    Raises:
        HTTPException: If session not found or cannot be cancelled.
    """
    session_repo = SessionRepository()
    session = await session_repo.get_session(task_id)

    if session is None:
        logger.warning(f"Cancel requested for non-existent session {task_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {task_id} not found")

    logger.info(f"Cancel requested for session {task_id} with status: {session.status}")

    if session.status != SessionStatus.RUNNING:
        logger.warning(f"Cannot cancel session {task_id}: not running (status: {session.status})")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session {task_id} is not running (status: {session.status})",
        )

    # Close all WebSocket connections for this task BEFORE cancelling
    await connection_manager.disconnect_all_for_task(session.operation_type, task_id)
    logger.info(f"Closed WebSocket connections for session {task_id}")

    # Cancel the task
    cancelled = await task_tracker.cancel_task(task_id)

    if cancelled:
        logger.info(f"Successfully cancelled task {task_id} in task tracker")
    else:
        # Task was not found in memory (e.g., server restart)
        logger.warning(f"Task {task_id} not found in memory but marked as RUNNING in DB. Force cancelling in DB.")

    # Delete the session from database immediately (instead of just marking as cancelled)
    deleted = await session_repo.delete_session(task_id)
    if deleted:
        logger.info(f"Deleted session {task_id} from database")
    else:
        logger.warning(f"Failed to delete session {task_id} from database")

    # Unregister task from tracker
    task_tracker.unregister_task(task_id)

    return OperationResponse(
        task_id=task_id,
        operation="cancellation",
        status="cancelled",
    )