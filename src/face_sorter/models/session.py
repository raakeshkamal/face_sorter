"""
Training session models for Face Sorter.

This module contains models for managing persistent training sessions with state tracking.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional


class SessionStatus(str, Enum):
    """Status of a training session."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TrainingSession:
    """
    Represents a training session with persistent state.

    Attributes:
        task_id: Unique identifier for the session (UUID).
        operation_type: Type of operation (training, cleaning, etc.).
        status: Current status of the session.
        source_dir: Source directory for the operation.
        config: Training configuration dictionary.
        progress: Current progress dictionary (current, total, status, current_item).
        started_at: Timestamp when the session started.
        completed_at: Timestamp when the session completed (optional).
        error: Error message if the session failed (optional).
    """

    def __init__(
        self,
        task_id: str,
        operation_type: str,
        status: SessionStatus,
        source_dir: str,
        config: dict[str, Any],
        progress: dict[str, Any],
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        error: Optional[str] = None,
    ) -> None:
        self.task_id = task_id
        self.operation_type = operation_type
        self.status = status
        self.source_dir = source_dir
        self.config = config
        self.progress = progress
        self.started_at = started_at if started_at else datetime.utcnow()
        self.completed_at = completed_at
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        data = {
            "task_id": self.task_id,
            "operation_type": self.operation_type,
            "status": self.status.value if isinstance(self.status, SessionStatus) else self.status,
            "source_dir": self.source_dir,
            "config": self.config,
            "progress": self.progress,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
        if self.error:
            data["error"] = self.error
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingSession":
        """Create a TrainingSession instance from a dictionary."""
        return cls(
            task_id=data["task_id"],
            operation_type=data["operation_type"],
            status=SessionStatus(data["status"]),
            source_dir=data["source_dir"],
            config=data["config"],
            progress=data["progress"],
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error=data.get("error"),
        )


class SessionProgress:
    """
    Represents progress during a training session.

    Attributes:
        current: Current progress value.
        total: Total progress value.
        status: Status message.
        current_item: Current item being processed (optional).
    """

    def __init__(
        self,
        current: int,
        total: int,
        status: str,
        current_item: Optional[str] = None,
    ) -> None:
        self.current = current
        self.total = total
        self.status = status
        self.current_item = current_item

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current": self.current,
            "total": self.total,
            "status": self.status,
            "current_item": self.current_item,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionProgress":
        """Create a SessionProgress instance from a dictionary."""
        return cls(
            current=data.get("current", 0),
            total=data.get("total", 0),
            status=data.get("status", ""),
            current_item=data.get("current_item"),
        )


class TrainingCancelledError(Exception):
    """Exception raised when training is cancelled."""

    pass


class CleaningCancelledError(Exception):
    """Exception raised when cleaning operation is cancelled."""

    pass


class DeduplicationCancelledError(Exception):
    """Exception raised when deduplication operation is cancelled."""

    pass


class StabilityScoreTrainingCancelledError(Exception):
    """Exception raised when stability score training is cancelled."""

    pass


class FaceDetectionTrainingCancelledError(Exception):
    """Exception raised when face detection training is cancelled."""

    pass
