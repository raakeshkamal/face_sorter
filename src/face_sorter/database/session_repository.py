"""
Session repository for managing training sessions.

This module provides a repository for CRUD operations on training sessions.
"""

import logging
from typing import Any, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo import ASCENDING

from face_sorter.config import get_settings
from face_sorter.database.connection import get_database
from face_sorter.models.session import SessionStatus, TrainingSession

logger = logging.getLogger(__name__)


class SessionRepository:
    """Repository for managing training sessions."""

    def __init__(self, db: Optional[AsyncIOMotorDatabase] = None) -> None:
        """
        Initialize the repository.

        Args:
            db: Database instance. If None, uses global connection.
        """
        self._db: Optional[AsyncIOMotorDatabase] = db
        self._collection: Optional[AsyncIOMotorCollection] = None

    async def _get_collection(self) -> AsyncIOMotorCollection:
        """Get the collection lazily."""
        if self._collection is None:
            if self._db is None:
                self._db = await get_database()
            settings = get_settings()
            # Use a sessions collection
            self._collection = self._db["sessions"]
        return self._collection

    async def create_session(self, session: TrainingSession) -> str:
        """
        Create a new training session.

        Args:
            session: TrainingSession instance.

        Returns:
            The task_id of the created session.
        """
        collection = await self._get_collection()
        await collection.insert_one(session.to_dict())
        logger.info(f"Created session {session.task_id}")
        return session.task_id

    async def get_session(self, task_id: str) -> Optional[TrainingSession]:
        """
        Get a session by task_id.

        Args:
            task_id: Unique identifier for the session.

        Returns:
            TrainingSession instance or None if not found.
        """
        collection = await self._get_collection()
        data = await collection.find_one({"task_id": task_id})
        if data:
            return TrainingSession.from_dict(data)
        return None

    async def get_active_session(
        self, operation_type: str, source_dir: str
    ) -> Optional[TrainingSession]:
        """
        Get an active session for a given operation type and source directory.

        Args:
            operation_type: Type of operation (training, cleaning, etc.).
            source_dir: Source directory path.

        Returns:
            TrainingSession instance or None if no active session found.
        """
        collection = await self._get_collection()
        data = await collection.find_one(
            {
                "operation_type": operation_type,
                "source_dir": source_dir,
                "status": SessionStatus.RUNNING.value,
            }
        )
        if data:
            return TrainingSession.from_dict(data)
        return None

    async def update_session(self, task_id: str, updates: dict[str, Any]) -> bool:
        """
        Update a session.

        Args:
            task_id: Unique identifier for the session.
            updates: Dictionary of fields to update.

        Returns:
            True if the session was updated, False otherwise.
        """
        collection = await self._get_collection()
        result = await collection.update_one({"task_id": task_id}, {"$set": updates})
        if result.modified_count > 0:
            logger.debug(f"Updated session {task_id}: {updates}")
        return result.modified_count > 0

    async def get_all_sessions(
        self,
        status: Optional[SessionStatus] = None,
        operation_type: Optional[str] = None,
        limit: int = 100,
    ) -> list[TrainingSession]:
        """
        Get all sessions, optionally filtered by status and operation type.

        Args:
            status: Optional status filter.
            operation_type: Optional operation type filter.
            limit: Maximum number of sessions to return.

        Returns:
            List of TrainingSession instances.
        """
        collection = await self._get_collection()
        query: dict[str, Any] = {}
        if status:
            query["status"] = status.value if isinstance(status, SessionStatus) else status
        if operation_type:
            query["operation_type"] = operation_type

        cursor = collection.find(query).sort("started_at", -1).limit(limit)
        return [TrainingSession.from_dict(doc) async for doc in cursor]

    async def delete_session(self, task_id: str) -> bool:
        """
        Delete a session by task_id.

        Args:
            task_id: Unique identifier for the session.

        Returns:
            True if the session was deleted, False otherwise.
        """
        collection = await self._get_collection()
        result = await collection.delete_one({"task_id": task_id})
        if result.deleted_count > 0:
            logger.info(f"Deleted session {task_id}")
        return result.deleted_count > 0

    async def create_indexes(self) -> None:
        """Create indexes for the sessions collection."""
        collection = await self._get_collection()
        await collection.create_index([("task_id", 1)], unique=True)
        await collection.create_index([("operation_type", 1), ("source_dir", 1)])
        await collection.create_index([("status", 1)])
        await collection.create_index([("started_at", -1)])
        logger.info("Created indexes for sessions collection")
