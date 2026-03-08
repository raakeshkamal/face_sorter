"""
Task tracker service for managing active background tasks.

This module provides a singleton service to track active asyncio tasks
for cancellation and monitoring.
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TaskTracker:
    """
    Tracks active background tasks for cancellation and monitoring.
    """

    _instance: Optional["TaskTracker"] = None

    def __new__(cls) -> "TaskTracker":
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the task tracker."""
        if not hasattr(self, "_initialized"):
            self.active_tasks: dict[str, asyncio.Task] = {}
            self._initialized = True

    def register_task(self, task_id: str, task: asyncio.Task) -> None:
        """
        Register a task for tracking.

        Args:
            task_id: Unique identifier for the task.
            task: The asyncio.Task to track.
        """
        self.active_tasks[task_id] = task
        logger.info(f"Registered task {task_id}")

    def unregister_task(self, task_id: str) -> None:
        """
        Unregister a task.

        Args:
            task_id: Unique identifier for the task.
        """
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
            logger.info(f"Unregistered task {task_id}")

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.

        Args:
            task_id: Unique identifier for the task.

        Returns:
            True if the task was cancelled, False if it was not found or already done.
        """
        task = self.get_active_task(task_id)
        if task and not task.done():
            task.cancel()
            try:
                await task
                logger.info(f"Task {task_id} was cancelled")
                return True
            except asyncio.CancelledError:
                logger.info(f"Task {task_id} was cancelled")
                return True
            except Exception as e:
                logger.error(f"Error cancelling task {task_id}: {e}")
                return False
        elif task and task.done():
            logger.warning(f"Task {task_id} already completed")
            return False
        else:
            logger.warning(f"Task {task_id} not found")
            return False

    def get_active_task(self, task_id: str) -> Optional[asyncio.Task]:
        """
        Get an active task by task_id.

        Args:
            task_id: Unique identifier for the task.

        Returns:
            The asyncio.Task if found and active, None otherwise.
        """
        task = self.active_tasks.get(task_id)
        if task and not task.done():
            return task
        return None

    def get_all_active_tasks(self) -> dict[str, asyncio.Task]:
        """
        Get all active tasks.

        Returns:
            Dictionary of task_id -> asyncio.Task for all active tasks.
        """
        # Filter out completed tasks
        return {
            task_id: task
            for task_id, task in self.active_tasks.items()
            if not task.done()
        }

    async def check_task_health(self, timeout_seconds: int = 300) -> list[str]:
        """
        Check task health and return list of stuck task IDs.

        This method checks if tasks have been inactive for too long.
        Tasks that exceed the timeout are considered "stuck" and should be cleaned up.

        Args:
            timeout_seconds: Timeout in seconds before a task is considered stuck.

        Returns:
            List of task IDs that are stuck.
        """
        stuck_tasks = []
        current_time = asyncio.get_event_loop().time()

        for task_id, task in self.active_tasks.items():
            if task.done():
                continue

            # Check if task has been running too long
            # Note: asyncio.Task doesn't track start time by default
            # This is a placeholder for more sophisticated health checking
            # In a real implementation, you'd track task start times and last progress updates
            pass

        return stuck_tasks

    def cleanup(self) -> None:
        """Remove completed tasks from the tracker."""
        completed_tasks = [
            task_id for task_id, task in self.active_tasks.items() if task.done()
        ]
        for task_id in completed_tasks:
            self.unregister_task(task_id)
        if completed_tasks:
            logger.info(f"Cleaned up {len(completed_tasks)} completed tasks")

    async def start_periodic_cleanup(self) -> None:
        """Start periodic task cleanup."""
        logger.info("Starting periodic task cleanup (every 60 seconds)")
        while True:
            try:
                await asyncio.sleep(60)
                self.cleanup()
            except asyncio.CancelledError:
                logger.info("Periodic cleanup stopped")
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")

    def start_cleanup_task(self) -> None:
        """Start the periodic cleanup task."""
        asyncio.create_task(self.start_periodic_cleanup())


# Global task tracker instance
task_tracker = TaskTracker()
