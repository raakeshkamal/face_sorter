"""
WebSocket connection manager.

This module manages WebSocket connections for real-time progress updates
during long-running operations.
"""

import json
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect


class ConnectionManager:
    """
    Manages WebSocket connections for broadcasting progress updates.
    """

    def __init__(self) -> None:
        """Initialize the connection manager with empty active connections."""
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, operation_type: str, task_id: str) -> None:
        """
        Accept a WebSocket connection and add it to the active connections.

        Args:
            websocket: WebSocket connection to accept
            operation_type: Type of operation (e.g., "training", "cleaning")
            task_id: Unique identifier for the task
        """
        await websocket.accept()
        key = f"{operation_type}:{task_id}"
        if key not in self.active_connections:
            self.active_connections[key] = []
        self.active_connections[key].append(websocket)

    def disconnect(self, websocket: WebSocket, operation_type: str, task_id: str) -> None:
        """
        Remove a WebSocket connection from active connections.

        Args:
            websocket: WebSocket connection to remove
            operation_type: Type of operation
            task_id: Unique identifier for the task
        """
        key = f"{operation_type}:{task_id}"
        if key in self.active_connections and websocket in self.active_connections[key]:
            self.active_connections[key].remove(websocket)
            if not self.active_connections[key]:
                del self.active_connections[key]

    async def broadcast(self, operation_type: str, task_id: str, message: dict[str, Any]) -> None:
        """
        Broadcast a message to all connected WebSocket clients for a specific task.

        Args:
            operation_type: Type of operation
            task_id: Unique identifier for the task
            message: Message to broadcast (will be JSON encoded)
        """
        key = f"{operation_type}:{task_id}"
        print(f"DEBUG(WebSocket): Broadcasting to key={key}. Active connections map keys: {list(self.active_connections.keys())}")
        if key in self.active_connections:
            disconnected = []
            for connection in self.active_connections[key]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    print(f"DEBUG(WebSocket): Error sending message: {e}")
                    disconnected.append(connection)

            # Clean up disconnected connections
            for connection in disconnected:
                self.disconnect(connection, operation_type, task_id)
        else:
            print(f"DEBUG(WebSocket): No active connections found for key={key}")

    async def send_progress(
        self,
        operation_type: str,
        task_id: str,
        current: int,
        total: int,
        status: str,
        current_item: str | None = None,
    ) -> None:
        """
        Send a progress update to connected clients.

        Args:
            operation_type: Type of operation
            task_id: Unique identifier for the task
            current: Current progress value
            total: Total value for progress calculation
            status: Current status message
            current_item: Optional name of the current item being processed
        """
        percentage = (current / total * 100) if total > 0 else 0.0
        message = {
            "type": "progress",
            "operation": operation_type,
            "task_id": task_id,
            "progress": {
                "current": current,
                "total": total,
                "percentage": round(percentage, 2),
                "status": status,
                "current_item": current_item,
            },
        }
        print(f"DEBUG(WebSocket): send_progress called for {operation_type}:{task_id} -> {current}/{total}")
        await self.broadcast(operation_type, task_id, message)

    async def send_complete(
        self, operation_type: str, task_id: str, result: dict[str, Any]
    ) -> None:
        """
        Send a completion message to connected clients.

        Args:
            operation_type: Type of operation
            task_id: Unique identifier for the task
            result: Result data from the operation
        """
        message = {
            "type": "complete",
            "operation": operation_type,
            "task_id": task_id,
            "result": result,
        }
        await self.broadcast(operation_type, task_id, message)

    async def send_error(
        self, operation_type: str, task_id: str, error: dict[str, Any]
    ) -> None:
        """
        Send an error message to connected clients.

        Args:
            operation_type: Type of operation
            task_id: Unique identifier for the task
            error: Error details (message, details)
        """
        message = {
            "type": "error",
            "operation": operation_type,
            "task_id": task_id,
            "error": error,
        }
        await self.broadcast(operation_type, task_id, message)


# Global connection manager instance
connection_manager = ConnectionManager()