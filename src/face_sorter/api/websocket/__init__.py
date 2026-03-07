"""
WebSocket connection manager for real-time updates.

This module provides WebSocket functionality for broadcasting progress updates
during long-running operations like training, cleaning, deduping, and sorting.
"""

from face_sorter.api.websocket.manager import ConnectionManager

__all__ = ["ConnectionManager"]