"""
FastAPI application for Face Sorter web UI.

This module provides the web interface for the face sorting system,
including REST API endpoints and WebSocket connections for real-time updates.
"""

from face_sorter.api.main import app

__all__ = ["app"]