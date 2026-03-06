"""
Face Sorter - A face recognition and sorting application.

This package provides tools for detecting faces in images, generating embeddings,
and sorting them into person-based classes.
"""

__version__ = "0.1.0"
__author__ = "Face Sorter Contributors"

from .config import get_settings, reset_settings

__all__ = [
    "__version__",
    "__author__",
    "get_settings",
    "reset_settings",
]
