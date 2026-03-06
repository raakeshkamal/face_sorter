"""Database module for Face Sorter."""

from .connection import get_connection, get_database, MongoDBConnection
from .repositories import (
    ClassRepository,
    ClusterRepository,
    FaceRepository,
    fetch_data_optimized,
)

__all__ = [
    "get_connection",
    "get_database",
    "MongoDBConnection",
    "FaceRepository",
    "ClassRepository",
    "ClusterRepository",
    "fetch_data_optimized",
]
