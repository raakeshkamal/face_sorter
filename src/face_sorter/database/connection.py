"""
MongoDB connection management.

This module provides a singleton connection manager for MongoDB connections.
"""

import logging
from typing import Optional

from pymongo import MongoClient
from pymongo.database import Database

from face_sorter.config import get_settings

logger = logging.getLogger(__name__)


class MongoDBConnection:
    """Singleton MongoDB connection manager."""

    _instance: Optional["MongoDBConnection"] = None

    def __new__(cls) -> "MongoDBConnection":
        """Ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the connection manager."""
        if not hasattr(self, "_initialized"):
            self._client: Optional[MongoClient] = None
            self._db: Optional[Database] = None
            self._initialized = True

    def connect(self, uri: Optional[str] = None, database: Optional[str] = None) -> Database:
        """
        Connect to MongoDB and return the database instance.

        Args:
            uri: MongoDB connection URI. If None, uses settings.
            database: Database name. If None, uses settings.

        Returns:
            Database: The MongoDB database instance.
        """
        settings = get_settings()

        if uri is None:
            uri = settings.mongodb_uri
        if database is None:
            database = settings.mongodb_database

        if self._client is None:
            try:
                self._client = MongoClient(uri)
                self._db = self._client[database]
                logger.info(f"Successfully connected to MongoDB: {uri}")
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                raise

        return self._db

    def disconnect(self) -> None:
        """Close the MongoDB connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._db = None
            logger.info("MongoDB connection closed")

    def get_database(self) -> Database:
        """
        Get the database instance without creating a new connection.

        Returns:
            Database: The MongoDB database instance.
        """
        if self._db is None:
            return self.connect()
        return self._db

    def get_client(self) -> MongoClient:
        """
        Get the MongoDB client instance.

        Returns:
            MongoClient: The MongoDB client instance.
        """
        if self._client is None:
            self.connect()
        return self._client


# Global connection instance
_connection: Optional[MongoDBConnection] = None


def get_connection() -> MongoDBConnection:
    """
    Get the global MongoDB connection instance.

    Returns:
        MongoDBConnection: The connection manager instance.
    """
    global _connection
    if _connection is None:
        _connection = MongoDBConnection()
    return _connection


def get_database() -> Database:
    """
    Get the MongoDB database instance.

    Returns:
        Database: The MongoDB database instance.
    """
    return get_connection().get_database()
