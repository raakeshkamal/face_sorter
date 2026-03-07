"""
MongoDB connection management.

This module provides a singleton connection manager for MongoDB connections.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from face_sorter.config import get_settings

logger = logging.getLogger(__name__)


class AsyncMongoDBConnection:
    """Singleton async MongoDB connection manager."""

    _instance: Optional["AsyncMongoDBConnection"] = None

    def __new__(cls) -> "AsyncMongoDBConnection":
        """Ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the connection manager."""
        if not hasattr(self, "_initialized"):
            self._client: Optional[AsyncIOMotorClient] = None
            self._db: Optional[AsyncIOMotorDatabase] = None
            self._initialized = True

    async def connect(
        self, uri: Optional[str] = None, database: Optional[str] = None
    ) -> AsyncIOMotorDatabase:
        """
        Connect to MongoDB and return the database instance.

        Args:
            uri: MongoDB connection URI. If None, uses settings.
            database: Database name. If None, uses settings.

        Returns:
            AsyncIOMotorDatabase: The MongoDB database instance.
        """
        settings = get_settings()

        if uri is None:
            uri = settings.mongodb_uri
        if database is None:
            database = settings.mongodb_database

        if self._client is None:
            try:
                self._client = AsyncIOMotorClient(
                    uri,
                    maxPoolSize=100,
                    minPoolSize=10,
                    maxIdleTimeMS=30000,
                )
                self._db = self._client[database]
                logger.info(f"Successfully connected to MongoDB: {uri}")
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                raise

        return self._db

    async def disconnect(self) -> None:
        """Close the MongoDB connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._db = None
            logger.info("MongoDB connection closed")

    async def get_database(self) -> AsyncIOMotorDatabase:
        """
        Get the database instance without creating a new connection.

        Returns:
            AsyncIOMotorDatabase: The MongoDB database instance.
        """
        if self._db is None:
            return await self.connect()
        return self._db

    async def get_client(self) -> AsyncIOMotorClient:
        """
        Get the MongoDB client instance.

        Returns:
            AsyncIOMotorClient: The MongoDB client instance.
        """
        if self._client is None:
            await self.connect()
        return self._client


# Global connection instance
_connection: Optional[AsyncMongoDBConnection] = None


async def get_connection() -> AsyncMongoDBConnection:
    """
    Get the global MongoDB connection instance.

    Returns:
        AsyncMongoDBConnection: The connection manager instance.
    """
    global _connection
    if _connection is None:
        _connection = AsyncMongoDBConnection()
    return _connection


async def get_database() -> AsyncIOMotorDatabase:
    """
    Get the MongoDB database instance.

    Returns:
        AsyncIOMotorDatabase: The MongoDB database instance.
    """
    connection = await get_connection()
    return await connection.get_database()


@asynccontextmanager
async def get_db_context():
    """
    Async context manager for database connection.

    Yields:
        AsyncIOMotorDatabase: The MongoDB database instance.
    """
    conn = await get_connection()
    db = await conn.connect()
    try:
        yield db
    finally:
        await conn.disconnect()


# For backward compatibility, create sync wrappers
class MongoDBConnection(AsyncMongoDBConnection):
    """Synchronous wrapper for backward compatibility."""

    def connect(self, uri: Optional[str] = None, database: Optional[str] = None):
        """Sync wrapper for connect method."""
        import asyncio

        return asyncio.run(super().connect(uri, database))

    def disconnect(self) -> None:
        """Sync wrapper for disconnect method."""
        import asyncio

        return asyncio.run(super().disconnect())

    def get_database(self):
        """Sync wrapper for get_database method."""
        import asyncio

        return asyncio.run(super().get_database())

    def get_client(self):
        """Sync wrapper for get_client method."""
        import asyncio

        return asyncio.run(super().get_client())


def get_connection_sync() -> MongoDBConnection:
    """
    Get the global MongoDB connection instance (sync wrapper).

    Returns:
        MongoDBConnection: The connection manager instance.
    """
    global _connection
    if _connection is None:
        _connection = MongoDBConnection()
    return _connection


def get_database_sync():
    """
    Get the MongoDB database instance (sync wrapper).

    Returns:
        The MongoDB database instance.
    """
    return get_connection_sync().get_database()
