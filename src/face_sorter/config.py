"""
Configuration management for Face Sorter.

This module uses Pydantic to manage application configuration from environment variables
and .env files.
"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database Configuration
    mongodb_uri: str = Field(
        default="mongodb://localhost:27017/",
        description="MongoDB connection URI",
    )
    mongodb_database: str = Field(
        default="facedatabase",
        description="MongoDB database name",
    )

    # MongoDB Collection Names
    collection_backup: str = Field(
        default="bkpcollection",
        description="Collection name for face embeddings backup",
    )
    collection_classes: str = Field(
        default="classcollection",
        description="Collection name for face classes",
    )
    collection_clusters: str = Field(
        default="clustercollection",
        description="Collection name for face clusters",
    )

    # Directory Paths
    source_dir: str = Field(
        default="/Volumes/data_sets/fresh_backup/images",
        description="Source directory containing images to process",
    )
    noface_dir: str = Field(
        default="/Volumes/data_sets/fresh_backup/noface",
        description="Directory for images with no faces detected",
    )
    broken_dir: str = Field(
        default="/Volumes/data_sets/fresh_backup/broken",
        description="Directory for broken/corrupted images",
    )
    cache_dir: str = Field(
        default="/Volumes/data_sets/fresh_backup/.cache",
        description="Directory for compressed image cache",
    )

    # Processing Settings
    cache_quality: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Quality setting for cached images (1-100)",
    )
    similarity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for face matching",
    )
    cluster_min_samples: int = Field(
        default=2,
        ge=1,
        description="Minimum samples for HDBSCAN clustering",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size for processing images",
    )

    # Model Settings
    insightface_providers: list[str] = Field(
        default_factory=lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        description="ONNX Runtime providers for InsightFace",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the global settings instance.

    Returns:
        Settings: The application settings.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """
    Reset the global settings instance.

    This is mainly useful for testing.
    """
    global _settings
    _settings = None
