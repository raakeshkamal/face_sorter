"""
Configuration management for Face Sorter.

This module uses Pydantic to manage application configuration from environment variables
and .env files.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
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
    duplicates_dir: str = Field(
        default="/Volumes/data_sets/fresh_backup/duplicates",
        description="Directory for duplicate images found by deduplication",
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

    # Deduplication Settings
    dedup_model_name: str = Field(
        default="openai/clip-vit-base-patch32",
        description="CLIP model name for deduplication",
    )
    dedup_threshold: float = Field(
        default=0.98,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for considering images as duplicates (0-1)",
    )
    dedup_batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size for deduplication embedding computation",
    )
    dedup_cache_file: Optional[str] = Field(
        default=None,
        description="Path to cache CLIP embeddings (defaults to cache_dir/clip_embeddings.npy)",
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

    # Dataset Cleaning Settings
    clean_output_dir: str = Field(
        default="/Volumes/data_sets/fresh_backup/cleaned",
        description="Output directory for cleaned images",
    )
    clean_broken_dir: str = Field(
        default="/Volumes/data_sets/fresh_backup/clean_broken",
        description="Directory for broken/invalid images from cleaning",
    )
    clean_batch_size: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Batch size for cleaning images",
    )
    clean_img_prefix: str = Field(
        default="IMG",
        description="Prefix for cleaned image filenames",
    )
    clean_quality: int = Field(
        default=95,
        ge=1,
        le=100,
        description="JPEG quality for cleaned images",
    )
    clean_recursive: bool = Field(
        default=True,
        description="Recursively scan source directory for images",
    )
    clean_start_index: int = Field(
        default=1,
        ge=0,
        description="Starting index for sequential naming",
    )
    clean_extensions: list[str] = Field(
        default_factory=lambda: [
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif', '.heic', '.heif'
        ],
        description="Image file extensions to process",
    )

    @field_validator("dedup_cache_file", mode="before")
    @classmethod
    def set_default_dedup_cache_file(cls, v: Optional[str], info) -> str:
        """Set default dedup cache file path based on cache_dir."""
        if v is None:
            cache_dir = info.data.get("cache_dir")
            if cache_dir:
                return str(Path(cache_dir) / "clip_embeddings.npy")
        return v or ""

    @field_validator("clean_extensions", mode="before")
    @classmethod
    def validate_clean_extensions(cls, v: list[str]) -> list[str]:
        """Remove HEIC/HEIF extensions if pillow-heif is not available."""
        heif_extensions = {'.heic', '.heif'}
        if any(ext.lower() in heif_extensions for ext in v):
            try:
                import pillow_heif
                pillow_heif.register_heif_opener()
                return v
            except ImportError:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning("pillow-heif not available, HEIC/HEIF files will be skipped")
                return [ext for ext in v if ext.lower() not in heif_extensions]
        return v

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
