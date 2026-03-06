"""
Image compression and caching service.

This module handles building and managing the image cache for faster processing.
"""

import asyncio
import logging
import os
from typing import Any, Optional

from PIL import Image

from face_sorter.config import get_settings
from face_sorter.database.repositories import FaceRepository
from face_sorter.models.face import CacheResult
from face_sorter.utils.file_async import async_makedirs, async_read_image

logger = logging.getLogger(__name__)


async def compress_image(bkp: dict[str, Any], quality: int = 50) -> bool:
    """
    Compress and save an image to the cache.

    Args:
        bkp: Dictionary containing image path and cache URL.
        quality: JPEG quality (1-100).

    Returns:
        True if successful, False otherwise.
    """
    try:
        img = await async_read_image(bkp["path"])

        # Ensure directory exists
        file_path = os.path.expanduser(bkp["cache_url"])
        await async_makedirs(os.path.dirname(file_path), exist_ok=True)

        # Resize and save (PIL is blocking)
        async def _compress(img_obj):
            img_resized = img_obj.resize(img_obj.size, Image.Resampling.LANCZOS)
            img_resized.save(file_path, quality=quality, optimize=True)

        await asyncio.to_thread(_compress, img)
        return True
    except Exception as e:
        logger.error(f"Error processing {bkp['path']}: {e}")
        return False


async def clear_and_recreate_cache(cache_dir: str) -> None:
    """
    Clear and recreate the cache directory.

    Args:
        cache_dir: Path to the cache directory.
    """
    from face_sorter.utils.file_async import async_delete_directory

    cache_path = os.path.expanduser(cache_dir)
    await async_delete_directory(cache_path, ignore_errors=True)
    await async_makedirs(cache_path, exist_ok=True)
    logger.info(f"Cache directory cleared and recreated: {cache_path}")


async def build_cache(
    cache_dir: Optional[str] = None, quality: Optional[int] = None
) -> CacheResult:
    """
    Build the image cache from the database.

    Args:
        cache_dir: Cache directory path.
        quality: JPEG quality for cached images.

    Returns:
        CacheResult: Information about the cache building process.
    """
    settings = get_settings()

    if cache_dir is None:
        cache_dir = settings.cache_dir
    if quality is None:
        quality = settings.cache_quality

    logger.info("Building cache...")

    # Clear and recreate cache directory
    await clear_and_recreate_cache(cache_dir)

    # Get unique records from database
    face_repo = FaceRepository()
    unique_path = set()
    unique_records = []

    # Get all faces from database (async)
    faces = await face_repo.get_all_faces(sort=[("idx", 1)])
    for bkp in faces:
        if bkp["path"] not in unique_path:
            unique_path.add(bkp["path"])
            unique_records.append(bkp)

    total = len(unique_records)
    logger.info(f"Found {total} unique images to cache")

    # Process images in parallel using asyncio.gather
    failed = 0

    # Process in batches to avoid memory issues
    batch_size = 50
    for i in range(0, total, batch_size):
        batch = unique_records[i : i + batch_size]

        # Create tasks for this batch
        tasks = [compress_image(bkp, quality) for bkp in batch]

        # Process batch in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count failures
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in batch processing: {result}")
                failed += 1
            elif result is False:
                failed += 1

        # Log progress
        processed = min(i + batch_size, total)
        if processed % 100 == 0 or processed == total:
            logger.info(f"Processed {processed}/{total} images")

    result = CacheResult(processed=total, total=total, failed=failed)
    logger.info(f"Cache built successfully. Success rate: {result.success_rate():.1f}%")

    return result


# Synchronous wrappers for backward compatibility
def compress_image_sync(bkp: dict[str, Any], quality: int = 50) -> bool:
    """
    Synchronous wrapper for compress_image.

    Args:
        bkp: Dictionary containing image path and cache URL.
        quality: JPEG quality (1-100).

    Returns:
        True if successful, False otherwise.
    """
    return asyncio.run(compress_image(bkp, quality))


def clear_and_recreate_cache_sync(cache_dir: str) -> None:
    """
    Synchronous wrapper for clear_and_recreate_cache.

    Args:
        cache_dir: Path to the cache directory.
    """
    return asyncio.run(clear_and_recreate_cache(cache_dir))


def build_cache_sync(
    cache_dir: Optional[str] = None, quality: Optional[int] = None
) -> CacheResult:
    """
    Synchronous wrapper for build_cache.

    Args:
        cache_dir: Cache directory path.
        quality: JPEG quality for cached images.

    Returns:
        CacheResult: Information about the cache building process.
    """
    return asyncio.run(build_cache(cache_dir, quality))
