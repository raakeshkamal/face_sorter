"""
Image compression and caching service.

This module handles building and managing the image cache for faster processing.
"""

import logging
import multiprocessing
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from PIL import Image

from face_sorter.config import get_settings
from face_sorter.database.repositories import FaceRepository
from face_sorter.models.face import CacheResult

logger = logging.getLogger(__name__)


def compress_image(bkp: dict[str, Any], quality: int = 50) -> bool:
    """
    Compress and save an image to the cache.

    Args:
        bkp: Dictionary containing image path and cache URL.
        quality: JPEG quality (1-100).

    Returns:
        True if successful, False otherwise.
    """
    try:
        img = Image.open(bkp["path"])
        img = img.resize(img.size, Image.Resampling.LANCZOS)
        file_path = os.path.expanduser(bkp["cache_url"])
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        img.save(file_path, quality=quality, optimize=True)
        return True
    except Exception as e:
        logger.error(f"Error processing {bkp['path']}: {e}")
        return False


def clear_and_recreate_cache(cache_dir: str) -> None:
    """
    Clear and recreate the cache directory.

    Args:
        cache_dir: Path to the cache directory.
    """
    cache_path = os.path.expanduser(cache_dir)
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path, ignore_errors=True)
    os.makedirs(cache_path, exist_ok=True)
    logger.info(f"Cache directory cleared and recreated: {cache_path}")


def build_cache(cache_dir: Optional[str] = None, quality: Optional[int] = None) -> CacheResult:
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
    clear_and_recreate_cache(cache_dir)

    # Get unique records from database
    face_repo = FaceRepository()
    unique_path = set()
    unique_records = []

    for bkp in face_repo.get_all_faces(sort=[("idx", 1)]):
        if bkp["path"] not in unique_path:
            unique_path.add(bkp["path"])
            unique_records.append(bkp)

    total = len(unique_records)
    logger.info(f"Found {total} unique images to cache")

    # Process images in parallel
    num_workers = min(multiprocessing.cpu_count() * 2, 16)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        # Submit all tasks
        for bkp in unique_records:
            futures.append(executor.submit(compress_image, bkp, quality))

        # Wait for all tasks to complete with progress tracking
        failed = 0
        for i, future in enumerate(futures, 1):
            success = future.result()
            if not success:
                failed += 1

            if i % 100 == 0 or i == total:
                logger.info(f"Processed {i}/{total} images")

    result = CacheResult(processed=total, total=total, failed=failed)
    logger.info(f"Cache built successfully. Success rate: {result.success_rate():.1f}%")

    return result
