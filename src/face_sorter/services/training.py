"""
Face detection and embedding generation service.

This module handles the training process: detecting faces in images and generating
face embeddings using InsightFace.
"""

import asyncio
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import psutil
from insightface.app import FaceAnalysis

from face_sorter.config import get_settings
from face_sorter.database.repositories import FaceRepository
from face_sorter.models.face import FaceEmbedding, TrainingProgress
from face_sorter.models.session import TrainingCancelledError
from face_sorter.utils.file_async import (
    async_file_exists,
    async_list_files,
    async_move_file,
    async_makedirs,
)
from face_sorter.utils.image import compress_image

logger = logging.getLogger(__name__)


def get_process_memory() -> float:
    """
    Get the current process memory usage in MB.

    Returns:
        Memory usage in megabytes.
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)


async def generate_embeddings(
    app: FaceAnalysis, img_path: str, noface_dir: Optional[str] = None
) -> list[Any]:
    """
    Generate face embeddings for an image.

    Args:
        app: InsightFace FaceAnalysis instance.
        img_path: Path to image file.
        noface_dir: Directory to move images without faces (optional, unused).

    Returns:
        List of face objects from InsightFace.
    """
    """
    Generate face embeddings for an image.

    Args:
        app: InsightFace FaceAnalysis instance.
        img_path: Path to image file.
        noface_dir: Directory to move images without faces (optional).

    Returns:
        List of face objects from InsightFace.
    """
    try:
        # cv2 doesn't have async support, use thread pool
        img = await asyncio.to_thread(cv2.imread, img_path)
        if img is None:
            logger.warning(f"Failed to read image: {img_path}")
            return []

        # InsightFace operations are blocking, use thread pool
        faces = await asyncio.to_thread(app.get, img)
        return faces
    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return []


async def get_file_list_filtered_and_sorted(
    bkpcollection: Any, src_dir: str, duplicates_path: Optional[Path] = None
) -> list[str]:
    """
    Get a filtered and sorted list of image files.

    Args:
        bkpcollection: MongoDB collection containing processed images.
        src_dir: Source directory to scan.
        duplicates_path: Path to duplicates directory (will be skipped).

    Returns:
        List of image filenames.
    """
    src_path = Path(src_dir)

    # Get all JPG files (async list files)
    all_files = await async_list_files(src_dir, "*.jpg")
    all_files.extend(await async_list_files(src_dir, "*.JPG"))

    # Sort by file size
    sort_list = sorted(all_files, key=lambda x: Path(x).stat().st_size)
    # Extract just filenames
    sort_list = [Path(f).name for f in sort_list]

    # Remove already processed images
    processed_items = set()
    async for img in bkpcollection.find():
        if "item" in img:
            processed_items.add(img["item"])

    sort_list = [f for f in sort_list if f not in processed_items]

    # Remove files in duplicates directory
    if duplicates_path:
        duplicates_set = set()
        try:
            # Get all files in duplicates directory
            duplicates_files = await async_list_files(str(duplicates_path), "*.jpg")
            duplicates_files.extend(await async_list_files(str(duplicates_path), "*.JPG"))
            duplicates_set = {Path(f).name for f in duplicates_files}
        except Exception as e:
            logger.warning(f"Could not read duplicates directory: {e}")

        sort_list = [f for f in sort_list if f not in duplicates_set]

    return sort_list


async def train(
    source_dir: Optional[str] = None,
    noface_dir: Optional[str] = None,
    broken_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    duplicates_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str, str, dict[str, Any] | None], None]] = None,
    cancellation_event: Optional[asyncio.Event] = None,
) -> TrainingProgress:
    """
    Train model by detecting faces and generating embeddings.

    Args:
        source_dir: Directory containing images to process.
        noface_dir: Directory for images without faces.
        broken_dir: Directory for corrupted images.
        cache_dir: Directory for cache.
        duplicates_dir: Directory for duplicate images (will be skipped).
        progress_callback: Optional callback function(current, total, status, current_item, image_data)
                        for reporting progress during training. image_data contains:
                        - filename: Name of the current image
                        - cache_url: URL to the cached image
                        - det_score: Detection confidence score (if face found)
                        - age: Estimated age (if face found)
                        - gender: Gender (0=male, 1=female, if face found)
        cancellation_event: Optional asyncio.Event that, when set, signals cancellation.

    Returns:
        TrainingProgress: Information about training progress.

    Raises:
        TrainingCancelledError: If cancellation_event is set during training.
    """
    settings = get_settings()

    # Use provided directories or defaults from settings
    if not source_dir:
        source_dir = settings.source_dir
    if not noface_dir:
        noface_dir = settings.noface_dir
    if not broken_dir:
        broken_dir = settings.broken_dir
    if not cache_dir:
        cache_dir = settings.cache_dir
    if not duplicates_dir:
        duplicates_dir = settings.duplicates_dir

    src_dir = Path(source_dir)
    duplicates_path = Path(duplicates_dir) if duplicates_dir else None

    # Check for cancellation before starting expensive operations
    if cancellation_event and cancellation_event.is_set():
        raise TrainingCancelledError("Training was cancelled before initialization")

    # Ensure output directories exist
    await async_makedirs(noface_dir, exist_ok=True)
    await async_makedirs(broken_dir, exist_ok=True)
    await async_makedirs(cache_dir, exist_ok=True)  # Ensure cache directory exists

    # Initialize face detection model
    logger.info("Initializing InsightFace model...")
    app = FaceAnalysis(providers=settings.insightface_providers)
    app.prepare(ctx_id=0)

    # Get database connection
    face_repo = FaceRepository()
    bkpcollection = await face_repo._get_collection()

    # Get file list
    logger.info("Loading image database...")
    file_list = await get_file_list_filtered_and_sorted(bkpcollection, source_dir, duplicates_path)
    random.shuffle(file_list)

    total_files = len(file_list)
    logger.info(f"Found {total_files} images to process")

    # Process images
    with_faces = 0
    without_faces = 0

    for i, item in enumerate(file_list, 1):
        # Check for cancellation before processing each image
        if cancellation_event and cancellation_event.is_set():
            logger.info("Training cancelled by user")
            raise TrainingCancelledError("Training was cancelled by user")

        item_path = src_dir.joinpath(item)

        if not await async_file_exists(str(item_path)):
            logger.warning(f"File not found, skipping: {item_path}")
            continue

        # Build cache BEFORE face detection so carousel can display image
        cache_filename = item  # Keep same filename for URL consistency
        cache_path = os.path.join(cache_dir, cache_filename)

        # Build cache synchronously (await it to ensure completion)
        cache_success = await compress_image(
            input_path=str(item_path),
            output_path=cache_path,
            quality=settings.cache_quality,
            optimize=True
        )
        if not cache_success:
            logger.warning(f"Failed to build cache for {item}, but continuing training")

        # Now generate face embeddings (cache file is guaranteed to exist)
        logger.info(f"Processing {i}/{total_files}: {item}")
        faces = await generate_embeddings(app, str(item_path), noface_dir)

        # Report progress
        if progress_callback and (i == 1 or i % 10 == 0 or i == total_files):
            status = "Processing images" if i < total_files else "Complete"
            # Include image data for the carousel
            image_data = {
                "filename": item,
                "cache_url": item,  # Just the filename, frontend handles URL construction
            }
            # Add face detection metadata if available
            if len(faces) > 0:
                face = faces[0]  # Use first face for metadata
                image_data.update({
                    "det_score": float(face.det_score),
                    "age": int(face.age),
                    "gender": int(face.gender),
                })
            progress_callback(i, total_files, status, item, image_data)

            # Explicitly yield to the event loop to ensure WebSockets flush
            await asyncio.sleep(0.01)

        if len(faces) == 0:
            logger.info(f"No face found, moving to noface directory: {item}")
            try:
                await async_move_file(str(item_path), noface_dir)
                without_faces += 1
            except Exception as e:
                logger.error(f"Error moving file to noface directory: {e}")
            continue

        with_faces += 1

        # Save face embeddings
        count = await face_repo.count_faces()
        for face in faces:
            face_data = FaceEmbedding(
                idx=count,
                item=item,
                path=str(item_path),
                age=face.age,
                gender=int(face.gender),
                bbox=face.bbox.tolist(),
                kps=face.kps.tolist(),
                det_score=float(face.det_score),
                landmark_3d_68=face.landmark_3d_68.tolist(),
                pose=face.pose.tolist(),
                landmark_2d_106=face.landmark_2d_106.tolist(),
                embedding=face.embedding.tolist(),
                cache_url=item,  # Just the filename, frontend handles URL construction
            )
            await face_repo.insert_face(face_data.to_dict())
            count += 1

        # Log memory usage periodically
        if i % 10 == 0:
            logger.info(f"Memory usage: {get_process_memory():.2f} MB")

    logger.info(f"Training complete. Processed {with_faces + without_faces} images")
    logger.info(f"With faces: {with_faces}, Without faces: {without_faces}")

    # Report final progress
    if progress_callback:
        progress_callback(with_faces + without_faces, total_files, "Complete", "", None)

    return TrainingProgress(
        processed=with_faces + without_faces,
        total=total_files,
        with_faces=with_faces,
        without_faces=without_faces,
    )


# Synchronous wrappers for backward compatibility
def train_sync(
    source_dir: Optional[str] = None,
    noface_dir: Optional[str] = None,
    broken_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    duplicates_dir: Optional[str] = None,
) -> TrainingProgress:
    """
    Synchronous wrapper for train function.

    Args:
        source_dir: Directory containing images to process.
        noface_dir: Directory for images without faces.
        broken_dir: Directory for corrupted images.
        cache_dir: Directory for cache.
        duplicates_dir: Directory for duplicate images (will be skipped).

    Returns:
        TrainingProgress: Information about training progress.
    """
    return asyncio.run(train(source_dir, noface_dir, broken_dir, cache_dir, duplicates_dir, None))
