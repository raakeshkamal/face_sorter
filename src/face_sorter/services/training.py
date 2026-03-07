"""
Face detection and embedding generation service.

This module handles the training process: detecting faces in images and generating
face embeddings using InsightFace.
"""

import asyncio
import logging
import os
import random
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import psutil
from insightface.app import FaceAnalysis

from face_sorter.config import get_settings
from face_sorter.database.repositories import FaceRepository
from face_sorter.models.face import FaceEmbedding, TrainingProgress
from face_sorter.utils.file_async import (
    async_file_exists,
    async_list_files,
    async_move_file,
    async_makedirs,
)

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
    progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
) -> TrainingProgress:
    """
    Train model by detecting faces and generating embeddings.

    Args:
        source_dir: Directory containing images to process.
        noface_dir: Directory for images without faces.
        broken_dir: Directory for corrupted images.
        cache_dir: Directory for cache.
        duplicates_dir: Directory for duplicate images (will be skipped).
        progress_callback: Optional callback function(current, total, status, current_item)
                        for reporting progress during training.

    Returns:
        TrainingProgress: Information about training progress.
    """
    settings = get_settings()

    # Use provided directories or defaults from settings
    if source_dir is None:
        source_dir = settings.source_dir
    if noface_dir is None:
        noface_dir = settings.noface_dir
    if broken_dir is None:
        broken_dir = settings.broken_dir
    if cache_dir is None:
        cache_dir = settings.cache_dir
    if duplicates_dir is None:
        duplicates_dir = settings.duplicates_dir

    src_dir = Path(source_dir)
    duplicates_path = Path(duplicates_dir) if duplicates_dir else None

    # Ensure output directories exist
    await async_makedirs(noface_dir, exist_ok=True)
    await async_makedirs(broken_dir, exist_ok=True)

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
        item_path = src_dir.joinpath(item)

        if not await async_file_exists(str(item_path)):
            logger.warning(f"File not found, skipping: {item_path}")
            continue

        logger.info(f"Processing {i}/{total_files}: {item}")
        faces = await generate_embeddings(app, str(item_path), noface_dir)

        # Report progress
        if progress_callback and (i == 1 or i % 10 == 0 or i == total_files):
            status = "Processing images" if i < total_files else "Complete"
            progress_callback(i, total_files, status, item)
            
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
                cache_url=os.path.join(cache_dir, item),
            )
            await face_repo.insert_face(face_data.to_dict())
            count += 1

        # Log memory usage periodically
        if i % 10 == 0:
            logger.info(f"Memory usage: {get_process_memory():.2f} MB")

    logger.info(f"Training complete. Processed {i} images")
    logger.info(f"With faces: {with_faces}, Without faces: {without_faces}")

    # Report final progress
    if progress_callback:
        progress_callback(i, total_files, "Complete", "")

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
    return asyncio.run(train(source_dir, noface_dir, broken_dir, cache_dir, duplicates_dir))
