"""
Face detection and embedding generation service.

This module handles the training process: detecting faces in images and generating
face embeddings using InsightFace, along with stability score calculation.
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
from face_sorter.services.stability_score import (
    evaluate_stability,
    load_stability_model,
)
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


async def generate_embeddings(app: FaceAnalysis, img_path: str) -> list[Any]:
    """
    Generate face embeddings for an image.

    Args:
        app: InsightFace FaceAnalysis instance.
        img_path: Path to image file.

    Returns:
        List of face objects from InsightFace.
    """
    try:
        img = await asyncio.to_thread(cv2.imread, img_path)
        if img is None:
            logger.warning(f"Failed to read image: {img_path}")
            return []

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

    all_files = await async_list_files(src_dir, "*.jpg")
    all_files.extend(await async_list_files(src_dir, "*.JPG"))

    sort_list = sorted(all_files, key=lambda x: Path(x).stat().st_size)
    sort_list = [Path(f).name for f in sort_list]

    processed_items = set()
    async for img in bkpcollection.find():
        if "item" in img:
            processed_items.add(img["item"])

    sort_list = [f for f in sort_list if f not in processed_items]

    if duplicates_path:
        duplicates_set = set()
        try:
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
    Train model by detecting faces and generating embeddings with stability scores.

    This unified function:
    1. Loads both InsightFace and stability models at startup
    2. For each image:
       - Runs face detection first
       - If face found: calculates stability score, saves full document
       - If no face: moves to noface directory (skips stability)

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
                        - stability_score: Stability score (if face found)
                        - classification_result: Classification result (if face found)
                        - content_probability: Content probability (if face found)
        cancellation_event: Optional asyncio.Event that, when set, signals cancellation.

    Returns:
        TrainingProgress: Information about training progress.

    Raises:
        TrainingCancelledError: If cancellation_event is set during training.
    """
    settings = get_settings()

    if not source_dir:
        source_dir = settings.source_dir

    src_path = Path(source_dir).resolve()
    parent_dir = src_path.parent

    if not noface_dir:
        noface_dir = str(parent_dir / "noface")
    if not broken_dir:
        broken_dir = str(parent_dir / "broken")
    if not cache_dir:
        cache_dir = str(parent_dir / ".cache")
    if not duplicates_dir:
        duplicates_dir = str(parent_dir / "duplicates")

    src_dir = Path(source_dir)
    duplicates_path = Path(duplicates_dir) if duplicates_dir else None

    if cancellation_event and cancellation_event.is_set():
        raise TrainingCancelledError("Training was cancelled before initialization")

    await async_makedirs(noface_dir, exist_ok=True)
    await async_makedirs(broken_dir, exist_ok=True)
    await async_makedirs(cache_dir, exist_ok=True)
    await async_makedirs(duplicates_dir, exist_ok=True)

    logger.info("Loading models...")
    logger.info("Initializing InsightFace model...")
    app = FaceAnalysis(providers=settings.insightface_providers)
    app.prepare(ctx_id=0)

    stability_session = None
    stability_processor = None
    if settings.stability_model_name:
        logger.info("Loading stability score classification model (ONNX)...")
        try:
            stability_session, stability_processor, model_desc = await load_stability_model()
            logger.info(f"Stability model loaded: {model_desc}")
        except Exception as e:
            logger.warning(f"Failed to load stability score model: {e}")
            stability_session = None
            stability_processor = None
    else:
        logger.info("Stability score detection disabled (no model configured)")

    face_repo = FaceRepository()
    bkpcollection = await face_repo._get_collection()

    logger.info("Loading image database...")
    file_list = await get_file_list_filtered_and_sorted(bkpcollection, source_dir, duplicates_path)
    random.shuffle(file_list)

    total_files = len(file_list)
    logger.info(f"Found {total_files} images to process")

    with_faces = 0
    without_faces = 0

    for i, item in enumerate(file_list, 1):
        if cancellation_event and cancellation_event.is_set():
            logger.info("Training cancelled by user")
            raise TrainingCancelledError("Training was cancelled by user")

        item_path = src_dir.joinpath(item)

        if not await async_file_exists(str(item_path)):
            logger.warning(f"File not found, skipping: {item_path}")
            continue

        cache_filename = item
        cache_path = os.path.join(cache_dir, cache_filename)

        cache_success = await compress_image(
            input_path=str(item_path),
            output_path=cache_path,
            quality=settings.cache_quality,
            optimize=True,
        )
        if not cache_success:
            logger.warning(f"Failed to build cache for {item}, but continuing training")

        logger.info(f"Processing {i}/{total_files}: {item}")
        faces = await generate_embeddings(app, str(item_path))

        if len(faces) == 0:
            logger.info(f"No face found, moving to noface directory: {item}")
            try:
                await async_move_file(str(item_path), noface_dir)
                without_faces += 1
            except Exception as e:
                logger.error(f"Error moving file to noface directory: {e}")
            continue

        with_faces += 1

        stability_score = None
        classification_result = None
        content_probability = None

        if stability_session is not None and stability_processor is not None:
            try:
                stability_score, details = await evaluate_stability(
                    stability_session, stability_processor, item_path
                )
                classification_result = details.get("classification_result")
                content_probability = details.get("content_probability")
            except Exception as e:
                logger.warning(f"Failed to calculate stability score for {item}: {e}")

        if progress_callback and (i == 1 or i % 10 == 0 or i == total_files):
            status = "Processing images" if i < total_files else "Complete"
            image_data: dict[str, Any] = {
                "filename": item,
                "cache_url": item,
            }
            face = faces[0]
            image_data["det_score"] = float(face.det_score)
            image_data["age"] = int(face.age)
            image_data["gender"] = int(face.gender)
            if stability_score is not None:
                image_data["stability_score"] = stability_score
                image_data["classification_result"] = classification_result
                image_data["content_probability"] = content_probability
            progress_callback(i, total_files, status, item, image_data)
            await asyncio.sleep(0.01)

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
                cache_url=item,
                stability_score=stability_score,
                classification_result=classification_result,
                content_probability=content_probability,
            )
            await face_repo.insert_face(face_data.to_dict())
            count += 1

        if i % 10 == 0:
            logger.info(f"Memory usage: {get_process_memory():.2f} MB")

    logger.info(f"Training complete. Processed {with_faces + without_faces} images")
    logger.info(f"With faces: {with_faces}, Without faces: {without_faces}")

    if progress_callback:
        progress_callback(with_faces + without_faces, total_files, "Complete", "", None)

    return TrainingProgress(
        processed=with_faces + without_faces,
        total=total_files,
        with_faces=with_faces,
        without_faces=without_faces,
    )


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
