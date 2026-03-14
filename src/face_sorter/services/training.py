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
from face_sorter.models.session import (
    FaceDetectionTrainingCancelledError,
    StabilityScoreTrainingCancelledError,
    TrainingCancelledError,
)
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


async def get_all_image_files(src_dir: str, duplicates_path: Optional[Path] = None) -> list[str]:
    """
    Get all image files from a directory without filtering by database entries.

    Args:
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

    # Use provided directories or derive from source_dir parent
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

    # Check for cancellation before starting expensive operations
    if cancellation_event and cancellation_event.is_set():
        raise TrainingCancelledError("Training was cancelled before initialization")

    # Ensure output directories exist
    await async_makedirs(noface_dir, exist_ok=True)
    await async_makedirs(broken_dir, exist_ok=True)
    await async_makedirs(cache_dir, exist_ok=True)  # Ensure cache directory exists
    await async_makedirs(duplicates_dir, exist_ok=True) # Ensure duplicates directory exists

    # Initialize face detection model
    logger.info("Initializing InsightFace model...")
    app = FaceAnalysis(providers=settings.insightface_providers)
    app.prepare(ctx_id=0)

    # Load stability score classification model
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

        # Calculate stability score (once per image, not per face)
        stability_score = None
        classification_result = None
        content_probability = None

        if stability_session is not None and stability_processor is not None:
            try:
                stability_score, details = await evaluate_stability(
                    stability_session,
                    stability_processor,
                    item_path
                )
                classification_result = details.get('classification_result')
                content_probability = details.get('content_probability')
            except Exception as e:
                logger.warning(f"Failed to calculate stability score for {item}: {e}")

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
            # Add stability score if available
            if stability_score is not None:
                image_data.update({
                    "stability_score": stability_score,
                    "classification_result": classification_result,
                    "content_probability": content_probability,
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
                stability_score=stability_score,
                classification_result=classification_result,
                content_probability=content_probability,
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


async def train_stability_scores(
    source_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    duplicates_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str, str, dict[str, Any] | None], None]] = None,
    cancellation_event: Optional[asyncio.Event] = None,
) -> TrainingProgress:
    """
    Train stability scores only (no face detection).

    Processes all images, calculates stability scores, and saves partial documents
    to MongoDB. Skips images that already have stability scores.

    Args:
        source_dir: Directory containing images to process.
        cache_dir: Directory for cache.
        duplicates_dir: Directory for duplicate images (will be skipped).
        progress_callback: Optional callback function(current, total, status, current_item, image_data)
                        for reporting progress during training.
        cancellation_event: Optional asyncio.Event that, when set, signals cancellation.

    Returns:
        TrainingProgress: Information about training progress.

    Raises:
        StabilityScoreTrainingCancelledError: If cancellation_event is set during training.
    """
    settings = get_settings()

    # Use provided directories or derive from source_dir parent
    if not source_dir:
        source_dir = settings.source_dir

    src_path = Path(source_dir).resolve()
    parent_dir = src_path.parent

    if not cache_dir:
        cache_dir = str(parent_dir / ".cache")
    if not duplicates_dir:
        duplicates_dir = str(parent_dir / "duplicates")

    src_dir = Path(source_dir)
    duplicates_path = Path(duplicates_dir) if duplicates_dir else None

    # Check for cancellation before starting expensive operations
    if cancellation_event and cancellation_event.is_set():
        from face_sorter.models.session import StabilityScoreTrainingCancelledError
        raise StabilityScoreTrainingCancelledError("Stability score training was cancelled before initialization")

    # Ensure output directories exist
    await async_makedirs(cache_dir, exist_ok=True)

    # Load stability score classification model only
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
    for i, item in enumerate(file_list, 1):
        # Check for cancellation before processing each image
        if cancellation_event and cancellation_event.is_set():
            from face_sorter.models.session import StabilityScoreTrainingCancelledError
            logger.info("Stability score training cancelled by user")
            raise StabilityScoreTrainingCancelledError("Stability score training was cancelled by user")

        item_path = src_dir.joinpath(item)

        if not await async_file_exists(str(item_path)):
            logger.warning(f"File not found, skipping: {item_path}")
            continue

        # Check if stability score already exists
        existing_score = await face_repo.get_stability_score_by_item(item)
        if existing_score and existing_score.get("stability_score") is not None:
            logger.info(f"Stability score already exists, skipping: {item}")
            continue

        # Build cache
        cache_filename = item  # Keep same filename for URL consistency
        cache_path = os.path.join(cache_dir, cache_filename)

        cache_success = await compress_image(
            input_path=str(item_path),
            output_path=cache_path,
            quality=settings.cache_quality,
            optimize=True
        )
        if not cache_success:
            logger.warning(f"Failed to build cache for {item}, but continuing")

        # Calculate stability score
        stability_score = None
        classification_result = None
        content_probability = None

        if stability_session is not None and stability_processor is not None:
            try:
                stability_score, details = await evaluate_stability(
                    stability_session,
                    stability_processor,
                    item_path
                )
                classification_result = details.get('classification_result')
                content_probability = details.get('content_probability')
                logger.info(f"Processing {i}/{total_files}: {item} - Stability: {stability_score:.3f}")
            except Exception as e:
                logger.warning(f"Failed to calculate stability score for {item}: {e}")

        # Save partial document to MongoDB
        # Query count to assign an idx if this is a new document, avoiding duplicate key error on idx: null
        count = await face_repo.count_faces()
        
        # Check if the document already exists to preserve its idx
        collection = await face_repo._get_collection()
        existing_doc = await collection.find_one({"item": item}, {"idx": 1})
        current_idx = existing_doc.get("idx") if existing_doc and "idx" in existing_doc else count
        
        partial_data = {
            "idx": current_idx,
            "item": item,
            "path": str(item_path),
            "cache_url": item,
            "stability_score": stability_score,
            "classification_result": classification_result,
            "content_probability": content_probability,
        }
        await face_repo.upsert_face_by_item(item, partial_data)

        # Report progress
        if progress_callback and (i == 1 or i % 10 == 0 or i == total_files):
            status = "Processing images" if i < total_files else "Complete"
            image_data = {
                "filename": item,
                "cache_url": item,
                "stability_score": stability_score,
                "classification_result": classification_result,
                "content_probability": content_probability,
            }
            progress_callback(i, total_files, status, item, image_data)
            await asyncio.sleep(0.01)

    logger.info(f"Stability score training complete. Processed {total_files} images")

    # Report final progress
    if progress_callback:
        progress_callback(total_files, total_files, "Complete", "", None)

    return TrainingProgress(
        processed=total_files,
        total=total_files,
        with_faces=0,  # Not applicable for stability scores
        without_faces=0,  # Not applicable for stability scores
    )


async def train_face_detections(
    source_dir: Optional[str] = None,
    noface_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    duplicates_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str, str, dict[str, Any] | None], None]] = None,
    cancellation_event: Optional[asyncio.Event] = None,
) -> TrainingProgress:
    """
    Train face detection only (no stability score calculation).

    Processes all images, detects faces, and merges with existing stability scores
    via upsert. Moves images without faces to noface directory.

    Args:
        source_dir: Directory containing images to process.
        noface_dir: Directory for images without faces.
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
        FaceDetectionTrainingCancelledError: If cancellation_event is set during training.
    """
    settings = get_settings()

    # Use provided directories or derive from source_dir parent
    if not source_dir:
        source_dir = settings.source_dir

    src_path = Path(source_dir).resolve()
    parent_dir = src_path.parent

    if not noface_dir:
        noface_dir = str(parent_dir / "noface")
    if not cache_dir:
        cache_dir = str(parent_dir / ".cache")
    if not duplicates_dir:
        duplicates_dir = str(parent_dir / "duplicates")

    src_dir = Path(source_dir)
    duplicates_path = Path(duplicates_dir) if duplicates_dir else None

    # Check for cancellation before starting expensive operations
    if cancellation_event and cancellation_event.is_set():
        from face_sorter.models.session import FaceDetectionTrainingCancelledError
        raise FaceDetectionTrainingCancelledError("Face detection training was cancelled before initialization")

    # Ensure output directories exist
    await async_makedirs(noface_dir, exist_ok=True)
    await async_makedirs(cache_dir, exist_ok=True)

    # Initialize face detection model only
    logger.info("Initializing InsightFace model...")
    app = FaceAnalysis(providers=settings.insightface_providers)
    app.prepare(ctx_id=0)

    # Get database connection
    face_repo = FaceRepository()

    # Get file list - get all files, will filter for existing embeddings inline
    logger.info("Loading image files...")
    file_list = await get_all_image_files(source_dir, duplicates_path)
    random.shuffle(file_list)

    total_files = len(file_list)
    logger.info(f"Found {total_files} images to process")

    # Process images
    with_faces = 0
    without_faces = 0

    for i, item in enumerate(file_list, 1):
        # Check for cancellation before processing each image
        if cancellation_event and cancellation_event.is_set():
            from face_sorter.models.session import FaceDetectionTrainingCancelledError
            logger.info("Face detection training cancelled by user")
            raise FaceDetectionTrainingCancelledError("Face detection training was cancelled by user")

        item_path = src_dir.joinpath(item)

        if not await async_file_exists(str(item_path)):
            logger.warning(f"File not found, skipping: {item_path}")
            continue

        # Check if face embedding already exists, skip if so
        collection = await face_repo._get_collection()
        existing_doc = await collection.find_one(
            {"item": item},
            projection={"embedding": 1, "_id": 0}
        )
        if existing_doc and "embedding" in existing_doc and existing_doc["embedding"]:
            logger.info(f"Face embedding already exists, skipping: {item}")
            continue

        # Build cache if not exists
        cache_filename = item
        cache_path = os.path.join(cache_dir, cache_filename)

        if not await async_file_exists(cache_path):
            cache_success = await compress_image(
                input_path=str(item_path),
                output_path=cache_path,
                quality=settings.cache_quality,
                optimize=True
            )
            if not cache_success:
                logger.warning(f"Failed to build cache for {item}, but continuing")

        # Generate face embeddings
        logger.info(f"Processing {i}/{total_files}: {item}")
        faces = await generate_embeddings(app, str(item_path), noface_dir)

        # Check if stability score exists in DB, merge if present
        existing_stability = await face_repo.get_stability_score_by_item(item)
        stability_score = None
        classification_result = None
        content_probability = None

        if existing_stability:
            stability_score = existing_stability.get("stability_score")
            classification_result = existing_stability.get("classification_result")
            content_probability = existing_stability.get("content_probability")

        # Report progress
        if progress_callback and (i == 1 or i % 10 == 0 or i == total_files):
            status = "Processing images" if i < total_files else "Complete"
            image_data = {
                "filename": item,
                "cache_url": item,
            }
            if len(faces) > 0:
                face = faces[0]
                image_data.update({
                    "det_score": float(face.det_score),
                    "age": int(face.age),
                    "gender": int(face.gender),
                })
            if stability_score is not None:
                image_data.update({
                    "stability_score": stability_score,
                    "classification_result": classification_result,
                    "content_probability": content_probability,
                })
            progress_callback(i, total_files, status, item, image_data)
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

        # Save face embeddings via upsert (merge with existing stability scores)
        count = await face_repo.count_faces()
        collection = await face_repo._get_collection()

        # Check if document already exists to preserve its idx
        existing_doc = await collection.find_one({"item": item}, {"idx": 1})
        current_idx = existing_doc.get("idx") if existing_doc and "idx" in existing_doc else None

        for face in faces:
            face_data = {
                "idx": current_idx if current_idx is not None else count,
                "item": item,
                "path": str(item_path),
                "age": face.age,
                "gender": int(face.gender),
                "bbox": face.bbox.tolist(),
                "kps": face.kps.tolist(),
                "det_score": float(face.det_score),
                "landmark_3d_68": face.landmark_3d_68.tolist(),
                "pose": face.pose.tolist(),
                "landmark_2d_106": face.landmark_2d_106.tolist(),
                "embedding": face.embedding.tolist(),
                "cache_url": item,
                "stability_score": stability_score,
                "classification_result": classification_result,
                "content_probability": content_probability,
            }
            await face_repo.upsert_face_by_item(item, face_data)
            if current_idx is None:  # Only increment count for new documents
                count += 1

        # Log memory usage periodically
        if i % 10 == 0:
            logger.info(f"Memory usage: {get_process_memory():.2f} MB")

    logger.info(f"Face detection training complete. Processed {with_faces + without_faces} images")
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


def train_stability_scores_sync(
    source_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    duplicates_dir: Optional[str] = None,
) -> TrainingProgress:
    """
    Synchronous wrapper for train_stability_scores function.

    Args:
        source_dir: Directory containing images to process.
        cache_dir: Directory for cache.
        duplicates_dir: Directory for duplicate images (will be skipped).

    Returns:
        TrainingProgress: Information about training progress.
    """
    return asyncio.run(train_stability_scores(source_dir, cache_dir, duplicates_dir, None, None))


def train_face_detections_sync(
    source_dir: Optional[str] = None,
    noface_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    duplicates_dir: Optional[str] = None,
) -> TrainingProgress:
    """
    Synchronous wrapper for train_face_detections function.

    Args:
        source_dir: Directory containing images to process.
        noface_dir: Directory for images without faces.
        cache_dir: Directory for cache.
        duplicates_dir: Directory for duplicate images (will be skipped).

    Returns:
        TrainingProgress: Information about training progress.
    """
    return asyncio.run(train_face_detections(source_dir, noface_dir, cache_dir, duplicates_dir, None, None))
