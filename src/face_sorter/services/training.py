"""
Face detection and embedding generation service.

This module handles the training process: detecting faces in images and generating
face embeddings using InsightFace.
"""

import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any, Optional

import cv2
import psutil
from insightface.app import FaceAnalysis

from face_sorter.config import get_settings
from face_sorter.database.repositories import FaceRepository
from face_sorter.models.face import FaceEmbedding, TrainingProgress

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


def generate_embeddings(
    app: FaceAnalysis, img_path: str, noface_dir: Optional[str] = None
) -> list[Any]:
    """
    Generate face embeddings for an image.

    Args:
        app: InsightFace FaceAnalysis instance.
        img_path: Path to the image file.
        noface_dir: Directory to move images without faces (optional).

    Returns:
        List of face objects from InsightFace.
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Failed to read image: {img_path}")
            return []

        faces = app.get(img)
        return faces
    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return []


def get_file_list_filtered_and_sorted(
    bkp_collection: Any, src_dir: str
) -> list[str]:
    """
    Get a filtered and sorted list of image files.

    Args:
        bkp_collection: MongoDB collection containing processed images.
        src_dir: Source directory to scan.

    Returns:
        List of image filenames.
    """
    src_path = Path(src_dir)

    # Get all JPG files
    sort_list = [
        f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f)) and f.lower().endswith(".jpg")
    ]

    # Sort by file size
    sort_list = sorted(sort_list, key=lambda x: os.stat(os.path.join(src_dir, x)).st_size)

    # Remove already processed images
    for img in bkp_collection.find():
        if img["item"] in sort_list:
            sort_list.remove(img["item"])

    return sort_list


def train(
    source_dir: Optional[str] = None,
    noface_dir: Optional[str] = None,
    broken_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> TrainingProgress:
    """
    Train the model by detecting faces and generating embeddings.

    Args:
        source_dir: Directory containing images to process.
        noface_dir: Directory for images without faces.
        broken_dir: Directory for corrupted images.
        cache_dir: Directory for cache.

    Returns:
        TrainingProgress: Information about the training progress.
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

    src_dir = Path(source_dir)

    # Ensure output directories exist
    os.makedirs(noface_dir, exist_ok=True)
    os.makedirs(broken_dir, exist_ok=True)

    # Initialize face detection model
    logger.info("Initializing InsightFace model...")
    app = FaceAnalysis(providers=settings.insightface_providers)
    app.prepare(ctx_id=0)

    # Get database connection
    face_repo = FaceRepository()
    bkpcollection = face_repo.collection

    # Get file list
    logger.info("Loading image database...")
    file_list = get_file_list_filtered_and_sorted(bkpcollection, source_dir)
    random.shuffle(file_list)

    total_files = len(file_list)
    logger.info(f"Found {total_files} images to process")

    # Process images
    with_faces = 0
    without_faces = 0

    for i, item in enumerate(file_list, 1):
        item_path = src_dir.joinpath(item)

        if not os.path.exists(item_path):
            logger.warning(f"File not found, skipping: {item_path}")
            continue

        logger.info(f"Processing {i}/{total_files}: {item}")
        faces = generate_embeddings(app, str(item_path), noface_dir)

        if len(faces) == 0:
            logger.info(f"No face found, moving to noface directory: {item}")
            try:
                shutil.move(str(item_path), noface_dir)
                without_faces += 1
            except Exception as e:
                logger.error(f"Error moving file to noface directory: {e}")
            continue

        with_faces += 1

        # Save face embeddings
        for face in faces:
            face_data = FaceEmbedding(
                idx=bkpcollection.count_documents({}),
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
            face_repo.insert_face(face_data.to_dict())

        # Log memory usage periodically
        if i % 10 == 0:
            logger.info(f"Memory usage: {get_process_memory():.2f} MB")

    logger.info(f"Training complete. Processed {i} images")
    logger.info(f"With faces: {with_faces}, Without faces: {without_faces}")

    return TrainingProgress(
        processed=with_faces + without_faces,
        total=total_files,
        with_faces=with_faces,
        without_faces=without_faces,
    )
