"""
Deduplication service for Face Sorter.

This module provides functionality to find and move duplicate images using CLIP embeddings.
"""

import asyncio
import glob
import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

from face_sorter.models.face import DeduplicationResult, DuplicateMoveResult
from face_sorter.utils.file_async import (
    async_makedirs,
    async_move_file,
    async_file_exists,
)

# Fix for DecompressionBombWarning
Image.MAX_IMAGE_PIXELS = None

# Set environment variable to potentially fix OpenMP conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger(__name__)


async def load_images(source_dir: str) -> List[str]:
    """
    Recursively find all images in the directory.

    Args:
        source_dir: Directory to search for images.

    Returns:
        List of image file paths.
    """
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.webp']
    image_files = []

    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(source_dir, '**', ext), recursive=True))

    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(source_dir, '**', ext.upper()), recursive=True))

    return sorted(list(set(image_files)))


async def get_image_quality(path: str) -> Tuple[int, int]:
    """
    Return (resolution_pixels, file_size_bytes) for quality comparison.

    Args:
        path: Path to the image file.

    Returns:
        Tuple of (resolution_pixels, file_size_bytes).
    """
    try:
        # Run in thread pool to avoid blocking
        result = await asyncio.to_thread(_get_image_quality_sync, path)
        return result
    except Exception as e:
        logger.error(f"Error getting image quality for {path}: {e}")
        return (0, 0)


def _get_image_quality_sync(path: str) -> Tuple[int, int]:
    """Synchronous helper for get_image_quality."""
    with Image.open(path) as img:
        pixels = img.width * img.height
    size = os.path.getsize(path)
    return (pixels, size)


async def compute_clip_embeddings(
    image_files: List[str],
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute CLIP embeddings for images in batches.

    Args:
        image_files: List of image file paths.
        model_name: CLIP model name.
        batch_size: Batch size for processing.
        device: Device to use (cpu, cuda, mps).

    Returns:
        Numpy array of embeddings.
    """
    logger.info(f"Loading CLIP model: {model_name}...")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    embeddings = []

    for i in tqdm(range(0, len(image_files), batch_size), desc="Computing embeddings"):
        batch_paths = image_files[i : i + batch_size]
        images = []

        for path in batch_paths:
            try:
                # Run in thread pool to avoid blocking
                image = await asyncio.to_thread(_load_image_sync, path)
                images.append(image)
            except Exception as e:
                logger.error(f"Error loading {path}: {e}")

        if not images:
            continue

        try:
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
                outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
                embeddings.append(outputs.cpu().numpy())
        except Exception as e:
            logger.error(f"Error processing batch starting at {i}: {e}")

    if embeddings:
        return np.vstack(embeddings).astype(np.float32)
    return np.array([], dtype=np.float32)


def _load_image_sync(path: str) -> Image.Image:
    """Synchronous helper for loading an image."""
    return Image.open(path).convert("RGB")


def find_duplicate_groups(
    image_files: List[str],
    embeddings: np.ndarray,
    threshold: float = 0.99,
    batch_size: int = 1000,
) -> List[Tuple[str, List[str]]]:
    """
    Find duplicates using PyTorch Matrix Multiplication.

    Args:
        image_files: List of image file paths.
        embeddings: Numpy array of embeddings.
        threshold: Similarity threshold (0-1).
        batch_size: Batch size for similarity computation.

    Returns:
        List of tuples (original_path, [duplicate_paths]).
    """
    logger.info("Preparing for similarity search...")
    if len(embeddings) == 0:
        logger.info("No embeddings to process.")
        return []

    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Check for NaNs
    if np.isnan(embeddings).any():
        logger.warning("Embeddings contain NaNs. Replacing with zeros.")
        embeddings = np.nan_to_num(embeddings)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"

    logger.info(f"Using device for search: {device}")

    # Convert to torch tensor
    try:
        embeddings_tensor = torch.from_numpy(embeddings).to(device)
    except Exception as e:
        logger.error(f"Error moving embeddings to device: {e}. Fallback to CPU.")
        device = "cpu"
        embeddings_tensor = torch.from_numpy(embeddings).to(device)

    num_images = len(embeddings_tensor)
    visited = set()
    duplicates = []

    logger.info(f"Computing similarity matrix in batches (Threshold: {threshold})...")

    for i in tqdm(range(0, num_images, batch_size), desc="Searching"):
        end_idx = min(i + batch_size, num_images)

        query_chunk = embeddings_tensor[i:end_idx]

        sim_matrix = torch.mm(query_chunk, embeddings_tensor.T)

        sim_matrix = sim_matrix.cpu().numpy()

        for local_row in range(sim_matrix.shape[0]):
            global_idx = i + local_row

            if global_idx in visited:
                continue

            scores = sim_matrix[local_row]

            matches = np.where(scores >= threshold)[0]

            group_indices = [m for m in matches if m not in visited]

            if len(group_indices) > 1:
                cluster_files = [image_files[idx] for idx in group_indices]

                scored_files = []
                for idx in group_indices:
                    path = image_files[idx]
                    quality = asyncio.run(get_image_quality(path))
                    scored_files.append((quality, path, idx))

                scored_files.sort(key=lambda x: (x[0][0], x[0][1], [-ord(c) for c in x[1]]), reverse=True)

                best_idx = scored_files[0][2]
                original_path = image_files[best_idx]

                duplicate_paths = []
                for _, path, idx in scored_files[1:]:
                    duplicate_paths.append(path)
                    visited.add(idx)

                visited.add(best_idx)
                duplicates.append((original_path, duplicate_paths))
            elif len(group_indices) == 1:
                visited.add(group_indices[0])

    return duplicates


async def move_duplicate_files(
    duplicates: List[Tuple[str, List[str]]],
    duplicates_dir: str,
) -> DuplicateMoveResult:
    """
    Move duplicate files to the duplicates directory.

    Args:
        duplicates: List of (original_path, [duplicate_paths]) tuples.
        duplicates_dir: Directory to move duplicates to.

    Returns:
        DuplicateMoveResult with statistics.
    """
    # Create duplicates directory if it doesn't exist
    await async_makedirs(duplicates_dir, exist_ok=True)

    total_duplicates = sum(len(dups) for _, dups in duplicates)
    moved = 0
    failed = 0

    logger.info(f"Moving {total_duplicates} duplicate files to {duplicates_dir}...")

    for original, duplicate_paths in tqdm(duplicates, desc="Moving duplicates"):
        for dup_path in duplicate_paths:
            try:
                # Create subdirectories based on original path
                relative_path = Path(dup_path).relative_to(Path(original).parent.parent)
                dest_path = os.path.join(duplicates_dir, str(relative_path))

                # Ensure destination directory exists
                dest_dir = os.path.dirname(dest_path)
                await async_makedirs(dest_dir, exist_ok=True)

                # Move the file
                await async_move_file(dup_path, dest_path)
                moved += 1
            except Exception as e:
                logger.error(f"Error moving {dup_path}: {e}")
                failed += 1

    logger.info(f"Successfully moved {moved} duplicate files.")
    if failed > 0:
        logger.warning(f"Failed to move {failed} duplicate files.")

    return DuplicateMoveResult(moved=moved, failed=failed, total_duplicates=total_duplicates)


async def build_dedup_cache(
    source_dir: str,
    duplicates_dir: str,
    model_name: str = "openai/clip-vit-base-patch32",
    threshold: float = 0.98,
    batch_size: int = 32,
    cache_file: Optional[str] = None,
    force_recompute: bool = False,
) -> DeduplicationResult:
    """
    Build deduplication cache and move duplicate images.

    Args:
        source_dir: Source directory containing images.
        duplicates_dir: Directory to move duplicates to.
        model_name: CLIP model name.
        threshold: Similarity threshold for duplicates.
        batch_size: Batch size for processing.
        cache_file: Path to cache embeddings.
        force_recompute: Force recompute embeddings even if cache exists.

    Returns:
        DeduplicationResult with statistics.
    """
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"

    logger.info(f"Using device: {device}")
    logger.info(f"Finding images in {source_dir}...")

    image_files = await load_images(source_dir)
    total_images = len(image_files)
    logger.info(f"Found {total_images} images.")

    if total_images == 0:
        return DeduplicationResult(
            total_images=0,
            duplicate_groups=0,
            total_duplicates=0,
            moved_duplicates=0,
            duplicates_dir=duplicates_dir,
        )

    embeddings = None
    cache_loaded = False
    cache_saved = False

    # Check cache
    if cache_file and not force_recompute and await async_file_exists(cache_file):
        logger.info(f"Loading embeddings from cache: {cache_file}")
        try:
            cached_data = await asyncio.to_thread(np.load, cache_file, allow_pickle=True)
            if len(cached_data) == total_images:
                embeddings = cached_data
                cache_loaded = True
                logger.info("Cache loaded successfully.")
            else:
                logger.info(f"Cache size ({len(cached_data)}) mismatch with found images ({total_images}). Recomputing...")
        except Exception as e:
            logger.error(f"Could not load cache: {e}")
            embeddings = None

    if embeddings is None:
        embeddings = await compute_clip_embeddings(image_files, model_name, batch_size, device)

        if cache_file:
            logger.info(f"Saving embeddings to {cache_file}...")
            try:
                await asyncio.to_thread(np.save, cache_file, embeddings)
                cache_saved = True
            except Exception as e:
                logger.error(f"Could not save cache: {e}")

    # Find duplicates
    duplicates = find_duplicate_groups(image_files, embeddings, threshold)

    if duplicates:
        duplicate_groups = len(duplicates)
        total_duplicates = sum(len(dups) for _, dups in duplicates)

        logger.info(f"Found {duplicate_groups} duplicate groups with {total_duplicates} duplicate images.")

        # Move duplicates
        move_result = await move_duplicate_files(duplicates, duplicates_dir)
        moved_duplicates = move_result.moved

        return DeduplicationResult(
            total_images=total_images,
            duplicate_groups=duplicate_groups,
            total_duplicates=total_duplicates,
            moved_duplicates=moved_duplicates,
            cache_loaded=cache_loaded,
            cache_saved=cache_saved,
            duplicates_dir=duplicates_dir,
        )
    else:
        logger.info("No duplicates found.")
        return DeduplicationResult(
            total_images=total_images,
            duplicate_groups=0,
            total_duplicates=0,
            moved_duplicates=0,
            cache_loaded=cache_loaded,
            cache_saved=cache_saved,
            duplicates_dir=duplicates_dir,
        )


def deduplicate_sync(
    source_dir: Optional[str] = None,
    duplicates_dir: Optional[str] = None,
    threshold: Optional[float] = None,
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    cache_file: Optional[str] = None,
    force_recompute: bool = False,
) -> DeduplicationResult:
    """
    Synchronous wrapper for deduplication.

    Args:
        source_dir: Source directory containing images.
        duplicates_dir: Directory to move duplicates to.
        threshold: Similarity threshold for duplicates.
        model_name: CLIP model name.
        batch_size: Batch size for processing.
        cache_file: Path to cache embeddings.
        force_recompute: Force recompute embeddings even if cache exists.

    Returns:
        DeduplicationResult with statistics.
    """
    from face_sorter.config import get_settings

    settings = get_settings()

    # Use provided values or fall back to config
    source_dir = source_dir or settings.source_dir
    duplicates_dir = duplicates_dir or settings.duplicates_dir
    threshold = threshold if threshold is not None else settings.dedup_threshold
    model_name = model_name or settings.dedup_model_name
    batch_size = batch_size if batch_size is not None else settings.dedup_batch_size
    cache_file = cache_file or settings.dedup_cache_file

    return asyncio.run(
        build_dedup_cache(
            source_dir=source_dir,
            duplicates_dir=duplicates_dir,
            model_name=model_name,
            threshold=threshold,
            batch_size=batch_size,
            cache_file=cache_file,
            force_recompute=force_recompute,
        )
    )
