"""
Dataset cleaning service for Face Sorter.

This module handles cleaning, validating, and standardizing image datasets
by converting all images to RGB JPEG format with sequential naming.
"""

import asyncio
import logging
from pathlib import Path
from typing import Callable, Optional

from PIL import ImageFile

from face_sorter.config import get_settings
from face_sorter.models.face import CleanResult
from face_sorter.utils.file_async import (
    async_list_files,
    async_makedirs,
    async_read_image,
)
from face_sorter.utils.image import is_valid_image

logger = logging.getLogger(__name__)

# Configure PIL to handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Register HEIF/HEIC support if available
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    logger.info("HEIF/HEIC support enabled via pillow-heif")
except ImportError:
    logger.warning("pillow-heif not available, HEIC/HEIF files will be skipped")


async def scan_images(
    source_dir: str,
    extensions: list[str],
    recursive: bool = True,
) -> list[str]:
    """
    Scan directory for image files with given extensions.

    Args:
        source_dir: Directory to scan.
        extensions: List of file extensions to include (with leading dot).
        recursive: Whether to scan recursively.

    Returns:
        List of image file paths.
    """
    image_files = []

    # Use async_list_files for each extension pattern
    patterns = []
    for ext in extensions:
        # Add both lowercase and uppercase patterns
        patterns.append(f"*{ext.lower()}")
        patterns.append(f"*{ext.upper()}")

    for pattern in patterns:
        if recursive:
            files = await async_list_files(source_dir, f"**/{pattern}")
        else:
            files = await async_list_files(source_dir, pattern)
        image_files.extend(files)

    # Remove duplicates and convert to absolute paths
    unique_files = list(set(str(Path(f).resolve()) for f in image_files))

    logger.info(f"Found {len(unique_files)} images in {source_dir}")
    return sorted(unique_files)


async def process_single_image(
    input_path: str,
    output_path: str,
    quality: int = 95,
) -> bool:
    """
    Process a single image: validate and convert to RGB JPEG.

    Args:
        input_path: Path to input image.
        output_path: Path to save processed image.
        quality: JPEG quality (1-100).

    Returns:
        True if successful, False otherwise.
    """
    try:
        # Validate image
        if not await is_valid_image(input_path):
            logger.warning(f"Invalid image: {input_path}")
            return False

        # Read image
        img = await async_read_image(input_path)

        # Ensure output directory exists
        output_path_abs = str(Path(output_path).resolve())
        await async_makedirs(str(Path(output_path_abs).parent), exist_ok=True)

        # Convert to RGB if needed (PIL is blocking)
        async def _convert():
            if img.mode != "RGB":
                rgb_img = img.convert("RGB")
            else:
                rgb_img = img
            rgb_img.save(output_path_abs, "JPEG", quality=quality, optimize=True)

        await asyncio.to_thread(_convert)
        return True

    except Exception as e:
        logger.error(f"Error processing {input_path}: {e}")
        return False


async def process_batch(
    batch: list[str],
    output_dir: str,
    prefix: str,
    start_index: int,
    quality: int = 95,
) -> tuple[int, int, int]:
    """
    Process a batch of images in parallel.

    Args:
        batch: List of input image paths.
        output_dir: Output directory for processed images.
        prefix: Filename prefix (e.g., "IMG").
        start_index: Starting index for naming.
        quality: JPEG quality (1-100).

    Returns:
        Tuple of (successful_count, failed_count, current_index).
    """
    successful = 0
    failed = 0
    current_index = start_index

    # Create tasks for batch processing
    tasks = []
    for input_path in batch:
        output_filename = f"{prefix}_{current_index:04d}.jpg"
        output_path = str(Path(output_dir) / output_filename)
        tasks.append((input_path, output_path))
        current_index += 1

    # Process batch in parallel
    results = await asyncio.gather(
        *[
            process_single_image(input_path, output_path, quality)
            for input_path, output_path in tasks
        ],
        return_exceptions=True,
    )

    # Count results
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Exception in batch processing: {result}")
            failed += 1
        elif result is True:
            successful += 1
        else:
            failed += 1

    return successful, failed, current_index


async def find_next_available_index(
    output_dir: str,
    prefix: str,
) -> int:
    """
    Find next available index for sequential naming.

    Args:
        output_dir: Output directory to check.
        prefix: Filename prefix to search for.

    Returns:
        Next available index (starting from 1 if no files found).
    """
    try:
        # List existing files matching pattern
        pattern = f"{prefix}_*.jpg"
        existing_files = await async_list_files(output_dir, pattern)

        if not existing_files:
            return 1

        # Extract indices from filenames
        max_index = 0
        for file_path in existing_files:
            filename = Path(file_path).name
            # Extract index from "IMG_XXXX.jpg"
            try:
                # Remove prefix and extension
                idx_str = filename.replace(f"{prefix}_", "").replace(".jpg", "")
                idx_str = idx_str.split(".")[0]  # Handle case sensitivity
                idx = int(idx_str)
                if idx > max_index:
                    max_index = idx
            except (ValueError, IndexError):
                continue

        return max_index + 1
    except Exception as e:
        logger.error(f"Error finding next index: {e}")
        return 1


async def clean_dataset(
    source_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    broken_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
    img_prefix: Optional[str] = None,
    quality: Optional[int] = None,
    recursive: Optional[bool] = None,
    start_index: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
) -> CleanResult:
    """
    Clean and standardize an image dataset.

    This function recursively scans source directory for images,
    validates their integrity, converts them to RGB JPEG format,
    and saves them to a flat output directory with sequential naming.

    Invalid or broken images are moved to a separate broken directory.
    Original files are not modified.

    Args:
        source_dir: Source directory containing images to clean.
        output_dir: Output directory for cleaned images.
        broken_dir: Directory for broken/invalid images.
        batch_size: Batch size for parallel processing.
        img_prefix: Prefix for output filenames.
        quality: JPEG quality (1-100).
        recursive: Whether to scan recursively.
        start_index: Starting index for sequential naming.
        progress_callback: Optional callback function(current, total, status, current_item)
                        for reporting progress during cleaning.

    Returns:
        CleanResult with statistics about cleaning operation.
    """
    settings = get_settings()

    # Use provided directories or defaults from settings
    if source_dir is None:
        source_dir = settings.source_dir
    if output_dir is None:
        output_dir = settings.clean_output_dir
    if broken_dir is None:
        broken_dir = settings.clean_broken_dir
    if batch_size is None:
        batch_size = settings.clean_batch_size
    if img_prefix is None:
        img_prefix = settings.clean_img_prefix
    if quality is None:
        quality = settings.clean_quality
    if recursive is None:
        recursive = settings.clean_recursive
    if start_index is None:
        start_index = settings.clean_start_index

    logger.info(f"Starting dataset cleaning from {source_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Broken directory: {broken_dir}")

    # Ensure output directories exist
    await async_makedirs(output_dir, exist_ok=True)
    await async_makedirs(broken_dir, exist_ok=True)

    # Find next available index
    if start_index == 1:
        # Auto-detect next index if not specified
        start_index = await find_next_available_index(output_dir, img_prefix)
        logger.info(f"Starting from index: {start_index}")

    # Scan for images
    logger.info("Scanning for images...")
    image_files = await scan_images(source_dir, settings.clean_extensions, recursive)
    total = len(image_files)

    if total == 0:
        logger.info("No images found to process")
        return CleanResult(
            processed=0,
            total=0,
            successful=0,
            failed=0,
            moved_to_broken=0,
            output_dir=output_dir,
            broken_dir=broken_dir,
            start_index=start_index,
            end_index=start_index,
        )

    logger.info(f"Found {total} images to process")

    # Process images in batches
    successful = 0
    failed = 0
    current_index = start_index
    processed = 0

    for i in range(0, total, batch_size):
        batch = image_files[i : i + batch_size]
        batch_start_index = current_index

        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        logger.info(
            f"Processing batch {batch_num}/{total_batches} ({len(batch)} images)"
        )

        batch_successful, batch_failed, next_index = await process_batch(
            batch,
            output_dir,
            img_prefix,
            batch_start_index,
            quality,
        )

        successful += batch_successful
        failed += batch_failed
        current_index = next_index
        processed += len(batch)

        # Log progress
        if processed % 100 == 0 or processed == total:
            logger.info(
                f"Progress: {processed}/{total} images processed "
                f"({successful} successful, {failed} failed)"
            )
            # Report progress via callback
            if progress_callback:
                status = "Processing" if processed < total else "Complete"
                progress_callback(processed, total, status, f"batch_{batch_num}")

    end_index = current_index - 1

    logger.info("Dataset cleaning complete")
    logger.info(f"  Processed: {processed}/{total}")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    if processed > 0:
        logger.info(f"  Success rate: {successful / processed * 100:.1f}%")
    logger.info(
        f"  Output range: {img_prefix}_{start_index:04d}.jpg "
        f"to {img_prefix}_{end_index:04d}.jpg"
    )

    # Report final progress
    if progress_callback:
        progress_callback(processed, total, "Complete", "")

    return CleanResult(
        processed=processed,
        total=total,
        successful=successful,
        failed=failed,
        moved_to_broken=0,  # Track if we implement moving broken files
        output_dir=output_dir,
        broken_dir=broken_dir,
        start_index=start_index,
        end_index=end_index,
    )


def clean_dataset_sync(
    source_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    broken_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
    img_prefix: Optional[str] = None,
    quality: Optional[int] = None,
    recursive: Optional[bool] = None,
    start_index: Optional[int] = None,
) -> CleanResult:
    """
    Synchronous wrapper for clean_dataset.

    Args:
        source_dir: Source directory containing images to clean.
        output_dir: Output directory for cleaned images.
        broken_dir: Directory for broken/invalid images.
        batch_size: Batch size for parallel processing.
        img_prefix: Prefix for output filenames.
        quality: JPEG quality (1-100).
        recursive: Whether to scan recursively.
        start_index: Starting index for sequential naming.

    Returns:
        CleanResult with statistics about cleaning operation.
    """
    return asyncio.run(
        clean_dataset(
            source_dir=source_dir,
            output_dir=output_dir,
            broken_dir=broken_dir,
            batch_size=batch_size,
            img_prefix=img_prefix,
            quality=quality,
            recursive=recursive,
            start_index=start_index,
        )
    )
