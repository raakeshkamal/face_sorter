"""
Image processing utilities.

This module provides reusable image processing functions.
"""

import asyncio
import logging
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


async def compress_image(
    input_path: str,
    output_path: str,
    quality: int = 50,
    optimize: bool = True,
) -> bool:
    """
    Compress and save an image.

    Args:
        input_path: Path to input image.
        output_path: Path to save compressed image.
        quality: JPEG quality (1-100).
        optimize: Whether to optimize image.

    Returns:
        True if successful, False otherwise.
    """
    try:
        # Read image asynchronously
        from .file_async import async_read_image, async_makedirs

        img = await async_read_image(input_path)

        # Ensure output directory exists
        await async_makedirs(str(output_path).rsplit("/", 1)[0], exist_ok=True)

        # Resize with high quality resampling (PIL is blocking)
        img = await asyncio.to_thread(img.resize, img.size, Image.Resampling.LANCZOS)

        # Save image (PIL is blocking)
        await asyncio.to_thread(
            img.save, output_path, "JPEG", quality=quality, optimize=optimize
        )
        return True
    except Exception as e:
        logger.error(f"Error compressing image {input_path}: {e}")
        return False


async def draw_bounding_box(
    image_path: str,
    bbox: Tuple[int, int, int, int],
    output_path: str,
    color: Tuple[int, int, int] = (255, 0, 0),
    width: int = 5,
) -> bool:
    """
    Draw a bounding box on an image.

    Args:
        image_path: Path to input image.
        bbox: Bounding box coordinates (x1, y1, x2, y2).
        output_path: Path to save image with bounding box.
        color: RGB color tuple.
        width: Line width in pixels.

    Returns:
        True if successful, False otherwise.
    """
    try:
        from .file_async import async_makedirs, async_write_image

        img = await async_read_image(image_path)

        # Ensure output directory exists
        await async_makedirs(str(output_path).rsplit("/", 1)[0], exist_ok=True)

        # Draw bounding box (PIL is blocking)
        async def _draw():
            draw = ImageDraw.Draw(img)
            draw.rectangle(
                [(bbox[0], bbox[1]), (bbox[2], bbox[3])],
                outline=color,
                width=width,
            )
            img.save(output_path, "JPEG", quality=75, optimize=True)

        await asyncio.to_thread(_draw)
        return True
    except Exception as e:
        logger.error(f"Error drawing bounding box on {image_path}: {e}")
        return False


async def crop_face(
    image_path: str,
    bbox: Tuple[int, int, int, int],
    output_path: str,
    padding: int = 20,
) -> bool:
    """
    Crop a face from an image.

    Args:
        image_path: Path to input image.
        bbox: Bounding box coordinates (x1, y1, x2, y2).
        output_path: Path to save cropped face.
        padding: Padding in pixels around bounding box.

    Returns:
        True if successful, False otherwise.
    """
    try:
        from .file_async import async_makedirs

        img = await async_read_image(image_path)

        # Ensure output directory exists
        await async_makedirs(str(output_path).rsplit("/", 1)[0], exist_ok=True)

        # Crop and save (PIL is blocking)
        async def _crop():
            width, height = img.size
            x1 = max(0, bbox[0] - padding)
            y1 = max(0, bbox[1] - padding)
            x2 = min(width, bbox[2] + padding)
            y2 = min(height, bbox[3] + padding)

            cropped = img.crop((x1, y1, x2, y2))
            cropped.save(output_path, "JPEG", quality=95)

        await asyncio.to_thread(_crop)
        return True
    except Exception as e:
        logger.error(f"Error cropping face from {image_path}: {e}")
        return False


async def get_image_size(image_path: str) -> Optional[Tuple[int, int]]:
    """
    Get size of an image.

    Args:
        image_path: Path to image.

    Returns:
        Tuple of (width, height) or None if error.
    """
    try:
        img = await async_read_image(image_path)
        return img.size
    except Exception as e:
        logger.error(f"Error getting size of {image_path}: {e}")
        return None


async def is_valid_image(image_path: str) -> bool:
    """
    Check if a file is a valid image.

    Args:
        image_path: Path to file.

    Returns:
        True if valid image, False otherwise.
    """
    try:
        img = await async_read_image(image_path)

        # Verify and load (PIL is blocking)
        async def _verify():
            img.verify()
            # Need to reload after verify
            return Image.open(image_path)

        await asyncio.to_thread(_verify)
        return True
    except Exception as e:
        logger.debug(f"Invalid image {image_path}: {e}")
        return False


async def convert_to_rgb(image_path: str, output_path: str) -> bool:
    """
    Convert an image to RGB format.

    Args:
        image_path: Path to input image.
        output_path: Path to save RGB image.

    Returns:
        True if successful, False otherwise.
    """
    try:
        from .file_async import async_makedirs

        img = await async_read_image(image_path)

        # Ensure output directory exists
        await async_makedirs(str(output_path).rsplit("/", 1)[0], exist_ok=True)

        # Convert and save (PIL is blocking)
        async def _convert():
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(output_path, "JPEG", quality=95)

        await asyncio.to_thread(_convert)
        return True
    except Exception as e:
        logger.error(f"Error converting {image_path} to RGB: {e}")
        return False


def compress_image_sync(
    input_path: str,
    output_path: str,
    quality: int = 50,
    optimize: bool = True,
) -> bool:
    """
    Synchronous wrapper for compress_image for backward compatibility.

    Args:
        input_path: Path to input image.
        output_path: Path to save compressed image.
        quality: JPEG quality (1-100).
        optimize: Whether to optimize image.

    Returns:
        True if successful, False otherwise.
    """
    return asyncio.run(compress_image(input_path, output_path, quality, optimize))


def draw_bounding_box_sync(
    image_path: str,
    bbox: Tuple[int, int, int, int],
    output_path: str,
    color: Tuple[int, int, int] = (255, 0, 0),
    width: int = 5,
) -> bool:
    """
    Synchronous wrapper for draw_bounding_box for backward compatibility.

    Args:
        image_path: Path to input image.
        bbox: Bounding box coordinates (x1, y1, x2, y2).
        output_path: Path to save image with bounding box.
        color: RGB color tuple.
        width: Line width in pixels.

    Returns:
        True if successful, False otherwise.
    """
    return asyncio.run(draw_bounding_box(image_path, bbox, output_path, color, width))


def crop_face_sync(
    image_path: str,
    bbox: Tuple[int, int, int, int],
    output_path: str,
    padding: int = 20,
) -> bool:
    """
    Synchronous wrapper for crop_face for backward compatibility.

    Args:
        image_path: Path to input image.
        bbox: Bounding box coordinates (x1, y1, x2, y2).
        output_path: Path to save cropped face.
        padding: Padding in pixels around bounding box.

    Returns:
        True if successful, False otherwise.
    """
    return asyncio.run(crop_face(image_path, bbox, output_path, padding))


def get_image_size_sync(image_path: str) -> Optional[Tuple[int, int]]:
    """
    Synchronous wrapper for get_image_size for backward compatibility.

    Args:
        image_path: Path to image.

    Returns:
        Tuple of (width, height) or None if error.
    """
    return asyncio.run(get_image_size(image_path))


def is_valid_image_sync(image_path: str) -> bool:
    """
    Synchronous wrapper for is_valid_image for backward compatibility.

    Args:
        image_path: Path to file.

    Returns:
        True if valid image, False otherwise.
    """
    return asyncio.run(is_valid_image(image_path))


def convert_to_rgb_sync(image_path: str, output_path: str) -> bool:
    """
    Synchronous wrapper for convert_to_rgb for backward compatibility.

    Args:
        image_path: Path to input image.
        output_path: Path to save RGB image.

    Returns:
        True if successful, False otherwise.
    """
    return asyncio.run(convert_to_rgb(image_path, output_path))
