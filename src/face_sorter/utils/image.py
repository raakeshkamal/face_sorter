"""
Image processing utilities.

This module provides reusable image processing functions.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


def compress_image(
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
        optimize: Whether to optimize the image.

    Returns:
        True if successful, False otherwise.
    """
    try:
        with Image.open(input_path) as img:
            # Resize with high quality resampling
            img = img.resize(img.size, Image.Resampling.LANCZOS)
            img.save(output_path, "JPEG", quality=quality, optimize=optimize)
        return True
    except Exception as e:
        logger.error(f"Error compressing image {input_path}: {e}")
        return False


def draw_bounding_box(
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
        with Image.open(image_path) as img:
            draw = ImageDraw.Draw(img)
            draw.rectangle(
                [(bbox[0], bbox[1]), (bbox[2], bbox[3])],
                outline=color,
                width=width,
            )
            img.save(output_path, "JPEG", quality=75, optimize=True)
        return True
    except Exception as e:
        logger.error(f"Error drawing bounding box on {image_path}: {e}")
        return False


def crop_face(
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
        padding: Padding in pixels around the bounding box.

    Returns:
        True if successful, False otherwise.
    """
    try:
        with Image.open(image_path) as img:
            # Add padding and ensure bounds
            width, height = img.size
            x1 = max(0, bbox[0] - padding)
            y1 = max(0, bbox[1] - padding)
            x2 = min(width, bbox[2] + padding)
            y2 = min(height, bbox[3] + padding)

            # Crop and save
            cropped = img.crop((x1, y1, x2, y2))
            cropped.save(output_path, "JPEG", quality=95)
        return True
    except Exception as e:
        logger.error(f"Error cropping face from {image_path}: {e}")
        return False


def get_image_size(image_path: str) -> Optional[Tuple[int, int]]:
    """
    Get the size of an image.

    Args:
        image_path: Path to image.

    Returns:
        Tuple of (width, height) or None if error.
    """
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        logger.error(f"Error getting size of {image_path}: {e}")
        return None


def is_valid_image(image_path: str) -> bool:
    """
    Check if a file is a valid image.

    Args:
        image_path: Path to file.

    Returns:
        True if valid image, False otherwise.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
            img.load()
        return True
    except Exception as e:
        logger.debug(f"Invalid image {image_path}: {e}")
        return False


def convert_to_rgb(image_path: str, output_path: str) -> bool:
    """
    Convert an image to RGB format.

    Args:
        image_path: Path to input image.
        output_path: Path to save RGB image.

    Returns:
        True if successful, False otherwise.
    """
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(output_path, "JPEG", quality=95)
        return True
    except Exception as e:
        logger.error(f"Error converting {image_path} to RGB: {e}")
        return False
