"""
Async file utilities for I/O operations.

This module provides async wrappers for file system operations.
"""

import asyncio
import logging
import os
import shutil
from io import BytesIO
from pathlib import Path
from typing import Optional

import aiofiles
from PIL import Image

logger = logging.getLogger(__name__)


async def async_read_image(path: str) -> Image.Image:
    """
    Asynchronously read an image file.

    Args:
        path: Path to the image file.

    Returns:
        PIL Image object.

    Raises:
        Exception: If image reading fails.
    """
    try:
        async with aiofiles.open(path, "rb") as f:
            img_data = await f.read()
        # PIL doesn't have async support, run in thread pool
        return await asyncio.to_thread(Image.open, BytesIO(img_data))
    except Exception as e:
        logger.error(f"Error reading image {path}: {e}")
        raise


async def async_write_image(
    img: Image.Image, path: str, format: str = "JPEG", quality: int = 75
) -> bool:
    """
    Asynchronously write an image file.

    Args:
        img: PIL Image object.
        path: Path to save the image.
        format: Image format (JPEG, PNG, etc.).
        quality: JPEG quality (1-100).

    Returns:
        True if successful, False otherwise.
    """
    try:
        # Ensure directory exists
        dir_path = str(Path(path).parent)
        await async_makedirs(dir_path, exist_ok=True)

        # PIL doesn't have async support, run in thread pool
        async def _write():
            return await asyncio.to_thread(img.save, path, format, quality=quality)

        await _write()
        return True
    except Exception as e:
        logger.error(f"Error writing image to {path}: {e}")
        return False


async def async_move_file(src: str, dst: str) -> None:
    """
    Asynchronously move a file.

    Note: shutil doesn't have async support, so we use asyncio.to_thread.

    Args:
        src: Source file path.
        dst: Destination file path.
    """
    try:
        # Ensure destination directory exists
        dir_path = str(Path(dst).parent)
        await async_makedirs(dir_path, exist_ok=True)

        await asyncio.to_thread(shutil.move, src, dst)
    except Exception as e:
        logger.error(f"Error moving file from {src} to {dst}: {e}")
        raise


async def async_makedirs(path: str, mode: int = 0o777, exist_ok: bool = False) -> None:
    """
    Asynchronously create directories.

    Args:
        path: Directory path to create.
        mode: Permission mode.
        exist_ok: Don't raise error if directory exists.
    """
    try:
        # os.makedirs is mostly CPU-bound, use thread pool
        await asyncio.to_thread(os.makedirs, path, mode, exist_ok)
    except Exception as e:
        if not (exist_ok and os.path.exists(path)):
            logger.error(f"Error creating directory {path}: {e}")
            raise


async def async_list_files(
    path: str, pattern: Optional[str] = None
) -> list[str]:
    """
    Asynchronously list files in a directory.

    Args:
        path: Directory path.
        pattern: Optional glob pattern to filter files.

    Returns:
        List of file paths.
    """
    try:
        # pathlib is CPU-bound for listing, use thread pool
        if pattern:
            files = await asyncio.to_thread(list, Path(path).glob(pattern))
        else:
            files = await asyncio.to_thread(list, Path(path).iterdir())
        return [str(f) for f in files if f.is_file()]
    except Exception as e:
        logger.error(f"Error listing files in {path}: {e}")
        return []


async def async_list_directories(path: str) -> list[str]:
    """
    Asynchronously list directories in a directory.

    Args:
        path: Directory path.

    Returns:
        List of directory paths.
    """
    try:
        dir_path = Path(path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        # List directories only
        items = await asyncio.to_thread(list, dir_path.iterdir())
        directories = [str(d) for d in items if d.is_dir()]
        return sorted(directories)
    except Exception as e:
        logger.error(f"Error listing directories in {path}: {e}")
        raise


async def async_file_exists(path: str) -> bool:
    """
    Asynchronously check if a file exists.

    Args:
        path: File path to check.

    Returns:
        True if file exists, False otherwise.
    """
    try:
        # os.path.exists is CPU-bound, use thread pool
        return await asyncio.to_thread(os.path.exists, path)
    except Exception as e:
        logger.error(f"Error checking if {path} exists: {e}")
        return False


async def async_read_file(path: str, mode: str = "r") -> str:
    """
    Asynchronously read a text file.

    Args:
        path: File path to read.
        mode: File mode (r, rb, etc.).

    Returns:
        File content as string.
    """
    async with aiofiles.open(path, mode) as f:
        return await f.read()


async def async_write_file(path: str, content: str, mode: str = "w") -> None:
    """
    Asynchronously write a text file.

    Args:
        path: File path to write.
        content: Content to write.
        mode: File mode (w, wb, etc.).
    """
    async with aiofiles.open(path, mode) as f:
        await f.write(content)


async def async_delete_file(path: str) -> None:
    """
    Asynchronously delete a file.

    Args:
        path: File path to delete.
    """
    try:
        # os.unlink is mostly I/O-bound but doesn't have async support
        await asyncio.to_thread(os.unlink, path)
    except Exception as e:
        logger.error(f"Error deleting file {path}: {e}")
        raise


async def async_delete_directory(path: str, ignore_errors: bool = False) -> None:
    """
    Asynchronously delete a directory and all its contents.

    Args:
        path: Directory path to delete.
        ignore_errors: If True, ignore errors during deletion.
    """
    try:
        await asyncio.to_thread(shutil.rmtree, path, ignore_errors=ignore_errors)
    except Exception as e:
        if not ignore_errors:
            logger.error(f"Error deleting directory {path}: {e}")
            raise


async def async_get_file_size(path: str) -> int:
    """
    Asynchronously get file size.

    Args:
        path: File path.

    Returns:
        File size in bytes.
    """
    try:
        return await asyncio.to_thread(os.path.getsize, path)
    except Exception as e:
        logger.error(f"Error getting size of {path}: {e}")
        raise


async def async_is_file(path: str) -> bool:
    """
    Asynchronously check if path is a file.

    Args:
        path: Path to check.

    Returns:
        True if path is a file, False otherwise.
    """
    try:
        return await asyncio.to_thread(os.path.isfile, path)
    except Exception as e:
        logger.error(f"Error checking if {path} is a file: {e}")
        return False


async def async_is_dir(path: str) -> bool:
    """
    Asynchronously check if path is a directory.

    Args:
        path: Path to check.

    Returns:
        True if path is a directory, False otherwise.
    """
    try:
        return await asyncio.to_thread(os.path.isdir, path)
    except Exception as e:
        logger.error(f"Error checking if {path} is a directory: {e}")
        return False
