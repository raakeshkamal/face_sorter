"""
Filesystem browse API endpoints.

This module provides endpoints for browsing the filesystem to select directories.
"""

import logging
import os
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from face_sorter.utils.file_async import async_list_directories

logger = logging.getLogger(__name__)

router = APIRouter()


class DirectoryInfo(BaseModel):
    """Information about a directory."""

    name: str
    path: str
    type: str  # "folder" or "parent"


class DirectoryListResponse(BaseModel):
    """Response containing directory listing."""

    current_path: str
    directories: List[DirectoryInfo]
    has_parent: bool


@router.get("/browse", response_model=DirectoryListResponse)
async def browse_directories(
    path: str = Query("", description="Directory path to browse (empty for user home)")
) -> DirectoryListResponse:
    """
    Browse directories in the filesystem.

    Args:
        path: Directory path to browse. Defaults to user home if empty.

    Returns:
        DirectoryListResponse containing current path and list of directories.

    Raises:
        HTTPException: If the path doesn't exist or is not accessible.
    """
    try:
        # Use user home directory if path is empty
        if not path or path.strip() == "":
            path = str(Path.home())

        # Resolve and validate path
        resolved_path = Path(path).resolve()

        # Security check: ensure path is absolute
        if not resolved_path.is_absolute():
            raise HTTPException(
                status_code=400,
                detail="Path must be absolute"
            )

        # Check if path exists and is a directory
        if not resolved_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Directory not found: {resolved_path}"
            )

        if not resolved_path.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"Not a directory: {resolved_path}"
            )

        # List directories
        directories = await async_list_directories(str(resolved_path))

        # Build directory info list
        dir_info_list = []
        for dir_path in directories:
            dir_path_obj = Path(dir_path)
            dir_info_list.append(
                DirectoryInfo(
                    name=dir_path_obj.name,
                    path=str(dir_path_obj),
                    type="folder"
                )
            )

        # Check if we have a parent directory (not at root)
        has_parent = str(resolved_path.parent) != str(resolved_path)

        response = DirectoryListResponse(
            current_path=str(resolved_path),
            directories=dir_info_list,
            has_parent=has_parent
        )

        logger.debug(f"Browsed directories in {resolved_path}: found {len(dir_info_list)} directories")

        return response

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except PermissionError as e:
        logger.error(f"Permission denied accessing {path}: {e}")
        raise HTTPException(
            status_code=403,
            detail=f"Permission denied: Unable to access directory"
        )
    except Exception as e:
        logger.error(f"Error browsing directories in {path}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to browse directory: {str(e)}"
        )
