"""
Face clustering and sorting service.

This module handles sorting faces into classes and clustering unknown faces
using FAISS and HDBSCAN.
"""

import asyncio
import logging
import os
from typing import Any, Optional

import faiss
import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import HDBSCAN

from face_sorter.config import get_settings
from face_sorter.database.repositories import (
    ClassRepository,
    ClusterRepository,
    FaceRepository,
    fetch_data_optimized,
)
from face_sorter.models.face import FaceClass
from face_sorter.utils.file_async import async_makedirs

logger = logging.getLogger(__name__)


async def add_new_class(class_name: str, cluster_id: int) -> None:
    """
    Add a new face class from a cluster.

    Args:
        class_name: Name of the new class.
        cluster_id: Cluster ID to create the class from.
    """
    cluster_repo = ClusterRepository()
    cluster = await cluster_repo.get_cluster(cluster_id)

    if cluster is None:
        logger.error(f"Cluster {cluster_id} not found")
        return

    class_repo = ClassRepository()
    await class_repo.insert_class(class_name, cluster["centroid"])
    logger.info(f"Added class '{class_name}' from cluster {cluster_id}")


async def remove_class(class_name: str) -> None:
    """
    Remove a face class.

    Args:
        class_name: Name of the class to remove.
    """
    class_repo = ClassRepository()
    await class_repo.delete_class(class_name)
    logger.info(f"Removed class '{class_name}'")


async def get_all_class_names() -> list[str]:
    """
    Get all class names.

    Returns:
        List of class names.
    """
    class_repo = ClassRepository()
    class_names = await class_repo.get_all_class_names()
    logger.info(f"Found {len(class_names)} classes")
    return class_names


async def process_image(
    img_url: str, expanded_path: str, bbox: np.ndarray
) -> None:
    """
    Process an image by drawing a bounding box and saving.

    Args:
        img_url: Output path for the processed image.
        expanded_path: Input path for the image.
        bbox: Bounding box coordinates.
    """
    try:
        # Use async file operations for reading
        from face_sorter.utils.file_async import async_read_image

        img = await async_read_image(expanded_path)

        # Ensure output directory exists
        await async_makedirs(os.path.dirname(os.path.expanduser(img_url)), exist_ok=True)

        # Draw bounding box and save (PIL is blocking)
        async def _draw():
            draw = ImageDraw.Draw(img)
            draw.rectangle(
                [(bbox[0], bbox[1]), (bbox[2], bbox[3])],
                outline=(255, 0, 0),
                width=5,
            )
            img.save(img_url, "JPEG", quality=75, optimize=True)

        await asyncio.to_thread(_draw)
    except Exception as e:
        logger.error(f"Error processing image {expanded_path}: {e}")


async def sort_faces_by_class(
    cache_dir: str,
    imgname: list[str],
    imgcache: list[str],
    imgbbox: list[list[int]],
    sorted_class_names: list[str],
    sorted_ids: list[int],
) -> None:
    """
    Sort faces into class directories.

    Args:
        cache_dir: Cache directory.
        imgname: List of image names.
        imgcache: List of cache paths.
        imgbbox: List of bounding boxes.
        sorted_class_names: Class names for sorted images.
        sorted_ids: Indices of sorted images.
    """
    # Create class directories
    unique_class_paths: dict[str, str] = {}
    for index, class_name in enumerate(sorted_class_names):
        path = os.path.expanduser(os.path.join(cache_dir, "faces", class_name))
        unique_class_paths[class_name] = path
        await async_makedirs(path, exist_ok=True)

    # Process files in parallel using asyncio.gather
    tasks = []
    for index, i in enumerate(sorted_ids):
        class_name = sorted_class_names[index]
        path = unique_class_paths[class_name]
        img_path = os.path.join(path, imgname[i])
        expanded_path = os.path.expanduser(imgcache[i])
        bbox = np.array(imgbbox[i]).astype(np.int32)

        tasks.append(process_image(img_path, expanded_path, bbox))

    # Wait for all tasks to complete
    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info(f"Sorted {len(sorted_ids)} faces into classes")


async def show_results(
    cache_dir: str,
    unsorted_imgs: list[str],
    unsorted_cache: list[str],
    unsorted_bbox: list[list[int]],
    label: int,
    indices: np.ndarray,
) -> None:
    """
    Show cluster results by saving images with bounding boxes.

    Args:
        cache_dir: Cache directory.
        unsorted_imgs: List of unsorted image names.
        unsorted_cache: List of cache paths.
        unsorted_bbox: List of bounding boxes.
        label: Cluster label.
        indices: Image indices in the cluster.
    """
    cache_path = os.path.expanduser(os.path.join(cache_dir, "clusters", str(label)))
    await async_makedirs(cache_path, exist_ok=True)

    # Process images in parallel using asyncio.gather
    tasks = []
    for i in indices:
        cache_url = os.path.join(cache_path, unsorted_imgs[i])
        expanded_path = os.path.expanduser(unsorted_cache[i])
        bbox = np.array(unsorted_bbox[i]).astype(np.int32)

        tasks.append(process_image(cache_url, expanded_path, bbox))

    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info(f"Saved {len(indices)} images for cluster {label}")


def match_faces_to_classes(
    imgembeddings: list[list[float]],
    classembeddings: list[np.ndarray],
    classname: list[str],
) -> tuple[list[int], list[str], list[list[float]]]:
    """
    Match faces to known classes using FAISS.

    Args:
        imgembeddings: List of image embeddings.
        classembeddings: List of class embeddings.
        classname: List of class names.

    Returns:
        Tuple containing:
            - sorted_ids: Indices of sorted images
            - sorted_class_names: Class names for sorted images
            - unsorted_embeddings: Embeddings of unsorted images
    """
    # Convert to numpy arrays
    imgembeddings_arr = np.asarray(imgembeddings, dtype=np.float32)
    classembeddings_arr = np.asarray(classembeddings, dtype=np.float32)

    # Create FAISS index for images
    index = faiss.IndexFlatIP(imgembeddings_arr.shape[1])
    faiss.normalize_L2(imgembeddings_arr)
    index.add(imgembeddings_arr)

    # Match faces to classes
    sorted_ids = set()
    sorted_class_mapping: dict[int, str] = {}

    for id, img in enumerate(classembeddings_arr):
        query = img.reshape(1, -1)
        Lims, Dist, Idx = index.range_search(query, get_settings().similarity_threshold)

        for i in Idx:
            if i not in sorted_ids:
                sorted_ids.add(i)
                sorted_class_mapping[i] = classname[id]

    # Get unsorted embeddings
    unsorted_ids = [i for i in range(len(imgembeddings_arr)) if i not in sorted_ids]
    unsorted_embeddings = [imgembeddings_arr[i].tolist() for i in unsorted_ids]

    return (
        list(sorted_ids),
        [sorted_class_mapping[i] for i in sorted_ids],
        unsorted_embeddings,
    )


def cluster_unknown_faces(
    unsorted_embeddings: list[list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster unknown faces using HDBSCAN.

    Args:
        unsorted_embeddings: List of unsorted face embeddings.

    Returns:
        Tuple containing cluster labels and centroids.
    """
    if not unsorted_embeddings:
        return np.array([]), np.array([])

    face_embeddings = np.array(unsorted_embeddings, dtype=np.float32)

    # Run HDBSCAN clustering
    dbscan = HDBSCAN(
        metric="cosine",
        min_samples=get_settings().cluster_min_samples,
        store_centers="centroid",
    )

    dbscan.fit(face_embeddings)
    cluster_labels = dbscan.labels_
    cluster_centers = dbscan.centroids_

    return cluster_labels, cluster_centers


async def sort(
    cache_dir: Optional[str] = None,
    max_results: int = 10,
) -> None:
    """
    Sort faces into classes and cluster unknown faces.

    Args:
        cache_dir: Cache directory.
        max_results: Maximum number of clusters to show.
    """
    if cache_dir is None:
        cache_dir = get_settings().cache_dir

    # Fetch data from database
    logger.info("Fetching data from database...")
    (
        refname,
        refembeddings,
        classname,
        classembeddings,
        imgname,
        imgpath,
        imgbbox,
        imgcache,
        imgembeddings,
    ) = await fetch_data_optimized()

    # Match faces to classes
    logger.info("Matching faces to known classes...")
    sorted_ids, sorted_class_names, unsorted_embeddings = match_faces_to_classes(
        imgembeddings, classembeddings, classname
    )

    logger.info(f"Sorted {len(sorted_ids)} faces into classes")
    logger.info(f"Found {len(unsorted_embeddings)} unsorted faces")

    # Sort faces by class
    if sorted_ids:
        await sort_faces_by_class(
            cache_dir, imgname, imgcache, imgbbox, sorted_class_names, sorted_ids
        )

    # Cluster unknown faces
    if unsorted_embeddings:
        logger.info("Clustering unknown faces...")
        cluster_labels, cluster_centers = cluster_unknown_faces(unsorted_embeddings)

        # Get unique labels and counts
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        sorted_indices = np.argsort(-counts)
        sorted_unique_labels = unique_labels[sorted_indices]

        logger.info(f"Found {len(sorted_unique_labels)} clusters")

        # Save clusters
        cluster_repo = ClusterRepository()
        await cluster_repo.clear_clusters()

        results = 0
        for i, label in enumerate(sorted_unique_labels):
            if label != -1 and results < max_results:  # Skip noise points
                logger.info(f"Processing cluster {i}")
                results += 1
                indices = np.where(cluster_labels == label)[0]
                centroid = cluster_centers[label].tolist()

                await cluster_repo.insert_cluster(
                    cluster_name=int(label),
                    cluster_id=i,
                    indices=indices.tolist(),
                    centroid=centroid,
                )

                # Get data for this cluster
                unsorted_imgs = [imgname[idx] for idx in indices]
                unsorted_cache = [imgcache[idx] for idx in indices]
                unsorted_bbox = [imgbbox[idx] for idx in indices]

                await show_results(
                    cache_dir, unsorted_imgs, unsorted_cache, unsorted_bbox, i, indices
                )

        logger.info(f"Saved {results} clusters")


# Synchronous wrappers for backward compatibility
def add_new_class_sync(class_name: str, cluster_id: int) -> None:
    """
    Synchronous wrapper for add_new_class.

    Args:
        class_name: Name of the new class.
        cluster_id: Cluster ID to create the class from.
    """
    return asyncio.run(add_new_class(class_name, cluster_id))


def remove_class_sync(class_name: str) -> None:
    """
    Synchronous wrapper for remove_class.

    Args:
        class_name: Name of the class to remove.
    """
    return asyncio.run(remove_class(class_name))


def get_all_class_names_sync() -> list[str]:
    """
    Synchronous wrapper for get_all_class_names.

    Returns:
        List of class names.
    """
    return asyncio.run(get_all_class_names())


def sort_sync(cache_dir: Optional[str] = None, max_results: int = 10) -> None:
    """
    Synchronous wrapper for sort.

    Args:
        cache_dir: Cache directory.
        max_results: Maximum number of clusters to show.
    """
    return asyncio.run(sort(cache_dir, max_results))
