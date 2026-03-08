"""
Face clustering and sorting service.

This module handles sorting faces into classes and clustering unknown faces
using FAISS and HDBSCAN.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional, Callable

import faiss
import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA

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
    settings = get_settings()
    for index, i in enumerate(sorted_ids):
        class_name = sorted_class_names[index]
        path = unique_class_paths[class_name]
        img_path = os.path.join(path, imgname[i])
        
        # Use filename from imgcache[i] and join with settings.cache_dir
        source_cache_path = os.path.join(settings.cache_dir, os.path.basename(imgcache[i]))
        expanded_path = os.path.expanduser(source_cache_path)
        bbox = np.array(imgbbox[i]).astype(np.int32)

        tasks.append(process_image(img_path, expanded_path, bbox))

    # Wait for all tasks to complete
    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info(f"Sorted {len(sorted_ids)} faces into classes")


async def show_results(
    cache_dir: str,
    cluster_imgs: list[str],
    cluster_cache: list[str],
    cluster_bbox: list[list[int]],
    label: int,
) -> None:
    """
    Show cluster results by saving images with bounding boxes.

    Args:
        cache_dir: Cache directory.
        cluster_imgs: List of image names in this cluster.
        cluster_cache: List of cache paths in this cluster.
        cluster_bbox: List of bounding boxes in this cluster.
        label: Cluster label.
    """
    cache_path = os.path.expanduser(os.path.join(cache_dir, "clusters", str(label)))
    await async_makedirs(cache_path, exist_ok=True)

    # Process images in parallel using asyncio.gather
    tasks = []
    settings = get_settings()
    for i in range(len(cluster_imgs)):
        cache_url = os.path.join(cache_path, cluster_imgs[i])
        # Only use the filename from cluster_cache[i], ignoring any absolute paths
        source_cache_path = os.path.join(settings.cache_dir, os.path.basename(cluster_cache[i]))
        expanded_path = os.path.expanduser(source_cache_path)
        bbox = np.array(cluster_bbox[i]).astype(np.int32)

        tasks.append(process_image(cache_url, expanded_path, bbox))

    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info(f"Saved {len(cluster_imgs)} images for cluster {label}")


def match_faces_to_classes(
    imgembeddings: list[list[float]],
    classembeddings: list[np.ndarray],
    classname: list[str],
) -> tuple[list[int], list[str], list[list[float]], list[int]]:
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
            - unsorted_ids: Indices of unsorted images
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
        unsorted_ids,
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
    # Re-normalize just in case
    faiss.normalize_L2(face_embeddings)

    # Use PCA to reduce dimensionality for faster clustering
    # 64 dimensions is usually enough to maintain face cluster integrity
    n_samples = face_embeddings.shape[0]
    n_features = face_embeddings.shape[1]
    
    # PCA n_components must be <= min(n_samples, n_features)
    if n_samples > 100:
        n_components = min(64, n_samples, n_features)
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(face_embeddings)
    else:
        reduced_embeddings = face_embeddings

    # Run HDBSCAN clustering on reduced embeddings
    # Using 'euclidean' on normalized vectors is much faster than 'cosine'
    dbscan = HDBSCAN(
        metric="euclidean",
        min_samples=get_settings().cluster_min_samples,
        min_cluster_size=get_settings().cluster_min_size,
    )

    dbscan.fit(reduced_embeddings)
    cluster_labels = dbscan.labels_
    
    # Calculate centroids in ORIGINAL space
    unique_labels = np.unique(cluster_labels)
    cluster_centers_dict = {}
    
    for label in unique_labels:
        if label == -1:
            continue
        mask = (cluster_labels == label)
        centroid = face_embeddings[mask].mean(axis=0)
        # Re-normalize centroid
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        cluster_centers_dict[label] = centroid

    # Convert to expected array format (indices matching label values)
    if not cluster_centers_dict:
        return cluster_labels, np.array([])
        
    max_label = max(cluster_centers_dict.keys())
    centers_arr = np.zeros((max_label + 1, face_embeddings.shape[1]), dtype=np.float32)
    for label, center in cluster_centers_dict.items():
        centers_arr[label] = center
        
    return cluster_labels, centers_arr


async def sort(
    cache_dir: Optional[str] = None,
    source_dir: Optional[str] = None,
    max_results: int = 10,
    progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
    cancellation_event: Optional[asyncio.Event] = None,
) -> None:
    """
    Sort faces into classes and cluster unknown faces.

    Args:
        cache_dir: Cache directory.
        source_dir: Source directory to derive cache_dir from if missing.
        max_results: Maximum number of clusters to show.
        progress_callback: Optional callback for progress reporting.
        cancellation_event: Optional event to signal cancellation.
    """
    settings = get_settings()

    # Derive cache_dir if missing
    if not cache_dir:
        if source_dir:
            src_path = Path(source_dir).resolve()
            cache_dir = str(src_path.parent / ".cache")
        else:
            cache_dir = settings.cache_dir

    if not cache_dir:
        logger.error("No cache directory specified and source_dir not provided")
        return

    # Ensure cache directory exists
    await async_makedirs(os.path.expanduser(cache_dir), exist_ok=True)

    if progress_callback:
        progress_callback(0, 100, "Initializing", "Fetching data from database...")

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

    total_imgs = len(imgembeddings)
    if total_imgs == 0:
        logger.info("No images found to sort")
        if progress_callback:
            progress_callback(100, 100, "Complete", "No images found")
        return

    if cancellation_event and cancellation_event.is_set():
        return

    if progress_callback:
        progress_callback(10, 100, "Matching", "Matching faces to known classes...")

    # Match faces to classes in a separate thread
    logger.info("Matching faces to known classes...")
    sorted_ids, sorted_class_names, unsorted_embeddings, unsorted_ids = await asyncio.to_thread(
        match_faces_to_classes, imgembeddings, classembeddings, classname
    )

    logger.info(f"Sorted {len(sorted_ids)} faces into classes")
    logger.info(f"Found {len(unsorted_embeddings)} unsorted faces")

    if cancellation_event and cancellation_event.is_set():
        return

    # Sort faces by class
    if sorted_ids:
        if progress_callback:
            progress_callback(30, 100, "Sorting", f"Sorting {len(sorted_ids)} faces into class directories...")
        await sort_faces_by_class(
            cache_dir, imgname, imgcache, imgbbox, sorted_class_names, sorted_ids
        )

    if cancellation_event and cancellation_event.is_set():
        return

    # Cluster unknown faces
    if unsorted_embeddings:
        if progress_callback:
            progress_callback(50, 100, "Clustering", "Clustering unknown faces...")
        
        # Yield to event loop to ensure progress message is sent
        await asyncio.sleep(0.01)

        import time
        start_time = time.time()
        logger.info(f"Starting HDBSCAN clustering for {len(unsorted_embeddings)} faces...")
        
        cluster_labels, cluster_centers = await asyncio.to_thread(
            cluster_unknown_faces, unsorted_embeddings
        )
        
        end_time = time.time()
        logger.info(f"HDBSCAN clustering completed in {end_time - start_time:.2f} seconds")

        # Get unique labels and counts
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        sorted_indices = np.argsort(-counts)
        sorted_unique_labels = unique_labels[sorted_indices]

        logger.info(f"Found {len(sorted_unique_labels)} clusters")

        # Save clusters
        cluster_repo = ClusterRepository()
        face_repo = FaceRepository()
        await cluster_repo.clear_clusters()
        await face_repo.clear_all_clusters()

        results = 0
        
        for i, label in enumerate(sorted_unique_labels):
            if label != -1 and results < max_results:  # Skip noise points
                if cancellation_event and cancellation_event.is_set():
                    return

                logger.info(f"Processing cluster {results}")
                
                # indices are relative to unsorted_embeddings
                indices = np.where(cluster_labels == label)[0]
                centroid = cluster_centers[label].tolist()

                # Map back to real database indices (idx)
                real_indices = [unsorted_ids[idx] for idx in indices]

                await cluster_repo.insert_cluster(
                    cluster_name=int(label),
                    cluster_id=results,
                    indices=real_indices,
                    centroid=centroid,
                )

                # Update face documents with cluster ID
                await face_repo.update_faces_cluster(real_indices, results)

                # Get data for this cluster
                cluster_imgs = [imgname[idx] for idx in real_indices]
                cluster_cache = [imgcache[idx] for idx in real_indices]
                cluster_bbox = [imgbbox[idx] for idx in real_indices]

                if progress_callback:
                    progress_callback(
                        50 + int((results / max_results) * 40), 
                        100, 
                        "Saving Clusters", 
                        f"Saving cluster {results} ({len(indices)} faces)..."
                    )

                await show_results(
                    cache_dir, cluster_imgs, cluster_cache, cluster_bbox, results
                )
                results += 1

        logger.info(f"Saved {results} clusters")

    if progress_callback:
        progress_callback(100, 100, "Complete", f"Successfully sorted faces and found {results if unsorted_embeddings else 0} clusters.")


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
