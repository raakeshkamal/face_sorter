"""
Data access layer for MongoDB collections.

This module provides repository classes for managing face embeddings, classes, and clusters.
"""

import logging
from typing import Any, Optional

import numpy as np
from pymongo import ASCENDING
from pymongo.collection import Collection
from pymongo.database import Database

from face_sorter.config import get_settings
from face_sorter.database.connection import get_database

logger = logging.getLogger(__name__)


class FaceRepository:
    """Repository for managing face embeddings."""

    def __init__(self, db: Optional[Database] = None) -> None:
        """
        Initialize the repository.

        Args:
            db: Database instance. If None, uses global connection.
        """
        if db is None:
            db = get_database()
        settings = get_settings()
        self.collection: Collection = db[settings.collection_backup]

    def insert_face(self, face_data: dict[str, Any]) -> None:
        """
        Insert a face embedding into the database.

        Args:
            face_data: Dictionary containing face information.
        """
        self.collection.insert_one(face_data)

    def get_all_faces(
        self,
        projection: Optional[dict[str, Any]] = None,
        sort: Optional[list[tuple[str, int]]] = None,
    ) -> list[dict[str, Any]]:
        """
        Get all faces from the database.

        Args:
            projection: Fields to include/exclude.
            sort: Sort specification.

        Returns:
            List of face documents.
        """
        if projection is None:
            projection = {}
        if sort is None:
            sort = [("idx", ASCENDING)]

        return list(self.collection.find(projection=projection).sort(sort))

    def count_faces(self) -> int:
        """
        Count the number of faces in the collection.

        Returns:
            Number of faces.
        """
        return self.collection.count_documents({})


class ClassRepository:
    """Repository for managing face classes."""

    def __init__(self, db: Optional[Database] = None) -> None:
        """
        Initialize the repository.

        Args:
            db: Database instance. If None, uses global connection.
        """
        if db is None:
            db = get_database()
        settings = get_settings()
        self.collection: Collection = db[settings.collection_classes]

    def insert_class(self, class_name: str, embedding: list[float]) -> None:
        """
        Insert a face class into the database.

        Args:
            class_name: Name of the class.
            embedding: Face embedding vector.
        """
        class_entry = {"class": class_name, "embed": embedding}
        self.collection.insert_one(class_entry)

    def get_all_classes(
        self, projection: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """
        Get all classes from the database.

        Args:
            projection: Fields to include/exclude.

        Returns:
            List of class documents.
        """
        if projection is None:
            projection = {}
        return list(self.collection.find(projection=projection))

    def get_class_embeddings(
        self,
    ) -> tuple[list[str], list[list[float]], list[str], list[np.ndarray]]:
        """
        Get all class embeddings with optimized data structure.

        Returns:
            Tuple of (class_names, embeddings, distinct_classes, mean_embeddings).
        """
        class_docs = self.get_all_classes(projection={"class": 1, "embed": 1, "_id": 0})

        refname = [doc["class"] for doc in class_docs]
        refembeddings = [doc["embed"] for doc in class_docs]

        # Get distinct classes
        distinct_classes = list(set(refname))

        # Group documents by class
        class_data: dict[str, dict[str, list]] = {}
        for doc in class_docs:
            cls = doc["class"]
            if cls not in class_data:
                class_data[cls] = {"embeds": []}
            class_data[cls]["embeds"].append(doc["embed"])

        # Compute mean embeddings
        classembeddings = [None] * len(distinct_classes)
        for i, cls in enumerate(distinct_classes):
            data = class_data[cls]
            embeds = np.array(data["embeds"])
            # Normalize embeddings
            norms = np.linalg.norm(embeds, axis=1, keepdims=True)
            normalized_embeds = embeds / norms
            # Compute mean and normalize
            mean_embed = np.mean(normalized_embeds, axis=0)
            classembeddings[i] = mean_embed / np.linalg.norm(mean_embed)

        return refname, refembeddings, distinct_classes, classembeddings

    def delete_class(self, class_name: str) -> None:
        """
        Delete a class from the database.

        Args:
            class_name: Name of the class to delete.
        """
        self.collection.delete_one({"class": class_name})

    def get_all_class_names(self) -> list[str]:
        """
        Get all class names.

        Returns:
            List of class names.
        """
        return [doc["class"] for doc in self.get_all_classes(projection={"class": 1, "_id": 0})]


class ClusterRepository:
    """Repository for managing face clusters."""

    def __init__(self, db: Optional[Database] = None) -> None:
        """
        Initialize the repository.

        Args:
            db: Database instance. If None, uses global connection.
        """
        if db is None:
            db = get_database()
        settings = get_settings()
        self.collection: Collection = db[settings.collection_clusters]

    def insert_cluster(
        self,
        cluster_name: int,
        cluster_id: int,
        indices: list[int],
        centroid: list[float],
    ) -> None:
        """
        Insert a cluster into the database.

        Args:
            cluster_name: Original cluster label.
            cluster_id: Sequential cluster ID.
            indices: List of image indices in the cluster.
            centroid: Cluster centroid embedding.
        """
        cluster_info = {
            "cluster_name": cluster_name,
            "cluster_id": cluster_id,
            "indices": indices,
            "centroid": centroid,
        }
        self.collection.insert_one(cluster_info)

    def get_cluster(self, cluster_id: int) -> Optional[dict[str, Any]]:
        """
        Get a cluster by ID.

        Args:
            cluster_id: Cluster ID.

        Returns:
            Cluster document or None if not found.
        """
        return self.collection.find_one({"cluster_id": cluster_id})

    def clear_clusters(self) -> None:
        """Remove all clusters from the collection."""
        self.collection.drop()

    def get_all_clusters(
        self, projection: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """
        Get all clusters from the database.

        Args:
            projection: Fields to include/exclude.

        Returns:
            List of cluster documents.
        """
        if projection is None:
            projection = {}
        return list(self.collection.find(projection=projection))


def fetch_data_optimized(
    db: Optional[Database] = None,
) -> tuple[
    list[str],
    list[list[float]],
    list[str],
    list[np.ndarray],
    list[str],
    list[str],
    list[list[int]],
    list[str],
    list[list[float]],
]:
    """
    Fetch optimized data from all collections for sorting.

    Args:
        db: Database instance. If None, uses global connection.

    Returns:
        Tuple containing:
            - refname: List of reference class names
            - refembeddings: List of reference embeddings
            - classname: List of distinct class names
            - classembeddings: List of mean class embeddings
            - imgname: List of image names
            - imgpath: List of image paths
            - imgbbox: List of bounding boxes
            - imgcache: List of cache URLs
            - imgembeddings: List of image embeddings
    """
    face_repo = FaceRepository(db)
    class_repo = ClassRepository(db)

    # Fetch class data
    (
        refname,
        refembeddings,
        classname,
        classembeddings,
    ) = class_repo.get_class_embeddings()

    # Fetch face data
    face_docs = face_repo.get_all_faces(
        projection={
            "item": 1,
            "path": 1,
            "cache_url": 1,
            "bbox": 1,
            "embedding": 1,
            "_id": 0,
        },
        sort=[("idx", ASCENDING)],
    )

    count = face_repo.count_faces()
    imgname = [None] * count
    imgpath = [None] * count
    imgcache = [None] * count
    imgbbox = [None] * count
    imgembeddings = [None] * count

    for i, doc in enumerate(face_docs):
        imgname[i] = doc["item"]
        imgpath[i] = doc["path"]
        imgcache[i] = doc["cache_url"]
        imgbbox[i] = doc["bbox"]
        imgembeddings[i] = doc["embedding"]

    return (
        refname,
        refembeddings,
        classname,
        classembeddings,
        imgname,
        imgpath,
        imgbbox,
        imgcache,
        imgembeddings,
    )
