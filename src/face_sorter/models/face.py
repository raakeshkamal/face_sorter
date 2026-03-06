"""
Data models for Face Sorter.

This module contains Pydantic models for type-safe data handling.
"""

from typing import Optional


class FaceEmbedding:
    """
    Represents a face embedding with associated metadata.

    Attributes:
        idx: Index of the face in the collection.
        item: Image filename.
        path: Full path to the image.
        age: Estimated age of the face.
        gender: Estimated gender (0 for male, 1 for female).
        bbox: Bounding box coordinates [x1, y1, x2, y2].
        kps: Keypoints for the face.
        det_score: Detection confidence score.
        landmark_3d_68: 68 3D facial landmarks.
        pose: Pose estimation data.
        landmark_2d_106: 106 2D facial landmarks.
        embedding: Face embedding vector (512-dimensional).
        cache_url: Path to cached image.
    """

    def __init__(
        self,
        idx: int,
        item: str,
        path: str,
        age: int,
        gender: int,
        bbox: list[int],
        kps: list[list[int]],
        det_score: float,
        landmark_3d_68: list[float],
        pose: list[float],
        landmark_2d_106: list[float],
        embedding: list[float],
        cache_url: str,
    ) -> None:
        self.idx = idx
        self.item = item
        self.path = path
        self.age = age
        self.gender = gender
        self.bbox = bbox
        self.kps = kps
        self.det_score = det_score
        self.landmark_3d_68 = landmark_3d_68
        self.pose = pose
        self.landmark_2d_106 = landmark_2d_106
        self.embedding = embedding
        self.cache_url = cache_url

    def to_dict(self) -> dict:
        """Convert to dictionary for MongoDB storage."""
        return {
            "idx": self.idx,
            "item": self.item,
            "path": self.path,
            "age": self.age,
            "gender": self.gender,
            "bbox": self.bbox,
            "kps": self.kps,
            "det_score": self.det_score,
            "landmark_3d_68": self.landmark_3d_68,
            "pose": self.pose,
            "landmark_2d_106": self.landmark_2d_106,
            "embedding": self.embedding,
            "cache_url": self.cache_url,
        }


class FaceClass:
    """
    Represents a known face class.

    Attributes:
        class_name: Name of the person.
        embedding: Mean face embedding for this class.
    """

    def __init__(self, class_name: str, embedding: list[float]) -> None:
        self.class_name = class_name
        self.embedding = embedding

    def to_dict(self) -> dict:
        """Convert to dictionary for MongoDB storage."""
        return {"class": self.class_name, "embed": self.embedding}


class FaceCluster:
    """
    Represents a cluster of similar faces.

    Attributes:
        cluster_name: Original cluster label from HDBSCAN.
        cluster_id: Sequential cluster ID.
        indices: List of image indices in the cluster.
        centroid: Cluster centroid embedding.
    """

    def __init__(
        self,
        cluster_name: int,
        cluster_id: int,
        indices: list[int],
        centroid: list[float],
    ) -> None:
        self.cluster_name = cluster_name
        self.cluster_id = cluster_id
        self.indices = indices
        self.centroid = centroid

    def to_dict(self) -> dict:
        """Convert to dictionary for MongoDB storage."""
        return {
            "cluster_name": self.cluster_name,
            "cluster_id": self.cluster_id,
            "indices": self.indices,
            "centroid": self.centroid,
        }


class ProcessedImage:
    """
    Represents an image that has been processed.

    Attributes:
        filename: Image filename.
        path: Full path to the image.
        faces: List of face embeddings found in the image.
    """

    def __init__(self, filename: str, path: str, faces: list[FaceEmbedding]) -> None:
        self.filename = filename
        self.path = path
        self.faces = faces


class SortResult:
    """
    Represents the result of a sorting operation.

    Attributes:
        sorted_ids: Indices of sorted images.
        sorted_class_names: Class names for sorted images.
        unsorted_indices: Indices of unsorted images.
        cluster_count: Number of clusters found.
    """

    def __init__(
        self,
        sorted_ids: list[int],
        sorted_class_names: list[str],
        unsorted_indices: list[int],
        cluster_count: int,
    ) -> None:
        self.sorted_ids = sorted_ids
        self.sorted_class_names = sorted_class_names
        self.unsorted_indices = unsorted_indices
        self.cluster_count = cluster_count


class TrainingProgress:
    """
    Represents progress during training.

    Attributes:
        processed: Number of images processed.
        total: Total number of images to process.
        with_faces: Number of images with faces.
        without_faces: Number of images without faces.
    """

    def __init__(self, processed: int, total: int, with_faces: int, without_faces: int) -> None:
        self.processed = processed
        self.total = total
        self.with_faces = with_faces
        self.without_faces = without_faces

    def percentage(self) -> float:
        """Return percentage of images processed."""
        if self.total == 0:
            return 0.0
        return (self.processed / self.total) * 100


class CacheResult:
    """
    Represents the result of a cache operation.

    Attributes:
        processed: Number of images processed.
        total: Total number of images to process.
        failed: Number of images that failed to process.
    """

    def __init__(self, processed: int, total: int, failed: int = 0) -> None:
        self.processed = processed
        self.total = total
        self.failed = failed

    def percentage(self) -> float:
        """Return percentage of images processed."""
        if self.total == 0:
            return 0.0
        return (self.processed / self.total) * 100

    def success_rate(self) -> float:
        """Return success rate of cache operation."""
        if self.processed == 0:
            return 0.0
        return ((self.processed - self.failed) / self.processed) * 100
