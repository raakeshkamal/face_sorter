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


class DuplicateMoveResult:
    """
    Represents the result of moving duplicate files.

    Attributes:
        moved: Number of duplicate files successfully moved.
        failed: Number of files that failed to move.
        total_duplicates: Total number of duplicates found.
    """

    def __init__(self, moved: int, failed: int, total_duplicates: int) -> None:
        self.moved = moved
        self.failed = failed
        self.total_duplicates = total_duplicates

    def success_rate(self) -> float:
        """Return success rate of move operation."""
        if self.total_duplicates == 0:
            return 0.0
        return (self.moved / self.total_duplicates) * 100


class DeduplicationResult:
    """
    Represents the result of a deduplication operation.

    Attributes:
        total_images: Total number of images processed.
        duplicate_groups: Number of duplicate groups found.
        total_duplicates: Total number of duplicate images found.
        moved_duplicates: Number of duplicate files moved to duplicates_dir.
        cache_loaded: Whether embeddings were loaded from cache.
        cache_saved: Whether embeddings were saved to cache.
        duplicates_dir: Directory where duplicates were moved.
    """

    def __init__(
        self,
        total_images: int,
        duplicate_groups: int,
        total_duplicates: int,
        moved_duplicates: int,
        cache_loaded: bool = False,
        cache_saved: bool = False,
        duplicates_dir: Optional[str] = None,
    ) -> None:
        self.total_images = total_images
        self.duplicate_groups = duplicate_groups
        self.total_duplicates = total_duplicates
        self.moved_duplicates = moved_duplicates
        self.cache_loaded = cache_loaded
        self.cache_saved = cache_saved
        self.duplicates_dir = duplicates_dir

    def deduplication_rate(self) -> float:
        """Return percentage of images that are duplicates."""
        if self.total_images == 0:
            return 0.0
        return (self.total_duplicates / self.total_images) * 100


class CleanResult:
    """
    Represents the result of a dataset cleaning operation.

    Attributes:
        processed: Number of images processed.
        total: Total number of images found.
        successful: Number of successfully cleaned images.
        failed: Number of images that failed to clean.
        moved_to_broken: Number of files moved to broken directory.
        output_dir: Directory where cleaned images were saved.
        broken_dir: Directory where broken images were moved.
        start_index: Starting index used for naming.
        end_index: Ending index used for naming.
    """

    def __init__(
        self,
        processed: int,
        total: int,
        successful: int,
        failed: int,
        moved_to_broken: int,
        output_dir: str,
        broken_dir: str,
        start_index: int,
        end_index: int,
    ) -> None:
        self.processed = processed
        self.total = total
        self.successful = successful
        self.failed = failed
        self.moved_to_broken = moved_to_broken
        self.output_dir = output_dir
        self.broken_dir = broken_dir
        self.start_index = start_index
        self.end_index = end_index

    def success_rate(self) -> float:
        """Return success rate of cleaning operation."""
        if self.processed == 0:
            return 0.0
        return (self.successful / self.processed) * 100

    def broken_rate(self) -> float:
        """Return percentage of images that were broken."""
        if self.processed == 0:
            return 0.0
        return (self.moved_to_broken / self.processed) * 100
