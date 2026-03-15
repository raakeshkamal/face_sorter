"""
Script to backfill class assignments to face documents.

This script updates face documents with their assigned class name
based on cluster assignments.
"""

import asyncio
from face_sorter.database.connection import get_database
from face_sorter.database.repositories import ClusterRepository, FaceRepository


async def backfill_classes():
    """Backfill class assignments to all faces."""
    print("Starting class backfill...")

    cluster_repo = ClusterRepository()
    face_repo = FaceRepository()

    # Get all clusters with class assignments
    clusters = await cluster_repo.get_all_clusters()
    clusters_with_class = [c for c in clusters if c.get("class_name")]

    print(f"Found {len(clusters_with_class)} clusters with class assignments")

    updated_count = 0
    for cluster in clusters_with_class:
        cluster_id = cluster["cluster_id"]
        class_name = cluster["class_name"]
        indices = cluster.get("indices", [])

        if not indices:
            print(f"  Cluster {cluster_id} ({class_name}): No faces")
            continue

        # Update all faces in the cluster with the class name
        await face_repo.update_faces_class(indices, class_name)
        updated_count += len(indices)
        print(f"  Cluster {cluster_id} ({class_name}): Updated {len(indices)} faces")

    print(
        f"\nBackfill complete! Updated {updated_count} faces across {len(clusters_with_class)} classes."
    )


if __name__ == "__main__":
    asyncio.run(backfill_classes())
