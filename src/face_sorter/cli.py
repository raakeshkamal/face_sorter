"""
Command-line interface for Face Sorter.

This module provides a Click-based CLI for interacting with the face sorting system.
"""

import logging

import click

from face_sorter.cli_web import web
from face_sorter.config import get_settings
from face_sorter.services.caching import build_cache_sync
from face_sorter.services.clean import clean_dataset_sync
from face_sorter.services.deduplication import deduplicate_sync
from face_sorter.services.sorting import (
    add_new_class_sync,
    get_all_class_names_sync,
    remove_class_sync,
    sort_sync,
)
from face_sorter.services.training import train_sync
from face_sorter.utils.logging import setup_logging

# Set up logging
settings = get_settings()
setup_logging(level=settings.log_level)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Face Sorter - A face recognition and sorting application."""
    pass


@cli.command()
@click.option("--source-dir", type=click.Path(exists=True), help="Source directory containing images")
@click.option("--noface-dir", type=click.Path(), help="Directory for images without faces")
@click.option("--broken-dir", type=click.Path(), help="Directory for broken images")
@click.option("--cache-dir", type=click.Path(), help="Cache directory")
@click.option("--duplicates-dir", type=click.Path(), help="Directory for duplicate images (will be skipped)")
def train(source_dir, noface_dir, broken_dir, cache_dir, duplicates_dir):
    """
    Train the model by detecting faces and generating embeddings.

    This processes all images in the source directory, detects faces using InsightFace,
    and generates embeddings that are stored in MongoDB.
    """
    logger.info("Starting training...")
    progress = train_sync(source_dir, noface_dir, broken_dir, cache_dir, duplicates_dir)
    logger.info(f"Training complete. Processed {progress.processed} images")
    logger.info(f"  With faces: {progress.with_faces}")
    logger.info(f"  Without faces: {progress.without_faces}")


@cli.command()
@click.option("--source-dir", type=click.Path(exists=True), help="Source directory containing images")
@click.option("--output-dir", type=click.Path(), help="Output directory for cleaned images")
@click.option("--broken-dir", type=click.Path(), help="Directory for broken/invalid images")
@click.option("--batch-size", type=int, help="Batch size for processing")
@click.option("--img-prefix", type=str, help="Prefix for output filenames (e.g., IMG)")
@click.option("--quality", type=int, help="JPEG quality (1-100)")
@click.option("--recursive/--no-recursive", default=None, help="Scan recursively")
@click.option("--start-index", type=int, help="Starting index for sequential naming")
def clean(source_dir, output_dir, broken_dir, batch_size, img_prefix, quality, recursive, start_index):
    """
    Clean and standardize an image dataset.

    This command recursively scans source directory for images,
    validates their integrity, converts them to RGB JPEG format,
    and saves them to a flat output directory with sequential naming.

    Invalid or broken images are moved to a separate directory.
    Original files are not modified.

    Example:
        face-sorter clean --source-dir /path/to/raw --output-dir /path/to/cleaned
    """
    logger.info("Starting dataset cleaning...")
    result = clean_dataset_sync(
        source_dir=source_dir,
        output_dir=output_dir,
        broken_dir=broken_dir,
        batch_size=batch_size,
        img_prefix=img_prefix,
        quality=quality,
        recursive=recursive,
        start_index=start_index,
    )
    logger.info(f"Cleaning complete. Processed {result.processed}/{result.total} images")
    logger.info(f"  Successful: {result.successful}")
    logger.info(f"  Failed: {result.failed}")
    logger.info(f"  Moved to broken: {result.moved_to_broken}")
    logger.info(f"  Success rate: {result.success_rate():.1f}%")
    logger.info(f"  Output: {result.output_dir}")
    logger.info(f"  Broken: {result.broken_dir}")


@cli.command()
@click.option("--cache-dir", type=click.Path(), help="Cache directory")
@click.option("--quality", type=int, help="JPEG quality (1-100)")
def build_cache(cache_dir, quality):
    """
    Build the image cache.

    This compresses and caches images from the database for faster processing.
    """
    logger.info("Building cache...")
    result = build_cache_sync(cache_dir, quality)
    logger.info(f"Cache built successfully. Processed {result.processed} images")
    logger.info(f"  Success rate: {result.success_rate():.1f}%")


@cli.command()
@click.option("--source-dir", type=click.Path(exists=True), help="Source directory containing images")
@click.option("--duplicates-dir", type=click.Path(), help="Directory for duplicate images")
@click.option("--threshold", type=float, help="Similarity threshold (0-1)")
@click.option("--model-name", type=str, help="CLIP model name")
@click.option("--batch-size", type=int, help="Batch size for processing")
@click.option("--force-recompute", is_flag=True, help="Force recompute embeddings even if cache exists")
def dedup(source_dir, duplicates_dir, threshold, model_name, batch_size, force_recompute):
    """
    Find and move duplicate images.

    This command uses CLIP embeddings to find visually similar images and moves
    duplicates to the duplicates directory, keeping only the highest quality version.
    """
    logger.info("Starting deduplication...")
    result = deduplicate_sync(
        source_dir=source_dir,
        duplicates_dir=duplicates_dir,
        threshold=threshold,
        model_name=model_name,
        batch_size=batch_size,
        force_recompute=force_recompute,
    )
    logger.info(f"Deduplication complete. Processed {result.total_images} images")
    logger.info(f"  Found {result.duplicate_groups} duplicate groups")
    logger.info(f"  Moved {result.moved_duplicates} duplicate images")
    if result.cache_loaded:
        logger.info("  Loaded embeddings from cache")
    if result.cache_saved:
        logger.info("  Saved embeddings to cache")
    logger.info(f"  Duplicates moved to: {result.duplicates_dir}")


@cli.command()
@click.option("--cache-dir", type=click.Path(), help="Cache directory")
@click.option("--max-results", type=int, default=10, help="Maximum number of clusters to show")
def sort_faces(cache_dir, max_results):
    """
    Sort faces into classes and cluster unknown faces.

    This matches faces to known classes using FAISS and clusters unknown faces
    using HDBSCAN.
    """
    logger.info("Sorting faces...")
    sort_sync(cache_dir, max_results)
    logger.info("Sorting complete")


@cli.command()
@click.argument("class_name", type=str)
@click.argument("cluster_id", type=int)
def add_class(class_name, cluster_id):
    """
    Add a new class from a cluster.

    Example: face-sorter add-class "John Doe" 42
    """
    logger.info(f"Adding class '{class_name}' from cluster {cluster_id}...")
    add_new_class_sync(class_name, cluster_id)
    logger.info(f"Class '{class_name}' added successfully")


@cli.command()
@click.argument("class_name", type=str)
def remove(class_name):
    """
    Remove a class.

    Example: face-sorter remove "John Doe"
    """
    logger.info(f"Removing class '{class_name}'...")
    remove_class_sync(class_name)
    logger.info(f"Class '{class_name}' removed successfully")


@cli.command("list-classes")
def list_classes():
    """List all face classes."""
    logger.info("Fetching all classes...")
    classes = get_all_class_names_sync()
    if classes:
        click.echo("Classes:")
        for class_name in classes:
            click.echo(f"  - {class_name}")
    else:
        click.echo("No classes found")


# Add aliases for backward compatibility
cli.add_command(add_class, name="add-class")
cli.add_command(remove, name="remove-class")
cli.add_command(sort_faces, name="sort")
cli.add_command(web)


if __name__ == "__main__":
    cli()
