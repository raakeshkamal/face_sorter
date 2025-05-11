#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["line-profiler", "click", "Pillow", "pymongo", "numpy", "psutil", "onnxruntime", "onnxruntime-silicon", "faiss-cpu", "scikit-learn", "opencv-python", "insightface"]
# ///

# TODO: implement line_profiler
import click
import line_profiler

from run_train import *
from build_cache import *
from run_sort import *
from mongo_db import *

@click.command()
@click.option("--build-cache", is_flag=True, help="Build the cache.")
@click.option("--run", nargs=1, type=int, help="Run the application. Example: --run 10")
@click.option("--train", is_flag=True, help="Train the application.")
@click.option(
    "--add-class",
    nargs=2,
    type=(str, int),
    help="Add a class with the given class_name and fileIndex. Example: --add-class MyClass 42",
)
@click.option(
    "--rm-class",
    nargs=1,
    type=str,
    help="Remove a class with the given class_name. Example: --rm-class MyClass",
)
@click.option("--get-all-class", is_flag=True, help="Get all class names.")

@line_profiler.profile
def cli(build_cache, run, train, add_class, rm_class, get_all_class):

    SRC_DIR = "/Volumes/data_sets/new_backup/images"
    NOFACE_DIR = "/Volumes/data_sets/new_backup/noface"
    BROKEN_DIR = "/Volumes/data_sets/new_backup/broken"
    CACHE_DIR = "/Volumes/data_sets/new_backup/.cache"

    """Command-line interface for the application."""
    # Check if any valid options were provided
    if not (build_cache or run or train or add_class or rm_class or get_all_class):
        click.echo(
            "No valid command specified. Use --help for usage information."
        )
        return

    if train:
        """ Training application takes 10mins for 10000 images """
        click.echo("Training application...")
        train_func(SRC_DIR, NOFACE_DIR, BROKEN_DIR, CACHE_DIR)
        click.echo("Application executed successfully.")

    if build_cache:
        """Build the cache. takes 2mins for 10000 images"""
        click.echo("Building cache...")
        build_cache_func()
        click.echo("Cache built successfully.")

    if add_class:
        class_name, file_index = add_class
        add_class_func(class_name, file_index)
    
    if rm_class:
        class_name = rm_class
        remove_class(class_name)
    
    if get_all_class:
        get_all_class_names()

    if run:
        max_results = run
        click.echo("Running application...")
        run_func(CACHE_DIR, max_results)
        click.echo("Application executed successfully.")

def add_class_func(class_name, class_num):
    """Function to add a class."""
    click.echo(f"Adding class '{class_name}' with file index '{class_num}'...")
    add_new_class(class_name, class_num)
    click.echo(f"Class '{class_name}' added successfully.")


if __name__ == "__main__":
    cli()
