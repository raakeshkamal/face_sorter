#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["Pillow"]
# ///

import line_profiler
import multiprocessing
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from mongo_db import *

@line_profiler.profile
def compress_image(bkp):
    """Process a single image."""
    try:
        img = Image.open(bkp["path"])
        img = img.resize(img.size, Image.Resampling.LANCZOS)
        file_path = os.path.expanduser(bkp["cache_url"])
        os.makedirs(
            os.path.dirname(file_path), exist_ok=True
        )  # Ensure directory exists
        img.save(file_path, quality=50, optimize=True)
    except Exception as e:
        print(f"Error processing {bkp['path']}: {e}")

@line_profiler.profile
def clear_and_recreate_cache():
    """Clear and recreate the cache directory safely"""
    cache_dir = os.path.expanduser("~/Documents/image_dedup/.cache")
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)

@line_profiler.profile
def build_cache_func():
    """Function to build the cache."""
    db = get_mongo_db()
    bkpcollection = db["bkpcollection"]

    # Clear cache directory
    clear_and_recreate_cache()

    # Use a set for faster lookups - O(1) vs O(n) for list
    unique_path = set()

    # Collect unique records first (avoids repeatedly checking the growing unique_path list)
    unique_records = []
    for bkp in bkpcollection.find().sort("idx", 1):
        if bkp["path"] not in unique_path:
            unique_path.add(bkp["path"])
            unique_records.append(bkp)

    # Process images in parallel
    num_workers = min(
        multiprocessing.cpu_count() * 2, 16
    )  # 2x CPU cores is good for I/O bound tasks
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        # Submit all tasks
        for bkp in unique_records:
            futures.append(executor.submit(compress_image, bkp))

        # Wait for all tasks to complete with progress tracking
        total = len(futures)
        for i, future in enumerate(futures, 1):
            future.result()  # Get result to ensure exceptions are raised
            if i % 100 == 0 or i == total:
                print(f"Processed {i}/{total} images")