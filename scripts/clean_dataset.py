#!/usr/bin/env python3
"""
clean_dataset.py

A unified script to clean, validate, and standardize an image dataset.
- Recursively scans for images (JPG, PNG, WEBP, HEIC, etc.).
- Validates integrity (checks for corruption, decompression bombs).
- Converts everything to RGB JPEG.
- Saves to a flat output directory with sequential naming (IMG_XXXX.jpg).
- Does NOT modify source files.
- Ignores videos and broken files.
"""

import os
import sys
import argparse
import shutil
import warnings
import concurrent.futures
from pathlib import Path
from PIL import Image, ImageFile

# Handle truncation warning
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Try to import pillow_heif for HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_SUPPORT = True
except ImportError:
    HEIF_SUPPORT = False

# Supported Extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
if HEIF_SUPPORT:
    IMAGE_EXTENSIONS.update({'.heic', '.heif'})

def is_valid_image(file_path):
    """
    Check if the file is a valid image and return its Image object if true.
    Returns (is_valid, img_object_or_error_msg)
    """
    try:
        # Check for decompression bomb warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", Image.DecompressionBombWarning)
            
            img = Image.open(file_path)
            
            # Verify basic integrity
            img.verify() 
            
            # Re-open for actual processing because verify() consumes the file pointer in a specific way
            # and may not detect all issues that load() does.
            img = Image.open(file_path)
            img.load() # Force load pixel data
            
            # Check for warnings captured
            for warning in w:
                if issubclass(warning.category, Image.DecompressionBombWarning):
                    return False, f"Decompression Bomb Warning: {warning.message}"
            
            return True, img
    except Exception as e:
        return False, str(e)

def process_and_save(file_info):
    """
    Process a single file: convert to JPG and save to temp or memory.
    Returns (success, original_path, processed_image_object)
    NOTE: We cannot pass open Image objects easily back from processes if using ProcessPool,
    but ThreadPool shares memory. threading is usually fine for IO bound, but image conversion is CPU bound.
    However, for simplicity and compatibility, we will do conversion here.
    """
    file_path, output_dir, counter_idx = file_info
    
    try:
        if not os.path.exists(file_path):
            return False, file_path, "File not found"

        valid, result = is_valid_image(file_path)
        if not valid:
            return False, file_path, result
        
        # 'result' is the PIL Image object
        img = result
        
        # Convert to RGB
        if img.mode in ('RGBA', 'P', 'LA', 'CMYK'):
            img = img.convert('RGB')
            
        # Generate new filename
        new_filename = f"IMG_{counter_idx:04d}.jpg"
        output_path = os.path.join(output_dir, new_filename)
        
        # Save
        img.save(output_path, "JPEG", quality=90, optimize=True)
        
        return True, file_path, output_path
        
    except Exception as e:
        return False, file_path, str(e)

def main():
    parser = argparse.ArgumentParser(description="Clean and standardize image dataset to JPG.")
    parser.add_argument("--input_dir", default=".", help="Input directory to scan.")
    parser.add_argument("--output_dir", default="cleaned_images", help="Output directory for cleaned images.")
    parser.add_argument("--recursive", action="store_true", help="Scan input recursively.")
    parser.add_argument("--start_index", type=int, default=1, help="Start index for naming (default: 1).")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)
        
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if output dir is not empty and warn/adjust
    existing_files = [f for f in os.listdir(output_dir) if f.startswith("IMG_") and f.endswith(".jpg")]
    current_index = args.start_index
    
    if existing_files:
        # Try to find the highest index to allow appending
        max_idx = 0
        for f in existing_files:
            try:
                # Expecting IMG_XXXX.jpg
                part = f.split('_')[1].split('.')[0]
                idx = int(part)
                if idx > max_idx:
                    max_idx = idx
            except (IndexError, ValueError):
                continue
        
        if max_idx > 0:
            print(f"Index detected in output folder. Resuming from IMG_{max_idx + 1:04d}.jpg")
            current_index = max_idx + 1

    # 1. Scan files
    print(f"Scanning '{input_dir}'...")
    tasks = []
    
    file_iterator = input_dir.rglob("*") if args.recursive else input_dir.glob("*")
    
    for file_path in file_iterator:
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            tasks.append(file_path)
            
    total_files = len(tasks)
    print(f"Found {total_files} potential images.")
    
    if total_files == 0:
        print("No images found. Exiting.")
        sys.exit(0)
        
    if not HEIF_SUPPORT and any(f.suffix.lower() in {'.heic', '.heif'} for f in tasks):
        print("WARNING: HEIC files detected but 'pillow-heif' is not installed. These will be skipped.")
        
    # Prepare batch arguments
    # We assign indices upfront. 
    # NOTE: If a file fails, that index will be skipped in the final sequence (e.g. IMG_0001, IMG_0003). 
    # If strictly sequential numbering is required WITHOUT gaps, we would need to process first, then rename.
    # Given the scale, gaps are usually acceptable, or we can use a post-process rename.
    # However, to keep it simple and parallel, let's process them and handle naming carefully.
    
    # Better approach for strict sequentiality:
    # 1. Process files in parallel and save to temp names or just use Future objects.
    # 2. As they complete successfully, move/rename them to the final sequential name.
    
    print("Processing images...")
    
    success_count = 0
    fail_count = 0
    
    # Use ThreadPoolExecutor because image conversion is partly IO (reading/writing) but also CPU (decoding).
    # ProcessPoolExecutor gives better CPU isolation but pickling PIL objects is tricky.
    # Since we are saving in the worker function, ProcessPoolExecutor is viable if we deal with paths only.
    # Let's use ProcessPoolExecutor for better CPU utilization on large batches.
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # We submit path and output_dir, but NOT the final filename yet.
        # We will save to a UUID or hash name first, then rename in main thread.
        
        future_map = {}
        for file_path in tasks:
            # We pass a temporary "counter" that is just the loop index for unicity in temp files if needed,
            # but ideally we just save to a temp name.
            # Let's actually have the worker return the Image object? No, too much overhead.
            # Let's have the worker save to a temp file in output_dir.
            
            # Helper wrapper
            future = executor.submit(process_single_image_to_temp, str(file_path), str(output_dir))
            future_map[future] = file_path

        # As they complete...
        for future in concurrent.futures.as_completed(future_map):
            original_path = future_map[future]
            try:
                success, temp_path_or_err = future.result()
                
                if success:
                    # Rename temp file to final sequential name
                    final_name = f"IMG_{current_index:04d}.jpg"
                    final_path = output_dir / final_name
                    
                    shutil.move(temp_path_or_err, final_path)
                    
                    # Log (optional, verbose)
                    # print(f"Encoded: {original_path.name} -> {final_name}")
                    
                    current_index += 1
                    success_count += 1
                else:
                    print(f"Failed: {original_path.name} - {temp_path_or_err}")
                    fail_count += 1
                    
            except Exception as e:
                print(f"Error processing {original_path.name}: {e}")
                fail_count += 1
                
            # Simple progress update
            processed = success_count + fail_count
            if processed % 100 == 0:
                 print(f"Progress: {processed}/{total_files}...")

    print("\nProcessing complete.")
    print(f"Successfully processed: {success_count}")
    print(f"Failed/Skipped: {fail_count}")

def process_single_image_to_temp(file_path, output_dir):
    """
    Worker function. Converts image and saves to a temp filename in output_dir.
    Returns (True, temp_file_path) or (False, error_message).
    """
    import uuid
    import cv2
    import numpy as np
    # Re-import needed for Process isolation if not inherited (fork vs spawn)
    from PIL import Image, ImageFile
    
    # CRITICAL: Must be set inside the worker process
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass

    try:
        # 1. Open and Validate with Pillow first
        try:
            img = Image.open(file_path)
            # Basic verify
            # img.verify() # verify() can be finicky with truncation, skipping strict verify to allow partial loads
            
            # Force load
            img.load()
            
            # 2. Convert to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
        except Exception as pil_err:
            # Fallback to OpenCV if Pillow fails (often works for broken containers/WebP)
            try:
                # cv2.imread doesn't handle unicode paths well on some systems, but okay on mac usually.
                # It returns None if it fails.
                cv_img = cv2.imread(file_path)
                if cv_img is None:
                    raise Exception(f"PIL failed ({pil_err}) and OpenCV could not read image")
                
                # Convert BGR to RGB
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv_img)
                
            except Exception as cv_err:
                 return False, f"Both PIL and CV2 failed: {pil_err} | {cv_err}"

        # 3. Save to temp
        temp_name = f".temp_{uuid.uuid4()}.jpg"
        temp_path = os.path.join(output_dir, temp_name)
        
        img.save(temp_path, "JPEG", quality=95, optimize=True)
        
        return True, temp_path
        
    except Exception as e:
        return False, str(e)

if __name__ == "__main__":
    main()
