import os
import argparse
import glob
import sys
from typing import List, Tuple
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from tqdm import tqdm

# Fix for DecompressionBombWarning
Image.MAX_IMAGE_PIXELS = None

# Set environment variable to potentially fix OpenMP conflicts if we were using FAISS, 
# but we are switching to Torch for search.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def load_images(image_dir: str) -> List[str]:
    """Recursively find all images in the directory."""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.webp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))
    
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, '**', ext.upper()), recursive=True))
        
    return sorted(list(set(image_files)))

def get_image_info(path: str) -> Tuple[int, int]:
    """Return (resolution_pixels, file_size_bytes) for quality comparison."""
    try:
        with Image.open(path) as img:
            pixels = img.width * img.height
        size = os.path.getsize(path)
        return (pixels, size)
    except Exception:
        return (0, 0)

def batch_process(model, processor, image_files: List[str], batch_size: int = 32, device: str = "cpu") -> np.ndarray:
    """Compute embeddings for images in batches."""
    embeddings = []
    
    for i in tqdm(range(0, len(image_files), batch_size), desc="Computing embeddings"):
        batch_paths = image_files[i : i + batch_size]
        images = []
        
        for path in batch_paths:
            try:
                image = Image.open(path).convert("RGB")
                images.append(image)
            except Exception as e:
                print(f"Error loading {path}: {e}")

        if not images:
            continue

        try:
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
                outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
                embeddings.append(outputs.cpu().numpy())
        except Exception as e:
            print(f"Error processing batch starting at {i}: {e}")
            
    if embeddings:
        return np.vstack(embeddings).astype(np.float32)
    return np.array([], dtype=np.float32)

def find_duplicates_torch(image_files: List[str], embeddings_np: np.ndarray, threshold: float = 0.99, batch_size: int = 1000):
    """Find duplicates using PyTorch Matrix Multiplication (more stable than FAISS on Mac)."""
    print("Preparing for similarity search...", flush=True)
    if len(embeddings_np) == 0:
        print("No embeddings to process.")
        return

    # Debug info
    print(f"Embeddings shape: {embeddings_np.shape}", flush=True)
    
    # Check for NaNs
    if np.isnan(embeddings_np).any():
        print("Warning: Embeddings contain NaNs. Replacing with zeros.", flush=True)
        embeddings_np = np.nan_to_num(embeddings_np)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    
    print(f"Using device for search: {device}", flush=True)
    
    # Convert to torch tensor
    try:
        embeddings = torch.from_numpy(embeddings_np).to(device)
    except Exception as e:
        print(f"Error moving embeddings to device: {e}. Fallback to CPU.")
        device = "cpu"
        embeddings = torch.from_numpy(embeddings_np).to(device)

    num_images = len(embeddings)
    visited = set()
    duplicates = []
    
    print(f"Computing similarity matrix in batches (Threshold: {threshold})...", flush=True)
    
    # Process in chunks to avoid OOM even with 25k images if VRAM is tight
    # We only need to compute rows i to i+batch vs all columns
    
    for i in tqdm(range(0, num_images, batch_size), desc="Searching"):
        end_idx = min(i + batch_size, num_images)
        
        # Chunk of queries: [batch, Dim]
        query_chunk = embeddings[i:end_idx]
        
        # Compute similarity: [batch, Dim] @ [Dim, Total] -> [batch, Total]
        # Transpose embeddings for MM
        sim_matrix = torch.mm(query_chunk, embeddings.T)
        
        # We need values > threshold
        # Since we only care about upper triangle for duplicates (undirected), 
        # but here we are computing full rows.
        # Let's iterate through the batch results on CPU to find matches
        
        # Move chunk result to CPU to parse
        sim_matrix = sim_matrix.cpu().numpy()
        
        for local_row in range(sim_matrix.shape[0]):
            global_idx = i + local_row
            
            if global_idx in visited:
                continue
                
            scores = sim_matrix[local_row]
            
            # Find matches > threshold
            matches = np.where(scores >= threshold)[0]
            
            # Consider all matches in the same group, including the current one
            # if they haven't been visited yet.
            group_indices = [m for m in matches if m not in visited]
            
            if len(group_indices) > 1:
                # We have a cluster. Find the best quality image as the original.
                cluster_files = [image_files[idx] for idx in group_indices]
                
                # Sort by (resolution, size) descending, then path ascending for stability
                scored_files = []
                for idx in group_indices:
                    path = image_files[idx]
                    quality = get_image_info(path)
                    scored_files.append((quality, path, idx))
                
                # Sort: reverse=True for quality (higher is better), then path (lower is better)
                # We can use a custom sort key
                scored_files.sort(key=lambda x: (x[0][0], x[0][1], [-ord(c) for c in x[1]]), reverse=True)
                
                best_idx = scored_files[0][2]
                original_path = image_files[best_idx]
                
                duplicate_paths = []
                for _, path, idx in scored_files[1:]:
                    duplicate_paths.append(path)
                    visited.add(idx)
                
                visited.add(best_idx)
                duplicates.append((original_path, duplicate_paths))
            elif len(group_indices) == 1:
                # Only the self-match or one image, just mark as visited
                visited.add(group_indices[0])

    return duplicates

def generate_report(duplicates: List[Tuple[str, List[str]]], output_file: str = "duplicates_report.txt"):
    """Generate a text report of found duplicates."""
    if not duplicates:
        print("No duplicates found.")
        return

    print(f"Found {len(duplicates)} sets of duplicates:\n")
    with open(output_file, "w") as f:
        for orig, dups in duplicates:
            print(f"Original: {orig}")
            f.write(f"Original: {orig}\n")
            for dup in dups:
                print(f"  Duplicate: {dup}")
                f.write(f"  Duplicate: {dup}\n")
            print("-" * 40)
            f.write("-" * 40 + "\n")
    print(f"\nReport saved to {output_file}")

def delete_duplicates(duplicates: List[Tuple[str, List[str]]]):
    """Delete the identified duplicate files."""
    count = 0
    total = sum(len(dups) for _, dups in duplicates)
    
    print(f"\nDeleting {total} duplicates...")
    for _, dups in duplicates:
        for dup in dups:
            try:
                if os.path.exists(dup):
                    os.remove(dup)
                    count += 1
                    # print(f"Deleted: {dup}") # process might be too noisy
            except Exception as e:
                print(f"Error deleting {dup}: {e}")
    
    print(f"Successfully deleted {count} images.")

def main():
    parser = argparse.ArgumentParser(description="Find duplicate images using CLIP.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--threshold", type=float, default=0.98, help="Similarity threshold (0-1). Default 0.98")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32", help="CLIP model name")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing/inference")
    parser.add_argument("--cache_file", type=str, default="embeddings_cache.npy", help="File to cache embeddings")
    parser.add_argument("--force_recompute", action="store_true", help="Force recomputing embeddings even if cache exists")
    parser.add_argument("--delete", action="store_true", help="Delete duplicate images after confirmation")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    
    print(f"Using device: {device}")
    print(f"Finding images in {args.image_dir}...")
    image_files = load_images(args.image_dir)
    print(f"Found {len(image_files)} images.")
    
    if len(image_files) == 0:
        return

    embeddings = None
    
    # Check cache
    if not args.force_recompute and os.path.exists(args.cache_file):
        print(f"Loading embeddings from cache: {args.cache_file}")
        try:
            cached_data = np.load(args.cache_file, allow_pickle=True)
            if len(cached_data) == len(image_files):
                embeddings = cached_data
                print("Cache loaded successfully.")
            else:
                print(f"Cache size ({len(cached_data)}) mismatch with found images ({len(image_files)}). Recomputing...")
        except Exception as e:
            print(f"Could not load cache: {e}")
            embeddings = None

    if embeddings is None:
        print(f"Loading CLIP model: {args.model_name}...")
        try:
            model = CLIPModel.from_pretrained(args.model_name).to(device)
            processor = CLIPProcessor.from_pretrained(args.model_name)
        except Exception as e:
            print(f"Failed to load model: {e}")
            return
            
        embeddings = batch_process(model, processor, image_files, args.batch_size, device)
        
        print(f"Saving embeddings to {args.cache_file}...")
        np.save(args.cache_file, embeddings)
    
    # Default to Torch-based search now since FAISS was crashing
    duplicates = find_duplicates_torch(image_files, embeddings, args.threshold)
    
    if duplicates:
        generate_report(duplicates)
        
        if args.delete:
            print(f"\nWARNING: You are about to DELETE {sum(len(d) for _, d in duplicates)} duplicate files.")
            confirm = input("Are you sure you want to continue? (y/n): ")
            if confirm.lower() == 'y':
                delete_duplicates(duplicates)
            else:
                print("Deletion cancelled.")
    else:
        print("No duplicates found.")

if __name__ == "__main__":
    main()
