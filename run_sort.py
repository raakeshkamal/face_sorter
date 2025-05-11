import line_profiler
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw
import faiss
import numpy as np
from sklearn.cluster import HDBSCAN

from build_cache import *
from mongo_db import *

@line_profiler.profile
def add_new_class(class_name, id):
    db = get_mongo_db()
    clustercollection = db["clustercollection"]
    cluster = clustercollection.find_one({"cluster_id": id})

    classcollection = db["classcollection"]
    class_entry = {
        "class": class_name,
        "embed": cluster["centroid"],
    }
    classcollection.insert_one(class_entry)

@line_profiler.profile
def get_all_class_names():
    db = get_mongo_db()
    classcollection = db["classcollection"]
    classes = classcollection.find({}, {"class": 1, "_id": 0})

    print([doc["class"] for doc in classes])

@line_profiler.profile
def remove_class(class_name):
    db = get_mongo_db()
    classcollection = db["classcollection"]
    classcollection.delete_one({"class": class_name})

@line_profiler.profile
def print_results(unsorted_imgs, unsorted_path, indices):
    # Batch string operations instead of multiple prints
    results = [
        f"idx: {i} name: {unsorted_imgs[i]} path: {unsorted_path[i]}"
        for i in indices
    ]
    print("\n".join(results))

@line_profiler.profile
def process_image(img_url, expanded_path, bbox):
    """Image processing using PIL"""
    with Image.open(expanded_path) as img:
        draw = ImageDraw.Draw(img)
        draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], 
                      outline=(255, 0, 0), width=5)
        img.save(img_url, 'JPEG', quality=75, optimize=True)

@line_profiler.profile
def sort_faces(CACHE_DIR, imgname, imgcache, imgbbox, sorted_class_names, indices):
    # Identify unique class paths to avoid redundant operations
    # TODO: Need only do the diff of file changes
    unique_class_paths = {}
    for index, _ in enumerate(indices):
        class_name = sorted_class_names[index]
        path = os.path.expanduser(
            f"{CACHE_DIR}/faces/{class_name}"
        )
        unique_class_paths[class_name] = path

    existing_files = {}
    # Ensure the directory exists
    for class_name,path in unique_class_paths.items():
        os.makedirs(path, exist_ok=True)
    
        # Get list of existing files (just the filenames, not full paths)
        existing_files[class_name] = ([f for f in os.listdir(path) 
                        if os.path.isfile(os.path.join(path, f))])

        final_files= []
        for index, i in enumerate(indices):
            if class_name == sorted_class_names[index]:
                final_files.append(imgname[i])

        # Convert lists to sets for efficient comparison
        existing_set = set(existing_files[class_name])
        final_set = set(final_files)
        
        # Calculate differences
        files_to_remove = existing_set - final_set

        for filename in files_to_remove:
            file_path = os.path.join(path, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error removing {filename}: {e}")

    # Use ThreadPoolExecutor for I/O bound tasks
    num_workers = min(
        multiprocessing.cpu_count() * 2, 16
    )  # More threads for I/O tasks
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for index, i in enumerate(indices):
            class_name = sorted_class_names[index]
            path = unique_class_paths[class_name]
            if imgname[i] not in existing_files[class_name]:
                img_path = f"{path}/{imgname[i]}"
                expanded_path = os.path.expanduser(imgcache[i])
                bbox = np.array(imgbbox[i]).astype(np.int32)
                futures.append(
                    executor.submit(process_image, img_path, expanded_path, bbox)
                )

        # Wait for all tasks to complete
        for future in futures:
            future.result()

@line_profiler.profile
def show_results(CACHE_DIR, unsorted_imgs, unsorted_cache, unsorted_bbox, label, indices):
    cache_path = os.path.expanduser(
        f"{CACHE_DIR}/clusters/{label}"
    )
    # Ensure the directory exists
    os.makedirs(cache_path, exist_ok=True)
    
    # Get list of existing files (just the filenames, not full paths)
    existing_files = ([f for f in os.listdir(cache_path) 
                    if os.path.isfile(os.path.join(cache_path, f))])

    final_files= []
    for i in indices:
        final_files.append(unsorted_imgs[i])

    # Convert lists to sets for efficient comparison
    existing_set = set(existing_files)
    final_set = set(final_files)
    
    # Calculate differences
    files_to_remove = existing_set - final_set

    for filename in files_to_remove:
        file_path = os.path.join(cache_path, filename)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error removing {filename}: {e}")

    # Use ThreadPoolExecutor for I/O bound tasks
    num_workers = min(
        multiprocessing.cpu_count() * 2, 16
    )  # More threads for I/O tasks
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in indices:
            if unsorted_imgs[i] not in existing_files:
                cache_url = f"{cache_path}/{unsorted_imgs[i]}"
                expanded_path = os.path.expanduser(unsorted_cache[i])
                bbox = np.array(unsorted_bbox[i]).astype(np.int32)
                futures.append(
                    executor.submit(process_image, cache_url, expanded_path, bbox)
                )

        # Wait for all tasks to complete
        for future in futures:
            future.result()

@line_profiler.profile
def run_optimized_search(
    imgembeddings,
    classembeddings,
    classname,
    imgname,
    imgpath,
    imgcache,
    imgbbox,
):
    # Convert to float32 arrays efficiently
    # Optimize 1: Direct conversion to float32 arrays (no need to normalize)

    # Optimize 1: Direct conversion to float32 arrays
    imgembeddings = np.asarray(imgembeddings, dtype="float32")
    classembeddings = np.asarray(classembeddings, dtype="float32")

    # Optimize 2: Parallel index creation
    # Create a Faiss index
    index = faiss.IndexFlatIP(imgembeddings.shape[1])  # Cosine distance
    faiss.normalize_L2(imgembeddings)
    index.add(imgembeddings)

    # Create a Faiss index
    if len(classembeddings) > 0:
        classindex = faiss.IndexFlatIP(classembeddings.shape[1])  # Cosine distance
        faiss.normalize_L2(classembeddings)
        classindex.add(classembeddings)

    # Optimize 3: Use sets for faster membership tests and operations
    total_range = set(range(len(imgembeddings)))
    sorted_ids = set()
    sorted_class_mapping = {}  # Maps image ID to class name

    # Optimize 4: Batch queries where possible
    for id, img in enumerate(classembeddings):
        # Reshape for batch operation
        query = img.reshape(1, -1)

        # Perform range search
        Lims, Dist, Idx = index.range_search(query, 0.5)

        # Optimize 5: Minimize print operations in loops
        # Uncomment for debugging only
        # print(f"Name: {classname[id]} Img: {classimg[id]} Indices: {Idx}")
        # print("Distances:", Dist)

        # Optimize 6: Batch update of sorted_ids
        for i, dist in zip(Idx, Dist):
            if i not in sorted_ids:
                sorted_ids.add(i)
                sorted_class_mapping[i] = classname[id]

            # Minimize prints for performance
            # print(f"name: {imgname[i]} dist:{dist} path: {imgpath[i]}")

    # Optimize 7: Use set difference for unsorted IDs
    unsorted_ids = list(total_range - sorted_ids)

    # Optimize 8: Use array indexing for batch extraction
    if unsorted_ids:
        unsorted_imgs = [imgname[i] for i in unsorted_ids]
        unsorted_path = [imgpath[i] for i in unsorted_ids]
        unsorted_cache = [imgcache[i] for i in unsorted_ids]
        unsorted_bbox = [imgbbox[i] for i in unsorted_ids]
        unsorted_embeddings = [imgembeddings[i] for i in unsorted_ids]
    else:
        unsorted_imgs = []
        unsorted_path = []
        unsorted_cache = []
        unsorted_bbox = []
        unsorted_embeddings = []

    # Convert sorted_ids to list for compatibility
    sorted_ids_list = list(sorted_ids)
    sorted_class_names = [sorted_class_mapping[i] for i in sorted_ids_list]

    return (
        sorted_ids_list,
        sorted_class_names,
        unsorted_imgs,
        unsorted_path,
        unsorted_cache,
        unsorted_bbox,
        unsorted_embeddings,
    )

@line_profiler.profile
def run_func(CACHE_DIR, max_results):
    """Function to run the application."""
    db = get_mongo_db()
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
    ) = fetch_data_optimized(db)
    (
        sorted_ids,
        sorted_class_names,
        unsorted_imgs,
        unsorted_path,
        unsorted_cache,
        unsorted_bbox,
        unsorted_embeddings,
    ) = run_optimized_search(
        imgembeddings,
        classembeddings,
        classname,
        imgname,
        imgpath,
        imgcache,
        imgbbox,
    )
    print(f"len(unsorted_imgs): {len(unsorted_imgs)}")
    print("Sorting faces...")
    sort_faces(CACHE_DIR, imgname, imgcache, imgbbox, sorted_class_names, sorted_ids)

    print("Saving clusters...")
    # Convert embeddings to numpy array once with the right data type
    face_embeddings = np.array(unsorted_embeddings, dtype=np.float32)

    # Optimize HDBSCAN for performance
    dbscan = HDBSCAN(metric="cosine", min_samples=2, store_centers="centroid") 

    # Fit the model once
    dbscan.fit(face_embeddings)
    cluster_labels = dbscan.labels_
    cluster_centers = dbscan.centroids_

    # Skip silhouette score calculation - it's not used
    # silhouette = silhouette_score(face_embeddings, dbscan.labels_)

    # Get unique labels and counts efficiently
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)

    # Sort in one vectorized operation (descending by count)
    sorted_indices = np.argsort(-counts)
    sorted_unique_labels = unique_labels[sorted_indices]
    print(f"len(sorted_unique_labels): {len(sorted_unique_labels)}")

    clustercollection = db["clustercollection"]
    clustercollection.drop()
    results = 0
    # Process clusters efficiently
    for i, label in enumerate(sorted_unique_labels):
        if label != -1 and results < max_results:  # Skip noise points
            print(f"Processing cluster {i}")
            results += 1
            indices = np.where(cluster_labels == label)[0]
            centroid = cluster_centers[label].tolist()
            cluster_info = {
                "cluster_name": int(label),
                "cluster_id" : i,
                "indices": indices.tolist(),
                "centroid": centroid,
            }
            clustercollection.insert_one(cluster_info)
            # print_results(unsorted_imgs, unsorted_path, indices)
            show_results( CACHE_DIR, unsorted_imgs, unsorted_cache, unsorted_bbox, i, indices)
