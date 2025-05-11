import line_profiler
import glob
import os
import random
import shutil
from pathlib import Path
import cv2
from insightface.app import FaceAnalysis
import psutil
import os

from mongo_db import *

@line_profiler.profile
def print_process_memory():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Process Memory: {memory_info.rss / (1024 * 1024):.2f} MB")


# Step 1: Generate face embeddings
@line_profiler.profile
def generate_embeddings(app, img_path, NOFACE_DIR):
    print_process_memory()
    embedding = []
    try:
        img = cv2.imread(img_path)
        embedding = app.get(img)
        # clear_gpu_memory()
    except Exception as e:
        print(e)
    return embedding

@line_profiler.profile
def get_file_list_filtered_and_sorted(bkpcollecttion, SRC_DIR, BROKEN_DIR):

    sort_list = filter(os.path.isfile, glob.glob(SRC_DIR + "/*"))

    # # Sort list of files in directory by size
    sort_list = sorted(sort_list, key=lambda x: os.stat(x).st_size)

    for item in sort_list:
        if not item.endswith(".jpg"):
            sort_list.remove(item)
            continue  # Skip to the next iteration of the loop
        # remove processed images
    
    for img in bkpcollecttion.find():
        if img["item"] in sort_list:
            sort_list.remove(img["item"])

    file_list_sorted = [os.path.basename(x) for x in sort_list]

    return file_list_sorted

@line_profiler.profile
def get_ref_img_lib(faces_list, NOFACE_DIR):
    ref_img_lib = []
    for item in faces_list:
        temp = {}
        temp["name"] = item["name"]
        temp["img"] = item["image"]
        temp["dst"] = item["dir"]
        ref_img_lib.append(temp)
    return ref_img_lib

@line_profiler.profile
def train_func(SRC_DIR, NOFACE_DIR, BROKEN_DIR, CACHE_DIR):

    src_dir = Path(SRC_DIR)
    db = get_mongo_db()

    app = FaceAnalysis(providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0)

    print("Loading Image database...")
    bkpcollection = db["bkpcollection"]
    file_list = get_file_list_filtered_and_sorted(bkpcollection, SRC_DIR, BROKEN_DIR)
    random.shuffle(file_list)
    print(file_list)
    i = 0
    for item in file_list:
        item_path = src_dir.joinpath(item)
        if not os.path.exists(item_path):
            print(f"Warning: File not found, skipping:{item_path}")
            continue  # Skip to the next iteration of the loopration of the loop
        i = i + 1
        print(f"iter: {i}")
        faces = generate_embeddings(app, item_path, NOFACE_DIR)
        if len(faces) == 0:
            print(f"Warning: No face found, skipping: {item_path}")
            shutil.move(item_path, NOFACE_DIR)
        for face in faces:
            embed = {}
            embed["idx"] = bkpcollection.count_documents({})
            embed["item"] = item
            embed["path"] = str(item_path)
            embed["age"] = face.age
            embed["gender"] = int(face.gender)
            embed["bbox"] = face.bbox.tolist()
            embed["kps"] = face.kps.tolist()
            embed["det_score"] = float(face.det_score)
            embed["landmark_3d_68"] = face.landmark_3d_68.tolist()
            embed["pose"] = face.pose.tolist()
            embed["landmark_2d_106"] = face.landmark_2d_106.tolist()
            embed["embedding"] = face.embedding.tolist()
            embed["cache_url"] = f"{CACHE_DIR}/{item}"
            bkpcollection.insert_one(embed)
