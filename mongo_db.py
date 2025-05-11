import line_profiler
from pymongo import MongoClient
import numpy as np

@line_profiler.profile
def get_mongo_db():
    client = None
    db = None

    try:
        # Connect to the local MongoDB server (default port 27017)
        client = MongoClient("mongodb://localhost:27017/")
        print("Successfully connected to MongoDB!")

        # Access (or create, if it doesn't exist) a database named "mydatabase"
        db = client.facedatabase  # Much cleaner than brackets for db access.

    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")

    return db

@line_profiler.profile
def fetch_data_optimized(db):
    # Get collections
    bkpcollection = db["bkpcollection"]
    classcollection = db["classcollection"]

    # === OPTIMIZATION 1: Fetch reference data efficiently ===
    # Use projection to retrieve only needed fields
    class_docs = list(
        classcollection.find(
            {}, {"class": 1, "embed": 1, "_id": 0}
        )
    )

    # Use list comprehensions instead of appends
    refname = [doc["class"] for doc in class_docs]
    refembeddings = [doc["embed"] for doc in class_docs]

    # === OPTIMIZATION 2: Process class data with vectorized operations ===
    # Get distinct classes once
    distinct_classes = list(set(refname))

    # Group documents by class efficiently
    class_data = {}
    for doc in class_docs:  # Reuse already fetched data
        cls = doc["class"]
        if cls not in class_data:
            class_data[cls] = {"items": [], "paths": [], "embeds": []}

        class_data[cls]["embeds"].append(doc["embed"])

    # Pre-allocate result arrays
    classembeddings = [None] * len(distinct_classes)

    # Vectorized processing
    for i, cls in enumerate(distinct_classes):
        data = class_data[cls]
        # Convert to NumPy array for vectorized operations
        embeds = np.array(data["embeds"])
        # Normalize all embeddings at once
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        normalized_embeds = embeds / norms
        # Compute mean and normalize in one step
        mean_embed = np.mean(normalized_embeds, axis=0)
        classembeddings[i] = mean_embed / np.linalg.norm(mean_embed)

    # === OPTIMIZATION 3: Efficiently fetch image data ===
    # Create index if it doesn't exist (run once)
    # bkpcollection.create_index([("idx", ASCENDING)])

    # Fetch docs with projection and sorting in a single query
    cursor = bkpcollection.find(
        {},
        {
            "item": 1,
            "path": 1,
            "cache_url": 1,
            "bbox": 1,
            "embedding": 1,
            "_id": 0,
        },
    ).sort("idx", 1)

    # Pre-allocate results with estimated size
    count = bkpcollection.count_documents({})
    imgname = [None] * count
    imgpath = [None] * count
    imgcache = [None] * count
    imgbbox = [None] * count
    imgembeddings = [None] * count

    # Populate arrays efficiently
    for i, doc in enumerate(cursor):
        imgname[i] = doc["item"]
        imgpath[i] = doc["path"]
        imgcache[i] = doc["cache_url"]
        imgbbox[i] = doc["bbox"]
        imgembeddings[i] = doc["embedding"]

    return (
        refname,
        refembeddings,
        distinct_classes,
        classembeddings,
        imgname,
        imgpath,
        imgbbox,
        imgcache,
        imgembeddings,
    )