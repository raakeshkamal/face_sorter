# Face Sorter

A Python face recognition and sorting application that uses machine learning to organize images by detected faces. The application processes large image datasets, extracts facial features, and sorts them into person-based classes.

## Features

- **Face Detection**: Uses InsightFace for robust face detection and embedding generation
- **Face Clustering**: Groups similar faces using HDBSCAN algorithm
- **Efficient Search**: Uses FAISS for fast similarity search
- **Image Caching**: Compresses and caches images for faster processing
- **Data Persistence**: Stores face embeddings and metadata in MongoDB
- **Modular Architecture**: Clean separation of concerns for maintainability

## Workflow

```
build-cache → train → sort → add-class → sort → (repeat until complete)
```

1. **build-cache**: Compress and cache images for faster processing
2. **train**: Detect faces and generate embeddings using InsightFace
3. **sort**: Match faces to known classes and cluster unknown faces
4. **add-class**: Manually add new face classes from clusters
5. **sort**: Re-sort remaining faces with new classes

## Installation

### Prerequisites

- Python 3.10 or higher
- MongoDB (running on localhost:27017)
- uv package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd face_sorter
```

2. Install dependencies with uv:
```bash
uv sync
```

3. Configure the application:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Install the package (optional, for CLI access):
```bash
uv pip install -e .
```

## Usage

### Command-Line Interface

The application provides a command-line interface with the following commands:

```bash
# Build cache
face-sorter build-cache

# Train model (detect faces and generate embeddings)
face-sorter train

# Sort faces (match to classes and cluster unknown faces)
face-sorter sort --max-results 10

# Add new class from cluster
face-sorter add-class "ClassName" 42

# Remove class
face-sorter remove-class "ClassName"

# List all classes
face-sorter list-classes

# Clean dataset (convert to standard format)
face-sorter clean-dataset --input-dir /path/to/images --output-dir cleaned

# Deduplicate images
face-sorter dedup --image-dir /path/to/images
```

### Configuration

Configure the application by copying `.env.example` to `.env` and editing the values:

```env
# Database Configuration
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=facedatabase

# Directory Paths
SOURCE_DIR=/path/to/your/images
NOFACE_DIR=/path/to/noface/images
BROKEN_DIR=/path/to/broken/images
CACHE_DIR=/path/to/cache/directory

# Processing Settings
SIMILARITY_THRESHOLD=0.5
CLUSTER_MIN_SAMPLES=2
BATCH_SIZE=32
```

### Development

For development with all tools:
```bash
uv sync --extra dev
```

Run tests:
```bash
pytest
```

Format code:
```bash
black src/
```

Type checking:
```bash
mypy src/
```

## Project Structure

```
face_sorter/
├── pyproject.toml          # Project configuration
├── README.md              # This file
├── .env.example           # Configuration template
├── src/
│   └── face_sorter/
│       ├── __init__.py    # Package exports
│       ├── cli.py         # Click CLI interface
│       ├── config.py      # Configuration management
│       ├── database/      # MongoDB operations
│       ├── models/        # Data models
│       ├── services/      # Business logic
│       └── utils/         # Utilities
├── scripts/               # Standalone utilities
├── notebooks/             # Jupyter notebooks
└── tests/                 # Unit tests
```

## Dependencies

- **Computer Vision**: OpenCV, Pillow, InsightFace
- **Machine Learning**: PyTorch, FAISS, scikit-learn
- **Database**: pymongo
- **CLI Framework**: Click
- **Configuration**: Pydantic, python-dotenv

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.
