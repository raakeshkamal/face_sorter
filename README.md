# Face Sorter

A Python face recognition and sorting application that uses machine learning to organize images by detected faces. The application processes large image datasets, extracts facial features, and sorts them into person-based classes. It includes a beautiful, fastdup-inspired Web UI for managing operations, built with FastAPI and Vue.js.

## 🌟 Features

### Core Capabilities
- **Face Detection**: Uses InsightFace for robust face detection and embedding generation
- **Face Clustering**: Groups similar faces using HDBSCAN algorithm
- **Efficient Search**: Uses FAISS for fast similarity search
- **Image Caching**: Compresses and caches images for faster processing
- **Data Persistence**: Stores face embeddings and metadata in MongoDB
- **Modular Architecture**: Clean separation of concerns for maintainability

### Web UI
- **Modern Dashboard**: Real-time statistics and quick action buttons
- **Image Galleries**: Beautiful, responsive galleries with lazy loading and modal viewing
- **Class Management**: Create, view, and delete face classes with live updates
- **Real-time Operations**: Training and cleaning with WebSocket-based progress tracking
- **Fastdup-Style Design**: Professional color scheme (#657BEC, #2E3E8E, #FFFCF3) and smooth animations
- **Responsive Layout**: Mobile-friendly sidebar navigation and adaptive grids

## 🔄 Workflow & Architecture

### Application Workflow
```text
build-cache → train → sort → add-class → sort → (repeat until complete)
```
1. **build-cache**: Compress and cache images for faster processing
2. **train**: Detect faces and generate embeddings using InsightFace
3. **sort**: Match faces to known classes and cluster unknown faces
4. **add-class**: Manually add new face classes from clusters
5. **sort**: Re-sort remaining faces with new classes

### System Architecture
```text
┌─────────────────────────────────────────────────────────────┐
│                  Vue.js Frontend                            │
│  - Dashboard with stats overview                            │
│  - Image galleries & Real-time progress bars                │
└──────────────────┬──────────────────────────────────────────┘
                   │ HTTP/WebSocket
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Server Layer                           │
│  - REST API & WebSocket for real-time updates               │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌───────────────┐    ┌───────────────┐
│ Service Layer │    │   Database    │
│ - training    │    │   MongoDB     │
│ - cleaning    │    │               │
└───────────────┘    └───────────────┘
```

## 🚀 Installation & Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+ (for Web UI development)
- MongoDB running locally (port 27017)
- `uv` package manager

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd face_sorter
```

2. Install backend dependencies:
```bash
uv sync
```

3. Install frontend dependencies:
```bash
cd src/face_sorter/web/frontend
npm install
```

4. Configure the application:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## 💻 Usage

### Web UI (Recommended)

**Development Mode:**
```bash
# Terminal 1: Backend
cd face_sorter
face-sorter web

# Terminal 2: Frontend
cd src/face_sorter/web/frontend
npm run dev
```
Access the UI at: **http://127.0.0.1:5173**

**Production Mode:**
```bash
cd src/face_sorter/web/frontend
npm run build
cd ../../../..
face-sorter web
```
Access the UI at: **http://127.0.0.1:8000**

### Command-Line Interface

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

## ⚙️ Configuration

Edit your `.env` file to configure the application:

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

# Web UI Settings
UI_HOST=127.0.0.1
UI_PORT=8000
UI_RELOAD=true
UI_LOG_LEVEL=info
```

## 🏭 Production Deployment

### Docker Deployment

A `docker-compose.yml` is provided for easy deployment:

```yaml
version: '3.8'

services:
  mongodb:
    image: mongo:7
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
    restart: unless-stopped

  face_sorter:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MONGODB_URI=mongodb://mongodb:27017/
      - MONGODB_DATABASE=facedatabase
      - UI_HOST=0.0.0.0
      - UI_PORT=8000
      - UI_RELOAD=false
    depends_on:
      - mongodb
    restart: unless-stopped
    volumes:
      - ./data:/app/data

volumes:
  mongodb_data:
```

```bash
# Build and start services
docker-compose up --build -d
```

### System Configuration (Nginx Reverse Proxy)

For bare-metal deployments, it is recommended to use Nginx as a reverse proxy:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    client_max_body_size 100M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## 🛠️ Development

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

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.
