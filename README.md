# Reranker Service

A high-performance reranker service using Sentence Transformer models, compatible with Jina AI and Cohere API formats.

## Features

- 🚀 **FastAPI-based REST API** - High-performance async API server
- 🔄 **API Compatibility** - Compatible with Jina AI and Cohere reranker APIs
- 🤖 **Multiple Model Support** - Supports BAAI/bge-reranker and Qwen3-reranker models
- 💻 **Multi-platform** - Optimized for CUDA, Apple Silicon (MPS), and CPU
- 📦 **Offline Mode** - Load models from local disk without internet access
- ⚡ **Production Ready** - Uvicorn ASGI server with multiple workers

## Supported Models

- **BAAI/bge-reranker-v2-m3** - Multilingual reranker (default)
- **BAAI/bge-reranker-large** - English reranker
- **BAAI/bge-reranker-base** - Smaller English reranker
- **Qwen3-reranker** - Qwen-based reranker models

## Quick Start

### Linux/macOS

1. **Setup the environment:**
   ```bash
   chmod +x setup.sh run.sh download_model.sh
   ./setup.sh
   ```

2. **Run the server:**
   ```bash
   ./run.sh
   ```

3. **Access the API:**
   - API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Manual Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Native API

```bash
POST /rerank
```

```json
{
  "query": "What is deep learning?",
  "documents": [
    "Deep learning is a subset of machine learning.",
    "The weather is nice today."
  ],
  "top_n": 2,
  "return_documents": true
}
```

### Cohere-Compatible API

```bash
POST /v1/rerank
```

```json
{
  "query": "What is deep learning?",
  "documents": ["Document 1", "Document 2"],
  "top_n": 2
}
```

### Jina AI-Compatible API

```bash
POST /api/v1/rerank
```

```json
{
  "query": "What is deep learning?",
  "documents": [
    {"text": "Document 1"},
    {"text": "Document 2"}
  ],
  "top_n": 2
}
```

## Configuration

Configuration is done through environment variables with the `RERANKER_` prefix:

| Variable | Description | Default |
|----------|-------------|---------|
| `RERANKER_HOST` | Server host | `0.0.0.0` |
| `RERANKER_PORT` | Server port | `8000` |
| `RERANKER_WORKERS` | Number of workers | `1` |
| `RERANKER_MODEL_NAME` | Model name or HuggingFace ID | `BAAI/bge-reranker-v2-m3` |
| `RERANKER_MODEL_PATH` | Local model path | `None` |
| `RERANKER_MODEL_CACHE_DIR` | Model cache directory | `./models` |
| `RERANKER_USE_OFFLINE_MODE` | Offline mode (no downloads) | `false` |
| `RERANKER_MAX_LENGTH` | Maximum sequence length | `512` |
| `RERANKER_BATCH_SIZE` | Inference batch size | `32` |
| `RERANKER_DEVICE` | Device: cuda, mps, cpu | Auto-detect |
| `RERANKER_USE_FP16` | Use FP16 precision | `true` |
| `RERANKER_API_KEY` | API key for auth | `None` |
| `RERANKER_LOG_LEVEL` | Logging level | `INFO` |

### Example `.env` file

```env
RERANKER_PORT=8000
RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3
RERANKER_DEVICE=mps
RERANKER_LOG_LEVEL=INFO
```

## Offline Model Usage

1. **Download the model:**
   ```bash
   ./download_model.sh BAAI/bge-reranker-v2-m3 ./models
   ```

2. **Configure for offline use:**
   ```bash
   export RERANKER_MODEL_PATH=./models/BAAI_bge-reranker-v2-m3
   export RERANKER_USE_OFFLINE_MODE=true
   ```

3. **Run the server:**
   ```bash
   ./run.sh
   ```

## Apple Silicon (MPS) Optimization

The service automatically detects and uses MPS on Apple Silicon Macs:

- Automatic MPS fallback to CPU for unsupported operations
- Optimized memory management for MPS
- FP32 precision for stability on MPS

To verify MPS support:
```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

## Development

### Install dev dependencies

```bash
./setup.sh --dev
```

### Run tests

```bash
pytest tests/ -v
```

### Run with auto-reload

```bash
./run.sh --dev
```

### Run as Daemon (Background Service)

```bash
# Start as daemon
./daemon.sh start

# Check status
./daemon.sh status

# View logs
./daemon.sh logs

# Stop daemon
./daemon.sh stop

# Restart daemon
./daemon.sh restart
```

## Project Structure

```
reranker-serve/
├── src/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py        # API routes
│   │   └── health.py        # Health endpoints
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py      # Configuration
│   ├── models/
│   │   ├── __init__.py
│   │   └── reranker.py      # Reranker model
│   └── schemas/
│       ├── __init__.py
│       └── rerank.py        # Pydantic schemas
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_config.py
│   └── test_reranker.py
├── setup.sh                 # Setup script
├── run.sh                   # Run script
├── download_model.sh        # Model download script
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── README.md
```

## License

MIT License
