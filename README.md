# Reranker Service

A high-performance reranker service using Sentence Transformer models, compatible with Jina AI and Cohere API formats.

## Features

- 🚀 **FastAPI-based REST API** - High-performance async API server
- 🔄 **API Compatibility** - Compatible with Jina AI and Cohere reranker APIs
- 🤖 **Multiple Model Support** - Supports BAAI/bge-reranker and Qwen3-reranker models
- 💻 **Multi-platform** - Optimized for CUDA, Apple Silicon (MPS), and CPU
- 🪟 **Cross-platform Scripts** - Full support for Windows, Linux, and macOS
- 📦 **Offline Mode** - Load models from local disk without internet access
- 🐳 **Docker Ready** - Includes Dockerfile and docker-compose.yml
- ⚡ **Production Ready** - Uvicorn ASGI server with multiple workers

## Supported Models

- **BAAI/bge-reranker-v2-m3** - Multilingual reranker (default)
- **BAAI/bge-reranker-large** - English reranker
- **BAAI/bge-reranker-base** - Smaller English reranker
- **Qwen3-reranker** - Qwen-based reranker models

## Quick Start

### Windows

1. **Setup the environment (PowerShell):**
   ```powershell
   .\setup.ps1
   ```

2. **Run the server:**
   ```powershell
   .\run.ps1
   ```

3. **Alternative (Batch files):**
   ```cmd
   setup.bat
   run.bat
   ```

### Linux/macOS

1. **Setup the environment:**
   ```bash
   chmod +x setup.sh run.sh download_model.sh daemon.sh
   ./setup.sh
   ```

2. **Run the server:**
   ```bash
   ./run.sh
   ```

### Access the API

- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Build and run with docker-compose
docker-compose up -d

# Or build manually
docker build -t reranker-serve .
docker run -p 8000:8000 reranker-serve
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
| `RERANKER_LOAD_BALANCER_ENABLED` | Enable load balancer mode | `false` |
| `RERANKER_CONFIG_PATH` | Path to YAML config file | `./reranker_config.yaml` |

### Example `.env` file

```env
RERANKER_PORT=8000
RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3
RERANKER_DEVICE=mps
RERANKER_LOG_LEVEL=INFO
```

## Load Balancer (LiteLLM-Style)

The service includes a load balancer that can route requests across multiple reranker backends, similar to LiteLLM's router.

### Enable Load Balancer

1. **Create a YAML configuration file** (`reranker_config.yaml`):

```yaml
model_list:
  - model_name: "primary-reranker"
    litellm_params:
      api_base: "http://server1:8000"
      api_key: "optional-key"
    weight: 2.0
    priority: 0
    
  - model_name: "backup-reranker"
    litellm_params:
      api_base: "http://server2:8000"
    weight: 1.0
    priority: 1

router_settings:
  routing_strategy: "weighted-random"  # or round-robin, least-busy, latency-based-routing, priority-failover
  num_retries: 3
  retry_delay: 1.0
  fallback_to_local: true
  health_check_interval: 60
```

2. **Enable in environment:**

```bash
export RERANKER_LOAD_BALANCER_ENABLED=true
export RERANKER_CONFIG_PATH=./reranker_config.yaml
```

3. **Use load-balanced endpoints:**

```bash
# Requests are automatically load-balanced
POST /rerank
POST /v1/rerank  
POST /api/v1/rerank
```

### Routing Strategies

| Strategy | Description |
|----------|-------------|
| `weighted-random` | Random selection weighted by model weights (default) |
| `round-robin` | Sequential rotation through backends |
| `least-busy` | Route to backend with fewest active requests |
| `latency-based-routing` | Route to backend with lowest average latency |
| `priority-failover` | Use highest priority backend, failover on error |
| `simple-shuffle` | Pure random selection |

### Load Balancer Features

- **Health Checks**: Automatic health monitoring of backends
- **Circuit Breaker**: Unhealthy backends are temporarily removed
- **Automatic Retries**: Failed requests retry on other backends
- **Rate Limiting**: Per-backend request rate limits (rpm, tpm)
- **Request Logging**: Detailed logging of routed requests
- **Fallback to Local**: Use local model if all backends fail

### Monitor Load Balancer

```bash
# Check load balancer statistics
curl http://localhost:8000/lb/stats
```

## Offline Model Usage

### Linux/macOS

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

### Windows

1. **Download the model (PowerShell):**
   ```powershell
   .\download_model.ps1 -ModelName "BAAI/bge-reranker-v2-m3" -OutputDir "./models"
   ```

2. **Configure for offline use:**
   ```powershell
   $env:RERANKER_MODEL_PATH="./models/BAAI_bge-reranker-v2-m3"
   $env:RERANKER_USE_OFFLINE_MODE="true"
   ```

3. **Run the server:**
   ```powershell
   .\run.ps1
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
# Linux/macOS
./setup.sh --dev

# Windows PowerShell
.\setup.ps1 -Dev
```

### Run tests

```bash
pytest tests/ -v
```

### Run with auto-reload

```bash
# Linux/macOS
./run.sh --dev

# Windows PowerShell
.\run.ps1 -Dev
```

### Run as Daemon (Background Service)

#### Linux/macOS

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

#### Windows (PowerShell)

```powershell
# Start as background job
.\daemon.ps1 -Action start

# Check status
.\daemon.ps1 -Action status

# View logs
.\daemon.ps1 -Action logs

# Stop daemon
.\daemon.ps1 -Action stop

# Restart daemon
.\daemon.ps1 -Action restart
```

## Troubleshooting

### Windows: PyTorch DLL Error

If you encounter `[WinError 1114] A dynamic link library (DLL) initialization routine failed`:

1. **Install Visual C++ Redistributable:**
   - Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Install and restart your terminal

2. **Run the fix script:**
   ```powershell
   .\fix-pytorch.ps1
   ```

### Verify PyTorch Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
```

### Verify Device Support

```bash
# CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# MPS (Apple Silicon)
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
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
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── rerank.py        # Pydantic schemas
│   └── load_balancer/
│       ├── __init__.py
│       ├── config.py        # YAML config loader
│       └── router.py        # Load balancer router
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_config.py
│   ├── test_load_balancer.py
│   └── test_reranker.py
├── models/                  # Model cache directory
├── reranker_config.yaml.example  # Load balancer config example
├── setup.sh                 # Setup script (Linux/macOS)
├── setup.ps1                # Setup script (Windows PowerShell)
├── setup.bat                # Setup script (Windows Batch)
├── run.sh                   # Run script (Linux/macOS)
├── run.ps1                  # Run script (Windows PowerShell)
├── run.bat                  # Run script (Windows Batch)
├── daemon.sh                # Daemon script (Linux/macOS)
├── daemon.ps1               # Daemon script (Windows PowerShell)
├── download_model.sh        # Model download (Linux/macOS)
├── download_model.ps1       # Model download (Windows PowerShell)
├── download_model.bat       # Model download (Windows Batch)
├── fix-pytorch.ps1          # PyTorch fix script (Windows)
├── Dockerfile               # Docker build file
├── docker-compose.yml       # Docker Compose config
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .env.example             # Environment variables template
├── .gitignore
└── README.md
```

## License

MIT License
