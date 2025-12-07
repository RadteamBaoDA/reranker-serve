# Reranker Service

A high-performance reranker service using Sentence Transformer models, compatible with Jina AI and Cohere API formats.

## Features

- 🚀 **FastAPI-based REST API** with async engine (vLLM-inspired architecture)
- 📊 **Dynamic Batching** - Automatic request batching for optimal GPU utilization
- 🔄 **API Compatibility** - Compatible with Jina AI and Cohere reranker APIs
- 🤖 **Multiple Models** - BAAI/bge-reranker and Qwen3-reranker support
- 💻 **Multi-platform** - CUDA, Apple Silicon (MPS), and CPU
-  **Load Balancing** - LiteLLM-style router for multiple backends
- 🐳 **Docker Ready** - Includes Dockerfile and docker-compose.yml

## Quick Start

### Windows

```powershell
.\setup.ps1
.\run.ps1
```

### Linux/macOS

```bash
chmod +x setup.sh run.sh
./setup.sh
./run.sh
```

### Docker

```bash
docker-compose up -d
```

### Access

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Basic Usage

```bash
curl -X POST http://localhost:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is deep learning?",
    "documents": ["Deep learning is a subset of ML.", "The weather is nice."],
    "top_n": 2
  }'
```

## Supported Models

| Model | Description |
|-------|-------------|
| `BAAI/bge-reranker-v2-m3` | Multilingual reranker (default) |
| `BAAI/bge-reranker-large` | English reranker |
| `BAAI/bge-reranker-base` | Smaller English reranker |
| `Qwen3-reranker` | Qwen-based reranker models |

## Configuration

Set environment variables with `RERANKER_` prefix:

```env
RERANKER_PORT=8000
RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3
RERANKER_DEVICE=cuda
```

See [Configuration Reference](docs/configuration.md) for all options.

## Documentation

| Document | Description |
|----------|-------------|
| [Configuration](docs/configuration.md) | Environment variables & settings |
| [API Reference](docs/api-reference.md) | All API endpoints with examples |
| [Load Balancer](docs/load-balancer.md) | Multi-backend routing setup |
| [Development](docs/development.md) | Dev setup, testing, project structure |
| [Deployment](docs/deployment.md) | Docker, offline mode, troubleshooting |

## License

MIT License
