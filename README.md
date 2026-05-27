# Reranker Service

A high-performance reranker service using Sentence Transformer models, compatible with Jina AI and Cohere API formats.

## Features

- 🚀 **FastAPI-based REST API** with async engine (vLLM-inspired architecture)
- 📊 **Dynamic Batching** - Automatic request batching for optimal GPU utilization
- 🔄 **API Compatibility** - Compatible with HuggingFace, Jina AI, and Cohere reranker APIs
- 🤖 **Multiple Models** - BAAI/bge-reranker and Qwen3-reranker support
- 💻 **Multi-platform** - CUDA, Apple Silicon (MPS), and CPU
- ⚖️ **Load Balancing** - LiteLLM-style router for multiple backends
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

### Native API (/rerank endpoint)

The `/rerank` endpoint now supports **both** native and HuggingFace formats!
**The response format automatically matches your request format.**

**Native format:**
```bash
curl -X POST http://localhost:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is deep learning?",
    "documents": ["Deep learning is a subset of ML.", "The weather is nice."],
    "top_n": 2
  }'
# Response: {"results": [{"index": 0, "relevance_score": 0.95, "document": {...}}], ...}
```

**HuggingFace format (same endpoint, different response!):**
```bash
curl -X POST http://localhost:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is deep learning?",
    "texts": ["Deep learning is a subset of ML.", "The weather is nice."],
    "top_k": 2
  }'
# Response: {"results": [{"index": 0, "score": 0.95, "text": "..."}], ...}
```

### HuggingFace-Compatible API (/reranking endpoint)

```bash
curl -X POST http://localhost:8000/reranking \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is deep learning?",
    "texts": ["Deep learning is a subset of ML.", "The weather is nice."],
    "top_k": 2
  }'
```

**Note:** 
- `/rerank` **automatically detects** your request format and returns the matching response:
  - Use `documents` → Get native response with `relevance_score` and `document` object
  - Use `texts` → Get HuggingFace response with `score` and `text` string
- `/reranking` always returns HuggingFace format regardless of input
- Both `top_n` and `top_k` work as aliases on either endpoint
- **Fix:** This solves the "'str' object has no attribute get" error by ensuring clients receive the expected response format

### Example Responses

**Native format request** (`documents` field):
```json
{
  "results": [
    {"index": 0, "relevance_score": 0.95, "document": {"text": "..."}}
  ],
  "model": "..."
}
```

**HuggingFace format request** (`texts` field):
```json
{
  "results": [
    {"index": 0, "score": 0.95, "text": "..."}
  ],
  "model": "..."
}
```

## Supported Models

| Model | Description |
|-------|-------------|
| `BAAI/bge-reranker-v2-m3` | Multilingual reranker (default) |
| `BAAI/bge-reranker-large` | English reranker |
| `BAAI/bge-reranker-base` | Smaller English reranker |
| `Qwen3-reranker` | Qwen-based reranker models |

## Configuration

The service supports multiple configuration methods with the following priority:

1. **Environment Variables** (highest priority)
2. **config.yml** - YAML configuration file
3. **.env file** - Environment file
4. **Default values** (lowest priority)

### Using config.yml (Recommended)

```yaml
server:
  host: 0.0.0.0
  port: 8000
  
model:
  name: BAAI/bge-reranker-v2-m3
  cache_dir: ./models

device:
  name: cuda  # cuda, mps, cpu, or null for auto-detect
```

Set the config file path with:
```bash
export RERANKER_CONFIG_PATH=./config.yml
```

### Using Environment Variables

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
| [Concurrency](docs/concurrency.md) | How batching works and which knobs to tune |
| [LiteLLM Integration](docs/integrations/litellm.md) | Wire this service into a LiteLLM proxy |
| [Operations](docs/operations.md) | Production runbook: supervisord, alerts, nginx, tuning |
| [Admin UI](docs/admin-ui.md) | Local password-gated dashboard: GPU quota, queue, config, logs |
| [Benchmark (4B)](docs/benchmark-qwen3-4b.md) | Measured throughput/latency of Qwen3-Reranker-4B on an RTX 4070 Ti SUPER + tuning levers |
| [Load Balancer](docs/load-balancer.md) | Multi-backend routing setup |
| [Development](docs/development.md) | Dev setup, testing, project structure |
| [Deployment](docs/deployment.md) | Docker, offline mode, troubleshooting |

## License

MIT License
