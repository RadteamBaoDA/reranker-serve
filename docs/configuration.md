# Configuration Reference

This document details all configuration options for the Reranker Service.

## Environment Variables

Configuration is done through environment variables with the `RERANKER_` prefix.

### Server Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `RERANKER_HOST` | Server host | `0.0.0.0` |
| `RERANKER_PORT` | Server port | `8000` |
| `RERANKER_WORKERS` | Number of uvicorn workers | `1` |
| `RERANKER_API_KEY` | API key for authentication | `None` |
| `RERANKER_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO` |
| `RERANKER_JSON_LOGS` | Enable JSON logs for production | `false` |

### Model Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `RERANKER_MODEL_NAME` | Model name or HuggingFace ID | `BAAI/bge-reranker-v2-m3` |
| `RERANKER_MODEL_PATH` | Local model path | `None` |
| `RERANKER_MODEL_CACHE_DIR` | Model cache directory | `./models` |
| `RERANKER_USE_OFFLINE_MODE` | Offline mode (no downloads) | `false` |
| `RERANKER_MAX_LENGTH` | Maximum sequence length | `512` |
| `RERANKER_BATCH_SIZE` | Inference batch size | `32` |
| `RERANKER_DEVICE` | Device: cuda, mps, cpu | Auto-detect |
| `RERANKER_USE_FP16` | Use FP16 precision | `true` |

### Async Engine Configuration (vLLM-inspired)

| Variable | Description | Default |
|----------|-------------|---------|
| `RERANKER_ENABLE_ASYNC_ENGINE` | Enable async engine for concurrent requests | `true` |
| `RERANKER_MAX_CONCURRENT_BATCHES` | Max batches processing simultaneously | `2` |
| `RERANKER_INFERENCE_THREADS` | Thread pool size for inference | `1` |
| `RERANKER_MAX_BATCH_SIZE` | Max requests per batch | `32` |
| `RERANKER_MAX_BATCH_PAIRS` | Max query-doc pairs per batch | `1024` |
| `RERANKER_BATCH_WAIT_TIMEOUT` | Wait time (seconds) to batch requests | `0.01` |
| `RERANKER_MAX_QUEUE_SIZE` | Max pending requests in queue | `1000` |
| `RERANKER_REQUEST_TIMEOUT` | Request timeout in seconds | `60.0` |

### Load Balancer Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `RERANKER_LOAD_BALANCER_ENABLED` | Enable load balancer mode | `false` |
| `RERANKER_CONFIG_PATH` | Path to YAML config file | `./reranker_config.yaml` |

## Example `.env` File

```env
# Server
RERANKER_PORT=8000
RERANKER_LOG_LEVEL=INFO

# Model
RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3
RERANKER_DEVICE=cuda

# High-performance async settings
RERANKER_ENABLE_ASYNC_ENGINE=true
RERANKER_MAX_CONCURRENT_BATCHES=4
RERANKER_MAX_BATCH_PAIRS=2048
RERANKER_BATCH_WAIT_TIMEOUT=0.005
```

## YAML Configuration (Load Balancer)

For load balancer configuration, see [load-balancer.md](load-balancer.md).

## Structured Logging

The service uses [structlog](https://www.structlog.org/) for structured, machine-readable logging.

### Console Output (Default)

```
2025-11-29T09:35:13.329710Z [info     ] starting_reranker_service      model=BAAI/bge-reranker-v2-m3 device=cuda
```

### JSON Output (Production)

Enable JSON logs for log aggregation systems (ELK, Splunk, etc.):

```bash
export RERANKER_JSON_LOGS=true
```

Output:
```json
{"event": "starting_reranker_service", "model": "BAAI/bge-reranker-v2-m3", "device": "cuda", "level": "info", "timestamp": "2025-11-29T09:35:13.329710Z"}
```

### Request Context

Each HTTP request automatically includes:
- `request_id`: Unique request identifier (also in response headers as `X-Request-ID`)
- `method`: HTTP method
- `path`: Request path
- `client_ip`: Client IP address
