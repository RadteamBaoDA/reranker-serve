# Configuration Guide

This guide explains all configuration options for the Reranker Service and how to use them.

## Configuration Priority

The service loads configuration from multiple sources with the following priority (highest to lowest):

1. **Environment Variables** - Set directly in your shell or process
2. **config.yml** - YAML configuration file (set via `RERANKER_CONFIG_PATH`)
3. **.env file** - Environment file in the project root
4. **Default values** - Built-in defaults

## Configuration Methods

### Method 1: config.yml (Recommended)

Create a `config.yml` file in your project root:

```yaml
# Server Configuration
server:
  host: 0.0.0.0
  port: 8000
  workers: 1
  reload: false

# Model Configuration
model:
  name: Qwen/Qwen3-Reranker-0.6B
  path: null  # or specify local path
  cache_dir: ./models
  use_offline_mode: false

# Inference Configuration
inference:
  max_length: 512
  batch_size: 32
  normalize_scores: true

# Device Configuration
device:
  name: null  # null = auto-detect, or cuda/mps/cpu
  force_cpu_only: false
  use_fp16: true
  mps_fallback_to_cpu: true

# API Configuration
api:
  key: null  # Set for authentication
  enable_cors: true
  cors_origins: "*"

# Async Engine Configuration
async_engine:
  enabled: true
  max_concurrent_batches: 2
  inference_threads: 1
  max_batch_size: 32
  max_batch_pairs: 1024
  batch_wait_timeout: 0.01
  max_queue_size: 1000
  request_timeout: 60.0

# Load Balancer Configuration
load_balancer:
  enabled: false
  config_path: ./reranker_config.yaml

# HTTP Configuration
http:
  trust_env: false

# Logging Configuration
logging:
  level: info
  json_logs: false
  log_dir: ./logs
  retention_days: 7
  max_bytes: 10485760
  backup_count: 5
```

**Usage:**

```bash
# Set config path (optional - defaults to ./config.yml)
export RERANKER_CONFIG_PATH=./config.yml

# Run the service
./run.sh
```

### Method 2: .env File

Create a `.env` file in your project root:

```env
RERANKER_HOST=0.0.0.0
RERANKER_PORT=8000
RERANKER_MODEL_NAME=Qwen/Qwen3-Reranker-0.6B
RERANKER_DEVICE=cuda
```

### Method 3: Environment Variables

```bash
export RERANKER_HOST=0.0.0.0
export RERANKER_PORT=8000
export RERANKER_MODEL_NAME=Qwen/Qwen3-Reranker-0.6B
./run.sh
```

### Method 4: Command Line (via uvicorn)

```bash
RERANKER_MODEL_NAME=Qwen/Qwen3-Reranker-0.6B uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## Configuration Options

### Server Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `host` / `RERANKER_HOST` | string | `0.0.0.0` | Server bind address |
| `port` / `RERANKER_PORT` | int | `8000` | Server port |
| `workers` / `RERANKER_WORKERS` | int | `1` | Number of uvicorn workers |
| `reload` / `RERANKER_RELOAD` | bool | `false` | Enable auto-reload (dev mode) |

### Model Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model.name` / `RERANKER_MODEL_NAME` | string | `BAAI/bge-reranker-v2-m3` | Model name or HuggingFace ID |
| `model.path` / `RERANKER_MODEL_PATH` | string | `null` | Local path to load model from |
| `model.cache_dir` / `RERANKER_MODEL_CACHE_DIR` | string | `./models` | Model cache directory |
| `model.use_offline_mode` / `RERANKER_USE_OFFLINE_MODE` | bool | `false` | Offline mode (no downloads) |

**Supported Models:**
- `BAAI/bge-reranker-v2-m3` - Multilingual (recommended)
- `BAAI/bge-reranker-large` - English
- `BAAI/bge-reranker-base` - Smaller/faster
- `Qwen/Qwen3-Reranker-0.6B` - Qwen3-based
- `Qwen/Qwen3-Reranker-4B` - Larger Qwen3
- `Qwen/Qwen3-Reranker-8B` - Best quality Qwen3

### Inference Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `inference.max_length` / `RERANKER_MAX_LENGTH` | int | `512` | Maximum sequence length |
| `inference.batch_size` / `RERANKER_BATCH_SIZE` | int | `32` | Batch size for inference |
| `inference.normalize_scores` / `RERANKER_NORMALIZE_SCORES` | bool | `true` | Normalize scores to 0-1 |

### Device Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `device.name` / `RERANKER_DEVICE` | string | `null` | Device: cuda/mps/cpu/null (auto) |
| `device.force_cpu_only` / `RERANKER_FORCE_CPU_ONLY` | bool | `false` | Force CPU-only mode |
| `device.use_fp16` / `RERANKER_USE_FP16` | bool | `true` | Use FP16 precision |
| `device.mps_fallback_to_cpu` / `RERANKER_MPS_FALLBACK_TO_CPU` | bool | `true` | Fallback to CPU on MPS errors |

### API Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `api.key` / `RERANKER_API_KEY` | string | `null` | API key for authentication |
| `api.enable_cors` / `RERANKER_ENABLE_CORS` | bool | `true` | Enable CORS |
| `api.cors_origins` / `RERANKER_CORS_ORIGINS` | string | `*` | Allowed CORS origins |

### Async Engine Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `async_engine.enabled` / `RERANKER_ENABLE_ASYNC_ENGINE` | bool | `true` | Enable async engine |
| `async_engine.max_concurrent_batches` / `RERANKER_MAX_CONCURRENT_BATCHES` | int | `2` | Max concurrent batches |
| `async_engine.inference_threads` / `RERANKER_INFERENCE_THREADS` | int | `1` | Model inference threads |
| `async_engine.max_batch_size` / `RERANKER_MAX_BATCH_SIZE` | int | `32` | Max requests per batch |
| `async_engine.max_batch_pairs` / `RERANKER_MAX_BATCH_PAIRS` | int | `1024` | Max query-doc pairs per batch |
| `async_engine.batch_wait_timeout` / `RERANKER_BATCH_WAIT_TIMEOUT` | float | `0.01` | Batch accumulation timeout (seconds) |
| `async_engine.max_queue_size` / `RERANKER_MAX_QUEUE_SIZE` | int | `1000` | Max pending requests |
| `async_engine.request_timeout` / `RERANKER_REQUEST_TIMEOUT` | float | `60.0` | Request timeout (seconds) |

### Load Balancer Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `load_balancer.enabled` / `RERANKER_LOAD_BALANCER_ENABLED` | bool | `false` | Enable load balancer |
| `load_balancer.config_path` / `RERANKER_CONFIG_PATH` | string | `./reranker_config.yaml` | Load balancer config file |

### HTTP Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `http.trust_env` / `RERANKER_TRUST_ENV` | bool | `false` | Use HTTP_PROXY/HTTPS_PROXY |

### Logging Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `logging.level` / `RERANKER_LOG_LEVEL` | string | `info` | Log level (debug/info/warning/error) |
| `logging.json_logs` / `RERANKER_JSON_LOGS` | bool | `false` | Output JSON logs |
| `logging.log_dir` / `RERANKER_LOG_DIR` | string | `./logs` | Log directory |
| `logging.retention_days` / `RERANKER_LOG_RETENTION_DAYS` | int | `7` | Days to keep logs |
| `logging.max_bytes` / `RERANKER_LOG_MAX_BYTES` | int | `10485760` | Max log file size (10MB) |
| `logging.backup_count` / `RERANKER_LOG_BACKUP_COUNT` | int | `5` | Number of backup log files |

## Testing Configuration

Test your configuration:

```bash
python test_config_loading.py
```

This will display all loaded configuration values.

## Example Configurations

### Production (GPU Server)

```yaml
server:
  host: 0.0.0.0
  port: 8000
  workers: 4

model:
  name: BAAI/bge-reranker-v2-m3
  cache_dir: /data/models

device:
  name: cuda
  use_fp16: true

async_engine:
  enabled: true
  max_concurrent_batches: 4
  max_batch_size: 64

logging:
  level: info
  json_logs: true
```

### Development (Local)

```yaml
server:
  host: 127.0.0.1
  port: 8000
  reload: true

model:
  name: BAAI/bge-reranker-base

device:
  name: cpu

logging:
  level: debug
```

### High Throughput (Multiple GPUs)

```yaml
server:
  workers: 8

async_engine:
  max_concurrent_batches: 8
  max_batch_size: 128
  max_batch_pairs: 2048
  inference_threads: 2

device:
  name: cuda
  use_fp16: true
```

## Troubleshooting

### Configuration not loading

1. Check file path: `echo $RERANKER_CONFIG_PATH`
2. Verify YAML syntax: `python -c "import yaml; yaml.safe_load(open('config.yml'))"`
3. Run test: `python test_config_loading.py`

### Environment variables override config.yml

This is expected behavior. To use only config.yml values:
- Remove conflicting variables from `.env`
- Unset environment variables: `unset RERANKER_PORT`

### Priority confusion

Remember the order:
1. `export RERANKER_PORT=9000` (highest)
2. `config.yml` → `server.port: 8000`
3. `.env` → `RERANKER_PORT=7000`
4. Default → `8000` (lowest)
