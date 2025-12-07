# Load Balancer Guide

The service includes a LiteLLM-style load balancer that routes requests across multiple reranker backends.

## Enable Load Balancer

1. **Set environment variables:**

```bash
export RERANKER_LOAD_BALANCER_ENABLED=true
export RERANKER_CONFIG_PATH=./reranker_config.yaml
```

2. **Create YAML configuration** (`reranker_config.yaml`):

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
  routing_strategy: "weighted-random"
  num_retries: 3
  retry_delay: 1.0
  fallback_to_local: true
  health_check_interval: 60
```

## Routing Strategies

| Strategy | Description |
|----------|-------------|
| `weighted-random` | Random selection weighted by model weights (default) |
| `round-robin` | Sequential rotation through backends |
| `least-busy` | Route to backend with fewest active requests |
| `latency-based-routing` | Route to backend with lowest average latency |
| `priority-failover` | Use highest priority backend, failover on error |
| `simple-shuffle` | Pure random selection |

## Features

- **Health Checks**: Automatic health monitoring of backends
- **Circuit Breaker**: Unhealthy backends are temporarily removed
- **Automatic Retries**: Failed requests retry on other backends
- **Rate Limiting**: Per-backend request rate limits (rpm, tpm)
- **Request Logging**: Detailed logging of routed requests
- **Fallback to Local**: Use local model if all backends fail

## Monitor Load Balancer

```bash
# Check load balancer statistics
curl http://localhost:8000/lb/stats
```

## Proxy Bypass

The service automatically configures proxy bypass for internal communications:

- Load balancer requests to backends bypass system proxy
- `NO_PROXY` environment variable is automatically configured
- Includes: `localhost`, `127.0.0.1`, `::1`, `0.0.0.0`, `.local`, `.internal`
