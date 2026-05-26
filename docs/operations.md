# Operations Runbook

## Process supervision

The service runs under supervisord. Restart with `supervisorctl restart reranker`. SIGTERM triggers a graceful drain: new requests are rejected with HTTP 503, in-flight requests complete (up to `RERANKER_GRACEFUL_SHUTDOWN_TIMEOUT` seconds), then uvicorn exits.

```bash
supervisorctl status reranker
supervisorctl restart reranker
supervisorctl tail -f reranker  # streams stdout/stderr
```

## TLS edge

nginx terminates TLS and applies per-IP rate-limit (`limit_req zone=ranker_rps rate=50r/s burst=20`) and per-IP connection cap (`limit_conn 16`). Use `examples/nginx.reranker.conf` as the starting point.

## Observability

### Enable

All observability is env-only:

```bash
RERANKER_EXPOSE_PROMETHEUS_METRICS=true
RERANKER_ENABLE_OTEL=true
OTEL_EXPORTER_OTLP_ENDPOINT=https://otel.internal/v1/traces
OTEL_SERVICE_NAME=reranker-serve
```

Restart the service. `curl http://localhost:8000/metrics` should now return Prometheus text. Traces appear in the OTel collector.

### Prometheus scrape

```yaml
scrape_configs:
  - job_name: reranker
    scrape_interval: 15s
    static_configs:
      - targets: ["reranker.host.internal:8000"]
    metrics_path: /metrics
```

Scrape from inside the firewall only — `/metrics` is intentionally not exposed through nginx.

### Suggested alerts

```yaml
groups:
  - name: reranker
    rules:
      - alert: RerankerQueueFull
        expr: rate(reranker_queue_full_total[5m]) > 0
        for: 2m
        annotations:
          summary: "Queue saturated — clients seeing HTTP 503"

      - alert: RerankerMpsFallback
        expr: increase(reranker_mps_fallback_total[1h]) > 0
        annotations:
          summary: "MPS→CPU fallback fired — investigate batch sizes"

      - alert: RerankerLatencyHigh
        expr: histogram_quantile(0.95, rate(reranker_request_duration_seconds_bucket[5m])) > 1.0
        for: 5m
        annotations:
          summary: "p95 request latency above 1s for 5 minutes"
```

## Tuning

| Symptom | Probable cause | Lever |
|---|---|---|
| `batch_occupancy_ratio < 0.5` | Traffic too thin to amortize batching | Raise `RERANKER_BATCH_WAIT_TIMEOUT` |
| Frequent `queue_full_total` | Producer faster than GPU | Raise `RERANKER_MAX_QUEUE_SIZE` or scale up |
| `inference_seconds` p95 climbing | VRAM pressure or thermal throttle | Check `nvidia-smi`, lower `RERANKER_MAX_BATCH_PAIRS` |
| `queue_wait_p95_ms` growing while inference flat | Behind the curve | Raise `RERANKER_MAX_BATCH_PAIRS` |

See `docs/concurrency.md` for full discussion of the knobs.
