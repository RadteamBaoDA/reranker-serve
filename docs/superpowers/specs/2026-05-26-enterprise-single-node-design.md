# Enterprise single-node hardening — design

**Date:** 2026-05-26
**Status:** Approved (pending spec review)
**Scope:** Wrap the existing Phase 1+2 reranker engine in the minimum enterprise scaffolding it needs to be operated as a production internal tool on a single bare-metal box at 10–50 RPS.

## Context

The Phase 1/2 work delivered a fast, well-instrumented reranker: dynamic batching, real per-percentile metrics, MPS hardening, multi-arch Docker, LiteLLM integration, and a startup device probe. What is missing is the operational shell around it: an HTTPS edge, a Prometheus scrape target, distributed traces, well-behaved backpressure, and graceful shutdown. Those are the four things that turn "fast code" into "a service you can put on PagerDuty."

The constraints are deliberately small:

- **Scale:** 10–50 RPS on 1–2 GPUs (internal tool, single team).
- **Deployment:** bare metal, supervisord. No Kubernetes, no fleet.
- **HA target:** ~99% — single node + supervisord auto-restart. Hardware failure is an accepted outage.
- **Auth:** the existing single shared bearer token (`RERANKER_API_KEY`). Network segmentation handles the rest.
- **Observability:** Prometheus `/metrics`, OpenTelemetry OTLP traces, structured JSON logs (already shipped), explicit operator-actionable counters.

No multi-tenancy, no cross-region, no service mesh.

## Goals

1. Add a TLS-terminating HTTPS edge in front of the FastAPI app without adding a second process to the reranker itself.
2. Convert the existing internal `/stats` numbers into a Prometheus `/metrics` exposition that operators can scrape.
3. Emit OTel traces so the reranker shows up in the larger RAG pipeline's traces.
4. Replace the existing "queue full → HTTP 500" path with proper backpressure (503 + `Retry-After`).
5. Make supervisord-managed shutdown actually graceful — in-flight requests drain before uvicorn exits.

## Non-goals

- Multi-node HA, autoscaling, sharding.
- Per-tenant identities, RBAC, audit logs.
- Replacing the dynamic batcher with continuous batching (rerankers are single-pass; not applicable).
- Migrating from supervisord to systemd or Kubernetes.
- Adding a database, message queue, or persistent state.

## Architecture

```
                                 Client (internal RAG service)
                                        │  HTTPS
                                        ▼
                              ┌──────────────────────┐
                              │  nginx  (systemd)    │   TLS termination
                              │                      │   limit_req per IP
                              │                      │   limit_conn for slow clients
                              └──────────┬───────────┘
                                         │  HTTP localhost:8000
                                         ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │  reranker-serve  (supervisord, uvicorn --workers 1)                    │
   │                                                                        │
   │   FastAPI routes (existing)                                            │
   │   ├── /rerank, /v1/rerank, …  (queue-full → 503 Retry-After)           │
   │   ├── /info, /stats          (existing + device_profile)               │
   │   ├── /metrics               (NEW — Prometheus exposition)             │
   │   └── /ready /live           (existing)                                │
   │                                                                        │
   │   Observability middleware (NEW)                                       │
   │   ├── prometheus_client      (histograms + counters)                   │
   │   ├── OpenTelemetry          (FastAPI spans → OTLP)                    │
   │   └── structlog              (already shipped)                         │
   │                                                                        │
   │   AsyncRerankerEngine        (Phase 1+2 — unchanged)                   │
   │   └── Graceful drain on SIGTERM (NEW — supervisord stopsignal hook)    │
   └────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
                               OTel collector + Prometheus
                               (existing infrastructure, out of scope)
```

Two components are new in this design: **nginx** (already on every linux box) and the **observability surface inside the FastAPI app**. Everything else is wiring.

## Request lifecycle

1. Client opens HTTPS to nginx. nginx applies `limit_req` (per-IP RPS cap) and `limit_conn` (slow-client cap), terminates TLS, forwards to `127.0.0.1:8000` with `X-Real-IP` and `X-Forwarded-For`.
2. Uvicorn passes the request to FastAPI. OTel `FastAPIInstrumentor` opens a server span. structlog binds `request_id`, `client_ip`, `method`, `path`.
3. Auth middleware compares the `Authorization: Bearer …` header to `RERANKER_API_KEY`. Missing or wrong → 401.
4. Route handler builds `RerankRequest` and calls `engine.rerank(...)`.
5. `RequestQueue.add_request()` either accepts the request (returns a future) or raises. If it raises with the queue-full message, the route returns **HTTP 503 with `Retry-After: 1`** instead of the current 500. `request_timeout` exhaustion → 504.
6. The fired batch task picks it up under the Phase 1 fire-and-forget dispatch; semaphore=1 gates GPU contention.
7. `_process_batch` opens an OTel child span around the executor call. Prometheus histograms record queue wait, inference latency, batch size, and pairs.
8. Result returned. Server span closed. `reranker_requests_total{route,status}` incremented.

On **SIGTERM** (supervisord stop, or operator restart):

- A handler flips a `_shutting_down` flag.
- New requests are rejected with 503 immediately so nginx stops sending traffic.
- Existing in-flight batches are awaited (already implemented in `Engine.stop()` via `_inflight_batches.gather`).
- Once the queue drains or `stopwaitsecs` expires, uvicorn exits.
- supervisord conf: `stopsignal=TERM`, `stopwaitsecs=70` (≥ `request_timeout`).

## Observability surface

### Prometheus `/metrics`

Mounted via `prometheus_client.make_asgi_app()` at `/metrics`. Populated by:

- A request middleware that records `Histogram` observations for total request duration, plus a `Counter` increment with `{route, status}` labels.
- A periodic snapshot (every 5 s) from `engine.get_stats()` into `Gauge` objects for batch occupancy, pending requests, in-flight batches, semaphore-available.
- Direct increments on specific events: `reranker_queue_full_total`, `reranker_mps_fallback_total`, `reranker_request_timeout_total`, `reranker_batch_processing_failed_total`.

Full metric family list:

| Name | Type | Labels | Source |
|---|---|---|---|
| `reranker_requests_total` | Counter | `route`, `status` | middleware |
| `reranker_request_duration_seconds` | Histogram | `route` | middleware |
| `reranker_queue_wait_seconds` | Histogram | — | engine on completion |
| `reranker_inference_seconds` | Histogram | — | engine in `_process_batch` |
| `reranker_batch_size` | Histogram | — | engine in `_process_batch` |
| `reranker_batch_occupancy_ratio` | Gauge | — | periodic snapshot |
| `reranker_pending_requests` | Gauge | — | periodic snapshot |
| `reranker_inflight_batches` | Gauge | — | periodic snapshot |
| `reranker_semaphore_available` | Gauge | — | periodic snapshot |
| `reranker_queue_full_total` | Counter | — | direct increment |
| `reranker_request_timeout_total` | Counter | — | direct increment |
| `reranker_mps_fallback_total` | Counter | — | direct increment |
| `reranker_batch_processing_failed_total` | Counter | — | direct increment |

### OpenTelemetry tracing

- `opentelemetry-instrumentation-fastapi` for server spans.
- A manual span in `_process_batch` with attributes `batch_size`, `pairs`, `device`, `queue_wait_ms`, `inference_ms`, `inflight_batches`.
- OTLP/HTTP exporter configured via env vars: `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME=reranker-serve`.
- Sampling: default tracestate; operator can override with `OTEL_TRACES_SAMPLER` and `OTEL_TRACES_SAMPLER_ARG`.

Crucially: if the OTel collector is unreachable, the SDK drops spans silently. The reranker keeps serving.

### Structured JSON logs

Unchanged — `structlog` is configured and rotates daily. Logs already include `request_id`, `client_ip`, `method`, `path`.

### Operator-actionable counters

The four `*_total` counters above (queue_full, request_timeout, mps_fallback, batch_processing_failed) plus the `*_duration_seconds` histograms give ops a clean alert palette. Example Alertmanager rule (documented in `docs/operations.md`, not shipped as runtime config):

```yaml
- alert: RerankerQueueFull
  expr: rate(reranker_queue_full_total[5m]) > 0
  for: 2m
  annotations:
    summary: "Reranker queue saturated — clients are seeing 503s"
```

## Configuration changes

### Observability switches — env-only

**Every observability toggle is controlled exclusively by environment variables.** Operators must be able to flip observability on or off without editing a versioned config file, and a single `env | grep RERANKER_` (plus the OTel-standard `OTEL_*` vars) must tell them precisely what is live. None of these switches are exposed through `config.yml`; if `config.yml` contains a matching key it is **ignored**.

| Env var | Default | Effect |
|---|---|---|
| `RERANKER_EXPOSE_PROMETHEUS_METRICS` | `false` | Mounts `/metrics`, starts the periodic engine-stats snapshot, registers all `reranker_*` metric families. When `false`, `prometheus_client` is not imported. |
| `RERANKER_PROMETHEUS_SNAPSHOT_INTERVAL_SECONDS` | `5` | Period at which the snapshot task copies `engine.get_stats()` into the Prometheus gauges. Ignored unless the switch above is on. |
| `RERANKER_ENABLE_OTEL` | `false` | Initializes the OTel SDK + `FastAPIInstrumentor`. When `false`, no `opentelemetry` packages are imported. |
| `RERANKER_OTEL_BATCH_SPAN` | `true` | Whether to emit the per-batch child span inside `_process_batch`. Lets operators dial back trace volume without disabling server spans. Ignored unless `RERANKER_ENABLE_OTEL=true`. |
| `RERANKER_LOG_LEVEL` | `info` | Existing — controls structlog level. Listed here so operators see the full observability picture in one place. |
| `RERANKER_JSON_LOGS` | `false` | Existing — JSON vs. console-friendly log format. |
| OTel-standard vars (`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`, `OTEL_TRACES_SAMPLER`, `OTEL_TRACES_SAMPLER_ARG`, …) | per SDK | Read directly by the OTel SDK; we do not shadow them. Only consulted when `RERANKER_ENABLE_OTEL=true`. |

Behavior summary: with no env vars set, the service runs exactly as it does today — no `/metrics` endpoint, no traces, no new imports. Each switch can be flipped independently without restarting any dependency other than the reranker process itself.

### Non-observability settings

These follow the standard pydantic-settings precedence (env var > YAML > default):

| Setting | Env var | Default | Effect |
|---|---|---|---|
| `queue_full_status_code` | `RERANKER_QUEUE_FULL_STATUS_CODE` | `503` | Status returned when `RequestQueue` rejects. |
| `graceful_shutdown_timeout` | `RERANKER_GRACEFUL_SHUTDOWN_TIMEOUT` | `60` | Seconds to wait for in-flight drain on SIGTERM. |

Phase-1 defaults (`max_concurrent_batches=1`, `batch_wait_timeout=0.005`, `enable_device_probe=true`) are unchanged.

## File-level plan

**New**
- `src/observability/__init__.py`, `src/observability/prometheus.py`, `src/observability/otel.py`, `src/observability/shutdown.py`
- `examples/nginx.reranker.conf` — drop-in config with TLS + rate-limit
- `tests/test_metrics.py`, `tests/test_backpressure.py`, `tests/test_otel.py`, `tests/test_graceful_shutdown.py`
- `docs/operations.md` — runbook: alert rules, nginx tuning, supervisord drain semantics

**Modified**
- `src/main.py` — wire up Prometheus + OTel at app create
- `src/api/routes.py` — 503 with `Retry-After` on queue full
- `src/engine/async_engine.py` — emit Prometheus observations from `_process_batch`; expose `_shutting_down` flag
- `src/engine/request_queue.py` — emit `queue_full_total` increment when `add_request` raises
- `src/engine/handlers/cross_encoder.py`, `src/models/qwen3_reranker.py` — emit `mps_fallback_total` on the existing fallback path
- `supervisord.conf` — `stopsignal=TERM`, `stopwaitsecs=70`
- `requirements.txt`, `pyproject.toml` — add `prometheus-client`, `opentelemetry-*`

## Testing

- **Unit:** `test_metrics.py` asserts every family in the table above appears in `/metrics` and that one request increments the right `{route,status}` cell. `test_otel.py` uses `InMemorySpanExporter` to assert one server span + one batch span per request, with the expected attributes. `test_backpressure.py` constructs an engine with `max_queue_size=2, request_timeout=0.05` and verifies 503 + `Retry-After`. `test_graceful_shutdown.py` sends SIGTERM mid-request and asserts the response completes with 200.
- **Integration:** existing pytest suite must stay green.
- **Manual:** spin nginx in front of a local reranker, hit it through HTTPS, scrape `/metrics`, verify the traces show up in the OTel collector.

## Rollout

Single bare-metal box, no fleet, so the rollout is sequential:

1. Land code with all four feature flags **off** in `main`. Existing behavior is preserved.
2. Flip `queue_full_status_code=503`. Smoke-test backpressure with a synthetic burst. Watch for client retry storms.
3. Flip `expose_prometheus_metrics=true`. Confirm the scrape target is healthy in Prometheus, dashboards populate.
4. Flip `enable_otel=true`. Confirm spans appear in the OTel collector; verify sampling honored.
5. Flip `graceful_shutdown_timeout=60` and update supervisord conf to `stopsignal=TERM`, `stopwaitsecs=70`. Send a `supervisorctl restart`, confirm zero error responses to in-flight clients.
6. After 24h of clean prod traffic, remove the flags and bake the new defaults.

## Open questions

None — all material decisions were elicited.

## Acceptance criteria

- `pytest tests/ -x` green, including the four new test files.
- `curl https://reranker.example/metrics` returns Prometheus-formatted text with every family above present.
- `supervisorctl restart reranker` produces zero non-200 status codes for requests already in flight at SIGTERM time, given they complete within `graceful_shutdown_timeout`.
- Filling the queue produces a 503 with `Retry-After: 1`, not a 500.
- An OTel collector configured via env vars receives one server span + one batch span per `/rerank` call, with `batch_size`, `pairs`, `device`, `inference_ms` attributes populated.
