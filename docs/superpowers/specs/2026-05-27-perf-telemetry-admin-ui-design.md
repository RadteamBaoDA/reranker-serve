# Phase 3 — Performance, telemetry, and admin UI — design

**Date:** 2026-05-27
**Status:** Draft (pending spec review)
**Scope:** Make the existing single-node reranker serve ~50 concurrent interactive-RAG users on one 16 GB Ada GPU with Qwen3-Reranker-4B at p95 < 800 ms, enhance the MPS and CPU paths with the same techniques, harden security, and add a local password-gated admin web UI for live monitoring, config editing, and restart.

## Context

Phase 1/2 delivered the async engine, dynamic request batching, device probe, and multi-platform handlers. An in-progress (uncommitted) "enterprise single-node hardening" layer adds Prometheus `/metrics`, OpenTelemetry, an `Observer` interface, HTTP-503 backpressure, and graceful shutdown (`docs/superpowers/specs/2026-05-26-enterprise-single-node-design.md`). **This phase builds on that observer/stats/backpressure layer — it does not replace it.**

The architecture review found one structural performance bottleneck and several gaps:

- **Cross-request batching is discarded at the handler.** The queue accumulates a multi-request `BatchedRequest`, but `QwenRerankerHandler.predict()` (`src/engine/handlers/qwen.py:27-40`) loops over requests one at a time and calls `Qwen3Reranker.rerank()` per request. 50 concurrent requests become up to 50 sequential GPU forward passes.
- **No GPU/device memory telemetry exists.** `device_probe.py` measures latency only; `/stats` has queue/latency/throughput but no VRAM. The requested "GPU quota remaining" needs new instrumentation.
- **No queue introspection.** `RequestQueue` exposes counts, not the list of running vs waiting requests the UI needs.
- **Security gaps.** CORS is `origins="*"` + `allow_credentials=True` (invalid combo, `src/main.py:331-337`); API key uses non-constant-time `!=` (`src/api/routes.py:147`); `/stats`, `/info`, `/docs` are unauthenticated and leak internals.
- **No web UI** of any kind exists.

## Locked decisions

| Decision | Value |
|---|---|
| Workload | Interactive RAG, latency-sensitive |
| Request size | Medium, ~20–50 docs/request |
| Latency target | p95 < 800 ms at 50 concurrent |
| GPU | NVIDIA Ada, 16 GB (flash-attn-2 + bf16 + FP8 capable) |
| Model | Qwen/Qwen3-Reranker-4B |
| Performance approach | Optimize in-place; keep FastAPI + queue + LB + observability shell |
| Device coverage | CUDA, MPS, and CPU all enhanced (not just CUDA) |
| Quantization default | Opt-in (bf16/fp32 default; FP8 on Ada / int8 on CPU enabled via `RERANKER_QUANTIZATION`) |
| `max_length` default | 256 (overridable) |
| Admin UI scope | Dashboard + edit config + restart |
| Admin UI auth | Single `RERANKER_ADMIN_PASSWORD`, session cookie, no users/roles |
| Admin UI stack | Jinja2 + HTMX + Chart.js, vendored locally, no Node build |
| Restart control | In-process engine reload for hot settings; Restart button (graceful drain → supervisord respawn) for host/port/workers |

## Goals

1. True cross-request batching so concurrent requests collapse into a few padded GPU passes instead of one-per-request.
2. VRAM/memory-aware batch sizing per device.
3. Device-specific enhancement for CUDA (flash-attn-2 + bf16 + opt-in FP8), MPS (SDPA + fp32 + unified-memory-aware sizing), CPU (intra-op threads + opt-in int8).
4. Uniform device-memory telemetry ("quota remaining") and queue introspection (running vs waiting) on all backends.
5. A local, password-gated admin UI for live monitoring, config editing with apply/restart, and log viewing.
6. Close the CORS, constant-time-compare, and unauthenticated-introspection findings.

## Non-goals

- Multi-GPU, multi-node, autoscaling, sharding.
- User accounts, roles, RBAC, audit logs (single shared admin password only).
- Replacing the engine with vLLM/Infinity/TEI (explicitly rejected).
- Continuous/paged-attention batching (rerankers are single-pass).
- Making 4B viable on CPU at 50 concurrent (CPU is a dev/functional path; 0.6B is the CPU-serving recommendation).

## Architecture

```
                     Browser (operator, local/VPN)        Client (RAG service)
                              │ HTTPS, session cookie            │ HTTPS, Bearer
                              ▼                                  ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │  reranker-serve  (FastAPI, uvicorn --workers 1, supervisord)               │
   │                                                                            │
   │   /admin/*  (NEW)                         /rerank, /v1/rerank, …           │
   │   ├── session auth middleware             ├── Bearer auth (constant-time)   │
   │   ├── Dashboard (HTMX + Chart.js)         └── 503 backpressure (existing)   │
   │   ├── Config editor → apply/restart                                        │
   │   ├── Log viewer (SSE tail)               /stats /info  → admin-or-bearer   │
   │   └── /admin/api/{resources,queue,        /metrics      → localhost scrape  │
   │        config,logs}                                                        │
   │                                                                            │
   │   AsyncRerankerEngine                                                      │
   │   ├── flat cross-request batching + length bucketing (NEW)                 │
   │   ├── score_pairs() primitive in handlers (NEW)                            │
   │   ├── queue snapshot: waiting[] + running[] (NEW)                          │
   │   ├── device resource probe: mem used/total/free + util (NEW)              │
   │   └── in-process reload on config apply (NEW)                              │
   │                                                                            │
   │   Observer (existing) → Prometheus gauges incl. device memory (extended)   │
   └──────────────────────────────────────────────────────────────────────────┘
```

---

## Part A — Performance core

### A1. `score_pairs()` primitive + flat cross-request batching

**Primitive.** Add to `Qwen3Reranker`:

```python
def score_pairs(self, pairs: list[tuple[str, str]],
                instruction: str | None = None) -> list[float]:
    """Score a flat list of (query, document) pairs; returns one float per pair,
    in input order. The batching/padding/length-bucketing happens here."""
```

The existing `rerank()` becomes a thin wrapper: build pairs for one query, call `score_pairs`, attach indices, sort, apply `top_k`. The `CrossEncoder` handler gets the analogous primitive so both model families share the batching path.

**Flat batching in the handler.** `QwenRerankerHandler.predict(batch)` no longer loops request-by-request. Instead:

1. Flatten every `(request.query, doc)` across all `batch.requests` into one list, recording `(request_index, doc_index)` for each.
2. Call `score_pairs(flat_pairs)` once.
3. Scatter scores back into per-request result lists using the recorded indices.
4. Per request: build result dicts, sort by score, apply `top_k`/`return_documents` (unchanged output shape).

This is the single change that fixes the 50-CCU bottleneck and is **device-independent** — CUDA, MPS, and CPU all benefit.

### A2. Length bucketing inside `score_pairs()`

Within the flat pair list, sort by tokenized length, form sub-batches up to `max_batch_pairs`, pad each sub-batch to its own max (not the global max), run the forward pass, and unscatter to input order. This removes the padding waste from mixing a 20-token pair with a 256-token pair in the same padded tensor. Pure win on every device; largest relative win on MPS/CPU.

### A3. VRAM/memory-aware batch sizing

Extend `src/engine/device_probe.py` (today latency-only) to also determine a safe `suggested_max_batch_pairs`:

- Run forward passes at increasing token-batch sizes, reading device free memory before/after.
- Stop when projected peak would cross a safety margin (default: keep ≥ 15% device memory free) or when latency-per-pair degrades past the existing 2× rule.
- Expose `suggested_max_batch_pairs` in `device_profile`; operator config caps it (`RERANKER_MAX_BATCH_PAIRS` still wins when pinned).

Memory source per backend (see B1): CUDA `torch.cuda.mem_get_info()`; MPS `torch.mps.recommended_max_memory()` / `current_allocated_memory()`; CPU `psutil.virtual_memory()`.

### A4. Device-specific precision & kernels

| | CUDA (Ada) | MPS | CPU |
|---|---|---|---|
| dtype default | bf16 | fp32 | fp32 |
| Quantization (opt-in) | FP8 e4m3 via `quantization_config` (torchao/transformers), Ada-gated | — (fp16 opt-in w/ auto-fallback) | int8 dynamic (`torch.ao.quantization.quantize_dynamic` on `nn.Linear`) |
| Attention | flash-attn-2 → SDPA fallback | SDPA | SDPA / math |
| Parallelism | 1 batch in-flight | 1 batch in-flight | `torch.set_num_threads(cpu_count)` (intra-op); 1 batch in-flight |
| Memory probe | `cuda.mem_get_info` | `mps.recommended_max_memory` + `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | `psutil` |
| Recovery | — | keep + instrument existing MPS→CPU fallback | — |

`RERANKER_QUANTIZATION` ∈ `{none (default), fp8, int8}`; an unsupported value for the active device logs a warning and falls back to the device default precision. `max_concurrent_batches` stays 1 for cuda/mps; CPU relies on torch intra-op threads, not multiple concurrent batches (which would contend). Default `max_length` becomes 256.

### A5. Expected outcome

- bf16 + flash-attn-2 + flat batching + `max_length=256` → p95 ≈ 0.9–1.4 s at 50 concurrent on Ada.
- `RERANKER_QUANTIZATION=fp8` roughly doubles pairs/pass → p95 < 800 ms reachable.
- MPS: interactive at lower concurrency; CPU: dev/functional. The code path is enhanced for all three.

---

## Part B — Telemetry surface

### B1. Device resource probe — `src/observability/resources.py` (NEW)

```python
def get_resource_stats() -> dict:
    """Uniform across backends:
    {device, backend, mem_used_mb, mem_total_mb, mem_free_mb, used_pct,
     util_pct?, temp_c?, power_w?}  (optional fields when available)."""
```

- CUDA: `torch.cuda.mem_get_info()` for used/total/free; optional `pynvml` for util %, temperature, power (degrades gracefully if `pynvml` absent).
- MPS: `recommended_max_memory()` as total, `current_allocated_memory()` as used.
- CPU: `psutil.virtual_memory()`.

Added to `engine.get_stats()` output under `device_resources`, and to new Prometheus gauges `reranker_device_memory_used_bytes`, `reranker_device_memory_total_bytes`, `reranker_device_utilization_ratio` (populated by the existing snapshot loop in `src/observability/prometheus.py`).

### B2. Queue introspection — running vs waiting

`asyncio.Queue` is not enumerable, so:

- `RequestQueue` maintains an ordered `dict[str, _WaitingEntry]` (`request_id, num_docs, enqueued_at, priority`) updated on `add_request` / `get_batch` / `complete`/`cancel`. Snapshot is a cheap copy.
- `AsyncRerankerEngine` records in-flight batch metadata (`batch_id, request_ids, total_pairs, started_at`) keyed alongside `_inflight_batches`.
- New `engine.get_queue_snapshot()` → `{waiting: [...], running: [...]}`. Exposed at `GET /admin/api/queue`.

### B3. Config snapshot + logs

- `GET /admin/api/config` → effective settings grouped by `config.yml` section; each entry `{value, source: env|yaml|default, needs_restart: bool}`; secrets (`api_key`, `admin_password`) redacted to `"***set***"`/`null`.
- `GET /admin/api/logs/tail?lines=&level=&q=` → recent lines from the rotating files in `settings.log_dir`. `GET /admin/api/logs/stream` → SSE live tail.

---

## Part C — Security hardening

1. **CORS.** In `create_app()`, if `get_cors_origins_list() == ["*"]`, set `allow_credentials=False`. Document that credentialed CORS requires explicit origins. Fixes the invalid wildcard-plus-credentials combo.
2. **Constant-time API key.** Replace `token != settings.api_key` with `hmac.compare_digest`.
3. **Admin auth** — `src/admin/auth.py`:
   - `RERANKER_ADMIN_PASSWORD` (env-only). If unset, `/admin/*` returns 503 "admin UI not configured".
   - `POST /admin/login` compares password with `hmac.compare_digest`; on success sets a signed, **HttpOnly, SameSite=Strict** session cookie (HMAC-signed token, server-side `SECRET_KEY` derived from password + a per-process random salt; expiry configurable, default 12 h).
   - Middleware guards all `/admin/*` (pages and `/admin/api/*`). Unauthenticated → redirect to login (HTML) or 401 (API).
   - Simple in-memory exponential backoff on failed logins per client IP.
4. **Close introspection leaks.** `/stats` and `/info` require admin-session **or** valid bearer. `/metrics` remains localhost/scrape-only (not nginx-proxied, per the enterprise spec). `/docs`, `/redoc`, `/openapi.json` gated by `RERANKER_ENABLE_DOCS` (default `true`; recommend `false` in prod).

---

## Part D — Admin UI

Served by the same FastAPI app under `/admin`. Stack: Jinja2 templates + HTMX (partial refresh) + Chart.js (gauges/sparklines), **all vendored under `src/admin/static/`** (no CDN, offline-safe). Live data: 1 s HTMX polling of `/admin/api/{resources,queue,stats}`; SSE for the log tail.

**Pages**

1. **Login** — single password field → session cookie.
2. **Dashboard** — device quota gauge (used/total/free, +util/temp/power when NVIDIA), throughput (pairs/s, req/s sparklines), latency p50/p95, batch occupancy, 503/timeout counters (last 5 m); **Batches running** table (`batch_id, #requests, pairs, elapsed`) and **Waiting queue** table (`request_id, #docs, waited_ms`).
3. **Config** — form grouped by `config.yml` section; each field shows current value + source badge + needs-restart badge. **Save** writes `config.yml` (pydantic-validated); then **Apply now** (in-process engine reload) for hot settings, or surfaces **Restart required** for host/port/workers.
4. **Logs** — live SSE tail, level filter, text search, download current file.

```
┌─ Reranker Admin ───────────────────────────[ Dashboard | Config | Logs | Logout ]┐
│ Model: Qwen3-Reranker-4B    Device: cuda (Ada)   Engine: ● running   p95: 740ms  │
│  GPU QUOTA                        THROUGHPUT                  LATENCY             │
│  ███████░░ 11.4 / 16.0 GB         pairs/s ▁▂▄▆█▆▅▄           p50 310 ms           │
│  71%  util 83%  54°C  88W         req/s   ▁▁▂▃▅▄▃▂           p95 740 ms           │
│  BATCHES RUNNING (1)                      WAITING QUEUE (6)                       │
│  b-1287  4 reqs  142 pairs  180ms         r-9f2… 32 docs  90ms   …                │
└───────────────────────────────────────────────────────────────────────────────┘
```

**Config apply / restart mechanism**

- `src/admin/config_io.py` maps `config.yml` sections ↔ settings, classifies each setting `hot` vs `needs_restart`. Hot: model/path, device, precision/quantization, batch params, `max_length`, queue sizes, timeouts, log level. Needs-restart: `host`, `port`, `workers`.
- **Apply now** rewrites `config.yml`, rebuilds the `Settings` object, and calls `reset_async_engine()` + `get_async_engine()` to reload the model with new settings. Returns load status to the UI.
- **Restart** triggers the existing graceful drain (`engine.begin_shutdown()` → drain → process exit); supervisord respawns. Endpoint `POST /admin/api/restart` (admin-only).

---

## Configuration changes

New settings in `src/config/settings.py` (env `RERANKER_*`, YAML where sensible):

| Setting | Env | Default | Notes |
|---|---|---|---|
| `max_length` | `RERANKER_MAX_LENGTH` | **256** (was 512) | rerank prompt cap |
| `quantization` | `RERANKER_QUANTIZATION` | `none` | `none\|fp8\|int8`, device-gated |
| `admin_password` | `RERANKER_ADMIN_PASSWORD` | `None` | env-only; unset disables `/admin` |
| `admin_session_ttl_hours` | `RERANKER_ADMIN_SESSION_TTL_HOURS` | `12` | |
| `enable_docs` | `RERANKER_ENABLE_DOCS` | `true` | gate `/docs` `/redoc` `/openapi.json` |
| `cpu_num_threads` | `RERANKER_CPU_NUM_THREADS` | `None` (=cpu_count) | `torch.set_num_threads` |
| `device_mem_safety_margin` | `RERANKER_DEVICE_MEM_SAFETY_MARGIN` | `0.15` | VRAM probe headroom |

`admin_password` is env-only — it is a secret and is never written to `config.yml`. The observability switches remain env-only per the enterprise spec. **All other settings above — including `quantization`, `max_length`, `enable_docs`, and the CPU/VRAM knobs — follow standard env > YAML > default precedence and are editable through the admin config editor** (otherwise the UI could not "manage all config").

## File-level plan

**New**
- `src/observability/resources.py` — device memory/util probe
- `src/admin/__init__.py`, `src/admin/auth.py`, `src/admin/routes.py`, `src/admin/config_io.py`
- `src/admin/templates/{base,login,dashboard,config,logs}.html`
- `src/admin/static/` — vendored htmx.min.js, chart.umd.min.js, app.css, app.js
- `tests/test_flat_batching.py`, `tests/test_resources.py`, `tests/test_queue_snapshot.py`, `tests/test_admin_auth.py`, `tests/test_admin_config_io.py`, `tests/test_security_cors.py`
- `docs/admin-ui.md`

**Modified**
- `src/models/qwen3_reranker.py` — add `score_pairs()`; `rerank()` delegates; quantization/dtype/attention selection; default `max_length=256`
- `src/models/reranker.py` / `src/engine/handlers/cross_encoder.py` — analogous `score_pairs()` + int8 path
- `src/engine/handlers/qwen.py` — flat cross-request batching + scatter
- `src/engine/device_probe.py` — VRAM-aware `suggested_max_batch_pairs`
- `src/engine/request_queue.py` — waiting registry + snapshot
- `src/engine/async_engine.py` — in-flight batch metadata, `get_queue_snapshot()`, `device_resources` in stats, in-process reload helper, CPU thread setup
- `src/observability/prometheus.py` — device-memory gauges in snapshot loop
- `src/api/routes.py` — `hmac.compare_digest`; gate `/stats` `/info` behind admin-or-bearer
- `src/api/health.py` — `device_resources` in `/stats`, `/info`
- `src/main.py` — CORS credentials fix; mount `/admin`; admin middleware; `enable_docs` gating
- `src/config/settings.py` — new settings above
- `config.yml`, `.env.example` — document new keys; set model to `Qwen/Qwen3-Reranker-4B`
- `requirements.txt`, `pyproject.toml` — `jinja2`, `psutil`, `itsdangerous`; extras `admin`, `gpu-metrics` (`pynvml`), `quant` (`torchao`)
- `README.md` — link `docs/admin-ui.md`

## Testing

- **Flat batching:** a `BatchedRequest` with 3 requests of differing doc counts yields per-request results identical (order + scores) to scoring each request alone; assert a single `score_pairs` call covers all pairs (spy/mock).
- **Length bucketing:** mixed-length pairs return scores in input order; padded width per sub-batch ≤ global max.
- **Resources:** `get_resource_stats()` returns the required keys with a mocked CUDA/MPS/CPU backend; missing `pynvml` omits optional keys without error.
- **Queue snapshot:** enqueue N, hold the processor, assert `waiting` lists them in arrival order; once running, they move to `running`.
- **Admin auth:** no password set → `/admin` 503; wrong password → 401 + backoff; correct → cookie set; tampered cookie → 401; `/admin/api/*` rejects unauthenticated.
- **Config IO:** round-trip a `config.yml` edit; `needs_restart` classification correct; secrets redacted in `/admin/api/config`.
- **Security:** `cors_origins="*"` ⇒ `allow_credentials=False`; API key compare uses `compare_digest`; `/stats` 401 without auth, 200 with bearer.
- **Regression:** full existing suite stays green; observability-off path imports nothing new beyond `psutil`/`jinja2` only when `/admin` enabled.

## Rollout

1. Land Part A behind unchanged defaults; benchmark `benchmark_concurrent.py` at 50 concurrent before/after to confirm the batching win.
2. Land Part B telemetry; verify `/stats.device_resources` and new gauges.
3. Land Part C security; confirm CORS + auth + introspection gating.
4. Land Part D admin UI; smoke-test login, dashboard live data, config apply (hot), restart.
5. Set `model = Qwen/Qwen3-Reranker-4B`, `RERANKER_QUANTIZATION=fp8`, validate p95 < 800 ms at 50 concurrent; tune `max_batch_pairs` from the VRAM probe.

## Acceptance criteria

- `benchmark_concurrent.py` at 50 concurrent shows materially higher throughput and lower p95 after A1/A2 than before (same hardware/model).
- With `RERANKER_QUANTIZATION=fp8`, `max_length=256`, Qwen3-Reranker-4B on Ada 16 GB: p95 < 800 ms at 50 concurrent, no OOM.
- `/stats` and `/admin/api/resources` report device memory used/total/free on CUDA, MPS, and CPU.
- `/admin` requires the password; dashboard shows live GPU quota, running batches, and waiting queue; config edit applies hot settings via engine reload; Restart drains gracefully and supervisord respawns.
- CORS no longer emits `*`+credentials; API-key compare is constant-time; `/stats` and `/info` reject unauthenticated requests.
- Full pytest suite green, including new test files.

## Open questions

None — all material decisions elicited and recorded under "Locked decisions".
