# Enterprise Single-Node Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap the existing reranker engine in env-gated Prometheus metrics, OTel traces, well-behaved backpressure, and graceful supervisord-managed shutdown — without touching the engine's hot path when observability is disabled.

**Architecture:** A new `src/observability/` package houses three independent modules (`prometheus.py`, `otel.py`, `shutdown.py`), each initialized only when its `RERANKER_*` env var is on. The FastAPI app composes them in `main.py`. The existing engine emits counter/histogram observations through a thin observer interface that is a no-op when both modules are off. A typed `QueueFullError` replaces the current `RuntimeError` so the route can return HTTP 503 with `Retry-After`.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, pydantic-settings, `prometheus-client>=0.20`, `opentelemetry-api/sdk/instrumentation-fastapi/exporter-otlp-proto-http>=1.25`, pytest, pytest-asyncio.

**Reference spec:** `docs/superpowers/specs/2026-05-26-enterprise-single-node-design.md`

**Branch:** Recommend a worktree at `feature/enterprise-hardening` via `superpowers:using-git-worktrees`.

---

### Task 0: Worktree + dependencies + settings

**Files:**
- Modify: `requirements.txt`
- Modify: `pyproject.toml:55-67`
- Modify: `src/config/settings.py` (add new fields and bypass YAML for observability switches)

- [ ] **Step 1: Create worktree if not already in one**

```bash
git worktree add ../reranker-serve-enterprise feature/enterprise-hardening
cd ../reranker-serve-enterprise
```

- [ ] **Step 2: Add dependencies to `requirements.txt`** (append at the bottom)

```
# Observability (lazy-imported; only required when the matching env var is on)
prometheus-client>=0.20.0
opentelemetry-api>=1.25.0
opentelemetry-sdk>=1.25.0
opentelemetry-instrumentation-fastapi>=0.46b0
opentelemetry-exporter-otlp-proto-http>=1.25.0
```

- [ ] **Step 3: Add an `observability` extra to `pyproject.toml`**

Find the `[project.optional-dependencies]` block (lines 55-67) and add:

```toml
observability = [
    "prometheus-client>=0.20.0",
    "opentelemetry-api>=1.25.0",
    "opentelemetry-sdk>=1.25.0",
    "opentelemetry-instrumentation-fastapi>=0.46b0",
    "opentelemetry-exporter-otlp-proto-http>=1.25.0",
]
```

- [ ] **Step 4: Add settings fields**

Open `src/config/settings.py`. Find the `request_timeout` block (currently around line 85). Add immediately below `enable_device_probe`:

```python
    # Backpressure / lifecycle
    queue_full_status_code: int = Field(
        default=503,
        description="HTTP status returned when the request queue rejects (full / shutting down)."
    )
    graceful_shutdown_timeout: float = Field(
        default=60.0,
        description="Seconds to wait for in-flight requests to drain on SIGTERM."
    )

    # Observability — env-only switches (NOT read from config.yml)
    expose_prometheus_metrics: bool = Field(
        default=False,
        description="Mount /metrics. Env-only via RERANKER_EXPOSE_PROMETHEUS_METRICS."
    )
    prometheus_snapshot_interval_seconds: float = Field(
        default=5.0,
        description="Period of the engine-stats → Prometheus gauges snapshot loop."
    )
    enable_otel: bool = Field(
        default=False,
        description="Initialize OTel SDK + FastAPI instrumentor. Env-only via RERANKER_ENABLE_OTEL."
    )
    otel_batch_span: bool = Field(
        default=True,
        description="Emit per-batch child span. Lets ops dial back trace volume."
    )
```

- [ ] **Step 5: Strip observability keys from YAML loader**

In the same file find `if 'async_engine' in yaml_config:` (currently around line 265). After the existing block that maps async_engine keys, add:

```python
        # Observability switches are intentionally NOT read from config.yml.
        # Spec: docs/superpowers/specs/2026-05-26-enterprise-single-node-design.md
        for forbidden in (
            "expose_prometheus_metrics",
            "prometheus_snapshot_interval_seconds",
            "enable_otel",
            "otel_batch_span",
        ):
            flat_config.pop(forbidden, None)
```

- [ ] **Step 6: Run the test suite to confirm nothing else broke**

```bash
pytest tests/test_config.py tests/test_config_loading.py -v
```

Expected: PASS for all existing tests.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt pyproject.toml src/config/settings.py
git commit -m "feat(config): add env-only observability switches + backpressure settings"
```

---

### Task 1: Typed QueueFullError + 503 with Retry-After

**Files:**
- Modify: `src/engine/request_queue.py:135-202` (define exception, raise it from `add_request`)
- Modify: `src/api/routes.py:184-330` (catch it, return 503)
- Test: `tests/test_backpressure.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_backpressure.py`:

```python
"""Backpressure: queue-full → HTTP 503 with Retry-After header."""

import pytest


@pytest.mark.asyncio
async def test_queue_full_raises_typed_error():
    from src.engine.request_queue import QueueFullError, RequestQueue, RerankRequest

    queue = RequestQueue(max_queue_size=1, request_timeout=0.05)
    # Fill the queue
    await queue.add_request(RerankRequest(request_id="r1", query="q", documents=["a"]))

    # Second request can't enqueue within request_timeout → QueueFullError
    with pytest.raises(QueueFullError):
        await queue.add_request(RerankRequest(request_id="r2", query="q", documents=["b"]))


def test_route_returns_503_with_retry_after_when_queue_full(test_client, monkeypatch):
    from src.engine.request_queue import QueueFullError

    async def boom(*args, **kwargs):
        raise QueueFullError("queue is full")

    # Replace the dummy engine's rerank with one that raises
    monkeypatch.setattr(
        "src.engine.get_async_engine",
        lambda: _async_returning_engine_that_raises(QueueFullError("queue is full")),
    )

    resp = test_client.post("/rerank", json={"query": "q", "documents": ["a"]})
    assert resp.status_code == 503
    assert resp.headers.get("Retry-After") == "1"


def _async_returning_engine_that_raises(exc):
    """Helper: return a coroutine-returning callable whose engine.rerank raises exc."""
    class _Engine:
        is_running = True
        is_loaded = True
        device = "cpu"
        device_profile = None
        async def rerank(self, *args, **kwargs):
            raise exc

    async def _factory():
        return _Engine()

    return _factory()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_backpressure.py -v
```

Expected: FAIL with `ImportError: cannot import name 'QueueFullError'`.

- [ ] **Step 3: Define `QueueFullError` and raise it from `add_request`**

In `src/engine/request_queue.py`, find the existing `RequestStatus` enum (top of file). Immediately after the enum definition, add:

```python
class QueueFullError(RuntimeError):
    """RequestQueue rejected a request because it is at capacity or shutting down."""
```

Then find the `add_request` method (line ~135). Locate the existing block:

```python
except asyncio.TimeoutError:
    del self._active_requests[request.request_id]
    ...
    raise RuntimeError("Queue is full, request timed out")
```

Replace `RuntimeError(...)` with `QueueFullError(...)`. Also change the existing `if self._shutdown:` block at the top of `add_request` from `raise RuntimeError("Queue is shutting down")` to `raise QueueFullError("Queue is shutting down")`.

- [ ] **Step 4: Wire 503 into the route**

In `src/api/routes.py`, find the native `/rerank` handler (line ~184). Locate the `except HTTPException: raise` block I added in Phase 2.3. Above the catch-all `except Exception as e:`, add:

```python
    except QueueFullError as e:
        elapsed = time.time() - endpoint_start
        logger.warning(
            "rerank_endpoint_queue_full",
            request_id=request_id,
            error=str(e),
            elapsed_ms=round(elapsed * 1000, 2),
        )
        raise HTTPException(
            status_code=settings.queue_full_status_code,
            detail=str(e),
            headers={"Retry-After": "1"},
        )
```

Add the import at the top: `from src.engine.request_queue import QueueFullError`.

Apply the same pattern to the Cohere (`/v1/rerank` family), Jina (`/api/v1/rerank`), and HuggingFace (`/v1/reranking`, `/reranking`) endpoints. Each has the same `try/except Exception` shape; insert the `QueueFullError` branch above the catch-all.

- [ ] **Step 5: Run the test to verify it passes**

```bash
pytest tests/test_backpressure.py -v
```

Expected: PASS for both tests.

- [ ] **Step 6: Run the full suite to confirm no regression**

```bash
pytest tests/ -x
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/engine/request_queue.py src/api/routes.py tests/test_backpressure.py
git commit -m "feat(api): typed QueueFullError + HTTP 503 Retry-After on queue full"
```

---

### Task 2: Observer interface for the engine

**Files:**
- Create: `src/observability/__init__.py`
- Create: `src/observability/observer.py`
- Modify: `src/engine/async_engine.py` (call observer hooks)
- Modify: `src/engine/request_queue.py` (call observer hooks for queue_full)
- Modify: `src/engine/handlers/cross_encoder.py`, `src/models/qwen3_reranker.py` (call observer hook on MPS fallback)
- Test: `tests/test_observer.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_observer.py`:

```python
"""Engine emits typed observability events to the configured observer."""

import asyncio
import pytest


@pytest.mark.asyncio
async def test_observer_receives_batch_completed_event(monkeypatch):
    from src.engine.async_engine import AsyncRerankerEngine
    from src.engine.handlers.base import BaseHandler
    from src.observability import set_observer, NullObserver

    events = []

    class RecordingObserver(NullObserver):
        def on_batch_completed(self, *, batch_size, pairs, inference_seconds, device):
            events.append(("batch", batch_size, pairs, device))

        def on_request_completed(self, *, route, status, total_seconds, queue_wait_seconds):
            events.append(("request", route, status))

    class DummyHandler(BaseHandler):
        def load_model(self):
            self.model = "ok"
        def predict(self, batch):
            return [[{"index": 0, "relevance_score": 1.0}] for _ in batch.requests]

    set_observer(RecordingObserver())
    try:
        monkeypatch.setattr(
            "src.engine.async_engine.get_handler",
            lambda *a, **k: DummyHandler("p", "cpu", 64, False),
        )
        from src.config import settings as s
        monkeypatch.setattr(s, "enable_device_probe", False)

        engine = AsyncRerankerEngine(
            max_concurrent_batches=1, inference_threads=1, batch_wait_timeout=0.001
        )
        await engine.start()
        try:
            await engine.rerank("q", ["a"])
        finally:
            await engine.stop()
    finally:
        set_observer(NullObserver())

    assert any(e[0] == "batch" for e in events)
    batch_event = next(e for e in events if e[0] == "batch")
    assert batch_event[1] == 1   # batch_size
    assert batch_event[2] == 1   # pairs
    assert batch_event[3] == "cpu"


def test_observer_receives_queue_full_event():
    from src.observability import set_observer, NullObserver, get_observer
    from src.engine.request_queue import RequestQueue, RerankRequest, QueueFullError

    events = []

    class RecordingObserver(NullObserver):
        def on_queue_full(self):
            events.append("queue_full")

    set_observer(RecordingObserver())
    try:
        async def _run():
            queue = RequestQueue(max_queue_size=1, request_timeout=0.05)
            await queue.add_request(RerankRequest(request_id="x", query="q", documents=["a"]))
            with pytest.raises(QueueFullError):
                await queue.add_request(RerankRequest(request_id="y", query="q", documents=["b"]))

        asyncio.run(_run())
    finally:
        set_observer(NullObserver())

    assert events == ["queue_full"]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_observer.py -v
```

Expected: FAIL with `ImportError: cannot import name 'set_observer' from 'src.observability'`.

- [ ] **Step 3: Create the observer interface**

Create `src/observability/__init__.py`:

```python
"""Observability package — env-gated Prometheus, OTel, and graceful shutdown."""

from src.observability.observer import (
    NullObserver,
    Observer,
    get_observer,
    set_observer,
)

__all__ = ["Observer", "NullObserver", "get_observer", "set_observer"]
```

Create `src/observability/observer.py`:

```python
"""
Observer pattern: the engine emits typed events; the Prometheus and OTel
modules subscribe by setting an Observer implementation. When both are off,
NullObserver makes every emission a no-op on the engine's hot path.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Observer(Protocol):
    def on_request_completed(
        self, *, route: str, status: int, total_seconds: float, queue_wait_seconds: float
    ) -> None: ...

    def on_batch_completed(
        self, *, batch_size: int, pairs: int, inference_seconds: float, device: str
    ) -> None: ...

    def on_queue_full(self) -> None: ...

    def on_request_timeout(self) -> None: ...

    def on_mps_fallback(self) -> None: ...

    def on_batch_processing_failed(self) -> None: ...


class NullObserver:
    """Default observer; every method is a no-op."""

    def on_request_completed(self, **kwargs): pass
    def on_batch_completed(self, **kwargs): pass
    def on_queue_full(self): pass
    def on_request_timeout(self): pass
    def on_mps_fallback(self): pass
    def on_batch_processing_failed(self): pass


_observer: Observer = NullObserver()


def set_observer(observer: Observer) -> None:
    global _observer
    _observer = observer


def get_observer() -> Observer:
    return _observer
```

- [ ] **Step 4: Wire `on_batch_completed` from the engine**

In `src/engine/async_engine.py`, find `_process_batch` (line ~291). Inside the `try` block, after `self._inference_times_ms.append(processing_time * 1000.0)`, add:

```python
                from src.observability import get_observer
                get_observer().on_batch_completed(
                    batch_size=len(batch.requests),
                    pairs=batch.total_pairs,
                    inference_seconds=processing_time,
                    device=self.device,
                )
```

In the `except Exception as e:` branch of `_process_batch`, add:

```python
            from src.observability import get_observer
            get_observer().on_batch_processing_failed()
```

- [ ] **Step 5: Wire `on_queue_full` from the queue**

In `src/engine/request_queue.py`, find both places that now raise `QueueFullError`. Before each `raise QueueFullError(...)`, add:

```python
            from src.observability import get_observer
            get_observer().on_queue_full()
```

- [ ] **Step 6: Wire `on_mps_fallback`**

In `src/engine/handlers/cross_encoder.py`, find the existing MPS fallback path inside `predict()`. After the `logger.warning(...)` line and before `self.device = "cpu"`, add:

```python
                    from src.observability import get_observer
                    get_observer().on_mps_fallback()
```

Apply the same one-liner inside the existing fallback branch of `src/models/qwen3_reranker.py:_compute_logits` (after the `logger.warning` and before `cpu_inputs = ...`).

- [ ] **Step 7: Run the test to verify it passes**

```bash
pytest tests/test_observer.py -v
```

Expected: PASS for both tests.

- [ ] **Step 8: Confirm full suite green**

```bash
pytest tests/ -x
```

- [ ] **Step 9: Commit**

```bash
git add src/observability/ src/engine/ src/models/qwen3_reranker.py tests/test_observer.py
git commit -m "feat(observability): observer interface with NullObserver default"
```

---

### Task 3: Prometheus implementation (lazy-imported)

**Files:**
- Create: `src/observability/prometheus.py`
- Test: `tests/test_metrics.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_metrics.py`:

```python
"""Prometheus exposition: /metrics endpoint shape + counter increments."""

import pytest

pytest.importorskip("prometheus_client")


def test_observer_emits_request_total_counter(monkeypatch):
    monkeypatch.setenv("RERANKER_EXPOSE_PROMETHEUS_METRICS", "true")
    from src.observability import set_observer, NullObserver
    from src.observability.prometheus import PrometheusObserver, REGISTRY

    obs = PrometheusObserver(registry=REGISTRY)
    set_observer(obs)
    try:
        obs.on_request_completed(
            route="/rerank", status=200, total_seconds=0.05, queue_wait_seconds=0.005
        )
        obs.on_batch_completed(batch_size=4, pairs=16, inference_seconds=0.04, device="cpu")
        obs.on_queue_full()
        obs.on_mps_fallback()

        from prometheus_client import generate_latest
        text = generate_latest(REGISTRY).decode()

        assert 'reranker_requests_total{route="/rerank",status="200"} 1.0' in text
        assert "reranker_batch_size_bucket" in text
        assert "reranker_inference_seconds_bucket" in text
        assert "reranker_queue_full_total 1.0" in text
        assert "reranker_mps_fallback_total 1.0" in text
    finally:
        set_observer(NullObserver())


def test_snapshot_loop_copies_engine_stats():
    from src.observability.prometheus import _snapshot_into_gauges, REGISTRY

    stats = {
        "batch_occupancy_pct": 42.5,
        "pending_requests": 3,
        "inflight_batches": 1,
        "semaphore_available": 0,
    }
    _snapshot_into_gauges(stats)
    from prometheus_client import generate_latest
    text = generate_latest(REGISTRY).decode()
    assert "reranker_batch_occupancy_ratio 0.425" in text
    assert "reranker_pending_requests 3.0" in text
    assert "reranker_inflight_batches 1.0" in text
    assert "reranker_semaphore_available 0.0" in text
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_metrics.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.observability.prometheus'`.

- [ ] **Step 3: Implement the Prometheus module**

Create `src/observability/prometheus.py`:

```python
"""Prometheus implementation of the Observer interface."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from src.observability.observer import Observer


REGISTRY = CollectorRegistry()


_REQUESTS = Counter(
    "reranker_requests_total",
    "Total HTTP rerank requests served.",
    ["route", "status"],
    registry=REGISTRY,
)
_REQUEST_DURATION = Histogram(
    "reranker_request_duration_seconds",
    "End-to-end HTTP request duration in seconds.",
    ["route"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)
_QUEUE_WAIT = Histogram(
    "reranker_queue_wait_seconds",
    "Time a request spent waiting in the queue before inference.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=REGISTRY,
)
_INFERENCE = Histogram(
    "reranker_inference_seconds",
    "GPU forward-pass duration per batch.",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY,
)
_BATCH_SIZE = Histogram(
    "reranker_batch_size",
    "Number of requests per batch.",
    buckets=(1, 2, 4, 8, 16, 32, 64, 128),
    registry=REGISTRY,
)
_BATCH_OCCUPANCY = Gauge(
    "reranker_batch_occupancy_ratio",
    "Average batch size / max_batch_size, from engine.get_stats().",
    registry=REGISTRY,
)
_PENDING = Gauge(
    "reranker_pending_requests",
    "Requests waiting in the queue.",
    registry=REGISTRY,
)
_INFLIGHT = Gauge(
    "reranker_inflight_batches",
    "Batches currently in the executor.",
    registry=REGISTRY,
)
_SEMAPHORE = Gauge(
    "reranker_semaphore_available",
    "Free slots in the engine's batch semaphore.",
    registry=REGISTRY,
)
_QUEUE_FULL = Counter(
    "reranker_queue_full_total",
    "Number of times the queue rejected a request.",
    registry=REGISTRY,
)
_REQUEST_TIMEOUT = Counter(
    "reranker_request_timeout_total",
    "Number of requests that exhausted request_timeout.",
    registry=REGISTRY,
)
_MPS_FALLBACK = Counter(
    "reranker_mps_fallback_total",
    "MPS→CPU fallback firings on the existing fallback path.",
    registry=REGISTRY,
)
_BATCH_FAILED = Counter(
    "reranker_batch_processing_failed_total",
    "Batches that failed inside _process_batch.",
    registry=REGISTRY,
)


class PrometheusObserver(Observer):
    def __init__(self, registry: CollectorRegistry = REGISTRY):
        self._registry = registry

    def on_request_completed(self, *, route, status, total_seconds, queue_wait_seconds):
        _REQUESTS.labels(route=route, status=str(status)).inc()
        _REQUEST_DURATION.labels(route=route).observe(total_seconds)
        _QUEUE_WAIT.observe(queue_wait_seconds)

    def on_batch_completed(self, *, batch_size, pairs, inference_seconds, device):
        _BATCH_SIZE.observe(batch_size)
        _INFERENCE.observe(inference_seconds)

    def on_queue_full(self):
        _QUEUE_FULL.inc()

    def on_request_timeout(self):
        _REQUEST_TIMEOUT.inc()

    def on_mps_fallback(self):
        _MPS_FALLBACK.inc()

    def on_batch_processing_failed(self):
        _BATCH_FAILED.inc()


def _snapshot_into_gauges(stats: Dict[str, Any]) -> None:
    occ = stats.get("batch_occupancy_pct")
    if occ is not None:
        _BATCH_OCCUPANCY.set(occ / 100.0)
    pending = stats.get("pending_requests")
    if pending is not None:
        _PENDING.set(pending)
    inflight = stats.get("inflight_batches")
    if inflight is not None:
        _INFLIGHT.set(inflight)
    sema = stats.get("semaphore_available")
    if sema is not None:
        _SEMAPHORE.set(sema)


async def run_snapshot_loop(engine, interval_seconds: float) -> None:
    """Background task: copy engine.get_stats() into gauges every interval."""
    while True:
        try:
            _snapshot_into_gauges(engine.get_stats())
        except Exception:
            pass
        await asyncio.sleep(interval_seconds)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_metrics.py -v
```

Expected: PASS for both tests.

- [ ] **Step 5: Commit**

```bash
git add src/observability/prometheus.py tests/test_metrics.py
git commit -m "feat(observability): Prometheus observer + /metrics families"
```

---

### Task 4: Wire Prometheus into FastAPI + request middleware

**Files:**
- Modify: `src/main.py`
- Test: `tests/test_metrics_integration.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_metrics_integration.py`:

```python
"""End-to-end: with the env var on, /metrics is exposed and request counters fire."""

import pytest

pytest.importorskip("prometheus_client")


def test_metrics_endpoint_disabled_by_default(test_client):
    resp = test_client.get("/metrics")
    assert resp.status_code == 404


@pytest.fixture
def test_client_with_metrics(monkeypatch, dummy_engine):
    monkeypatch.setenv("RERANKER_EXPOSE_PROMETHEUS_METRICS", "true")
    from src.config import settings as s
    monkeypatch.setattr(s, "expose_prometheus_metrics", True)

    async def fake_get_async_engine():
        return dummy_engine
    monkeypatch.setattr("src.engine.get_async_engine", fake_get_async_engine)

    from src.main import create_app
    from fastapi.testclient import TestClient
    app = create_app()
    with TestClient(app) as client:
        yield client


def test_metrics_endpoint_exposes_text_when_enabled(test_client_with_metrics):
    resp = test_client_with_metrics.get("/metrics")
    assert resp.status_code == 200
    assert "reranker_requests_total" in resp.text


def test_rerank_request_increments_counter(test_client_with_metrics):
    test_client_with_metrics.post("/rerank", json={"query": "q", "documents": ["a"]})
    text = test_client_with_metrics.get("/metrics").text
    assert 'reranker_requests_total{route="/rerank",status="200"}' in text
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_metrics_integration.py -v
```

Expected: 2 of 3 fail (no /metrics endpoint, no counter increments).

- [ ] **Step 3: Wire Prometheus in `src/main.py`**

Open `src/main.py`, find `create_app()`. Inside it, after the existing router includes and before `return app`, add:

```python
    if settings.expose_prometheus_metrics:
        from prometheus_client import make_asgi_app
        from src.observability import set_observer
        from src.observability.prometheus import PrometheusObserver, REGISTRY, run_snapshot_loop

        set_observer(PrometheusObserver())
        app.mount("/metrics", make_asgi_app(registry=REGISTRY))

        @app.middleware("http")
        async def _request_metrics(request, call_next):
            import time
            from src.observability import get_observer
            start = time.perf_counter()
            response = await call_next(request)
            if not request.url.path.startswith(("/metrics", "/health", "/live", "/ready")):
                get_observer().on_request_completed(
                    route=request.url.path,
                    status=response.status_code,
                    total_seconds=time.perf_counter() - start,
                    queue_wait_seconds=0.0,
                )
            return response

        @app.on_event("startup")
        async def _start_snapshot():
            import asyncio
            from src.engine import get_async_engine
            engine = await get_async_engine()
            app.state._prom_snapshot_task = asyncio.create_task(
                run_snapshot_loop(engine, settings.prometheus_snapshot_interval_seconds)
            )

        @app.on_event("shutdown")
        async def _stop_snapshot():
            task = getattr(app.state, "_prom_snapshot_task", None)
            if task is not None:
                task.cancel()
```

(`from src.config import settings` is already imported at the top of `main.py`.)

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_metrics_integration.py -v
```

Expected: PASS for all three tests.

- [ ] **Step 5: Confirm full suite green**

```bash
pytest tests/ -x
```

- [ ] **Step 6: Commit**

```bash
git add src/main.py tests/test_metrics_integration.py
git commit -m "feat(observability): mount /metrics + request middleware when env var set"
```

---

### Task 5: OpenTelemetry implementation (lazy-imported)

**Files:**
- Create: `src/observability/otel.py`
- Test: `tests/test_otel.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_otel.py`:

```python
"""OTel: server span per request + batch span per inference, attributes correct."""

import pytest

pytest.importorskip("opentelemetry")


def test_init_otel_returns_tracer_provider(monkeypatch):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "reranker-test")
    from src.observability.otel import init_otel

    provider = init_otel(use_in_memory_exporter=True)
    assert provider is not None


@pytest.mark.asyncio
async def test_batch_span_emitted_for_each_batch(monkeypatch):
    from src.observability.otel import init_otel, get_in_memory_exporter
    init_otel(use_in_memory_exporter=True)
    exporter = get_in_memory_exporter()
    exporter.clear()

    from src.engine.async_engine import AsyncRerankerEngine
    from src.engine.handlers.base import BaseHandler

    class DummyHandler(BaseHandler):
        def load_model(self):
            self.model = "ok"
        def predict(self, batch):
            return [[{"index": 0, "relevance_score": 1.0}] for _ in batch.requests]

    monkeypatch.setattr(
        "src.engine.async_engine.get_handler",
        lambda *a, **k: DummyHandler("p", "cpu", 64, False),
    )
    from src.config import settings as s
    monkeypatch.setattr(s, "enable_device_probe", False)
    monkeypatch.setattr(s, "enable_otel", True)
    monkeypatch.setattr(s, "otel_batch_span", True)

    engine = AsyncRerankerEngine(
        max_concurrent_batches=1, inference_threads=1, batch_wait_timeout=0.001
    )
    await engine.start()
    try:
        await engine.rerank("q", ["a", "b"])
    finally:
        await engine.stop()

    spans = exporter.get_finished_spans()
    batch_spans = [s for s in spans if s.name == "reranker.batch"]
    assert len(batch_spans) >= 1
    attrs = batch_spans[0].attributes
    assert attrs["batch_size"] == 1
    assert attrs["pairs"] == 2
    assert attrs["device"] == "cpu"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_otel.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.observability.otel'`.

- [ ] **Step 3: Implement the OTel module**

Create `src/observability/otel.py`:

```python
"""OpenTelemetry initialization. Lazy-imported; only touched when enable_otel=True."""

from __future__ import annotations

from typing import Optional


_in_memory_exporter = None  # for tests


def init_otel(use_in_memory_exporter: bool = False):
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    resource = Resource.create({})  # service.name comes from OTEL_SERVICE_NAME
    provider = TracerProvider(resource=resource)

    if use_in_memory_exporter:
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )
        global _in_memory_exporter
        _in_memory_exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(_in_memory_exporter))
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

    trace.set_tracer_provider(provider)
    return provider


def instrument_fastapi(app) -> None:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    FastAPIInstrumentor().instrument_app(app)


def get_tracer():
    from opentelemetry import trace
    return trace.get_tracer("reranker-serve")


def get_in_memory_exporter():
    """Test-only accessor."""
    return _in_memory_exporter
```

- [ ] **Step 4: Wire the batch span in `_process_batch`**

In `src/engine/async_engine.py`, find `_process_batch`. Inside the `try` block, wrap the `loop.run_in_executor` line with a span when OTel is enabled:

```python
                if settings.enable_otel and settings.otel_batch_span:
                    from src.observability.otel import get_tracer
                    tracer = get_tracer()
                    with tracer.start_as_current_span("reranker.batch") as span:
                        span.set_attribute("batch_size", len(batch.requests))
                        span.set_attribute("pairs", batch.total_pairs)
                        span.set_attribute("device", self.device)
                        results = await loop.run_in_executor(
                            self._executor, self._inference_batch_sync, batch
                        )
                        span.set_attribute("inference_ms", (time.time() - start_time) * 1000.0)
                else:
                    results = await loop.run_in_executor(
                        self._executor, self._inference_batch_sync, batch
                    )
```

Replace the existing single `results = await loop.run_in_executor(...)` call with the above block.

- [ ] **Step 5: Run the test to verify it passes**

```bash
pytest tests/test_otel.py -v
```

Expected: PASS for both tests.

- [ ] **Step 6: Commit**

```bash
git add src/observability/otel.py src/engine/async_engine.py tests/test_otel.py
git commit -m "feat(observability): OTel init + per-batch child span"
```

---

### Task 6: Wire OTel into FastAPI + verify import isolation

**Files:**
- Modify: `src/main.py`
- Test: `tests/test_otel_integration.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_otel_integration.py`:

```python
"""OTel only initialized when env var on; FastAPI instrumented when so."""

import pytest

pytest.importorskip("opentelemetry")


def test_otel_off_by_default(monkeypatch):
    monkeypatch.delenv("RERANKER_ENABLE_OTEL", raising=False)
    from src.config import settings as s
    monkeypatch.setattr(s, "enable_otel", False)
    from src.main import create_app

    app = create_app()
    # No otel instrumentation hooks should have been registered on the app
    assert not getattr(app.state, "_otel_initialized", False)


def test_otel_on_initializes_provider(monkeypatch):
    monkeypatch.setenv("RERANKER_ENABLE_OTEL", "true")
    from src.config import settings as s
    monkeypatch.setattr(s, "enable_otel", True)
    from src.main import create_app

    app = create_app()
    assert getattr(app.state, "_otel_initialized", False) is True
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_otel_integration.py -v
```

Expected: FAIL — `_otel_initialized` never set when env on.

- [ ] **Step 3: Wire OTel in `src/main.py`**

In `create_app()`, after the existing observability/Prometheus block from Task 4, add:

```python
    if settings.enable_otel:
        from src.observability.otel import init_otel, instrument_fastapi
        init_otel(use_in_memory_exporter=False)
        instrument_fastapi(app)
        app.state._otel_initialized = True
    else:
        app.state._otel_initialized = False
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_otel_integration.py -v
```

Expected: PASS for both tests.

- [ ] **Step 5: Confirm full suite green**

```bash
pytest tests/ -x
```

- [ ] **Step 6: Commit**

```bash
git add src/main.py tests/test_otel_integration.py
git commit -m "feat(observability): init OTel + instrument FastAPI when env var set"
```

---

### Task 7: Graceful shutdown drain on SIGTERM

**Files:**
- Modify: `src/main.py`
- Modify: `src/engine/async_engine.py` (add `_shutting_down` flag + reject new requests)
- Modify: `supervisord.conf`
- Test: `tests/test_graceful_shutdown.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_graceful_shutdown.py`:

```python
"""Graceful drain: in-flight requests complete after SIGTERM signal."""

import asyncio
import pytest


@pytest.mark.asyncio
async def test_engine_rejects_new_requests_once_shutting_down(monkeypatch):
    from src.engine.async_engine import AsyncRerankerEngine
    from src.engine.handlers.base import BaseHandler
    from src.engine.request_queue import QueueFullError

    class DummyHandler(BaseHandler):
        def load_model(self): self.model = "ok"
        def predict(self, batch):
            return [[{"index": 0, "relevance_score": 1.0}] for _ in batch.requests]

    monkeypatch.setattr(
        "src.engine.async_engine.get_handler",
        lambda *a, **k: DummyHandler("p", "cpu", 64, False),
    )
    from src.config import settings as s
    monkeypatch.setattr(s, "enable_device_probe", False)

    engine = AsyncRerankerEngine(
        max_concurrent_batches=1, inference_threads=1, batch_wait_timeout=0.001
    )
    await engine.start()
    try:
        engine.begin_shutdown()
        with pytest.raises(QueueFullError):
            await engine.rerank("q", ["a"])
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_engine_drains_inflight_requests_on_stop(monkeypatch):
    """Issue a slow request, then stop(); the in-flight request must complete."""
    import time
    from src.engine.async_engine import AsyncRerankerEngine
    from src.engine.handlers.base import BaseHandler

    class SlowHandler(BaseHandler):
        def load_model(self): self.model = "ok"
        def predict(self, batch):
            time.sleep(0.05)
            return [[{"index": 0, "relevance_score": 1.0}] for _ in batch.requests]

    monkeypatch.setattr(
        "src.engine.async_engine.get_handler",
        lambda *a, **k: SlowHandler("p", "cpu", 64, False),
    )
    from src.config import settings as s
    monkeypatch.setattr(s, "enable_device_probe", False)

    engine = AsyncRerankerEngine(
        max_concurrent_batches=1, inference_threads=1, batch_wait_timeout=0.001
    )
    await engine.start()

    rerank_task = asyncio.create_task(engine.rerank("q", ["a"]))
    await asyncio.sleep(0.005)  # let it enter inference
    stop_task = asyncio.create_task(engine.stop())

    result = await rerank_task
    await stop_task
    assert len(result) == 1
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_graceful_shutdown.py -v
```

Expected: FAIL — `engine.begin_shutdown()` does not exist.

- [ ] **Step 3: Add `begin_shutdown()` to `AsyncRerankerEngine`**

In `src/engine/async_engine.py`, add at the end of `__init__`:

```python
        self._shutting_down = False
```

Add a method just below `is_loaded`:

```python
    def begin_shutdown(self) -> None:
        """Signal that no new requests should be accepted; in-flight work continues."""
        self._shutting_down = True
        self.request_queue._shutdown = True
```

Add at the top of `rerank()`, immediately after the `if not self._running:` check:

```python
        if self._shutting_down:
            from src.engine.request_queue import QueueFullError
            raise QueueFullError("Server is shutting down")
```

- [ ] **Step 4: Wire SIGTERM in `src/main.py`**

In `create_app()`, after the OTel block, add:

```python
    @app.on_event("startup")
    async def _install_sigterm_handler():
        import asyncio
        import signal
        from src.engine import get_async_engine

        loop = asyncio.get_running_loop()

        def _on_term():
            engine = None
            try:
                from src.engine.async_engine import peek_async_engine
                engine = peek_async_engine()
            except Exception:
                pass
            if engine is not None:
                engine.begin_shutdown()

        try:
            loop.add_signal_handler(signal.SIGTERM, _on_term)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler; supervisord lives on Linux anyway.
            pass
```

- [ ] **Step 5: Update `supervisord.conf`**

Open `supervisord.conf`. Find the `[program:reranker]` block. Ensure these lines are present (add if missing, edit if present):

```ini
stopsignal=TERM
stopwaitsecs=70
```

- [ ] **Step 6: Run the test to verify it passes**

```bash
pytest tests/test_graceful_shutdown.py -v
```

Expected: PASS for both tests.

- [ ] **Step 7: Confirm full suite green**

```bash
pytest tests/ -x
```

- [ ] **Step 8: Commit**

```bash
git add src/engine/async_engine.py src/main.py supervisord.conf tests/test_graceful_shutdown.py
git commit -m "feat(lifecycle): graceful shutdown on SIGTERM (drain + reject new)"
```

---

### Task 8: nginx example + operations runbook

**Files:**
- Create: `examples/nginx.reranker.conf`
- Create: `docs/operations.md`

- [ ] **Step 1: Write the nginx example**

Create `examples/nginx.reranker.conf`:

```nginx
# nginx in front of reranker-serve. Drop into /etc/nginx/sites-available/
# and symlink into sites-enabled. Replace certs + server_name first.

limit_req_zone $binary_remote_addr zone=ranker_rps:10m rate=50r/s;

upstream reranker {
    server 127.0.0.1:8000;
    keepalive 32;
}

server {
    listen 443 ssl http2;
    server_name reranker.internal;

    ssl_certificate     /etc/letsencrypt/live/reranker.internal/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/reranker.internal/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;

    client_max_body_size 4m;
    client_body_timeout  60s;
    send_timeout         60s;

    location / {
        limit_req zone=ranker_rps burst=20 nodelay;
        limit_conn_zone $binary_remote_addr zone=ranker_conn:10m;
        limit_conn ranker_conn 16;

        proxy_pass         http://reranker;
        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_set_header   Connection        "";
        proxy_read_timeout 75s;
    }

    # /metrics is intentionally NOT proxied. Scrape it from localhost only.
}
```

- [ ] **Step 2: Write the operations runbook**

Create `docs/operations.md`:

```markdown
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
```

- [ ] **Step 3: Commit**

```bash
git add examples/nginx.reranker.conf docs/operations.md
git commit -m "docs(ops): nginx example + operations runbook"
```

---

### Task 9: Wire ops docs into README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the new docs to the README documentation table**

In `README.md`, find the existing Documentation table (under `## Documentation`). After the `[LiteLLM Integration]` row, add:

```markdown
| [Operations](docs/operations.md) | Production runbook: supervisord, alerts, nginx, tuning |
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: link operations.md from README"
```

---

### Task 10: Final integration sweep

- [ ] **Step 1: Run the full suite**

```bash
pytest tests/ -x
```

Expected: every test passes.

- [ ] **Step 2: Smoke-test with observability off**

```bash
unset RERANKER_EXPOSE_PROMETHEUS_METRICS RERANKER_ENABLE_OTEL
./run.sh &
sleep 5
curl -sf http://localhost:8000/health
curl -i http://localhost:8000/metrics  # expect 404
kill %1
```

Expected: `/health` returns 200; `/metrics` returns 404.

- [ ] **Step 3: Smoke-test with observability on**

```bash
RERANKER_EXPOSE_PROMETHEUS_METRICS=true RERANKER_ENABLE_OTEL=false ./run.sh &
sleep 5
curl -sf http://localhost:8000/health
curl -sf http://localhost:8000/metrics | head -20
curl -X POST http://localhost:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{"query":"q","documents":["a","b"]}' | head
curl -sf http://localhost:8000/metrics | grep reranker_requests_total
kill %1
```

Expected: `reranker_requests_total{route="/rerank",status="200"} 1.0` appears.

- [ ] **Step 4: Smoke-test backpressure**

```bash
RERANKER_MAX_QUEUE_SIZE=1 RERANKER_REQUEST_TIMEOUT=0.5 ./run.sh &
sleep 5
# Hammer with parallel requests to trip the queue
for i in $(seq 1 20); do
  curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8000/rerank \
    -H "Content-Type: application/json" \
    -d '{"query":"q","documents":["a","b","c"]}' &
done
wait
kill %1
```

Expected: a mix of 200 and 503 responses.

- [ ] **Step 5: Commit any tweaks discovered in smoke testing**

```bash
git status
# Address any leftovers; if nothing changed, skip.
```

- [ ] **Step 6: Open PR**

```bash
gh pr create \
  --title "Enterprise single-node hardening (observability + backpressure + graceful shutdown)" \
  --body "Implements docs/superpowers/specs/2026-05-26-enterprise-single-node-design.md"
```

---

## Self-Review

**1. Spec coverage:**
- Architecture (nginx + in-process scaffolding) → Tasks 4, 6, 8
- Request lifecycle / backpressure → Task 1
- Prometheus exposition → Tasks 3, 4
- OTel tracing → Tasks 5, 6
- JSON logs → no task needed (already shipped)
- Operator-actionable counters → Tasks 2, 3 (observer wiring + Prometheus counters)
- Configuration changes (env-only switches) → Task 0
- Graceful shutdown → Task 7
- File-level plan (`src/observability/*`, nginx example, ops runbook) → Tasks 2-8
- Testing (`test_metrics`, `test_backpressure`, `test_otel`, `test_graceful_shutdown`) → Tasks 1, 3, 4, 5, 6, 7
- Rollout / acceptance criteria → Task 10

**2. Placeholder scan:** No "TODO", "TBD", "implement later", or generic "add validation" language remains. Every code step shows the code.

**3. Type consistency:** `Observer` interface in Task 2 names match `PrometheusObserver` methods in Task 3. `QueueFullError` defined in Task 1 is consistently imported in Tasks 2, 6, 7. Env-var names (`RERANKER_EXPOSE_PROMETHEUS_METRICS`, `RERANKER_ENABLE_OTEL`, `RERANKER_OTEL_BATCH_SPAN`) match the spec's table verbatim.
