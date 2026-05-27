# Phase 3B — Telemetry Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the live telemetry the admin dashboard needs — uniform device memory ("GPU quota") across CUDA/MPS/CPU, plus a queue snapshot of batches *running* and requests *waiting* — through the existing `/stats` endpoint and Prometheus `/metrics`.

**Architecture:** A new `src/observability/resources.py` reads device memory/utilization uniformly (CUDA via `torch.cuda.mem_get_info` + optional `pynvml`; MPS via `torch.mps`; CPU via `psutil`) and returns a flat dict. The engine includes it in `get_stats()` under `device_resources`, and the existing Prometheus snapshot loop copies it into new gauges. `RequestQueue` keeps an ordered registry of *waiting* requests; the engine records *in-flight* batch metadata; `engine.get_queue_snapshot()` returns `{waiting, running}`. No new HTTP surface and no auth changes here — those land in Phase 3C/3D.

**Tech Stack:** Python 3.11, PyTorch, prometheus-client, psutil, optional pynvml; pytest.

**Reference spec:** `docs/superpowers/specs/2026-05-27-perf-telemetry-admin-ui-design.md` (Part B, items B1–B2).

**Scope note:** B3 (config-snapshot + log-tail HTTP APIs) and the `/admin/api/*` endpoints are intentionally deferred to Phase 3D (admin UI), where session auth gates them. Phase 3B keeps everything behind the already-existing `/stats` (and `/metrics`) so it ships and tests independently.

**Branch:** `feature/phase3b-telemetry`, created off `feature/phase3a-performance-core` (Phase 3B depends on Phase 3A).

## Environment (read before every task)
- Work from `/mnt/d/Project/RAG/reranker-serve`.
- ONLY working interpreter: the Windows venv via WSL interop — **`./.venv/Scripts/python.exe`** (system `python3` has no deps).
- Test command: `./.venv/Scripts/python.exe -m pytest -p no:cacheprovider <targets>`
- Full-suite regression check: `./.venv/Scripts/python.exe -m pytest tests/ -p no:cacheprovider --ignore=tests/test_dual_format_rerank.py --ignore=tests/test_huggingface_api.py -q`
- `tests/conftest.py` globally mocks `torch` (so torch reads return MagicMocks in tests — device-specific readers must be monkeypatched in tests, never called for real).
- KNOWN-FAILING baseline (4 pre-existing tests, NOT this work's concern; must remain the only failures): `test_async_engine.py::test_batch_accumulation_overlaps_with_inference`, `test_graceful_shutdown.py::test_engine_drains_inflight_requests_on_stop`, `test_handlers.py::test_qwen3_compute_logits_falls_back_from_mps_to_cpu`, `test_handlers.py::test_qwen3_compute_logits_does_not_fall_back_when_disabled`.

---

### Task 0: Branch + optional GPU-metrics dependency

**Files:** Modify `pyproject.toml`.

- [ ] **Step 1: Create the branch off Phase 3A**

```bash
git checkout feature/phase3a-performance-core
git checkout -b feature/phase3b-telemetry
```

- [ ] **Step 2: Add the optional `gpu-metrics` extra** to `pyproject.toml` under `[project.optional-dependencies]`:

```toml
gpu-metrics = [
    "pynvml>=11.5.0",
]
```

- [ ] **Step 3: Ensure psutil is importable in the venv** (it is listed in requirements.txt from Phase 3A; install it into the venv so the CPU path works at runtime):

```bash
./.venv/Scripts/python.exe -m pip install -q psutil
./.venv/Scripts/python.exe -c "import psutil; print('psutil', psutil.__version__)"
```

Expected: prints a version.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build(telemetry): optional pynvml extra for GPU metrics"
```

End every commit body in this plan with this trailer on its own line:
`Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`

---

### Task 1: Device resource probe module

**Files:** Create `src/observability/resources.py`; Create `tests/test_resources.py`.

- [ ] **Step 1: Write the failing test.** Create `tests/test_resources.py`:

```python
"""Uniform device resource stats across CUDA/MPS/CPU."""

from src.observability import resources as R


def test_assemble_computes_free_and_pct():
    s = R._assemble_resource_stats("cuda", "cuda", mem_used_mb=4000.0, mem_total_mb=16000.0)
    assert s["mem_free_mb"] == 12000.0
    assert s["used_pct"] == 25.0
    assert s["device"] == "cuda" and s["backend"] == "cuda"
    # optional fields omitted when not provided
    assert "util_pct" not in s and "temp_c" not in s and "power_w" not in s


def test_assemble_includes_optional_fields_when_present():
    s = R._assemble_resource_stats(
        "cuda", "cuda", 1000.0, 16000.0, util_pct=83.0, temp_c=54.0, power_w=88.0
    )
    assert s["util_pct"] == 83.0 and s["temp_c"] == 54.0 and s["power_w"] == 88.0


def test_assemble_handles_zero_total():
    s = R._assemble_resource_stats("cpu", "cpu", 0.0, 0.0)
    assert s["used_pct"] == 0.0 and s["mem_free_mb"] == 0.0


def test_get_resource_stats_dispatches_to_reader(monkeypatch):
    monkeypatch.setitem(R._READERS, "cuda", lambda: {"device": "cuda", "ok": True})
    assert R.get_resource_stats("cuda") == {"device": "cuda", "ok": True}


def test_get_resource_stats_unknown_device_uses_cpu_reader(monkeypatch):
    monkeypatch.setitem(R._READERS, "cpu", lambda: {"device": "cpu", "ok": True})
    assert R.get_resource_stats("something-else")["device"] == "cpu"


def test_get_resource_stats_degrades_on_reader_error(monkeypatch):
    def boom():
        raise RuntimeError("no device")
    monkeypatch.setitem(R._READERS, "cuda", boom)
    s = R.get_resource_stats("cuda")
    # Never raises; returns a shaped dict with the required keys.
    assert s["device"] == "cuda"
    assert s["mem_total_mb"] == 0.0 and s["mem_used_mb"] == 0.0
    assert s["error"] == "unavailable"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest -p no:cacheprovider tests/test_resources.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.observability.resources'`.

- [ ] **Step 3: Implement the module.** Create `src/observability/resources.py`:

```python
"""Device resource probe: uniform memory/utilization stats across CUDA, MPS, CPU.

Returns a flat dict so /stats and Prometheus can surface "how much GPU quota
is left" identically regardless of backend. Reads are best-effort: any failure
yields a shaped degraded dict rather than raising, because telemetry must never
take down the serving path.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

_BYTES_PER_MB = 1024 * 1024
_nvml_ready = False


def _assemble_resource_stats(
    device: str,
    backend: str,
    mem_used_mb: float,
    mem_total_mb: float,
    util_pct: Optional[float] = None,
    temp_c: Optional[float] = None,
    power_w: Optional[float] = None,
) -> Dict[str, Any]:
    free_mb = max(0.0, mem_total_mb - mem_used_mb)
    used_pct = (mem_used_mb / mem_total_mb * 100.0) if mem_total_mb > 0 else 0.0
    stats: Dict[str, Any] = {
        "device": device,
        "backend": backend,
        "mem_used_mb": round(mem_used_mb, 1),
        "mem_total_mb": round(mem_total_mb, 1),
        "mem_free_mb": round(free_mb, 1),
        "used_pct": round(used_pct, 1),
    }
    if util_pct is not None:
        stats["util_pct"] = round(util_pct, 1)
    if temp_c is not None:
        stats["temp_c"] = round(temp_c, 1)
    if power_w is not None:
        stats["power_w"] = round(power_w, 1)
    return stats


def _nvml_extras() -> Dict[str, Optional[float]]:
    """Utilization/temp/power via pynvml, if installed. Init once, never raise."""
    global _nvml_ready
    try:
        import pynvml
        import torch
        if not _nvml_ready:
            pynvml.nvmlInit()
            _nvml_ready = True
        handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        return {
            "util_pct": float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu),
            "temp_c": float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)),
            "power_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
        }
    except Exception:
        return {"util_pct": None, "temp_c": None, "power_w": None}


def _cuda_stats() -> Dict[str, Any]:
    import torch
    free, total = torch.cuda.mem_get_info()
    used = total - free
    extras = _nvml_extras()
    return _assemble_resource_stats(
        "cuda", "cuda", used / _BYTES_PER_MB, total / _BYTES_PER_MB,
        util_pct=extras["util_pct"], temp_c=extras["temp_c"], power_w=extras["power_w"],
    )


def _mps_stats() -> Dict[str, Any]:
    import torch
    total = float(torch.mps.recommended_max_memory())
    used = float(torch.mps.current_allocated_memory())
    return _assemble_resource_stats("mps", "mps", used / _BYTES_PER_MB, total / _BYTES_PER_MB)


def _cpu_stats() -> Dict[str, Any]:
    import psutil
    vm = psutil.virtual_memory()
    used = (vm.total - vm.available) / _BYTES_PER_MB
    return _assemble_resource_stats(
        "cpu", "cpu", used, vm.total / _BYTES_PER_MB,
        util_pct=psutil.cpu_percent(interval=None),
    )


_READERS = {"cuda": _cuda_stats, "mps": _mps_stats, "cpu": _cpu_stats}


def get_resource_stats(device: str) -> Dict[str, Any]:
    """Best-effort uniform resource stats for the active device. Never raises."""
    reader = _READERS.get(device, _cpu_stats)
    try:
        return reader()
    except Exception:
        degraded = _assemble_resource_stats(device, device, 0.0, 0.0)
        degraded["error"] = "unavailable"
        return degraded
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest -p no:cacheprovider tests/test_resources.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Run the full suite.** Expected: only the 4 known-failing tests fail.

- [ ] **Step 6: Commit**

```bash
git add src/observability/resources.py tests/test_resources.py
git commit -m "feat(observability): uniform device resource probe (CUDA/MPS/CPU)"
```

---

### Task 2: Device-memory Prometheus gauges

**Files:** Modify `src/observability/prometheus.py`; Modify `tests/test_metrics.py`.

- [ ] **Step 1: Write the failing test.** Append to `tests/test_metrics.py`:

```python
def test_snapshot_loop_copies_device_resources():
    from src.observability.prometheus import _snapshot_into_gauges, REGISTRY

    stats = {
        "device_resources": {
            "device": "cuda", "backend": "cuda",
            "mem_used_mb": 4096.0, "mem_total_mb": 16384.0,
            "mem_free_mb": 12288.0, "used_pct": 25.0, "util_pct": 80.0,
        }
    }
    _snapshot_into_gauges(stats)
    from prometheus_client import generate_latest
    text = generate_latest(REGISTRY).decode()
    assert "reranker_device_memory_used_bytes 4.294967296e+09" in text
    assert "reranker_device_memory_total_bytes 1.7179869184e+10" in text
    assert "reranker_device_utilization_ratio 0.8" in text
```

- [ ] **Step 2: Run it to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest -p no:cacheprovider tests/test_metrics.py::test_snapshot_loop_copies_device_resources -v`
Expected: FAIL — gauges not defined / values absent.

- [ ] **Step 3: Add the gauges and snapshot logic.** In `src/observability/prometheus.py`, after the existing `_SEMAPHORE` gauge definition, add:

```python
_DEVICE_MEM_USED = Gauge(
    "reranker_device_memory_used_bytes",
    "Active device memory in use, in bytes.",
    registry=REGISTRY,
)
_DEVICE_MEM_TOTAL = Gauge(
    "reranker_device_memory_total_bytes",
    "Active device total memory, in bytes.",
    registry=REGISTRY,
)
_DEVICE_UTIL = Gauge(
    "reranker_device_utilization_ratio",
    "Active device compute utilization ratio (0-1), when available.",
    registry=REGISTRY,
)
```

Then, inside `_snapshot_into_gauges`, after the existing gauge copies, add:

```python
    res = stats.get("device_resources")
    if res:
        used_mb = res.get("mem_used_mb")
        total_mb = res.get("mem_total_mb")
        util_pct = res.get("util_pct")
        if used_mb is not None:
            _DEVICE_MEM_USED.set(used_mb * 1024 * 1024)
        if total_mb is not None:
            _DEVICE_MEM_TOTAL.set(total_mb * 1024 * 1024)
        if util_pct is not None:
            _DEVICE_UTIL.set(util_pct / 100.0)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest -p no:cacheprovider tests/test_metrics.py -v`
Expected: PASS (existing metrics tests + the new one).

- [ ] **Step 5: Run the full suite.** Expected: only the 4 known-failing tests fail.

- [ ] **Step 6: Commit**

```bash
git add src/observability/prometheus.py tests/test_metrics.py
git commit -m "feat(observability): device-memory + utilization Prometheus gauges"
```

---

### Task 3: Include device_resources in engine.get_stats()

**Files:** Modify `src/engine/async_engine.py`; Create `tests/test_engine_device_resources.py`.

- [ ] **Step 1: Write the failing test.** Create `tests/test_engine_device_resources.py`:

```python
"""engine.get_stats() carries a device_resources block."""

from src.engine.async_engine import AsyncRerankerEngine


def test_get_stats_includes_device_resources(monkeypatch):
    # Don't load a model; just exercise get_stats() shape.
    import src.observability.resources as R
    monkeypatch.setitem(
        R._READERS, "cpu",
        lambda: {"device": "cpu", "backend": "cpu", "mem_used_mb": 1.0,
                 "mem_total_mb": 2.0, "mem_free_mb": 1.0, "used_pct": 50.0},
    )

    engine = AsyncRerankerEngine(max_concurrent_batches=1, inference_threads=1)
    engine.device = "cpu"
    stats = engine.get_stats()
    assert "device_resources" in stats
    assert stats["device_resources"]["device"] == "cpu"
    assert stats["device_resources"]["used_pct"] == 50.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest -p no:cacheprovider tests/test_engine_device_resources.py -v`
Expected: FAIL — `device_resources` not in stats.

- [ ] **Step 3: Wire it in.** In `src/engine/async_engine.py`, in `get_stats()`, add a `device_resources` entry to the returned dict (place it just before the `**self.request_queue.get_stats(),` spread):

```python
            "device_resources": _safe_resource_stats(self.device),
```

Then add this module-level helper near `resolve_cpu_threads` (top of the file):

```python
def _safe_resource_stats(device: str) -> Dict[str, Any]:
    """Resource stats for /stats; never let telemetry break get_stats()."""
    try:
        from src.observability.resources import get_resource_stats
        return get_resource_stats(device)
    except Exception:
        return {"device": device, "error": "unavailable"}
```

(`Dict` and `Any` are already imported from typing at the top of the file.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest -p no:cacheprovider tests/test_engine_device_resources.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite.** Expected: only the 4 known-failing tests fail. (The existing `/stats` health test still passes — it just gains a new key.)

- [ ] **Step 6: Commit**

```bash
git add src/engine/async_engine.py tests/test_engine_device_resources.py
git commit -m "feat(engine): surface device_resources in get_stats()"
```

---

### Task 4: Queue introspection — waiting registry + running snapshot

**Files:** Modify `src/engine/request_queue.py`; Modify `src/engine/async_engine.py`; Create `tests/test_queue_snapshot.py`.

- [ ] **Step 1: Write the failing test.** Create `tests/test_queue_snapshot.py`:

```python
"""Queue introspection: waiting registry + engine running snapshot."""

import asyncio
import time

import pytest

from src.engine.request_queue import RequestQueue, RerankRequest


@pytest.mark.asyncio
async def test_waiting_registry_lists_enqueued_in_order():
    q = RequestQueue(max_queue_size=10, request_timeout=5.0)
    await q.add_request(RerankRequest(request_id="r1", query="q", documents=["a", "b"]))
    await q.add_request(RerankRequest(request_id="r2", query="q", documents=["c"]))

    waiting = q.get_waiting_snapshot()
    assert [w["request_id"] for w in waiting] == ["r1", "r2"]
    assert waiting[0]["num_docs"] == 2 and waiting[1]["num_docs"] == 1
    assert all(w["waited_ms"] >= 0 for w in waiting)


@pytest.mark.asyncio
async def test_batched_requests_leave_the_waiting_registry():
    q = RequestQueue(max_queue_size=10, batch_wait_timeout=0.005, request_timeout=5.0)
    await q.add_request(RerankRequest(request_id="r1", query="q", documents=["a"]))
    await q.add_request(RerankRequest(request_id="r2", query="q", documents=["b"]))
    batch = await q.get_batch()
    assert batch is not None and len(batch.requests) == 2
    assert q.get_waiting_snapshot() == []


def test_engine_queue_snapshot_shape():
    from src.engine.async_engine import AsyncRerankerEngine
    engine = AsyncRerankerEngine(max_concurrent_batches=1, inference_threads=1)
    # Simulate one in-flight batch.
    engine._inflight_meta["b1"] = {
        "batch_id": "b1", "request_ids": ["r1", "r2"],
        "num_requests": 2, "pairs": 5, "started_at": time.time(),
    }
    snap = engine.get_queue_snapshot()
    assert snap["waiting"] == []
    assert len(snap["running"]) == 1
    assert snap["running"][0]["batch_id"] == "b1"
    assert snap["running"][0]["pairs"] == 5
    assert snap["running"][0]["elapsed_ms"] >= 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest -p no:cacheprovider tests/test_queue_snapshot.py -v`
Expected: FAIL — `get_waiting_snapshot` / `_inflight_meta` / `get_queue_snapshot` don't exist.

- [ ] **Step 3: Add the waiting registry to `RequestQueue`.** In `src/engine/request_queue.py`:

(a) Add `from collections import OrderedDict` (the file already imports `from collections import deque`; add OrderedDict to that line or a new import).

(b) In `RequestQueue.__init__`, after `self._active_requests: Dict[str, RerankRequest] = {}`, add:

```python
        # Ordered registry of requests currently waiting in the queue (not yet batched).
        self._waiting: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
```

(c) In `add_request`, immediately after the successful `await asyncio.wait_for(self._queue.put(request), ...)` block (after the `request_queued_success` debug log, before `return request.result_future`), add:

```python
        self._waiting[request.request_id] = {
            "request_id": request.request_id,
            "num_docs": len(request.documents),
            "enqueued_at": time.time(),
            "priority": request.priority,
        }
```

(d) In `get_batch`, remove a request from `_waiting` the moment it is pulled into a batch. After `requests.append(first_request)` add `self._waiting.pop(first_request.request_id, None)`. After the loop's `requests.append(request)` add `self._waiting.pop(request.request_id, None)`. In the deferral branch (where a request is put back via `await self._queue.put(request)` because it would exceed `max_batch_pairs`), do NOT pop it (it stays waiting) — leave `_waiting` as-is for that request.

(e) In `cancel_request`, add `self._waiting.pop(request_id, None)` where the request is removed. In `complete_request`, add `self._waiting.pop(request_id, None)` right after the `request = self._active_requests.pop(request_id, None)` line (defensive — a completed request must not linger as waiting).

(f) Add the snapshot method to `RequestQueue`:

```python
    def get_waiting_snapshot(self) -> List[Dict[str, Any]]:
        """Requests currently waiting in the queue, oldest first, with wait time."""
        now = time.time()
        return [
            {**entry, "waited_ms": round((now - entry["enqueued_at"]) * 1000.0, 1)}
            for entry in self._waiting.values()
        ]
```

- [ ] **Step 4: Add in-flight metadata + snapshot to the engine.** In `src/engine/async_engine.py`:

(a) In `AsyncRerankerEngine.__init__`, after `self._inflight_batches: set[asyncio.Task] = set()`, add:

```python
        self._inflight_meta: Dict[str, Dict[str, Any]] = {}
```

(b) In `_process_batch`, at the very start of the method (before the `async with self._batch_semaphore:` line), record the batch; and remove it when done. Wrap the existing body so the metadata is always cleaned up:

```python
        self._inflight_meta[batch.batch_id] = {
            "batch_id": batch.batch_id,
            "request_ids": [r.request_id for r in batch.requests],
            "num_requests": len(batch.requests),
            "pairs": batch.total_pairs,
            "started_at": time.time(),
        }
        try:
            async with self._batch_semaphore:
                # ... (existing body of _process_batch unchanged, re-indented under this try)
        finally:
            self._inflight_meta.pop(batch.batch_id, None)
```

IMPORTANT: keep the entire existing `_process_batch` body exactly as-is, just moved inside the new `try:`/`finally:`. Do not change any inner logic.

(c) Add the snapshot method (place near `get_stats`):

```python
    def get_queue_snapshot(self) -> Dict[str, Any]:
        """Live view of batches running and requests waiting, for the admin dashboard."""
        now = time.time()
        running = [
            {**meta, "elapsed_ms": round((now - meta["started_at"]) * 1000.0, 1)}
            for meta in self._inflight_meta.values()
        ]
        return {"waiting": self.request_queue.get_waiting_snapshot(), "running": running}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest -p no:cacheprovider tests/test_queue_snapshot.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Run the full suite.** Expected: only the 4 known-failing tests fail. Pay attention: the `_process_batch` re-indentation must not change behavior — if any `test_async_engine`/`test_backpressure` test beyond the known 4 fails, the re-indent broke something; fix it.

- [ ] **Step 7: Commit**

```bash
git add src/engine/request_queue.py src/engine/async_engine.py tests/test_queue_snapshot.py
git commit -m "feat(engine): queue introspection (waiting registry + running snapshot)"
```

---

### Task 5: Integration sweep

**Files:** None (validation only).

- [ ] **Step 1: Full suite green-except-baseline**

Run: `./.venv/Scripts/python.exe -m pytest tests/ -p no:cacheprovider --ignore=tests/test_dual_format_rerank.py --ignore=tests/test_huggingface_api.py -q`
Expected: exactly the 4 known-failing tests fail; everything else (incl. all new telemetry tests) passes.

- [ ] **Step 2: Confirm the telemetry shape end-to-end** (no model needed — get_stats resource block + queue snapshot are independent of model load):

```bash
./.venv/Scripts/python.exe - <<'PY'
from src.engine.async_engine import AsyncRerankerEngine
e = AsyncRerankerEngine(max_concurrent_batches=1, inference_threads=1)
e.device = "cpu"
s = e.get_stats()
print("device_resources keys:", sorted(s["device_resources"].keys()))
print("queue snapshot:", e.get_queue_snapshot())
PY
```

Expected: `device_resources` has at least `device, backend, mem_used_mb, mem_total_mb, mem_free_mb, used_pct`; queue snapshot prints `{'waiting': [], 'running': []}`.

- [ ] **Step 3: Report** the branch state and that Phase 3B is code-complete. Do not push or open a PR unless asked (consistent with the Phase 3A decision to keep branches local).

---

## Self-Review

**1. Spec coverage (Part B, B1–B2):**
- B1 device resource probe (CUDA/MPS/CPU, optional pynvml) → Task 1; surfaced in `/stats` → Task 3; Prometheus gauges → Task 2.
- B2 queue introspection (waiting registry + running snapshot, `get_queue_snapshot()`) → Task 4.
- B3 (config snapshot + log APIs) and `/admin/api/*` endpoints → intentionally deferred to Phase 3D (documented in Scope note).

**2. Placeholder scan:** No TBD/TODO. Every code step shows complete code. The degraded-stats path in `get_resource_stats` is a deliberate, justified best-effort design (telemetry must never crash serving), not a placeholder.

**3. Type/signature consistency:**
- `get_resource_stats(device) -> dict`, `_assemble_resource_stats(...)`, `_READERS` defined Task 1; consumed by `_safe_resource_stats` (Task 3) and the gauges read `device_resources` keys produced by `_assemble_resource_stats` (Task 2).
- `RequestQueue.get_waiting_snapshot() -> List[Dict]` and `self._waiting` defined Task 4; `engine.get_queue_snapshot()` and `engine._inflight_meta` defined Task 4 and consume `get_waiting_snapshot`.
- Prometheus gauge names (`reranker_device_memory_used_bytes`, `_total_bytes`, `reranker_device_utilization_ratio`) match the spec's metric table.
