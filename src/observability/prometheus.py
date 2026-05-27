"""Prometheus implementation of the Observer interface."""

from __future__ import annotations

import asyncio
from typing import Any, Dict

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
    "MPS to CPU recovery firings on the existing recovery path.",
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

    def on_request_completed(self, *, route, status, total_seconds, queue_wait_seconds=None):
        _REQUESTS.labels(route=route, status=str(status)).inc()
        _REQUEST_DURATION.labels(route=route).observe(total_seconds)
        if queue_wait_seconds is not None:
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


async def run_snapshot_loop(engine, interval_seconds: float) -> None:
    """Background task: copy engine.get_stats() into gauges every interval."""
    while True:
        try:
            _snapshot_into_gauges(engine.get_stats())
        except Exception:
            pass
        await asyncio.sleep(interval_seconds)
