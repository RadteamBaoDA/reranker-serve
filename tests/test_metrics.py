"""Prometheus exposition: /metrics endpoint shape + counter increments."""

import pytest

pytest.importorskip("prometheus_client")


def test_observer_emits_request_total_counter():
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
