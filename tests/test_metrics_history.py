"""Time-series metrics buffer that backs the admin dashboard charts."""

from src.observability.metrics_history import MetricsHistory, build_sample


_FULL_STATS = {
    "inference_latency_p50_ms": 5.0,
    "inference_latency_p95_ms": 9.0,
    "requests_per_second": 3.0,
    "throughput_pairs_per_sec": 12.0,
    "inflight_batches": 2,
    "pending_requests": 4,
    "batch_occupancy_pct": 40.0,
    "device_resources": {
        "device": "cuda:0", "mem_used_mb": 1000.0, "mem_total_mb": 2000.0,
        "used_pct": 50.0, "util_pct": 73.0, "temp_c": 61.0, "power_w": 120.0,
    },
}


def test_build_sample_flattens_full_stats():
    s = build_sample(_FULL_STATS, now=100.0)
    assert s["t"] == 100.0
    assert s["p50_ms"] == 5.0
    assert s["p95_ms"] == 9.0
    assert s["rps"] == 3.0
    assert s["pairs_s"] == 12.0
    assert s["running"] == 2
    assert s["waiting"] == 4
    assert s["batch_occupancy_pct"] == 40.0
    assert s["gpu_util"] == 73.0
    assert s["gpu_mem_pct"] == 50.0
    assert s["gpu_temp_c"] == 61.0
    assert s["gpu_power_w"] == 120.0


def test_build_sample_cpu_has_no_gpu_fields():
    stats = {**_FULL_STATS, "device_resources": {"device": "cpu", "used_pct": 0.0}}
    s = build_sample(stats, now=1.0)
    assert s["gpu_util"] is None
    assert s["gpu_temp_c"] is None
    assert s["gpu_power_w"] is None
    # core perf numbers still present
    assert s["p50_ms"] == 5.0


def test_build_sample_missing_keys_default_safely():
    s = build_sample({}, now=1.0)
    assert s["p50_ms"] == 0.0
    assert s["rps"] == 0.0
    assert s["running"] == 0
    assert s["waiting"] == 0
    assert s["gpu_util"] is None


def test_record_and_get_all():
    h = MetricsHistory(maxlen=10)
    h.record(build_sample(_FULL_STATS, now=1.0))
    h.record(build_sample(_FULL_STATS, now=2.0))
    assert len(h) == 2
    assert [x["t"] for x in h.get()] == [1.0, 2.0]


def test_get_since_returns_strictly_newer():
    h = MetricsHistory(maxlen=10)
    for t in (1.0, 2.0, 3.0):
        h.record(build_sample(_FULL_STATS, now=t))
    assert [x["t"] for x in h.get(since_ts=2.0)] == [3.0]


def test_get_window_filters_old(monkeypatch):
    import src.observability.metrics_history as mod
    h = MetricsHistory(maxlen=10)
    for t in (100.0, 150.0, 200.0):
        h.record(build_sample(_FULL_STATS, now=t))
    monkeypatch.setattr(mod.time, "time", lambda: 210.0)
    # window=30s -> only samples with t >= 180
    assert [x["t"] for x in h.get(window_seconds=30)] == [200.0]


def test_maxlen_evicts_oldest():
    h = MetricsHistory(maxlen=3)
    for t in range(5):
        h.record(build_sample(_FULL_STATS, now=float(t)))
    assert [x["t"] for x in h.get()] == [2.0, 3.0, 4.0]
