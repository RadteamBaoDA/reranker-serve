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
