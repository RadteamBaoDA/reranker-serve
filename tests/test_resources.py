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
