"""CPU intra-op thread count is resolved from settings (None = all cores)."""

from src.engine.async_engine import resolve_cpu_threads


def test_explicit_thread_count_wins():
    assert resolve_cpu_threads(device="cpu", configured=6, cpu_count=12) == 6


def test_none_uses_cpu_count():
    assert resolve_cpu_threads(device="cpu", configured=None, cpu_count=12) == 12


def test_non_cpu_device_returns_none():
    # GPU paths must not touch torch.set_num_threads.
    assert resolve_cpu_threads(device="cuda", configured=8, cpu_count=12) is None
    assert resolve_cpu_threads(device="mps", configured=None, cpu_count=12) is None


def test_zero_or_negative_configured_falls_back_to_cpu_count():
    assert resolve_cpu_threads(device="cpu", configured=0, cpu_count=12) == 12
