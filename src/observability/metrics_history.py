"""In-memory time-series buffer of engine performance samples.

The engine exposes only instantaneous stats. This module keeps a rolling
window of flattened samples so the admin dashboard can draw trend charts that
survive page reloads and are shared across clients. It is a module-level
singleton, independent of the engine, so it persists across hot config
reloads (which recreate the engine).
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional


def _num(value: Any) -> Optional[float]:
    """Coerce to float, or None if missing / non-numeric (so charts can gap it)."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, 2)


def build_sample(stats: Dict[str, Any], now: float) -> Dict[str, Any]:
    """Flatten an engine ``get_stats()`` dict into one compact chart sample.

    Pure function (no clock, no I/O) so it is trivially testable. Core perf
    numbers default to 0.0 when absent; GPU fields default to None (CPU hosts
    and probe failures leave them out) so the charts render gaps rather than 0.
    """
    res = stats.get("device_resources") or {}
    return {
        "t": round(float(now), 2),
        "p50_ms": _num(stats.get("inference_latency_p50_ms")) or 0.0,
        "p95_ms": _num(stats.get("inference_latency_p95_ms")) or 0.0,
        "rps": _num(stats.get("requests_per_second")) or 0.0,
        "pairs_s": _num(stats.get("throughput_pairs_per_sec")) or 0.0,
        "running": int(stats.get("inflight_batches") or 0),
        "waiting": int(stats.get("pending_requests") or 0),
        "batch_occupancy_pct": _num(stats.get("batch_occupancy_pct")) or 0.0,
        "gpu_util": _num(res.get("util_pct")),
        "gpu_mem_pct": _num(res.get("used_pct")),
        "gpu_temp_c": _num(res.get("temp_c")),
        "gpu_power_w": _num(res.get("power_w")),
    }


class MetricsHistory:
    """Bounded ring buffer of samples with window/since filtering."""

    def __init__(self, maxlen: int) -> None:
        self._samples: Deque[Dict[str, Any]] = deque(maxlen=max(1, maxlen))

    def record(self, sample: Dict[str, Any]) -> None:
        self._samples.append(sample)

    def get(
        self,
        window_seconds: Optional[float] = None,
        since_ts: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Return samples, optionally limited to a recent window and/or to
        those strictly newer than ``since_ts`` (used for incremental polling)."""
        items = list(self._samples)
        if window_seconds is not None:
            cutoff = time.time() - window_seconds
            items = [s for s in items if s["t"] >= cutoff]
        if since_ts is not None:
            items = [s for s in items if s["t"] > since_ts]
        return items

    def clear(self) -> None:
        self._samples.clear()

    def __len__(self) -> int:
        return len(self._samples)


_history: Optional[MetricsHistory] = None


def _maxlen_from_settings() -> int:
    from src.config import settings
    minutes = max(1, int(getattr(settings, "metrics_history_minutes", 60)))
    interval = max(1, int(getattr(settings, "metrics_sample_interval_s", 5)))
    return math.ceil(minutes * 60 / interval) + 1


def get_history() -> MetricsHistory:
    """Lazily create the singleton, sized from settings."""
    global _history
    if _history is None:
        _history = MetricsHistory(maxlen=_maxlen_from_settings())
    return _history


def reset_history() -> None:
    """Drop the singleton so the next access rebuilds it (config reload/tests)."""
    global _history
    _history = None
