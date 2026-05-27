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
