"""
Startup device probe.

Runs a few warmup forward passes at increasing batch sizes after the model
loads, so operators see what the hardware actually delivers and so the engine
can pick a reasonable default batch size when the user hasn't pinned one.

Probe results are exposed via /info and /stats as `device_profile`.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

from src.config import get_logger
from src.engine.handlers.base import BaseHandler
from src.engine.request_queue import BatchedRequest, RerankRequest


logger = get_logger(__name__)


PROBE_BATCH_SIZES: tuple[int, ...] = (1, 8, 32)
PROBE_DOCS_PER_REQUEST: int = 4
PROBE_QUERY: str = "device probe query for warmup"
PROBE_DOC: str = (
    "Synthetic warmup document. This text exists only to exercise the model "
    "forward pass during startup so the engine can measure per-pair latency."
)


@dataclass
class ProbeResult:
    batch_size: int
    pairs: int
    elapsed_ms: float

    @property
    def ms_per_pair(self) -> float:
        return self.elapsed_ms / self.pairs if self.pairs > 0 else 0.0


@dataclass
class DeviceProfile:
    device: str
    probes: List[ProbeResult]
    suggested_batch_size: int
    user_pinned_batch_size: bool
    suggested_max_batch_pairs: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "probes": [
                {
                    "batch_size": p.batch_size,
                    "pairs": p.pairs,
                    "elapsed_ms": round(p.elapsed_ms, 2),
                    "ms_per_pair": round(p.ms_per_pair, 3),
                }
                for p in self.probes
            ],
            "suggested_batch_size": self.suggested_batch_size,
            "user_pinned_batch_size": self.user_pinned_batch_size,
            "suggested_max_batch_pairs": self.suggested_max_batch_pairs,
        }


def _build_probe_batch(batch_size: int) -> BatchedRequest:
    requests = [
        RerankRequest(
            request_id=f"probe-{uuid.uuid4().hex[:8]}",
            query=PROBE_QUERY,
            documents=[PROBE_DOC] * PROBE_DOCS_PER_REQUEST,
            return_documents=False,
        )
        for _ in range(batch_size)
    ]
    return BatchedRequest(batch_id=f"probe-batch-{batch_size}", requests=requests)


def _pick_suggested_batch_size(probes: List[ProbeResult]) -> int:
    """Largest probed batch whose ms-per-pair stays within 2x of the smallest."""
    if not probes:
        return PROBE_BATCH_SIZES[0]
    baseline = probes[0].ms_per_pair
    if baseline <= 0:
        return probes[-1].batch_size
    chosen = probes[0].batch_size
    for probe in probes[1:]:
        if probe.ms_per_pair <= baseline * 2.0:
            chosen = probe.batch_size
        else:
            break
    return chosen


def _device_free_fraction(device: str) -> float:
    """Fraction of the active device's memory currently free (0.0-1.0)."""
    try:
        if device == "cuda":
            import torch
            free, total = torch.cuda.mem_get_info()
            return free / total if total else 0.0
        if device == "mps":
            import torch
            total = torch.mps.recommended_max_memory()
            used = torch.mps.current_allocated_memory()
            return (total - used) / total if total else 0.0
        import psutil
        vm = psutil.virtual_memory()
        return vm.available / vm.total if vm.total else 0.0
    except Exception:
        return 1.0  # unknown -> do not constrain batch size


def suggest_max_batch_pairs(
    candidates: List[int],
    free_fraction_after: Callable[[int], float],
    safety_margin: float,
) -> int:
    """Largest candidate batch (in pairs) that still leaves >= safety_margin
    of device memory free. If none qualifies, return the smallest candidate."""
    ordered = sorted(candidates)
    chosen = ordered[0]
    for pairs in ordered:
        if free_fraction_after(pairs) >= safety_margin:
            chosen = pairs
        else:
            break
    return chosen


def run_device_probe(
    handler: BaseHandler,
    device: str,
    explicit_batch_size_env: str = "RERANKER_BATCH_SIZE",
) -> Optional[DeviceProfile]:
    """
    Exercise the loaded handler at PROBE_BATCH_SIZES and return a DeviceProfile.

    Returns None if probing raises — never fatal at startup.
    """
    user_pinned = explicit_batch_size_env in os.environ
    probes: List[ProbeResult] = []

    for batch_size in PROBE_BATCH_SIZES:
        batch = _build_probe_batch(batch_size)
        try:
            start = time.perf_counter()
            handler.predict(batch)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        except Exception as exc:
            logger.warning(
                "device_probe_batch_failed",
                batch_size=batch_size,
                error=str(exc),
            )
            break
        probes.append(
            ProbeResult(
                batch_size=batch_size,
                pairs=batch.total_pairs,
                elapsed_ms=elapsed_ms,
            )
        )
        logger.info(
            "device_probe_result",
            device=device,
            batch_size=batch_size,
            pairs=batch.total_pairs,
            elapsed_ms=round(elapsed_ms, 2),
            ms_per_pair=round(elapsed_ms / max(1, batch.total_pairs), 3),
        )

    if not probes:
        return None

    from src.config import settings as _settings

    pair_candidates = [64, 128, 256, 512, 1024]

    def _free_after(pairs: int) -> float:
        n_requests = max(1, pairs // PROBE_DOCS_PER_REQUEST)
        probe_batch = BatchedRequest(
            batch_id=f"mem-probe-{pairs}",
            requests=[
                RerankRequest(
                    request_id=f"mem-{uuid.uuid4().hex[:6]}",
                    query=PROBE_QUERY,
                    documents=[PROBE_DOC] * PROBE_DOCS_PER_REQUEST,
                    return_documents=False,
                )
                for _ in range(n_requests)
            ],
        )
        try:
            handler.predict(probe_batch)
        except Exception:
            return 0.0  # did not fit / errored -> treat as over budget
        return _device_free_fraction(device)

    suggested_pairs = suggest_max_batch_pairs(
        candidates=pair_candidates,
        free_fraction_after=_free_after,
        safety_margin=_settings.device_mem_safety_margin,
    )

    profile = DeviceProfile(
        device=device,
        probes=probes,
        suggested_batch_size=_pick_suggested_batch_size(probes),
        user_pinned_batch_size=user_pinned,
        suggested_max_batch_pairs=suggested_pairs,
    )
    logger.info(
        "device_probe_complete",
        device=device,
        suggested_batch_size=profile.suggested_batch_size,
        user_pinned_batch_size=profile.user_pinned_batch_size,
    )
    return profile
