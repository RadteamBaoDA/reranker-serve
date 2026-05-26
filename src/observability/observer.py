"""
Observer pattern: the engine emits typed events; the Prometheus and OTel
modules subscribe by setting an Observer implementation. When both are off,
NullObserver makes every emission a no-op on the engine's hot path.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Observer(Protocol):
    def on_request_completed(
        self, *, route: str, status: int, total_seconds: float,
        queue_wait_seconds: float | None = None,
    ) -> None: ...

    def on_batch_completed(
        self, *, batch_size: int, pairs: int, inference_seconds: float, device: str
    ) -> None: ...

    def on_queue_full(self) -> None: ...

    def on_request_timeout(self) -> None: ...

    def on_mps_fallback(self) -> None: ...

    def on_batch_processing_failed(self) -> None: ...


class NullObserver:
    """Default observer; every method is a no-op."""

    def on_request_completed(self, **kwargs): pass
    def on_batch_completed(self, **kwargs): pass
    def on_queue_full(self): pass
    def on_request_timeout(self): pass
    def on_mps_fallback(self): pass
    def on_batch_processing_failed(self): pass


_observer: Observer = NullObserver()


def set_observer(observer: Observer) -> None:
    global _observer
    _observer = observer


def get_observer() -> Observer:
    return _observer
