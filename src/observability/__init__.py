"""Observability package — env-gated Prometheus, OTel, and graceful shutdown."""

from src.observability.observer import (
    NullObserver,
    Observer,
    get_observer,
    set_observer,
)

__all__ = ["Observer", "NullObserver", "get_observer", "set_observer"]
