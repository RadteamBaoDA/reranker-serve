"""Unit tests for the startup device probe."""

import os

import pytest

from src.engine.device_probe import (
    DeviceProfile,
    ProbeResult,
    PROBE_BATCH_SIZES,
    _pick_suggested_batch_size,
    run_device_probe,
)
from src.engine.handlers.base import BaseHandler


class FastHandler(BaseHandler):
    def __init__(self):
        super().__init__("path", "cpu", 64, False)
        self.calls = 0

    def load_model(self):
        self.model = "ok"

    def predict(self, batch):
        self.calls += 1
        return [
            [{"index": i, "relevance_score": 0.5} for i, _ in enumerate(req.documents)]
            for req in batch.requests
        ]


def test_pick_suggested_batch_size_picks_largest_within_2x():
    probes = [
        ProbeResult(batch_size=1, pairs=4, elapsed_ms=4.0),    # 1.0 ms/pair
        ProbeResult(batch_size=8, pairs=32, elapsed_ms=48.0),  # 1.5 ms/pair (within 2x)
        ProbeResult(batch_size=32, pairs=128, elapsed_ms=384.0),  # 3.0 ms/pair (over 2x)
    ]
    assert _pick_suggested_batch_size(probes) == 8


def test_pick_suggested_batch_size_picks_largest_when_all_scale():
    probes = [
        ProbeResult(batch_size=1, pairs=4, elapsed_ms=4.0),
        ProbeResult(batch_size=8, pairs=32, elapsed_ms=32.0),
        ProbeResult(batch_size=32, pairs=128, elapsed_ms=128.0),
    ]
    assert _pick_suggested_batch_size(probes) == 32


def test_pick_suggested_batch_size_single_probe():
    probes = [ProbeResult(batch_size=1, pairs=4, elapsed_ms=4.0)]
    assert _pick_suggested_batch_size(probes) == 1


def test_pick_suggested_batch_size_empty():
    assert _pick_suggested_batch_size([]) == PROBE_BATCH_SIZES[0]


def test_run_device_probe_returns_profile_for_each_batch_size():
    handler = FastHandler()
    profile = run_device_probe(handler, "cpu")
    assert profile is not None
    assert isinstance(profile, DeviceProfile)
    assert [p.batch_size for p in profile.probes] == list(PROBE_BATCH_SIZES)
    assert handler.calls == len(PROBE_BATCH_SIZES)
    assert profile.device == "cpu"
    assert profile.suggested_batch_size in PROBE_BATCH_SIZES


def test_run_device_probe_records_user_pinned(monkeypatch):
    monkeypatch.setenv("RERANKER_BATCH_SIZE", "16")
    profile = run_device_probe(FastHandler(), "cpu")
    assert profile.user_pinned_batch_size is True


def test_run_device_probe_user_unpinned_by_default(monkeypatch):
    monkeypatch.delenv("RERANKER_BATCH_SIZE", raising=False)
    profile = run_device_probe(FastHandler(), "cpu")
    assert profile.user_pinned_batch_size is False


def test_run_device_probe_returns_none_when_handler_raises_immediately():
    class FailingHandler(FastHandler):
        def predict(self, batch):
            raise RuntimeError("synthetic failure")

    profile = run_device_probe(FailingHandler(), "cpu")
    assert profile is None


def test_device_profile_to_dict_roundtrip():
    profile = DeviceProfile(
        device="cpu",
        probes=[ProbeResult(batch_size=1, pairs=4, elapsed_ms=5.0)],
        suggested_batch_size=1,
        user_pinned_batch_size=False,
    )
    payload = profile.to_dict()
    assert payload["device"] == "cpu"
    assert payload["suggested_batch_size"] == 1
    assert payload["probes"][0]["ms_per_pair"] == pytest.approx(1.25)
