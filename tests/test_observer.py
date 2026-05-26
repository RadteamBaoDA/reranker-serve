"""Engine emits typed observability events to the configured observer."""

import asyncio
import pytest


@pytest.mark.asyncio
async def test_observer_receives_batch_completed_event(monkeypatch):
    from src.engine.async_engine import AsyncRerankerEngine
    from src.engine.handlers.base import BaseHandler
    from src.observability import set_observer, NullObserver

    events = []

    class RecordingObserver(NullObserver):
        def on_batch_completed(self, *, batch_size, pairs, inference_seconds, device):
            events.append(("batch", batch_size, pairs, device))

        def on_request_completed(self, *, route, status, total_seconds, queue_wait_seconds):
            events.append(("request", route, status))

    class DummyHandler(BaseHandler):
        def load_model(self):
            self.model = "ok"
        def predict(self, batch):
            return [[{"index": 0, "relevance_score": 1.0}] for _ in batch.requests]

    set_observer(RecordingObserver())
    try:
        monkeypatch.setattr(
            "src.engine.async_engine.get_handler",
            lambda *a, **k: DummyHandler("p", "cpu", 64, False),
        )
        from src.config import settings as s
        monkeypatch.setattr(s, "enable_device_probe", False)

        engine = AsyncRerankerEngine(
            max_concurrent_batches=1, inference_threads=1, batch_wait_timeout=0.001
        )
        await engine.start()
        try:
            await engine.rerank("q", ["a"])
        finally:
            await engine.stop()
    finally:
        set_observer(NullObserver())

    assert any(e[0] == "batch" for e in events)
    batch_event = next(e for e in events if e[0] == "batch")
    assert batch_event[1] == 1   # batch_size
    assert batch_event[2] == 1   # pairs
    assert batch_event[3] == "cpu"


def test_observer_receives_queue_full_event():
    from src.observability import set_observer, NullObserver
    from src.engine.request_queue import RequestQueue, RerankRequest, QueueFullError

    events = []

    class RecordingObserver(NullObserver):
        def on_queue_full(self):
            events.append("queue_full")

    set_observer(RecordingObserver())
    try:
        async def _run():
            queue = RequestQueue(max_queue_size=1, request_timeout=0.05)
            await queue.add_request(RerankRequest(request_id="x", query="q", documents=["a"]))
            with pytest.raises(QueueFullError):
                await queue.add_request(RerankRequest(request_id="y", query="q", documents=["b"]))

        asyncio.run(_run())
    finally:
        set_observer(NullObserver())

    assert events == ["queue_full"]
