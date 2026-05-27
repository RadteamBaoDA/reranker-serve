"""Graceful drain: in-flight requests complete after SIGTERM signal."""

import asyncio
import pytest


@pytest.mark.asyncio
async def test_engine_rejects_new_requests_once_shutting_down(monkeypatch):
    from src.engine.async_engine import AsyncRerankerEngine
    from src.engine.handlers.base import BaseHandler
    from src.engine.request_queue import QueueFullError

    class DummyHandler(BaseHandler):
        def load_model(self): self.model = "ok"
        def predict(self, batch):
            return [[{"index": 0, "relevance_score": 1.0}] for _ in batch.requests]

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
        engine.begin_shutdown()
        with pytest.raises(QueueFullError):
            await engine.rerank("q", ["a"])
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_engine_drains_inflight_requests_on_stop(monkeypatch):
    """Issue a slow request, then stop(); the in-flight request must complete."""
    import time
    from src.engine.async_engine import AsyncRerankerEngine
    from src.engine.handlers.base import BaseHandler

    class SlowHandler(BaseHandler):
        def load_model(self): self.model = "ok"
        def predict(self, batch):
            time.sleep(0.05)
            return [[{"index": 0, "relevance_score": 1.0}] for _ in batch.requests]

    monkeypatch.setattr(
        "src.engine.async_engine.get_handler",
        lambda *a, **k: SlowHandler("p", "cpu", 64, False),
    )
    from src.config import settings as s
    monkeypatch.setattr(s, "enable_device_probe", False)

    engine = AsyncRerankerEngine(
        max_concurrent_batches=1, inference_threads=1, batch_wait_timeout=0.001
    )
    await engine.start()

    rerank_task = asyncio.create_task(engine.rerank("q", ["a"]))
    await asyncio.sleep(0.005)  # let it enter inference
    stop_task = asyncio.create_task(engine.stop())

    result = await rerank_task
    await stop_task
    assert len(result) == 1
