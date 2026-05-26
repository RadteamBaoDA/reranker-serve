"""Backpressure: queue-full → HTTP 503 with Retry-After header."""

import pytest


@pytest.mark.asyncio
async def test_queue_full_raises_typed_error():
    from src.engine.request_queue import QueueFullError, RequestQueue, RerankRequest

    queue = RequestQueue(max_queue_size=1, request_timeout=0.05)
    # Fill the queue
    await queue.add_request(RerankRequest(request_id="r1", query="q", documents=["a"]))

    # Second request can't enqueue within request_timeout → QueueFullError
    with pytest.raises(QueueFullError):
        await queue.add_request(RerankRequest(request_id="r2", query="q", documents=["b"]))


def test_route_returns_503_with_retry_after_when_queue_full(test_client, monkeypatch):
    from src.engine.request_queue import QueueFullError

    class _Engine:
        is_running = True
        is_loaded = True
        device = "cpu"
        device_profile = None
        async def rerank(self, *args, **kwargs):
            raise QueueFullError("queue is full")

    async def fake_get_async_engine():
        return _Engine()

    monkeypatch.setattr("src.engine.get_async_engine", fake_get_async_engine)

    resp = test_client.post("/rerank", json={"query": "q", "documents": ["a"]})
    assert resp.status_code == 503
    assert resp.headers.get("Retry-After") == "1"
