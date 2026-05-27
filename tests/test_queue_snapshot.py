"""Queue introspection: waiting registry + engine running snapshot."""

import time

import pytest

from src.engine.request_queue import RequestQueue, RerankRequest


@pytest.mark.asyncio
async def test_waiting_registry_lists_enqueued_in_order():
    q = RequestQueue(max_queue_size=10, request_timeout=5.0)
    await q.add_request(RerankRequest(request_id="r1", query="q", documents=["a", "b"]))
    await q.add_request(RerankRequest(request_id="r2", query="q", documents=["c"]))

    waiting = q.get_waiting_snapshot()
    assert [w["request_id"] for w in waiting] == ["r1", "r2"]
    assert waiting[0]["num_docs"] == 2 and waiting[1]["num_docs"] == 1
    assert all(w["waited_ms"] >= 0 for w in waiting)


@pytest.mark.asyncio
async def test_batched_requests_leave_the_waiting_registry():
    q = RequestQueue(max_queue_size=10, batch_wait_timeout=0.005, request_timeout=5.0)
    await q.add_request(RerankRequest(request_id="r1", query="q", documents=["a"]))
    await q.add_request(RerankRequest(request_id="r2", query="q", documents=["b"]))
    batch = await q.get_batch()
    assert batch is not None and len(batch.requests) == 2
    assert q.get_waiting_snapshot() == []


def test_engine_queue_snapshot_shape():
    from src.engine.async_engine import AsyncRerankerEngine
    engine = AsyncRerankerEngine(max_concurrent_batches=1, inference_threads=1)
    # Simulate one in-flight batch.
    engine._inflight_meta["b1"] = {
        "batch_id": "b1", "request_ids": ["r1", "r2"],
        "num_requests": 2, "pairs": 5, "started_at": time.time(),
    }
    snap = engine.get_queue_snapshot()
    assert snap["waiting"] == []
    assert len(snap["running"]) == 1
    assert snap["running"][0]["batch_id"] == "b1"
    assert snap["running"][0]["pairs"] == 5
    assert snap["running"][0]["elapsed_ms"] >= 0
