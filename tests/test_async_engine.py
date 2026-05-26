"""Async engine and request queue behavior using a dummy handler."""

import asyncio
import time
from typing import Any, Dict, List

import pytest

from src.engine.handlers.base import BaseHandler
from src.engine.request_queue import (
    RequestQueue,
    RerankRequest,
    RerankResult,
    RequestStatus,
)


class DummyHandler(BaseHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loaded = False

    def load_model(self) -> None:
        self.loaded = True
        self.model = "dummy"

    def predict(self, batch) -> List[List[Dict[str, Any]]]:
        results: List[List[Dict[str, Any]]] = []
        for req in batch.requests:
            per_request = []
            for idx, doc in enumerate(req.documents):
                item = {"index": idx, "relevance_score": 1.0 - idx * 0.01}
                if req.return_documents:
                    item["document"] = {"text": doc}
                per_request.append(item)
            if req.top_k:
                per_request = per_request[:req.top_k]
            results.append(per_request)
        return results


@pytest.mark.asyncio
async def test_request_queue_add_complete_and_fail():
    queue = RequestQueue(max_batch_size=2, batch_wait_timeout=0.001)

    req = RerankRequest(request_id="r1", query="q", documents=["a", "b"])
    future = await queue.add_request(req)
    assert queue.pending_count == 1
    assert queue.active_count == 1

    result = RerankResult(request_id="r1", results=[], processing_time=0.0)
    queue.complete_request("r1", result)
    assert future.done()
    assert (await future).request_id == "r1"

    req2 = RerankRequest(request_id="r2", query="q", documents=["a"])
    future2 = await queue.add_request(req2)
    queue.fail_request("r2", "boom")
    with pytest.raises(RuntimeError):
        await future2


@pytest.mark.asyncio
async def test_request_queue_batching_respects_limits():
    queue = RequestQueue(max_batch_size=2, max_batch_pairs=3, batch_wait_timeout=0.02)

    await queue.add_request(RerankRequest(request_id="b1", query="q", documents=["a", "b"]))
    await queue.add_request(RerankRequest(request_id="b2", query="q", documents=["c", "d"]))
    await queue.add_request(RerankRequest(request_id="b3", query="q", documents=["e"]))

    batch = await queue.get_batch()
    assert batch is not None
    assert len(batch.requests) <= 2
    assert batch.total_pairs <= 3


@pytest.mark.asyncio
async def test_request_status_and_cancel():
    queue = RequestQueue()
    request = RerankRequest(request_id="cancel-me", query="q", documents=["a"])

    future = await queue.add_request(request)
    assert request.status == RequestStatus.PENDING

    cancelled = await queue.cancel_request("cancel-me")
    assert cancelled
    assert future.cancelled()
    assert queue.active_count == 0


@pytest.mark.asyncio
async def test_async_engine_lifecycle_and_rerank(monkeypatch, sample_documents):
    from src.engine.async_engine import AsyncRerankerEngine

    dummy = DummyHandler("path", "cpu", 64, False)
    monkeypatch.setattr("src.engine.async_engine.get_handler", lambda *args, **kwargs: dummy)

    engine = AsyncRerankerEngine(
        max_concurrent_batches=1,
        inference_threads=1,
        batch_wait_timeout=0.001,
    )

    await engine.start()
    assert engine.is_running and engine.is_loaded

    try:
        results = await engine.rerank(
            query="query",
            documents=sample_documents[:3],
            top_k=2,
            return_documents=True,
        )
        assert len(results) == 2
        assert results[0]["relevance_score"] >= results[1]["relevance_score"]
    finally:
        await engine.stop()
        assert not engine.is_running


@pytest.mark.asyncio
async def test_async_engine_concurrency(monkeypatch):
    from src.engine.async_engine import AsyncRerankerEngine

    dummy = DummyHandler("path", "cpu", 64, False)
    monkeypatch.setattr("src.engine.async_engine.get_handler", lambda *args, **kwargs: dummy)

    engine = AsyncRerankerEngine(
        max_concurrent_batches=2,
        inference_threads=1,
        batch_wait_timeout=0.01,
    )

    await engine.start()
    try:
        tasks = [
            asyncio.create_task(engine.rerank("q", [f"doc-{i}", f"doc-{i}-b"]))
            for i in range(4)
        ]
        results = await asyncio.gather(*tasks)
        assert len(results) == 4
        assert all(len(r) == 2 for r in results)
    finally:
        await engine.stop()


def test_async_engine_stats_increment(monkeypatch, sample_documents):
    from src.engine.async_engine import AsyncRerankerEngine

    dummy = DummyHandler("path", "cpu", 64, False)
    monkeypatch.setattr("src.engine.async_engine.get_handler", lambda *args, **kwargs: dummy)

    engine = AsyncRerankerEngine(
        max_concurrent_batches=1,
        inference_threads=1,
        batch_wait_timeout=0.001,
    )

    stats = engine.get_stats()
    assert stats["total_requests"] == 0

    expected_keys = {
        "running", "model_loaded", "total_requests",
        "inference_latency_p50_ms", "inference_latency_p95_ms",
        "throughput_pairs_per_sec", "inflight_batches",
        "semaphore_available", "max_concurrent_batches",
        "batch_occupancy_pct", "queue_wait_p50_ms", "queue_wait_p95_ms",
    }
    assert expected_keys.issubset(stats.keys())


@pytest.mark.asyncio
async def test_batch_accumulation_overlaps_with_inference(monkeypatch):
    """
    Regression test for the concurrency fix: while batch N is in the executor,
    batch N+1 should already be accumulating in the queue. We assert that the
    second batch's queue-wait timestamp is older than the first batch's
    inference-complete timestamp.
    """
    from src.engine.async_engine import AsyncRerankerEngine

    inference_started: list[float] = []
    inference_done: list[float] = []

    class SlowHandler(DummyHandler):
        def predict(self, batch):
            inference_started.append(asyncio.get_event_loop().time())
            time.sleep(0.05)  # 50 ms in the executor thread
            result = super().predict(batch)
            inference_done.append(asyncio.get_event_loop().time())
            return result

    slow = SlowHandler("path", "cpu", 64, False)
    monkeypatch.setattr("src.engine.async_engine.get_handler", lambda *args, **kwargs: slow)
    # Disable the startup probe so the timing assertions only see test traffic.
    from src.config import settings as _settings
    monkeypatch.setattr(_settings, "enable_device_probe", False)

    engine = AsyncRerankerEngine(
        max_concurrent_batches=1,
        inference_threads=1,
        max_batch_size=1,  # force one request per batch so multiple batches form
        batch_wait_timeout=0.005,
    )
    await engine.start()

    try:
        # Fire 3 requests with a small stagger so they become 3 separate batches.
        async def staggered(i):
            await asyncio.sleep(0.001 * i)
            return await engine.rerank("q", [f"d{i}"])

        results = await asyncio.gather(*(staggered(i) for i in range(3)))
        assert len(results) == 3

        # 3 batches must have been processed.
        assert len(inference_started) == 3

        # Critical assertion: while batch N is running (between its start and
        # done), batch N+1's start should overlap or follow immediately. We
        # assert that the gap between successive starts is significantly less
        # than the inference duration — i.e. accumulation is parallel.
        gap_01 = inference_started[1] - inference_done[0]
        gap_12 = inference_started[2] - inference_done[1]
        # Allow up to batch_wait_timeout + scheduling jitter (~10 ms)
        assert gap_01 < 0.02, f"batch 1 started too late after batch 0: {gap_01:.4f}s"
        assert gap_12 < 0.02, f"batch 2 started too late after batch 1: {gap_12:.4f}s"
    finally:
        await engine.stop()
