"""Async engine and request queue behavior using a dummy handler."""

import asyncio
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

    # Do not start engine; just ensure stats keys exist
    assert set(["running", "model_loaded", "total_requests"]).issubset(stats.keys())
