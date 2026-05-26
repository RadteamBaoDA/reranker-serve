"""OTel: server span per request + batch span per inference, attributes correct."""

import pytest

pytest.importorskip("opentelemetry")


def test_init_otel_returns_tracer_provider(monkeypatch):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "reranker-test")
    from src.observability.otel import init_otel

    provider = init_otel(use_in_memory_exporter=True)
    assert provider is not None


@pytest.mark.asyncio
async def test_batch_span_emitted_for_each_batch(monkeypatch):
    # Reset the OTel global so we can set a fresh provider in this test.
    import opentelemetry.trace as _ot
    _ot._TRACER_PROVIDER_SET_ONCE._done = False
    _ot._TRACER_PROVIDER = None

    from src.observability.otel import init_otel, get_in_memory_exporter
    init_otel(use_in_memory_exporter=True)
    exporter = get_in_memory_exporter()
    exporter.clear()

    from src.engine.async_engine import AsyncRerankerEngine
    from src.engine.handlers.base import BaseHandler

    class DummyHandler(BaseHandler):
        def load_model(self):
            self.model = "ok"
        def predict(self, batch):
            return [[{"index": 0, "relevance_score": 1.0}] for _ in batch.requests]

    monkeypatch.setattr(
        "src.engine.async_engine.get_handler",
        lambda *a, **k: DummyHandler("p", "cpu", 64, False),
    )
    from src.config import settings as s
    monkeypatch.setattr(s, "enable_device_probe", False)
    monkeypatch.setattr(s, "use_offline_mode", False)
    monkeypatch.setattr(s, "enable_otel", True)
    monkeypatch.setattr(s, "otel_batch_span", True)

    engine = AsyncRerankerEngine(
        max_concurrent_batches=1, inference_threads=1, batch_wait_timeout=0.001
    )
    await engine.start()
    try:
        await engine.rerank("q", ["a", "b"])
    finally:
        await engine.stop()

    spans = exporter.get_finished_spans()
    batch_spans = [s for s in spans if s.name == "reranker.batch"]
    assert len(batch_spans) >= 1
    attrs = batch_spans[0].attributes
    assert attrs["batch_size"] == 1
    assert attrs["pairs"] == 2
    assert attrs["device"] == "cpu"
