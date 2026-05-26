"""End-to-end: with the env var on, /metrics is exposed and request counters fire."""

import pytest

pytest.importorskip("prometheus_client")


def test_metrics_endpoint_disabled_by_default(test_client):
    resp = test_client.get("/metrics")
    assert resp.status_code == 404


@pytest.fixture
def test_client_with_metrics(monkeypatch, dummy_engine):
    monkeypatch.setenv("RERANKER_EXPOSE_PROMETHEUS_METRICS", "true")
    from src.config import settings as s
    monkeypatch.setattr(s, "expose_prometheus_metrics", True)

    async def fake_get_async_engine():
        return dummy_engine
    monkeypatch.setattr("src.engine.get_async_engine", fake_get_async_engine)

    from src.main import create_app
    from fastapi.testclient import TestClient
    app = create_app()
    with TestClient(app) as client:
        yield client


def test_metrics_endpoint_exposes_text_when_enabled(test_client_with_metrics):
    resp = test_client_with_metrics.get("/metrics")
    assert resp.status_code == 200
    assert "reranker_requests_total" in resp.text


def test_rerank_request_increments_counter(test_client_with_metrics):
    test_client_with_metrics.post("/rerank", json={"query": "q", "documents": ["a"]})
    text = test_client_with_metrics.get("/metrics").text
    assert 'reranker_requests_total{route="/rerank",status="200"}' in text
