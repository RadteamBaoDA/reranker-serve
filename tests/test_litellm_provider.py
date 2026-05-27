"""LiteLLM custom-provider adapter — request shape and response mapping."""

import json

import pytest

pytest.importorskip("litellm")

import httpx

from src.integrations.litellm_provider import RerankerLiteLLMProvider


_FAKE_RESPONSE = {
    "id": "rr-test-1",
    "results": [
        {"index": 0, "relevance_score": 0.91, "document": {"text": "a"}},
        {"index": 2, "relevance_score": 0.42, "document": {"text": "c"}},
    ],
    "meta": {"api_version": {"version": "1"}},
}


def _capture_transport(captured: dict) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["json"] = json.loads(request.content.decode())
        return httpx.Response(200, json=_FAKE_RESPONSE)

    return httpx.MockTransport(handler)


def test_rerank_posts_to_native_endpoint_and_maps_response(monkeypatch):
    provider = RerankerLiteLLMProvider()
    captured: dict = {}

    # Patch httpx.Client to use a MockTransport that captures the request
    original_client = httpx.Client

    def patched_client(*args, **kwargs):
        kwargs["transport"] = _capture_transport(captured)
        return original_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", patched_client)

    response = provider.rerank(
        model="local",
        query="what is deep learning",
        documents=["a", "b", "c"],
        top_n=2,
        return_documents=True,
        api_base="http://reranker.local:8000",
        api_key="secret-token",
    )

    assert captured["url"] == "http://reranker.local:8000/rerank"
    assert captured["headers"].get("authorization") == "Bearer secret-token"
    assert captured["json"] == {
        "query": "what is deep learning",
        "documents": ["a", "b", "c"],
        "return_documents": True,
        "top_n": 2,
    }
    assert response.id == "rr-test-1"
    assert len(response.results) == 2
    assert response.results[0].index == 0
    assert response.results[0].relevance_score == pytest.approx(0.91)
    assert response.results[1].index == 2


def test_rerank_omits_auth_header_for_dummy_key(monkeypatch):
    provider = RerankerLiteLLMProvider()
    captured: dict = {}
    original_client = httpx.Client

    def patched_client(*args, **kwargs):
        kwargs["transport"] = _capture_transport(captured)
        return original_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", patched_client)

    provider.rerank(
        model="local",
        query="q",
        documents=["d"],
        api_base="http://reranker.local:8000",
        api_key="dummy",
    )

    assert "authorization" not in captured["headers"]


def test_rerank_omits_top_n_when_none(monkeypatch):
    provider = RerankerLiteLLMProvider()
    captured: dict = {}
    original_client = httpx.Client

    def patched_client(*args, **kwargs):
        kwargs["transport"] = _capture_transport(captured)
        return original_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", patched_client)

    provider.rerank(
        model="local",
        query="q",
        documents=["d"],
        api_base="http://reranker.local:8000",
    )

    assert "top_n" not in captured["json"]


@pytest.mark.asyncio
async def test_arerank_uses_async_client(monkeypatch):
    provider = RerankerLiteLLMProvider()
    captured: dict = {}

    original_async_client = httpx.AsyncClient

    def patched_async_client(*args, **kwargs):
        kwargs["transport"] = _capture_transport(captured)
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_async_client)

    response = await provider.arerank(
        model="local",
        query="q",
        documents=["a", "b"],
        top_n=1,
        api_base="http://reranker.local:8000",
        api_key="dummy",
    )

    assert captured["url"] == "http://reranker.local:8000/rerank"
    assert response.id == "rr-test-1"
