"""API endpoint tests using a stubbed async engine."""

import pytest

from src.config import settings


class TestHealthEndpoints:
    def test_health_check(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model"] == settings.model_name

    def test_root_endpoint(self, test_client):
        resp = test_client.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_model_info(self, test_client):
        resp = test_client.get("/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == settings.model_name
        assert "max_length" in data
        assert data["async_engine_enabled"] is True


class TestNativeRerankEndpoint:
    def test_rerank_basic(self, test_client, sample_query, sample_documents):
        resp = test_client.post(
            "/rerank",
            json={"query": sample_query, "documents": sample_documents[:3]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 3
        assert {r["index"] for r in data["results"]} == {0, 1, 2}

    def test_rerank_with_top_n(self, test_client, sample_query, sample_documents):
        resp = test_client.post(
            "/rerank",
            json={"query": sample_query, "documents": sample_documents, "top_n": 2},
        )
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 2

    def test_rerank_without_documents_return(self, test_client, sample_query, sample_documents):
        resp = test_client.post(
            "/rerank",
            json={
                "query": sample_query,
                "documents": sample_documents[:2],
                "return_documents": False,
            },
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 2
        assert all("document" not in r or r["document"] is None for r in results)


class TestCohereCompatibleEndpoint:
    def test_cohere_rerank_string_documents(self, test_client, sample_query, sample_documents):
        resp = test_client.post(
            "/v1/rerank",
            json={"query": sample_query, "documents": sample_documents[:3]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data and data["id"]
        assert len(data["results"]) == 3
        assert "meta" in data

    def test_cohere_rerank_dict_documents(self, test_client, sample_query):
        documents = [{"text": "Doc 1"}, {"text": "Doc 2"}]
        resp = test_client.post(
            "/v1/rerank",
            json={"query": sample_query, "documents": documents},
        )
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 2

    def test_cohere_rerank_with_model_param(self, test_client, sample_query, sample_documents):
        resp = test_client.post(
            "/v1/rerank",
            json={"query": sample_query, "documents": sample_documents[:2], "model": "other"},
        )
        assert resp.status_code == 200


class TestJinaCompatibleEndpoint:
    def test_jina_rerank_string_documents(self, test_client, sample_query, sample_documents):
        resp = test_client.post(
            "/api/v1/rerank",
            json={"query": sample_query, "documents": sample_documents[:3]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 3
        assert data["usage"]["total_tokens"] >= 0

    def test_jina_rerank_dict_documents(self, test_client, sample_query):
        documents = [{"text": "Doc 1"}, {"text": "Doc 2"}]
        resp = test_client.post(
            "/api/v1/rerank",
            json={"query": sample_query, "documents": documents, "top_n": 1},
        )
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 1


class TestAPIAuthentication:
    def test_rerank_with_api_key_header(self, monkeypatch, test_client, sample_query, sample_documents):
        monkeypatch.setattr(settings, "api_key", "secret")
        resp = test_client.post(
            "/rerank",
            headers={"Authorization": "Bearer secret"},
            json={"query": sample_query, "documents": sample_documents[:2]},
        )
        assert resp.status_code == 200

    def test_rerank_missing_api_key(self, monkeypatch, test_client, sample_query, sample_documents):
        monkeypatch.setattr(settings, "api_key", "secret")
        resp = test_client.post(
            "/rerank",
            json={"query": sample_query, "documents": sample_documents[:1]},
        )
        assert resp.status_code == 401


class TestEdgeCases:
    def test_empty_query(self, test_client, sample_documents):
        resp = test_client.post(
            "/rerank",
            json={"query": "", "documents": sample_documents[:2]},
        )
        assert resp.status_code == 200

    def test_single_document(self, test_client, sample_query):
        resp = test_client.post(
            "/rerank",
            json={"query": sample_query, "documents": ["Only doc"]},
        )
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 1

    def test_empty_documents(self, test_client, sample_query):
        resp = test_client.post(
            "/rerank",
            json={"query": sample_query, "documents": []},
        )
        assert resp.status_code == 200
        assert resp.json()["results"] == []
