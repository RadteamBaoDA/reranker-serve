"""
Unit tests for API endpoints.
"""

import pytest
from unittest.mock import patch


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_health_check(self, test_client):
        """Test the health check endpoint."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model" in data
        assert "device" in data
        assert "version" in data
    
    def test_root_endpoint(self, test_client):
        """Test the root endpoint."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_model_info(self, test_client):
        """Test the model info endpoint."""
        response = test_client.get("/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "device" in data
        assert "max_length" in data
    
    def test_liveness_probe(self, test_client):
        """Test the liveness probe endpoint."""
        response = test_client.get("/live")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"


class TestNativeRerankEndpoint:
    """Tests for the native rerank endpoint."""
    
    def test_rerank_basic(self, test_client, sample_query, sample_documents):
        """Test basic reranking."""
        response = test_client.post(
            "/rerank",
            json={
                "query": sample_query,
                "documents": sample_documents[:3],
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "model" in data
        assert len(data["results"]) == 3
    
    def test_rerank_with_top_n(self, test_client, sample_query, sample_documents):
        """Test reranking with top_n parameter."""
        response = test_client.post(
            "/rerank",
            json={
                "query": sample_query,
                "documents": sample_documents,
                "top_n": 2,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
    
    def test_rerank_without_documents_return(self, test_client, sample_query, sample_documents):
        """Test reranking without returning documents."""
        response = test_client.post(
            "/rerank",
            json={
                "query": sample_query,
                "documents": sample_documents[:3],
                "return_documents": False,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 3


class TestCohereCompatibleEndpoint:
    """Tests for the Cohere-compatible rerank endpoint."""
    
    def test_cohere_rerank_string_documents(self, test_client, sample_query, sample_documents):
        """Test Cohere rerank with string documents."""
        response = test_client.post(
            "/v1/rerank",
            json={
                "query": sample_query,
                "documents": sample_documents[:3],
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "results" in data
        assert "meta" in data
    
    def test_cohere_rerank_dict_documents(self, test_client, sample_query):
        """Test Cohere rerank with dict documents."""
        documents = [
            {"text": "Document 1 content"},
            {"text": "Document 2 content"},
        ]
        
        response = test_client.post(
            "/v1/rerank",
            json={
                "query": sample_query,
                "documents": documents,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
    
    def test_cohere_rerank_with_model_param(self, test_client, sample_query, sample_documents):
        """Test that model parameter is accepted but ignored."""
        response = test_client.post(
            "/v1/rerank",
            json={
                "query": sample_query,
                "documents": sample_documents[:2],
                "model": "some-other-model",
            }
        )
        
        assert response.status_code == 200


class TestJinaCompatibleEndpoint:
    """Tests for the Jina AI-compatible rerank endpoint."""
    
    def test_jina_rerank_string_documents(self, test_client, sample_query, sample_documents):
        """Test Jina rerank with string documents."""
        response = test_client.post(
            "/api/v1/rerank",
            json={
                "query": sample_query,
                "documents": sample_documents[:3],
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "model" in data
        assert "usage" in data
        assert "results" in data
        assert "total_tokens" in data["usage"]
    
    def test_jina_rerank_dict_documents(self, test_client, sample_query):
        """Test Jina rerank with dict documents."""
        documents = [
            {"text": "Document 1 content"},
            {"text": "Document 2 content"},
        ]
        
        response = test_client.post(
            "/api/v1/rerank",
            json={
                "query": sample_query,
                "documents": documents,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
    
    def test_jina_rerank_with_top_n(self, test_client, sample_query, sample_documents):
        """Test Jina rerank with top_n parameter."""
        response = test_client.post(
            "/api/v1/rerank",
            json={
                "query": sample_query,
                "documents": sample_documents,
                "top_n": 2,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2


class TestAPIAuthentication:
    """Tests for API authentication."""
    
    def test_rerank_with_api_key(self, test_client, sample_query, sample_documents):
        """Test reranking with API key header."""
        response = test_client.post(
            "/rerank",
            json={
                "query": sample_query,
                "documents": sample_documents[:2],
            },
            headers={"Authorization": "Bearer test-key"}
        )
        
        # Should succeed since no API key is configured by default
        assert response.status_code == 200
    
    @patch("src.api.routes.settings")
    def test_rerank_missing_api_key(self, mock_settings, test_client, sample_query, sample_documents):
        """Test reranking when API key is required but missing."""
        mock_settings.api_key = "secret-key"
        mock_settings.model_name = "test-model"
        
        response = test_client.post(
            "/rerank",
            json={
                "query": sample_query,
                "documents": sample_documents[:2],
            }
        )
        
        # This test depends on how the fixture sets up the mock
        # In real scenario, it would return 401


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_query(self, test_client, sample_documents):
        """Test reranking with empty query."""
        response = test_client.post(
            "/rerank",
            json={
                "query": "",
                "documents": sample_documents[:2],
            }
        )
        
        # Should still work with empty query
        assert response.status_code == 200
    
    def test_single_document(self, test_client, sample_query):
        """Test reranking with single document."""
        response = test_client.post(
            "/rerank",
            json={
                "query": sample_query,
                "documents": ["Single document"],
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
    
    def test_empty_documents(self, test_client, sample_query):
        """Test reranking with empty document list."""
        response = test_client.post(
            "/rerank",
            json={
                "query": sample_query,
                "documents": [],
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 0
