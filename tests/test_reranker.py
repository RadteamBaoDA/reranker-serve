"""
Unit tests for the reranker model.
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np


class TestRerankerModel:
    """Tests for the RerankerModel class."""
    
    def test_model_initialization(self, mock_cross_encoder):
        """Test model initialization with default settings."""
        from src.models.reranker import RerankerModel
        
        model = RerankerModel()
        
        assert model.model_name_or_path is not None
        assert model.device in ["cuda", "mps", "cpu"]
        assert model.max_length > 0
    
    def test_model_load(self, mock_cross_encoder):
        """Test model loading."""
        from src.models.reranker import RerankerModel
        
        model = RerankerModel()
        model.load()
        
        assert model.is_loaded
        mock_cross_encoder.assert_called_once()
    
    def test_model_rerank(self, mock_reranker_model, sample_query, sample_documents):
        """Test reranking functionality."""
        results = mock_reranker_model.rerank(
            query=sample_query,
            documents=sample_documents[:3],
            top_k=None,
            return_documents=True,
        )
        
        assert len(results) == 3
        assert all("index" in r for r in results)
        assert all("relevance_score" in r for r in results)
        assert all("document" in r for r in results)
        
        # Results should be sorted by score descending
        scores = [r["relevance_score"] for r in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_model_rerank_top_k(self, mock_reranker_model, sample_query, sample_documents):
        """Test reranking with top_k limit."""
        results = mock_reranker_model.rerank(
            query=sample_query,
            documents=sample_documents,
            top_k=2,
            return_documents=True,
        )
        
        assert len(results) == 2
    
    def test_model_rerank_empty_documents(self, mock_reranker_model, sample_query):
        """Test reranking with empty document list."""
        results = mock_reranker_model.rerank(
            query=sample_query,
            documents=[],
            top_k=None,
            return_documents=True,
        )
        
        assert len(results) == 0
    
    def test_model_rerank_without_documents(self, mock_reranker_model, sample_query, sample_documents):
        """Test reranking without returning documents."""
        results = mock_reranker_model.rerank(
            query=sample_query,
            documents=sample_documents[:3],
            top_k=None,
            return_documents=False,
        )
        
        assert len(results) == 3
        assert all("document" not in r or r["document"] is None for r in results)
    
    def test_model_unload(self, mock_reranker_model):
        """Test model unloading."""
        assert mock_reranker_model.is_loaded
        
        mock_reranker_model.unload()
        
        assert not mock_reranker_model.is_loaded
    
    def test_singleton_pattern(self, mock_cross_encoder):
        """Test that get_reranker_model returns singleton."""
        from src.models.reranker import get_reranker_model, reset_reranker_model
        
        reset_reranker_model()
        
        model1 = get_reranker_model()
        model2 = get_reranker_model()
        
        assert model1 is model2
        
        reset_reranker_model()


class TestScoreNormalization:
    """Tests for score normalization."""
    
    def test_normalize_scores(self, mock_cross_encoder):
        """Test score normalization."""
        from src.models.reranker import RerankerModel
        
        model = RerankerModel()
        
        # Test with sample scores
        scores = np.array([2.0, 0.0, -2.0])
        normalized = model._normalize_scores(scores)
        
        # All scores should be between 0 and 1
        assert all(0 <= s <= 1 for s in normalized)
        
        # Order should be preserved
        assert normalized[0] > normalized[1] > normalized[2]
