"""
Test fixtures and configuration for pytest.
"""

import os
import pytest
from unittest.mock import MagicMock, patch

# Set test environment variables before importing app modules
os.environ["RERANKER_USE_OFFLINE_MODE"] = "false"
os.environ["RERANKER_MODEL_NAME"] = "BAAI/bge-reranker-v2-m3"


@pytest.fixture
def mock_cross_encoder():
    """Mock CrossEncoder for testing without loading actual model."""
    with patch("src.models.reranker.CrossEncoder") as mock:
        # Create a mock model instance
        mock_instance = MagicMock()
        mock_instance.predict.return_value = [0.9, 0.3, 0.7]
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_reranker_model(mock_cross_encoder):
    """Create a mock RerankerModel for testing."""
    from src.models.reranker import RerankerModel, reset_reranker_model
    
    # Reset any existing singleton
    reset_reranker_model()
    
    model = RerankerModel()
    model.load()
    
    yield model
    
    # Cleanup
    reset_reranker_model()


@pytest.fixture
def test_client(mock_cross_encoder):
    """Create a test client for the FastAPI app."""
    from fastapi.testclient import TestClient
    from src.models.reranker import reset_reranker_model
    
    # Reset model before creating client
    reset_reranker_model()
    
    # Import app after mocking
    from src.main import app
    
    client = TestClient(app)
    yield client
    
    # Cleanup
    reset_reranker_model()


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        "Deep learning is a subset of machine learning that uses neural networks.",
        "The weather is sunny and warm today.",
        "Natural language processing enables computers to understand human language.",
        "I enjoy playing chess on weekends.",
        "Transformers are a type of neural network architecture.",
    ]


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "What is deep learning and neural networks?"
