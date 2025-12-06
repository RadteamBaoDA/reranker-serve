"""
Test fixtures and configuration for pytest.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Set test environment variables before importing app modules
os.environ["RERANKER_USE_OFFLINE_MODE"] = "false"
os.environ["RERANKER_MODEL_NAME"] = "BAAI/bge-reranker-v2-m3"
# Disable async engine for tests to use simple mocking
os.environ["RERANKER_ENABLE_ASYNC_ENGINE"] = "false"

# Create mock torch module to avoid DLL loading issues on Windows
mock_torch = MagicMock()
mock_torch.__version__ = "2.0.0"
mock_torch.cuda.is_available.return_value = False
mock_torch.backends.mps.is_available.return_value = False
mock_torch.backends.mps.is_built.return_value = False
mock_torch.float16 = "float16"
mock_torch.float32 = "float32"

# Mock sentence_transformers
mock_sentence_transformers = MagicMock()
mock_cross_encoder_class = MagicMock()
mock_sentence_transformers.CrossEncoder = mock_cross_encoder_class

# Insert mocks into sys.modules before any imports
sys.modules["torch"] = mock_torch
sys.modules["sentence_transformers"] = mock_sentence_transformers


@pytest.fixture
def mock_cross_encoder():
    """Mock CrossEncoder for testing without loading actual model."""
    with patch("src.models.reranker.CrossEncoder") as mock:
        # Create a mock model instance
        mock_instance = MagicMock()
        
        # Make predict return dynamic scores based on input length
        def dynamic_predict(pairs, *args, **kwargs):
            # Return a score for each pair
            num_pairs = len(pairs)
            # Generate scores that are decreasing to simulate ranking
            import random
            random.seed(42)  # For reproducibility
            scores = [0.9 - (i * 0.1) + random.uniform(-0.05, 0.05) for i in range(num_pairs)]
            return scores
        
        mock_instance.predict.side_effect = dynamic_predict
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
