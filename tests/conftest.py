"""Shared pytest fixtures and lightweight dependency mocks."""

import os
import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# Keep tests deterministic and offline
os.environ.setdefault("RERANKER_USE_OFFLINE_MODE", "false")
os.environ.setdefault("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
os.environ.setdefault("RERANKER_ENABLE_ASYNC_ENGINE", "true")
os.environ.setdefault("RERANKER_ENABLE_LOAD_BALANCER", "false")

# Stub heavy dependencies before application imports
mock_torch = MagicMock()
mock_torch.__version__ = "2.0.0"
mock_torch.cuda.is_available.return_value = False
mock_torch.cuda.empty_cache = MagicMock()
mock_torch.backends.mps.is_available.return_value = False
mock_torch.backends.mps.is_built.return_value = False
mock_torch.float16 = "float16"
mock_torch.float32 = "float32"
sys.modules["torch"] = mock_torch

mock_sentence_transformers = MagicMock()
mock_sentence_transformers.CrossEncoder = MagicMock()
sys.modules["sentence_transformers"] = mock_sentence_transformers

# Transformers is only referenced when Qwen loads; stub to avoid import errors
sys.modules.setdefault("transformers", MagicMock())


@pytest.fixture
def sample_documents():
    return [
        "Deep learning is a subset of machine learning that uses neural networks.",
        "The weather is sunny and warm today.",
        "Natural language processing enables computers to understand human language.",
        "I enjoy playing chess on weekends.",
        "Transformers are a type of neural network architecture.",
    ]


@pytest.fixture
def sample_query():
    return "What is deep learning and neural networks?"


@pytest.fixture
def dummy_engine():
    class DummyEngine:
        def __init__(self):
            self.is_running = True
            self.is_loaded = True

        async def rerank(self, query, documents, top_k=None, return_documents=True, request_id=None):
            limit = top_k or len(documents)
            results = []
            for idx, doc in enumerate(documents[:limit]):
                entry = {"index": idx, "relevance_score": 1.0 / (idx + 1)}
                if return_documents:
                    entry["document"] = {"text": doc}
                results.append(entry)
            return results

        def get_stats(self):
            return {"pending_requests": 0, "total_requests": 0}

    return DummyEngine()


@pytest.fixture
def test_client(monkeypatch, dummy_engine):
    """Lightweight FastAPI test client with async engine stubbed."""

    async def fake_get_async_engine():
        return dummy_engine

    async def fake_reset_async_engine():
        dummy_engine.is_running = False

    monkeypatch.setattr("src.engine.get_async_engine", fake_get_async_engine)
    monkeypatch.setattr("src.engine.reset_async_engine", fake_reset_async_engine)

    from src.main import create_app

    app = create_app()
    with TestClient(app) as client:
        yield client
