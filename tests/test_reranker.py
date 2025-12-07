"""Unit coverage for handler factory, handlers, and reranker utilities."""

import numpy as np

from src.engine.handlers.cross_encoder import CrossEncoderHandler
from src.engine.handlers.factory import get_handler
from src.engine.handlers.qwen import QwenRerankerHandler
from src.engine.request_queue import BatchedRequest, RerankRequest
from src.models.reranker import RerankerModel
from src.models.qwen3_reranker import is_qwen3_reranker


def _batched_request(query: str, docs):
    return BatchedRequest(batch_id="b1", requests=[RerankRequest("r1", query, docs)])


def test_factory_selects_qwen_handler():
    handler = get_handler("Qwen3-Reranker-0.5B", device="cpu", max_length=128, use_fp16=False)
    assert isinstance(handler, QwenRerankerHandler)


def test_factory_defaults_to_cross_encoder():
    handler = get_handler("BAAI/bge-reranker-v2-m3", device="cpu", max_length=128, use_fp16=False)
    assert isinstance(handler, CrossEncoderHandler)


def test_cross_encoder_handler_predict_sorts_and_truncates():
    handler = CrossEncoderHandler("model", "cpu", 16, False)
    handler.model = type(
        "Dummy",
        (),
        {"predict": lambda self, pairs, **kwargs: [0.2, 0.8]},
    )()
    batch = _batched_request("q", ["doc-a", "doc-b"])
    results = handler.predict(batch)[0]
    assert results[0]["relevance_score"] >= results[1]["relevance_score"]
    assert {r["document"]["text"] for r in results} == {"doc-a", "doc-b"}


def test_qwen_handler_predict_delegates_to_model():
    handler = QwenRerankerHandler("qwen", "cpu", 32, False)

    class DummyQwen:
        def rerank(self, query, documents, top_k=None, return_documents=True):
            return [
                {"index": idx, "relevance_score": 0.5 + idx, "document": {"text": doc}}
                for idx, doc in enumerate(documents)
            ][: top_k or len(documents)]

    handler.model = DummyQwen()
    batch = _batched_request("q", ["a", "b", "c"])
    results = handler.predict(batch)[0]
    assert len(results) == 3
    assert results[0]["relevance_score"] <= results[-1]["relevance_score"]


def test_reranker_model_normalize_scores():
    model = RerankerModel()
    normalized = model._normalize_scores(np.array([2.0, 0.0, -2.0]))
    assert all(0 <= s <= 1 for s in normalized)
    assert normalized[0] > normalized[1] > normalized[2]


def test_is_qwen3_reranker_helper():
    assert is_qwen3_reranker("Qwen3-Reranker-0.5B")
    assert not is_qwen3_reranker("BAAI/bge-reranker-v2-m3")
