"""Unit tests for engine handlers."""

import pytest

from src.engine.handlers.base import BaseHandler
from src.engine.handlers.cross_encoder import CrossEncoderHandler
from src.engine.handlers.factory import get_handler
from src.engine.handlers.qwen import QwenRerankerHandler
from src.engine.request_queue import RerankRequest, BatchedRequest
from src.models.qwen3_reranker import Qwen3Reranker
from src.config import settings


class DummyHandler(BaseHandler):
    def load_model(self):
        self.model = "loaded"

    def predict(self, batch):
        return [[{"index": 0, "relevance_score": 1.0} for _ in batch.requests]]


def make_batch(top_k=None, return_documents=True):
    req = RerankRequest(
        request_id="r1",
        query="q",
        documents=["d1", "d2", "d3"],
        top_k=top_k,
        return_documents=return_documents,
    )
    return BatchedRequest(batch_id="b1", requests=[req])


def test_base_handler_unload_sets_model_none():
    handler = DummyHandler("path", "cpu", 128, False)
    handler.model = object()
    handler.unload()
    assert handler.model is None


def test_cross_encoder_handler_loads_with_trust_remote(monkeypatch):
    created = []

    class FakeTokenizer:
        def __init__(self):
            self.pad_token = None
            self.pad_token_id = None
            self.eos_token = "</s>"
            self.eos_token_id = 1
            self.padding_side = None

    class FakeCrossEncoder:
        def __init__(self, path, **kwargs):
            created.append((path, kwargs))
            self.tokenizer = FakeTokenizer()

    monkeypatch.setattr("src.engine.handlers.cross_encoder.CrossEncoder", FakeCrossEncoder)

    handler = CrossEncoderHandler("qwen-model", "cpu", 128, False)
    handler.load_model()

    assert isinstance(handler.model, FakeCrossEncoder)
    assert created[0][1]["trust_remote_code"] is True
    assert handler.model.tokenizer.pad_token == "</s>"
    assert handler.model.tokenizer.padding_side == "right"


def test_cross_encoder_handler_predict_sorts_and_limits(monkeypatch):
    class FakeCrossEncoder:
        def predict(self, pairs, batch_size, show_progress_bar):
            return [0.1, 0.9, 0.2]

    monkeypatch.setattr(settings, "normalize_scores", False)

    handler = CrossEncoderHandler("model", "cpu", 128, False)
    handler.model = FakeCrossEncoder()

    batch = make_batch(top_k=2, return_documents=True)
    results = handler.predict(batch)[0]

    assert len(results) == 2
    assert [r["index"] for r in results] == [1, 2]
    assert all("document" in r for r in results)


def test_cross_encoder_handler_predict_normalizes_scores(monkeypatch):
    class FakeCrossEncoder:
        def predict(self, pairs, batch_size, show_progress_bar):
            return [-1, 1]

    monkeypatch.setattr(settings, "normalize_scores", True)

    handler = CrossEncoderHandler("model", "cpu", 128, False)
    handler.model = FakeCrossEncoder()

    batch = BatchedRequest(
        batch_id="b2",
        requests=[RerankRequest(request_id="r2", query="q", documents=["a", "b"], top_k=None)],
    )
    results = handler.predict(batch)[0]

    assert results[0]["relevance_score"] > results[1]["relevance_score"]
    assert 0 < results[0]["relevance_score"] <= 1


def test_qwen_handler_load_and_predict(monkeypatch):
    loaded = {
        "load_called": False,
        "rerank_called": False,
    }

    class FakeQwen(Qwen3Reranker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            loaded["init_args"] = args

        def load(self):
            loaded["load_called"] = True
            return self

        def rerank(self, query, documents, top_k=None, return_documents=True, instruction=None):
            loaded["rerank_called"] = True
            return [
                {"index": 0, "relevance_score": 0.5},
                {"index": 1, "relevance_score": 0.4},
            ]

    monkeypatch.setattr("src.engine.handlers.qwen.Qwen3Reranker", FakeQwen)

    handler = QwenRerankerHandler("Qwen3-reranker", "cpu", 64, False)
    handler.load_model()

    batch = make_batch(top_k=None)
    results = handler.predict(batch)[0]

    assert loaded["load_called"] is True
    assert loaded["rerank_called"] is True
    assert results[0]["index"] == 0


def test_handler_factory_selects_qwen():
    handler = get_handler("qwen3-reranker-0.5b", "cpu", 128, False)
    assert isinstance(handler, QwenRerankerHandler)

    handler2 = get_handler("baai/bge", "cpu", 128, False)
    assert isinstance(handler2, CrossEncoderHandler)


def test_qwen3_compute_logits_falls_back_from_mps_to_cpu(monkeypatch):
    """
    Trigger the documented MPS kernel limitation (MPSGraph error) and verify
    that _compute_logits transparently retries on CPU when
    settings.mps_fallback_to_cpu is enabled.
    """
    import torch

    monkeypatch.setattr(settings, "mps_fallback_to_cpu", True)

    reranker = Qwen3Reranker(model_name_or_path="fake/qwen3-reranker", device="mps")
    reranker._token_true_id = 1
    reranker._token_false_id = 0

    call_count = {"forward": 0}

    class FakeOutputs:
        def __init__(self, batch_size: int):
            self.logits = torch.zeros(batch_size, 4, 8)
            self.logits[:, -1, 1] = 5.0  # "yes" token gets a high score
            self.logits[:, -1, 0] = -5.0

    class FakeModel:
        def __init__(self, device_name: str = "mps"):
            self._device_name = device_name

        @property
        def device(self):
            return torch.device(self._device_name)

        def __call__(self, **inputs):
            call_count["forward"] += 1
            if self._device_name == "mps":
                raise RuntimeError("MPSGraph: tensor exceeds INT_MAX")
            input_ids = inputs["input_ids"]
            return FakeOutputs(batch_size=input_ids.shape[0])

        def cpu(self):
            return FakeModel(device_name="cpu")

    fake_inputs = {
        "input_ids": torch.zeros((2, 4), dtype=torch.long),
        "attention_mask": torch.ones((2, 4), dtype=torch.long),
    }
    reranker._model = FakeModel(device_name="mps")

    scores = reranker._compute_logits(fake_inputs)

    assert len(scores) == 2
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert reranker.device == "cpu"
    assert call_count["forward"] == 2  # mps raised, cpu succeeded


def test_qwen3_compute_logits_does_not_fall_back_when_disabled(monkeypatch):
    """When mps_fallback_to_cpu=False, the MPS kernel error must propagate."""
    import torch

    monkeypatch.setattr(settings, "mps_fallback_to_cpu", False)

    reranker = Qwen3Reranker(model_name_or_path="fake/qwen3-reranker", device="mps")
    reranker._token_true_id = 1
    reranker._token_false_id = 0

    class FailingModel:
        @property
        def device(self):
            return torch.device("mps")

        def __call__(self, **inputs):
            raise RuntimeError("MPSGraph: tensor exceeds INT_MAX")

    reranker._model = FailingModel()
    fake_inputs = {"input_ids": torch.zeros((1, 2), dtype=torch.long)}

    with pytest.raises(RuntimeError, match="MPSGraph"):
        reranker._compute_logits(fake_inputs)
