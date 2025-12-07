"""Unit tests for model wrappers."""

import pytest

from src.models.reranker import RerankerModel, _is_qwen3_reranker
from src.models.qwen3_reranker import Qwen3Reranker, is_qwen3_reranker
from src.config import settings


class FakeCrossEncoder:
    def __init__(self, scores=None):
        self._scores = scores or []

    def predict(self, pairs, batch_size, show_progress_bar):
        return list(self._scores)


def test_reranker_normalize_scores():
    model = RerankerModel(model_name_or_path="dummy", device="cpu", max_length=16, use_fp16=False)
    scores = model._normalize_scores([0, 1, -1])
    assert all(0 <= s <= 1 for s in scores)
    assert scores[1] > scores[2]


def test_reranker_rerank_sorts_and_limits(monkeypatch):
    model = RerankerModel(model_name_or_path="dummy", device="cpu", max_length=8, use_fp16=False)
    model._model = FakeCrossEncoder(scores=[0.2, 0.9, 0.5])

    results = model.rerank("q", ["a", "b", "c"], top_k=2, return_documents=False)
    assert [r["index"] for r in results] == [1, 2]
    assert all("document" not in r for r in results)


def test_reranker_handles_empty_documents(monkeypatch):
    model = RerankerModel(model_name_or_path="dummy", device="cpu", max_length=8, use_fp16=False)
    model._model = FakeCrossEncoder(scores=[])
    assert model.rerank("q", [], top_k=None) == []


def test_is_qwen3_reranker_detection():
    assert _is_qwen3_reranker("Qwen3-reranker-0.5B") is True
    assert is_qwen3_reranker("qwen3-RERANKER-small") is True
    assert _is_qwen3_reranker("baai/bge") is False


def test_qwen3_formatting_helpers():
    model = Qwen3Reranker(model_name_or_path="qwen3-reranker", device="cpu", max_length=64, use_fp16=False)
    text = model._format_instruction("q?", "doc", instruction="instr")
    assert "<Query>: q?" in text and "<Document>: doc" in text

    model._prefix = "PRE-"
    model._suffix = "-POST"
    full = model._build_full_prompt("content")
    assert full.startswith("PRE-")
    assert full.endswith("-POST")


def test_qwen3_rerank_uses_stubbed_paths(monkeypatch):
    model = Qwen3Reranker(model_name_or_path="qwen3-reranker", device="cpu", max_length=32, use_fp16=False)

    def fake_load(self):
        self._model = object()
        return self

    monkeypatch.setattr(Qwen3Reranker, "load", fake_load)
    monkeypatch.setattr(model, "_process_inputs", lambda pairs: {"pairs": pairs})
    monkeypatch.setattr(model, "_compute_logits", lambda inputs: [0.8, 0.1, 0.5])

    results = model.rerank("q?", ["a", "b", "c"], top_k=2, return_documents=False)
    assert [r["index"] for r in results] == [0, 2]
    assert all("document" not in r for r in results)
