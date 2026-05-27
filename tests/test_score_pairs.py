"""score_pairs(): flat (query, doc) scoring, scores returned in input order."""

import pytest

from src.models.qwen3_reranker import Qwen3Reranker


def _make_reranker(monkeypatch):
    """Build a Qwen3Reranker without loading real weights; stub the GPU bits."""
    r = Qwen3Reranker.__new__(Qwen3Reranker)
    r._model = object()          # non-None so score_pairs won't try to load
    r._tokenizer = object()
    r.default_instruction = "inst"
    r.max_length = 256
    r.device = "cpu"

    # _process_inputs just passes the formatted prompts straight through;
    # _compute_logits scores by prompt length so we can assert ordering.
    monkeypatch.setattr(r, "_process_inputs", lambda prompts: prompts)
    monkeypatch.setattr(
        r, "_compute_logits", lambda prompts: [float(len(p)) for p in prompts]
    )
    return r


def test_score_pairs_returns_one_score_per_pair_in_input_order(monkeypatch):
    r = _make_reranker(monkeypatch)
    pairs = [("q", "short"), ("q", "a much longer document here"), ("q", "mid len")]
    scores = r.score_pairs(pairs)

    assert len(scores) == 3
    # Score is the formatted-prompt length; the longest doc must score highest,
    # and crucially the result is in INPUT order, not bucketed order.
    assert scores[1] == max(scores)
    assert scores[0] < scores[2] < scores[1]


def test_score_pairs_empty_returns_empty(monkeypatch):
    r = _make_reranker(monkeypatch)
    assert r.score_pairs([]) == []
