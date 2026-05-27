"""Handler flattens all pairs across requests into ONE score_pairs() call."""

from src.engine.handlers.qwen import QwenRerankerHandler
from src.engine.request_queue import BatchedRequest, RerankRequest


class _SpyModel:
    """score_pairs scores each pair as len(query)+len(doc); records call count."""
    def __init__(self):
        self.calls = 0
        self.last_pairs = None

    def score_pairs(self, pairs, instruction=None):
        self.calls += 1
        self.last_pairs = list(pairs)
        return [float(len(q) + len(d)) for (q, d) in pairs]


def _handler_with_spy():
    h = QwenRerankerHandler.__new__(QwenRerankerHandler)
    h.model = _SpyModel()
    return h


def test_predict_makes_single_score_pairs_call_across_requests():
    h = _handler_with_spy()
    batch = BatchedRequest(
        batch_id="b1",
        requests=[
            RerankRequest(request_id="r1", query="qq", documents=["a", "bb"]),
            RerankRequest(request_id="r2", query="q", documents=["ccc"]),
        ],
    )

    results = h.predict(batch)

    # One flat call covering all 3 pairs (2 + 1), not one call per request.
    assert h.model.calls == 1
    assert len(h.model.last_pairs) == 3
    assert len(results) == 2
    assert len(results[0]) == 2 and len(results[1]) == 1


def test_predict_scatters_scores_back_to_correct_request_and_sorts():
    h = _handler_with_spy()
    batch = BatchedRequest(
        batch_id="b2",
        requests=[
            RerankRequest(request_id="r1", query="q", documents=["x", "longer"]),
        ],
    )
    results = h.predict(batch)
    r1 = results[0]
    # score = len(q)+len(doc): "longer"(6)+1=7 beats "x"(1)+1=2, so sorted desc.
    assert r1[0]["index"] == 1
    assert r1[0]["relevance_score"] == 7.0
    assert r1[1]["index"] == 0


def test_predict_respects_top_k_per_request():
    h = _handler_with_spy()
    batch = BatchedRequest(
        batch_id="b3",
        requests=[
            RerankRequest(request_id="r1", query="q", documents=["a", "bb", "ccc"], top_k=1),
        ],
    )
    results = h.predict(batch)
    assert len(results[0]) == 1
    assert results[0][0]["index"] == 2  # "ccc" highest


def test_predict_empty_batch_returns_empty():
    h = _handler_with_spy()
    batch = BatchedRequest(batch_id="b4", requests=[])
    assert h.predict(batch) == []
