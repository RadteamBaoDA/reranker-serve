"""CrossEncoder handler also flattens pairs across requests into one predict()."""

import numpy as np

from src.engine.handlers.cross_encoder import CrossEncoderHandler
from src.engine.request_queue import BatchedRequest, RerankRequest


class _SpyCrossEncoder:
    def __init__(self):
        self.calls = 0
        self.last_pairs = None

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        self.calls += 1
        self.last_pairs = list(pairs)
        # score = len(query) + len(doc)
        return np.array([len(p[0]) + len(p[1]) for p in pairs], dtype=float)


def _handler_with_spy():
    h = CrossEncoderHandler.__new__(CrossEncoderHandler)
    h.model = _SpyCrossEncoder()
    h.device = "cpu"
    return h


def test_cross_encoder_single_predict_call_across_requests(monkeypatch):
    from src.config import settings as s
    monkeypatch.setattr(s, "normalize_scores", False)

    h = _handler_with_spy()
    batch = BatchedRequest(
        batch_id="b1",
        requests=[
            RerankRequest(request_id="r1", query="q", documents=["a", "bb"]),
            RerankRequest(request_id="r2", query="qq", documents=["ccc"]),
        ],
    )
    results = h.predict(batch)
    assert h.model.calls == 1
    assert len(h.model.last_pairs) == 3
    assert len(results) == 2
    assert len(results[0]) == 2 and len(results[1]) == 1


def test_cross_encoder_scatter_and_sort(monkeypatch):
    from src.config import settings as s
    monkeypatch.setattr(s, "normalize_scores", False)
    h = _handler_with_spy()
    batch = BatchedRequest(
        batch_id="b2",
        requests=[RerankRequest(request_id="r1", query="q", documents=["x", "longer"])],
    )
    results = h.predict(batch)
    assert results[0][0]["index"] == 1  # "longer" wins
