"""Admin live-data endpoints return engine telemetry behind auth."""

import importlib

from fastapi import FastAPI
from fastapi.testclient import TestClient


class _FakeEngine:
    device = "cpu"
    def get_stats(self):
        return {"device_resources": {"device": "cpu", "mem_used_mb": 1.0, "mem_total_mb": 2.0,
                                     "mem_free_mb": 1.0, "used_pct": 50.0},
                "requests_per_second": 3.0, "throughput_pairs_per_sec": 12.0,
                "inference_latency_p50_ms": 5.0, "inference_latency_p95_ms": 9.0,
                "batch_occupancy_pct": 40.0, "pending_requests": 0}
    def get_queue_snapshot(self):
        return {"waiting": [{"request_id": "r1", "num_docs": 3, "waited_ms": 12.0}],
                "running": [{"batch_id": "b1", "num_requests": 2, "pairs": 5, "elapsed_ms": 30.0}]}


def _app(monkeypatch):
    from src.config import settings as s
    monkeypatch.setattr(s, "admin_password", "pw")
    import src.admin.routes as routes
    importlib.reload(routes)

    async def fake_get_engine():
        return _FakeEngine()
    monkeypatch.setattr(routes, "_get_engine", fake_get_engine, raising=False)

    app = FastAPI()
    app.include_router(routes.router)
    c = TestClient(app)
    c.post("/admin/login", data={"password": "pw"})
    return c


def test_resources_endpoint(monkeypatch):
    c = _app(monkeypatch)
    r = c.get("/admin/api/resources")
    assert r.status_code == 200
    assert r.json()["device_resources"]["used_pct"] == 50.0


def test_queue_endpoint(monkeypatch):
    c = _app(monkeypatch)
    r = c.get("/admin/api/queue")
    body = r.json()
    assert r.status_code == 200
    assert body["waiting"][0]["request_id"] == "r1"
    assert body["running"][0]["batch_id"] == "b1"


def test_resources_requires_auth(monkeypatch):
    from src.config import settings as s
    monkeypatch.setattr(s, "admin_password", "pw")
    import src.admin.routes as routes
    importlib.reload(routes)
    app = FastAPI(); app.include_router(routes.router)
    r = TestClient(app).get("/admin/api/resources", follow_redirects=False)
    assert r.status_code in (303, 401)


def test_stats_partial_renders_kpi_cards(monkeypatch):
    c = _app(monkeypatch)
    r = c.get("/admin/partials/stats")
    assert r.status_code == 200
    assert "kpi-grid" in r.text
    assert "Latency p50" in r.text


def test_metrics_history_endpoint(monkeypatch):
    from src.observability import metrics_history as mh
    mh.reset_history()
    mh.get_history().record(mh.build_sample({"inference_latency_p50_ms": 5.0}, now=1.0))
    c = _app(monkeypatch)
    r = c.get("/admin/api/metrics/history")
    assert r.status_code == 200
    body = r.json()
    assert "server_now" in body
    assert body["samples"][-1]["p50_ms"] == 5.0


def test_metrics_history_requires_auth(monkeypatch):
    from src.config import settings as s
    monkeypatch.setattr(s, "admin_password", "pw")
    import src.admin.routes as routes
    importlib.reload(routes)
    app = FastAPI(); app.include_router(routes.router)
    r = TestClient(app).get("/admin/api/metrics/history", follow_redirects=False)
    assert r.status_code in (303, 401)
