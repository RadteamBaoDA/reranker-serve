"""GET /admin/api/logs/tail returns recent log lines, filterable."""

import importlib

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _client(monkeypatch, tmp_path):
    from src.config import settings as s
    monkeypatch.setattr(s, "admin_password", "pw")
    monkeypatch.setattr(s, "log_dir", str(tmp_path))
    (tmp_path / "reranker.log").write_text(
        "\n".join(f"line {i} INFO hello" for i in range(100)) + "\n"
        + "line 100 ERROR boom\n"
    )
    import src.admin.routes as routes
    importlib.reload(routes)
    app = FastAPI(); app.include_router(routes.router)
    c = TestClient(app)
    c.post("/admin/login", data={"password": "pw"})
    return c


def test_tail_returns_last_n(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.get("/admin/api/logs/tail?lines=10")
    assert r.status_code == 200
    lines = r.json()["lines"]
    assert len(lines) == 10
    assert "line 100 ERROR boom" in lines[-1]


def test_tail_filters_by_query(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.get("/admin/api/logs/tail?lines=200&q=ERROR")
    lines = r.json()["lines"]
    assert all("ERROR" in ln for ln in lines)
    assert len(lines) == 1
