"""Admin login sets a session cookie; require_admin guards protected routes."""

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _app(monkeypatch, password):
    from src.config import settings as s
    monkeypatch.setattr(s, "admin_password", password)
    monkeypatch.setattr(s, "admin_session_ttl_hours", 12)
    import importlib
    import src.admin.routes as routes
    importlib.reload(routes)
    app = FastAPI()
    app.include_router(routes.router)
    return app, routes


def test_login_rejects_wrong_password(monkeypatch):
    app, _ = _app(monkeypatch, "pw")
    c = TestClient(app)
    r = c.post("/admin/login", data={"password": "wrong"}, follow_redirects=False)
    assert r.status_code in (401, 303)
    if r.status_code == 303:
        assert "error" in r.headers["location"]


def test_login_accepts_correct_password_and_sets_cookie(monkeypatch):
    app, _ = _app(monkeypatch, "pw")
    c = TestClient(app)
    r = c.post("/admin/login", data={"password": "pw"}, follow_redirects=False)
    assert r.status_code == 303
    assert "reranker_admin" in r.cookies or any("reranker_admin" in h for h in r.headers.get_list("set-cookie"))


def test_admin_disabled_when_no_password(monkeypatch):
    app, _ = _app(monkeypatch, None)
    c = TestClient(app)
    r = c.post("/admin/login", data={"password": "anything"}, follow_redirects=False)
    assert r.status_code == 503


def test_protected_partial_requires_session(monkeypatch):
    app, _ = _app(monkeypatch, "pw")
    c = TestClient(app)
    r = c.get("/admin/api/config", follow_redirects=False)
    assert r.status_code in (303, 401)
    c.post("/admin/login", data={"password": "pw"})
    r2 = c.get("/admin/api/config")
    assert r2.status_code == 200
