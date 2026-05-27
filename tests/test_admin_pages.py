"""Admin pages render; unauthenticated dashboard redirects to login."""

import importlib

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _client(monkeypatch):
    from src.config import settings as s
    monkeypatch.setattr(s, "admin_password", "pw")
    import src.admin.routes as routes
    importlib.reload(routes)
    app = FastAPI(); app.include_router(routes.router)
    return TestClient(app)


def test_login_page_renders(monkeypatch):
    c = _client(monkeypatch)
    r = c.get("/admin/login")
    assert r.status_code == 200
    assert "password" in r.text.lower()


def test_dashboard_redirects_when_unauthenticated(monkeypatch):
    c = _client(monkeypatch)
    r = c.get("/admin", follow_redirects=False)
    assert r.status_code in (303, 307, 401)


def test_dashboard_renders_after_login(monkeypatch):
    c = _client(monkeypatch)
    c.post("/admin/login", data={"password": "pw"})
    r = c.get("/admin")
    assert r.status_code == 200
    assert "GPU" in r.text or "Quota" in r.text or "Dashboard" in r.text
