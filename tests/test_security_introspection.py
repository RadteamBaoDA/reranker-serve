"""/stats and /info require the bearer token when an api_key is configured."""

from fastapi.testclient import TestClient


def _client(monkeypatch, api_key):
    from src.config import settings as s
    monkeypatch.setattr(s, "api_key", api_key)
    from src.main import create_app
    return TestClient(create_app())


def test_info_open_when_no_key(monkeypatch):
    c = _client(monkeypatch, None)
    assert c.get("/info").status_code == 200


def test_info_requires_key_when_configured(monkeypatch):
    c = _client(monkeypatch, "secret")
    assert c.get("/info").status_code == 401
    assert c.get("/info", headers={"Authorization": "Bearer secret"}).status_code == 200


def test_stats_requires_key_when_configured(monkeypatch):
    c = _client(monkeypatch, "secret")
    assert c.get("/stats").status_code == 401
