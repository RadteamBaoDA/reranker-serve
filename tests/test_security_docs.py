"""When enable_docs is false, /docs, /redoc, /openapi.json are not served."""

from fastapi.testclient import TestClient


def _client(monkeypatch, enable_docs):
    from src.config import settings as s
    monkeypatch.setattr(s, "enable_docs", enable_docs)
    from src.main import create_app
    return TestClient(create_app())


def test_docs_served_when_enabled(monkeypatch):
    c = _client(monkeypatch, True)
    assert c.get("/openapi.json").status_code == 200


def test_docs_hidden_when_disabled(monkeypatch):
    c = _client(monkeypatch, False)
    assert c.get("/docs").status_code == 404
    assert c.get("/openapi.json").status_code == 404
