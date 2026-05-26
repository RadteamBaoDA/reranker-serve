"""OTel only initialized when env var on; FastAPI instrumented when so."""

import pytest

pytest.importorskip("opentelemetry")


def test_otel_off_by_default(monkeypatch):
    monkeypatch.delenv("RERANKER_ENABLE_OTEL", raising=False)
    from src.config import settings as s
    monkeypatch.setattr(s, "enable_otel", False)
    from src.main import create_app

    app = create_app()
    assert getattr(app.state, "_otel_initialized", None) is False


def test_otel_on_initializes_provider(monkeypatch):
    monkeypatch.setenv("RERANKER_ENABLE_OTEL", "true")
    from src.config import settings as s
    monkeypatch.setattr(s, "enable_otel", True)
    from src.main import create_app

    app = create_app()
    assert app.state._otel_initialized is True
