"""API key verification uses constant-time compare and the right status codes."""

import pytest
from fastapi import HTTPException

from src.api.routes import verify_api_key
from src.config import settings as s


def test_no_key_configured_allows(monkeypatch):
    monkeypatch.setattr(s, "api_key", None)
    assert verify_api_key(authorization=None) is True


def test_correct_bearer_key_allows(monkeypatch):
    monkeypatch.setattr(s, "api_key", "secret")
    assert verify_api_key(authorization="Bearer secret") is True


def test_correct_bare_key_allows(monkeypatch):
    monkeypatch.setattr(s, "api_key", "secret")
    assert verify_api_key(authorization="secret") is True


def test_wrong_key_rejected(monkeypatch):
    monkeypatch.setattr(s, "api_key", "secret")
    with pytest.raises(HTTPException) as ei:
        verify_api_key(authorization="Bearer nope")
    assert ei.value.status_code == 401


def test_missing_header_rejected(monkeypatch):
    monkeypatch.setattr(s, "api_key", "secret")
    with pytest.raises(HTTPException) as ei:
        verify_api_key(authorization=None)
    assert ei.value.status_code == 401
