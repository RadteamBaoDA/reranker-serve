"""Wildcard CORS must not be combined with allow_credentials=True."""

from src.main import cors_allow_credentials


def test_wildcard_disables_credentials():
    assert cors_allow_credentials(["*"]) is False


def test_explicit_origins_allow_credentials():
    assert cors_allow_credentials(["https://app.internal"]) is True


def test_wildcard_anywhere_disables_credentials():
    assert cors_allow_credentials(["https://a", "*"]) is False
