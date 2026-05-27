"""Admin auth primitives: constant-time password check, signed session tokens, throttle."""

from src.admin import auth


def test_verify_password_correct():
    assert auth.verify_password("hunter2", "hunter2") is True


def test_verify_password_wrong():
    assert auth.verify_password("nope", "hunter2") is False


def test_verify_password_empty_configured_is_false():
    assert auth.verify_password("anything", "") is False
    assert auth.verify_password("anything", None) is False


def test_session_token_roundtrip():
    tok = auth.create_session_token("pw", ttl_seconds=3600, now=1000.0)
    assert auth.verify_session_token(tok, "pw", now=1000.0) is True


def test_session_token_expires():
    tok = auth.create_session_token("pw", ttl_seconds=10, now=1000.0)
    assert auth.verify_session_token(tok, "pw", now=1011.0) is False


def test_session_token_rejects_wrong_password():
    tok = auth.create_session_token("pw", ttl_seconds=3600, now=1000.0)
    assert auth.verify_session_token(tok, "different", now=1000.0) is False


def test_session_token_rejects_tamper():
    tok = auth.create_session_token("pw", ttl_seconds=3600, now=1000.0)
    payload, sig = tok.split(".", 1)
    forged = payload + "." + ("0" * len(sig))
    assert auth.verify_session_token(forged, "pw", now=1000.0) is False


def test_session_token_malformed_is_false():
    assert auth.verify_session_token("garbage", "pw") is False


def test_login_throttle_blocks_after_threshold():
    t = auth.LoginThrottle(max_attempts=3, window_seconds=60)
    ip = "1.2.3.4"
    for _ in range(3):
        t.record_failure(ip, now=0.0)
    assert t.is_blocked(ip, now=1.0) is True


def test_login_throttle_resets_after_window():
    t = auth.LoginThrottle(max_attempts=3, window_seconds=60)
    ip = "1.2.3.4"
    for _ in range(3):
        t.record_failure(ip, now=0.0)
    assert t.is_blocked(ip, now=61.0) is False


def test_login_throttle_clear_on_success():
    t = auth.LoginThrottle(max_attempts=3, window_seconds=60)
    ip = "1.2.3.4"
    t.record_failure(ip, now=0.0)
    t.clear(ip)
    assert t.is_blocked(ip, now=1.0) is False
