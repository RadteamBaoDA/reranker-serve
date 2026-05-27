"""Admin authentication primitives — stdlib only, no HTTP, fully unit-testable.

A session token is `base64url(json({"exp": <unix>})) + "." + hex(HMAC-SHA256)`,
signed with a key derived from the configured admin password. Changing the
password invalidates all existing tokens.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from collections import defaultdict
from typing import Dict, List, Optional


def verify_password(candidate: str, configured: Optional[str]) -> bool:
    """Constant-time password check. No configured password => always False."""
    if not configured:
        return False
    return hmac.compare_digest(candidate.encode("utf-8"), configured.encode("utf-8"))


def _secret_key(password: str) -> bytes:
    return hashlib.sha256(("reranker-admin:" + password).encode("utf-8")).digest()


def create_session_token(password: str, ttl_seconds: int, now: Optional[float] = None) -> str:
    exp = int((now if now is not None else time.time()) + ttl_seconds)
    payload = base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode("utf-8")).decode("ascii")
    sig = hmac.new(_secret_key(password), payload.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def verify_session_token(token: str, password: Optional[str], now: Optional[float] = None) -> bool:
    if not password or not token or "." not in token:
        return False
    try:
        payload, sig = token.split(".", 1)
        expected = hmac.new(_secret_key(password), payload.encode("ascii"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return False
        data = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")))
        return (now if now is not None else time.time()) < float(data["exp"])
    except Exception:
        return False


class LoginThrottle:
    """In-memory per-IP failed-login throttle. Process-local (single worker)."""

    def __init__(self, max_attempts: int = 5, window_seconds: int = 300):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._failures: Dict[str, List[float]] = defaultdict(list)

    def _recent(self, ip: str, now: float) -> List[float]:
        cutoff = now - self.window_seconds
        recent = [t for t in self._failures.get(ip, []) if t >= cutoff]
        self._failures[ip] = recent
        return recent

    def record_failure(self, ip: str, now: Optional[float] = None) -> None:
        n = now if now is not None else time.time()
        self._failures[ip].append(n)

    def is_blocked(self, ip: str, now: Optional[float] = None) -> bool:
        n = now if now is not None else time.time()
        return len(self._recent(ip, n)) >= self.max_attempts

    def clear(self, ip: str) -> None:
        self._failures.pop(ip, None)
