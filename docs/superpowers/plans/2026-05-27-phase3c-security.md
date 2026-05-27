# Phase 3C — Security Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Close the security findings from the architecture review — fix the invalid CORS wildcard+credentials combo, make the API-key compare constant-time, gate the unauthenticated introspection endpoints, gate the docs, and add the (UI-agnostic) admin session-auth primitives that Phase 3D will wire into the admin UI.

**Architecture:** Small, well-bounded changes plus one new module `src/admin/auth.py` containing pure, stdlib-only auth primitives (constant-time password check, HMAC-signed session token with expiry, in-memory login throttle). No new HTTP routes here — the login route, session middleware, and `/admin/*` come in Phase 3D and consume these primitives.

**Tech Stack:** Python 3.11 stdlib (`hmac`, `hashlib`, `base64`, `json`, `time`, `secrets`), FastAPI, pytest. No new third-party dependency.

**Reference spec:** `docs/superpowers/specs/2026-05-27-perf-telemetry-admin-ui-design.md` (Part C).

**Branch:** `feature/phase3c-security`, created off `feature/phase3b-telemetry`.

## Environment (read before every task)
- Work from `/mnt/d/Project/RAG/reranker-serve`. Already on the branch (don't create worktrees).
- ONLY interpreter: **`./.venv/Scripts/python.exe`** (system python3 has no deps). Test: `./.venv/Scripts/python.exe -m pytest -p no:cacheprovider <targets>`.
- Full-suite regression check: `./.venv/Scripts/python.exe -m pytest tests/ -p no:cacheprovider --ignore=tests/test_dual_format_rerank.py --ignore=tests/test_huggingface_api.py -q`
- `tests/conftest.py` globally mocks `torch`.
- KNOWN-FAILING baseline (4 pre-existing tests; must remain the ONLY failures): `test_async_engine.py::test_batch_accumulation_overlaps_with_inference`, `test_graceful_shutdown.py::test_engine_drains_inflight_requests_on_stop`, `test_handlers.py::test_qwen3_compute_logits_falls_back_from_mps_to_cpu`, `test_handlers.py::test_qwen3_compute_logits_does_not_fall_back_when_disabled`.
- End every commit body with: `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`

---

### Task 0: Branch + settings

**Files:** Modify `src/config/settings.py`.

- [ ] **Step 1: Branch**

```bash
git checkout feature/phase3b-telemetry && git checkout -b feature/phase3c-security
```

- [ ] **Step 2: Add settings.** In `src/config/settings.py`, after the `cors_origins` field, add:

```python
    enable_docs: bool = Field(
        default=True,
        description="Expose /docs, /redoc, /openapi.json. Set false in production."
    )
    admin_password: Optional[str] = Field(
        default=None,
        description="Password gating the admin UI (env-only secret; never written to config.yml). Unset disables /admin."
    )
    admin_session_ttl_hours: int = Field(
        default=12,
        description="Admin session cookie lifetime in hours."
    )
```

In `load_yaml_config`, the `if 'api' in yaml_config:` block currently maps `api_key`, `enable_cors`, `cors_origins`. Add `enable_docs` and `admin_session_ttl_hours` (but NOT `admin_password` — it stays env-only):

```python
        if 'api' in yaml_config:
            api_cfg = yaml_config['api']
            flat_config['api_key'] = api_cfg.get('key')
            flat_config['enable_cors'] = api_cfg.get('enable_cors')
            flat_config['cors_origins'] = api_cfg.get('cors_origins')
            flat_config['enable_docs'] = api_cfg.get('enable_docs')
            flat_config['admin_session_ttl_hours'] = api_cfg.get('admin_session_ttl_hours')
```

- [ ] **Step 3: Verify + commit**

```bash
./.venv/Scripts/python.exe -c "from src.config import settings; print(settings.enable_docs, settings.admin_password, settings.admin_session_ttl_hours)"
# expect: True None 12
git add src/config/settings.py
git commit -m "feat(config): admin auth + docs-gating settings"
```

---

### Task 1: Constant-time API key compare

**Files:** Modify `src/api/routes.py`; Create `tests/test_security_apikey.py`.

- [ ] **Step 1: Write the failing test.** Create `tests/test_security_apikey.py`:

```python
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
```

- [ ] **Step 2: Run it.** `./.venv/Scripts/python.exe -m pytest -p no:cacheprovider tests/test_security_apikey.py -v` — most pass already, but verify the wrong-key path. (If all pass already, the constant-time change in Step 3 keeps them green.)

- [ ] **Step 3: Make the compare constant-time.** In `src/api/routes.py`, add `import hmac` at the top. In `verify_api_key`, replace:

```python
    if token != settings.api_key:
        logger.debug("verify_api_key_invalid")
        raise HTTPException(status_code=401, detail="Invalid API key")
```

with:

```python
    if not hmac.compare_digest(token, settings.api_key):
        logger.debug("verify_api_key_invalid")
        raise HTTPException(status_code=401, detail="Invalid API key")
```

- [ ] **Step 4:** Run the test (PASS, 5 tests) and the full suite (only the 4 known failures).

- [ ] **Step 5: Commit**

```bash
git add src/api/routes.py tests/test_security_apikey.py
git commit -m "fix(security): constant-time API key comparison"
```

---

### Task 2: CORS credentials fix

**Files:** Modify `src/main.py`; Create `tests/test_security_cors.py`.

- [ ] **Step 1: Write the failing test.** Create `tests/test_security_cors.py`:

```python
"""Wildcard CORS must not be combined with allow_credentials=True."""

from src.main import cors_allow_credentials


def test_wildcard_disables_credentials():
    assert cors_allow_credentials(["*"]) is False


def test_explicit_origins_allow_credentials():
    assert cors_allow_credentials(["https://app.internal"]) is True


def test_wildcard_anywhere_disables_credentials():
    assert cors_allow_credentials(["https://a", "*"]) is False
```

- [ ] **Step 2: Run it.** Expected: FAIL — `cannot import name 'cors_allow_credentials'`.

- [ ] **Step 3: Implement + use the helper.** In `src/main.py`, add a module-level function (before `create_app`):

```python
def cors_allow_credentials(origins: list[str]) -> bool:
    """Browsers reject Access-Control-Allow-Origin '*' with credentials; never
    combine them. Credentials are only allowed with explicit origins."""
    return "*" not in origins
```

Then in `create_app`, change the CORS middleware registration from `allow_credentials=True` to:

```python
        origins = settings.get_cors_origins_list()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=cors_allow_credentials(origins),
            allow_methods=["*"],
            allow_headers=["*"],
        )
```

(Keep the surrounding `if settings.enable_cors:` / logging as-is; only the `allow_origins`/`allow_credentials` wiring changes.)

- [ ] **Step 4:** Run the test (PASS, 3) and the full suite (only the 4 known failures).

- [ ] **Step 5: Commit**

```bash
git add src/main.py tests/test_security_cors.py
git commit -m "fix(security): never combine wildcard CORS with credentials"
```

---

### Task 3: Gate the API docs behind enable_docs

**Files:** Modify `src/main.py`; Create `tests/test_security_docs.py`.

- [ ] **Step 1: Write the failing test.** Create `tests/test_security_docs.py`:

```python
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
```

- [ ] **Step 2: Run it.** Expected: `test_docs_hidden_when_disabled` FAILS (docs always on).

- [ ] **Step 3: Implement.** In `src/main.py` `create_app`, change the `FastAPI(...)` constructor so the doc URLs are conditional:

```python
    app = FastAPI(
        title="Reranker Service",
        description=(...existing description unchanged...),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
    )
```

(Only the three `*_url` lines change; keep title/description/version/lifespan.)

- [ ] **Step 4:** Run the test (PASS, 2) and the full suite (only the 4 known failures). NOTE: if an existing test asserts `/docs` or `/openapi.json` is reachable, it still passes because `enable_docs` defaults to True.

- [ ] **Step 5: Commit**

```bash
git add src/main.py tests/test_security_docs.py
git commit -m "feat(security): gate /docs /redoc /openapi.json behind enable_docs"
```

---

### Task 4: Require auth on /stats and /info

**Files:** Modify `src/api/health.py`; Create `tests/test_security_introspection.py`.

- [ ] **Step 1: Write the failing test.** Create `tests/test_security_introspection.py`:

```python
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
```

- [ ] **Step 2: Run it.** Expected: the `requires_key` tests FAIL (currently open).

- [ ] **Step 3: Add the dependency.** In `src/api/health.py`, add at the top: `from fastapi import APIRouter, Depends` (it currently imports only `APIRouter`) and `from src.api.routes import verify_api_key`. Then add `_: bool = Depends(verify_api_key)` as a parameter to BOTH `model_info()` (the `/info` handler) and `engine_stats()` (the `/stats` handler). Example:

```python
@health_router.get("/info", response_model=ModelInfoResponse)
async def model_info(_: bool = Depends(verify_api_key)) -> ModelInfoResponse:
```

```python
@health_router.get("/stats", response_model=EngineStatsResponse)
async def engine_stats(_: bool = Depends(verify_api_key)) -> EngineStatsResponse:
```

Do NOT gate `/health`, `/ready`, `/live`, or `/` — those are liveness/readiness probes and must stay open.

Watch for a circular import: `health.py` importing from `routes.py`. If it occurs, move the import inside the handler functions (local import) instead of module top.

- [ ] **Step 4:** Run the test (PASS, 3) and the full suite. NOTE: existing /stats or /info tests use no api_key (default None) so they stay 200.

- [ ] **Step 5: Commit**

```bash
git add src/api/health.py tests/test_security_introspection.py
git commit -m "feat(security): require bearer auth for /stats and /info when key set"
```

---

### Task 5: Admin auth primitives (stdlib, UI-agnostic)

**Files:** Create `src/admin/__init__.py`; Create `src/admin/auth.py`; Create `tests/test_admin_auth.py`.

- [ ] **Step 1: Write the failing test.** Create `tests/test_admin_auth.py`:

```python
"""Admin auth primitives: constant-time password check, signed session tokens, throttle."""

from src.admin import auth


def test_verify_password_correct():
    assert auth.verify_password("hunter2", "hunter2") is True


def test_verify_password_wrong():
    assert auth.verify_password("nope", "hunter2") is False


def test_verify_password_empty_configured_is_false():
    # No password configured => nothing authenticates.
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
```

- [ ] **Step 2: Run it.** Expected: FAIL — `No module named 'src.admin'`.

- [ ] **Step 3: Implement.** Create `src/admin/__init__.py`:

```python
"""Admin subsystem: auth primitives (Phase 3C) and the admin UI (Phase 3D)."""
```

Create `src/admin/auth.py`:

```python
"""Admin authentication primitives — stdlib only, no HTTP, fully unit-testable.

A session token is `base64url(json({"exp": <unix>})) + "." + hex(HMAC-SHA256)`,
signed with a key derived from the configured admin password. Changing the
password invalidates all existing tokens. These primitives are consumed by the
Phase 3D login route + session middleware.
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
```

- [ ] **Step 4:** Run the test (PASS, 11) and the full suite (only the 4 known failures).

- [ ] **Step 5: Commit**

```bash
git add src/admin/__init__.py src/admin/auth.py tests/test_admin_auth.py
git commit -m "feat(admin): stdlib admin auth primitives (password, session token, throttle)"
```

---

### Task 6: Integration sweep

- [ ] **Step 1:** Full suite — `./.venv/Scripts/python.exe -m pytest tests/ -p no:cacheprovider --ignore=tests/test_dual_format_rerank.py --ignore=tests/test_huggingface_api.py -q`. Expected: only the 4 known-failing tests fail; all new Phase 3C tests pass.
- [ ] **Step 2:** Report branch state. Do not push/PR unless asked.

---

## Self-Review

**Spec coverage (Part C):** CORS fix → Task 2; constant-time API key → Task 1; admin auth (password + session) → Task 5 (login route/middleware deferred to 3D); close introspection leaks (/stats, /info) → Task 4; gate /docs → Task 3; settings (admin_password env-only, enable_docs, ttl) → Task 0.

**Placeholder scan:** none. All code shown in full.

**Type consistency:** `cors_allow_credentials(list)->bool` (Task 2); `verify_api_key` unchanged signature (Task 1); `verify_password/create_session_token/verify_session_token/LoginThrottle` (Task 5) are consumed by Phase 3D. `enable_docs`, `admin_password`, `admin_session_ttl_hours` settings (Task 0) used in Tasks 3 and (3D).
