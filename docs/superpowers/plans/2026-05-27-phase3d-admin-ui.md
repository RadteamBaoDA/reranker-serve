# Phase 3D — Admin UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A local, password-gated admin web UI at `/admin` for live monitoring (GPU/device quota, batches running/waiting, throughput, latency), config editing with apply/restart, and log viewing — consuming the Phase 3B telemetry and Phase 3C auth primitives.

**Architecture:** A new FastAPI router `src/admin/routes.py` mounted at `/admin`. Auth is a session cookie (`require_admin` dependency built on `src/admin/auth.py` from Phase 3C). `src/admin/config_io.py` reads/writes `config.yml` and classifies each setting hot vs needs-restart. Pages are Jinja2 templates; live data refreshes via **HTMX polling of server-rendered partial endpoints** (GPU gauge + queue tables + throughput as HTML); HTMX is vendored locally for offline use. Config "apply" reloads the engine in-process; "restart" triggers the existing graceful drain so supervisord respawns.

**Stack deviation (intentional):** The spec named Jinja2+HTMX+Chart.js. We keep Jinja2+HTMX but render charts/gauges as CSS/HTML partials instead of Chart.js, and use HTMX polling instead of SSE. Rationale: fully offline (no 200 KB canvas lib to vendor), zero client-side build, and the partial endpoints are unit-testable without a browser. Same UX for an ops dashboard.

**Tech Stack:** FastAPI, Jinja2, HTMX (vendored), PyYAML, pytest. Builds on Phase 3B (`engine.get_stats().device_resources`, `engine.get_queue_snapshot()`) and Phase 3C (`src/admin/auth.py`, `settings.admin_password`, `settings.admin_session_ttl_hours`, `settings.enable_docs`).

**Reference spec:** `docs/superpowers/specs/2026-05-27-perf-telemetry-admin-ui-design.md` (Part D + B3).

**Branch:** `feature/phase3d-admin-ui`, created off `feature/phase3c-security`.

## Environment (read before every task)
- Work from `/mnt/d/Project/RAG/reranker-serve`. Already on the branch (no worktrees).
- ONLY interpreter: **`./.venv/Scripts/python.exe`**. Test: `./.venv/Scripts/python.exe -m pytest -p no:cacheprovider <targets>`.
- Full-suite regression check: `./.venv/Scripts/python.exe -m pytest tests/ -p no:cacheprovider --ignore=tests/test_dual_format_rerank.py --ignore=tests/test_huggingface_api.py -q`
- `tests/conftest.py` globally mocks `torch`.
- KNOWN-FAILING baseline (4 pre-existing; must remain the ONLY failures): `test_async_engine.py::test_batch_accumulation_overlaps_with_inference`, `test_graceful_shutdown.py::test_engine_drains_inflight_requests_on_stop`, `test_handlers.py::test_qwen3_compute_logits_falls_back_from_mps_to_cpu`, `test_handlers.py::test_qwen3_compute_logits_does_not_fall_back_when_disabled`.
- End every commit body with: `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`

---

### Task 0: Branch + deps + vendor HTMX

**Files:** Modify `requirements.txt`, `pyproject.toml`; add `src/admin/static/htmx.min.js`.

- [ ] **Step 1: Branch**

```bash
git checkout feature/phase3c-security && git checkout -b feature/phase3d-admin-ui
```

- [ ] **Step 2: Add deps.** Append to `requirements.txt`:

```
# Admin UI templates
jinja2>=3.1.0
```

In `pyproject.toml` `[project.optional-dependencies]`, add:

```toml
admin = [
    "jinja2>=3.1.0",
]
```

- [ ] **Step 3: Install jinja2 into the venv + vendor HTMX**

```bash
./.venv/Scripts/python.exe -m pip install -q "jinja2>=3.1.0"
mkdir -p src/admin/static
curl -sSL https://unpkg.com/htmx.org@1.9.12/dist/htmx.min.js -o src/admin/static/htmx.min.js
test -s src/admin/static/htmx.min.js && head -c 60 src/admin/static/htmx.min.js
```

Expected: the file is non-empty (starts with HTMX banner). If `curl` has no network, write a minimal note and STOP — report BLOCKED (vendoring requires one-time network access).

- [ ] **Step 4: Commit**

```bash
git add requirements.txt pyproject.toml src/admin/static/htmx.min.js
git commit -m "build(admin): jinja2 dep + vendored HTMX"
```

---

### Task 1: config_io — effective config snapshot + writeback

**Files:** Create `src/admin/config_io.py`; Create `tests/test_admin_config_io.py`.

- [ ] **Step 1: Write the failing test.** Create `tests/test_admin_config_io.py`:

```python
"""Config snapshot (value/source/needs_restart, secrets redacted) + YAML writeback."""

import os
import textwrap

from src.admin import config_io


def test_secrets_are_redacted(monkeypatch):
    from src.config import settings as s
    monkeypatch.setattr(s, "api_key", "supersecret")
    snap = {row["name"]: row for row in config_io.get_config_snapshot()}
    assert snap["api_key"]["value"] == "***set***"
    # admin_password is env-only but if present must also redact
    assert "supersecret" not in str(snap["api_key"]["value"])


def test_needs_restart_classification():
    snap = {row["name"]: row for row in config_io.get_config_snapshot()}
    assert snap["model_name"]["needs_restart"] is True
    assert snap["log_level"]["needs_restart"] is False


def test_source_reports_env(monkeypatch):
    monkeypatch.setenv("RERANKER_MAX_LENGTH", "256")
    snap = {row["name"]: row for row in config_io.get_config_snapshot()}
    assert snap["max_length"]["source"] == "env"


def test_write_updates_roundtrip(tmp_path):
    cfg = tmp_path / "config.yml"
    cfg.write_text(textwrap.dedent("""
        model:
          name: Qwen/Qwen3-Reranker-4B
        logging:
          level: info
    """).strip())
    result = config_io.write_config_updates({"log_level": "debug", "max_length": 256}, path=str(cfg))
    assert result["written"] is True
    import yaml
    data = yaml.safe_load(cfg.read_text())
    assert data["logging"]["level"] == "debug"
    assert data["inference"]["max_length"] == 256  # created the section


def test_write_rejects_unknown_key(tmp_path):
    cfg = tmp_path / "config.yml"
    cfg.write_text("model:\n  name: x\n")
    result = config_io.write_config_updates({"not_a_setting": 1}, path=str(cfg))
    assert result["written"] is False
    assert "not_a_setting" in result["rejected"]
```

- [ ] **Step 2: Run it.** Expected: FAIL — `No module named 'src.admin.config_io'`.

- [ ] **Step 3: Implement.** Create `src/admin/config_io.py`:

```python
"""Read the effective config (value + source + needs_restart, secrets redacted)
and write edits back to config.yml. Drives the admin Config page."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import yaml

from src.config import settings

# (setting_attr, yaml_section, yaml_key, needs_restart)
_FIELD_MAP = [
    ("host", "server", "host", True),
    ("port", "server", "port", True),
    ("workers", "server", "workers", True),
    ("model_name", "model", "name", True),
    ("model_path", "model", "path", True),
    ("use_offline_mode", "model", "use_offline_mode", True),
    ("max_length", "inference", "max_length", True),
    ("batch_size", "inference", "batch_size", True),
    ("normalize_scores", "inference", "normalize_scores", False),
    ("device", "device", "name", True),
    ("use_fp16", "device", "use_fp16", True),
    ("quantization", "device", "quantization", True),
    ("cpu_num_threads", "device", "cpu_num_threads", True),
    ("device_mem_safety_margin", "device", "device_mem_safety_margin", False),
    ("max_batch_size", "async_engine", "max_batch_size", False),
    ("max_batch_pairs", "async_engine", "max_batch_pairs", False),
    ("batch_wait_timeout", "async_engine", "batch_wait_timeout", False),
    ("max_queue_size", "async_engine", "max_queue_size", False),
    ("request_timeout", "async_engine", "request_timeout", False),
    ("enable_docs", "api", "enable_docs", True),
    ("api_key", "api", "key", True),
    ("log_level", "logging", "level", False),
]
_SECRETS = {"api_key", "admin_password"}
_BY_ATTR = {attr: (section, key, restart) for (attr, section, key, restart) in _FIELD_MAP}


def _source(attr: str, yaml_cfg: Dict[str, Any]) -> str:
    if f"RERANKER_{attr.upper()}" in os.environ:
        return "env"
    section, key, _ = _BY_ATTR[attr]
    if section in yaml_cfg and key in (yaml_cfg.get(section) or {}):
        return "yaml"
    return "default"


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    p = path or os.environ.get("RERANKER_CONFIG_PATH", "config.yml")
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_config_snapshot(path: Optional[str] = None) -> List[Dict[str, Any]]:
    yaml_cfg = _load_yaml(path)
    rows: List[Dict[str, Any]] = []
    for attr, section, key, needs_restart in _FIELD_MAP:
        if attr in _SECRETS:
            value: Any = "***set***" if getattr(settings, attr, None) else None
        else:
            value = getattr(settings, attr, None)
        rows.append({
            "name": attr,
            "section": section,
            "key": key,
            "value": value,
            "source": _source(attr, yaml_cfg),
            "needs_restart": needs_restart,
            "secret": attr in _SECRETS,
        })
    return rows


def write_config_updates(updates: Dict[str, Any], path: Optional[str] = None) -> Dict[str, Any]:
    """Apply {setting_attr: value} edits to config.yml. Unknown or secret-attr
    keys are rejected. Returns {written, rejected, needs_restart}."""
    p = path or os.environ.get("RERANKER_CONFIG_PATH", "config.yml")
    rejected = [k for k in updates if k not in _BY_ATTR or k in _SECRETS]
    if rejected:
        return {"written": False, "rejected": rejected, "needs_restart": False}

    data = _load_yaml(p)
    needs_restart = False
    for attr, value in updates.items():
        section, key, restart = _BY_ATTR[attr]
        data.setdefault(section, {})
        data[section][key] = value
        needs_restart = needs_restart or restart

    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    return {"written": True, "rejected": [], "needs_restart": needs_restart}
```

- [ ] **Step 4:** Run the test (5 passed) and the full suite (only the 4 known failures).

- [ ] **Step 5: Commit**

```bash
git add src/admin/config_io.py tests/test_admin_config_io.py
git commit -m "feat(admin): config snapshot + config.yml writeback"
```

---

### Task 2: Session auth dependency + login/logout routes

**Files:** Create `src/admin/routes.py`; Create `tests/test_admin_login.py`.

- [ ] **Step 1: Write the failing test.** Create `tests/test_admin_login.py`:

```python
"""Admin login sets a session cookie; require_admin guards protected routes."""

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _app(monkeypatch, password):
    from src.config import settings as s
    monkeypatch.setattr(s, "admin_password", password)
    monkeypatch.setattr(s, "admin_session_ttl_hours", 12)
    # rebuild router fresh so module-level throttle is clean
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
    # No cookie -> redirect to login (303) or 401
    r = c.get("/admin/api/config", follow_redirects=False)
    assert r.status_code in (303, 401)
    # With login -> allowed
    c.post("/admin/login", data={"password": "pw"})
    r2 = c.get("/admin/api/config")
    assert r2.status_code == 200
```

- [ ] **Step 2: Run it.** Expected: FAIL — `No module named 'src.admin.routes'`.

- [ ] **Step 3: Implement the router skeleton with auth.** Create `src/admin/routes.py`:

```python
"""Admin UI router: session login + guarded API/page routes. Mounted at /admin."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from src.config import settings, get_logger
from src.admin import auth, config_io

logger = get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])
_COOKIE = "reranker_admin"
_throttle = auth.LoginThrottle(max_attempts=5, window_seconds=300)


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def require_admin(request: Request) -> bool:
    """Dependency: valid session cookie required. 503 if admin not configured."""
    if not settings.admin_password:
        raise HTTPException(status_code=503, detail="Admin UI not configured (set RERANKER_ADMIN_PASSWORD)")
    token = request.cookies.get(_COOKIE, "")
    if not auth.verify_session_token(token, settings.admin_password):
        raise HTTPException(status_code=401, detail="Admin session required")
    return True


@router.post("/login")
async def login(request: Request, password: str = Form(...)):
    if not settings.admin_password:
        raise HTTPException(status_code=503, detail="Admin UI not configured")
    ip = _client_ip(request)
    if _throttle.is_blocked(ip):
        raise HTTPException(status_code=429, detail="Too many attempts; try later")
    if not auth.verify_password(password, settings.admin_password):
        _throttle.record_failure(ip)
        return RedirectResponse("/admin/login?error=1", status_code=303)
    _throttle.clear(ip)
    token = auth.create_session_token(settings.admin_password, settings.admin_session_ttl_hours * 3600)
    resp = RedirectResponse("/admin", status_code=303)
    resp.set_cookie(_COOKIE, token, httponly=True, samesite="strict", max_age=settings.admin_session_ttl_hours * 3600)
    return resp


@router.post("/logout")
async def logout():
    resp = RedirectResponse("/admin/login", status_code=303)
    resp.delete_cookie(_COOKIE)
    return resp


@router.get("/api/config")
async def api_config(_: bool = Depends(require_admin)) -> JSONResponse:
    return JSONResponse({"config": config_io.get_config_snapshot()})
```

(Page routes and the rest of the API come in later tasks; this task establishes auth + login/logout + one guarded endpoint so the auth model is tested first.)

- [ ] **Step 4:** Run the test (4 passed) and the full suite (only the 4 known failures).

- [ ] **Step 5: Commit**

```bash
git add src/admin/routes.py tests/test_admin_login.py
git commit -m "feat(admin): session login/logout + require_admin dependency"
```

---

### Task 3: Live data API — resources + queue

**Files:** Modify `src/admin/routes.py`; Create `tests/test_admin_api.py`.

- [ ] **Step 1: Write the failing test.** Create `tests/test_admin_api.py`:

```python
"""Admin live-data endpoints return engine telemetry behind auth."""

import importlib

from fastapi import FastAPI
from fastapi.testclient import TestClient


class _FakeEngine:
    device = "cpu"
    def get_stats(self):
        return {"device_resources": {"device": "cpu", "mem_used_mb": 1.0, "mem_total_mb": 2.0,
                                     "mem_free_mb": 1.0, "used_pct": 50.0},
                "requests_per_second": 3.0, "throughput_pairs_per_sec": 12.0,
                "inference_latency_p50_ms": 5.0, "inference_latency_p95_ms": 9.0,
                "batch_occupancy_pct": 40.0, "pending_requests": 0}
    def get_queue_snapshot(self):
        return {"waiting": [{"request_id": "r1", "num_docs": 3, "waited_ms": 12.0}],
                "running": [{"batch_id": "b1", "num_requests": 2, "pairs": 5, "elapsed_ms": 30.0}]}


def _app(monkeypatch):
    from src.config import settings as s
    monkeypatch.setattr(s, "admin_password", "pw")
    import src.admin.routes as routes
    importlib.reload(routes)

    async def fake_get_async_engine():
        return _FakeEngine()
    monkeypatch.setattr("src.engine.get_async_engine", fake_get_async_engine, raising=False)
    monkeypatch.setattr(routes, "_get_engine", fake_get_async_engine, raising=False)

    app = FastAPI()
    app.include_router(routes.router)
    c = TestClient(app)
    c.post("/admin/login", data={"password": "pw"})
    return c


def test_resources_endpoint(monkeypatch):
    c = _app(monkeypatch)
    r = c.get("/admin/api/resources")
    assert r.status_code == 200
    assert r.json()["device_resources"]["used_pct"] == 50.0


def test_queue_endpoint(monkeypatch):
    c = _app(monkeypatch)
    r = c.get("/admin/api/queue")
    body = r.json()
    assert r.status_code == 200
    assert body["waiting"][0]["request_id"] == "r1"
    assert body["running"][0]["batch_id"] == "b1"


def test_resources_requires_auth(monkeypatch):
    from src.config import settings as s
    monkeypatch.setattr(s, "admin_password", "pw")
    import src.admin.routes as routes
    importlib.reload(routes)
    app = FastAPI(); app.include_router(routes.router)
    r = TestClient(app).get("/admin/api/resources", follow_redirects=False)
    assert r.status_code in (303, 401)
```

- [ ] **Step 2: Run it.** Expected: FAIL (endpoints missing).

- [ ] **Step 3: Implement.** In `src/admin/routes.py`, add an engine accessor and two endpoints:

```python
async def _get_engine():
    from src.engine import get_async_engine
    return await get_async_engine()


@router.get("/api/resources")
async def api_resources(_: bool = Depends(require_admin)) -> JSONResponse:
    engine = await _get_engine()
    stats = engine.get_stats()
    return JSONResponse({
        "device_resources": stats.get("device_resources", {}),
        "throughput": {
            "requests_per_second": stats.get("requests_per_second"),
            "pairs_per_second": stats.get("throughput_pairs_per_sec"),
        },
        "latency": {
            "p50_ms": stats.get("inference_latency_p50_ms"),
            "p95_ms": stats.get("inference_latency_p95_ms"),
        },
        "batch_occupancy_pct": stats.get("batch_occupancy_pct"),
        "pending_requests": stats.get("pending_requests"),
    })


@router.get("/api/queue")
async def api_queue(_: bool = Depends(require_admin)) -> JSONResponse:
    engine = await _get_engine()
    return JSONResponse(engine.get_queue_snapshot())
```

- [ ] **Step 4:** Run the test (3 passed) and the full suite (only the 4 known failures).

- [ ] **Step 5: Commit**

```bash
git add src/admin/routes.py tests/test_admin_api.py
git commit -m "feat(admin): live resources + queue API endpoints"
```

---

### Task 4: Config apply + restart

**Files:** Modify `src/admin/routes.py`; Create `tests/test_admin_apply.py`.

- [ ] **Step 1: Write the failing test.** Create `tests/test_admin_apply.py`:

```python
"""POST /admin/api/config writes config; /admin/api/restart triggers drain."""

import importlib

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _client(monkeypatch, tmp_path):
    from src.config import settings as s
    monkeypatch.setattr(s, "admin_password", "pw")
    cfg = tmp_path / "config.yml"
    cfg.write_text("logging:\n  level: info\n")
    monkeypatch.setenv("RERANKER_CONFIG_PATH", str(cfg))

    import src.admin.routes as routes
    importlib.reload(routes)

    calls = {"reload": 0, "drain": 0}

    async def fake_reset():
        calls["reload"] += 1
    monkeypatch.setattr(routes, "_reload_engine", fake_reset, raising=False)

    class _Eng:
        def begin_shutdown(self):
            calls["drain"] += 1
    async def fake_get_engine():
        return _Eng()
    monkeypatch.setattr(routes, "_get_engine", fake_get_engine, raising=False)
    monkeypatch.setattr(routes, "_schedule_exit", lambda: calls.__setitem__("exit", True), raising=False)

    app = FastAPI(); app.include_router(routes.router)
    c = TestClient(app)
    c.post("/admin/login", data={"password": "pw"})
    return c, calls, cfg


def test_apply_writes_config_and_reloads(monkeypatch, tmp_path):
    c, calls, cfg = _client(monkeypatch, tmp_path)
    r = c.post("/admin/api/config", json={"updates": {"log_level": "debug"}})
    assert r.status_code == 200
    assert r.json()["written"] is True
    import yaml
    assert yaml.safe_load(cfg.read_text())["logging"]["level"] == "debug"
    assert calls["reload"] == 1


def test_apply_rejects_unknown_key(monkeypatch, tmp_path):
    c, calls, cfg = _client(monkeypatch, tmp_path)
    r = c.post("/admin/api/config", json={"updates": {"bogus": 1}})
    assert r.status_code == 400


def test_restart_triggers_drain(monkeypatch, tmp_path):
    c, calls, cfg = _client(monkeypatch, tmp_path)
    r = c.post("/admin/api/restart")
    assert r.status_code == 200
    assert calls["drain"] == 1
    assert calls.get("exit") is True
```

- [ ] **Step 2: Run it.** Expected: FAIL (endpoints missing).

- [ ] **Step 3: Implement.** In `src/admin/routes.py` add:

```python
import asyncio
import os
import signal


async def _reload_engine():
    """Re-create the engine so config changes (model/device/batch) take effect."""
    from src.engine import reset_async_engine, get_async_engine
    await reset_async_engine()
    await get_async_engine()


def _schedule_exit():
    """Ask the process to exit so supervisord respawns with new config."""
    os.kill(os.getpid(), signal.SIGTERM)


@router.post("/api/config")
async def api_apply_config(request: Request, _: bool = Depends(require_admin)) -> JSONResponse:
    body = await request.json()
    updates = body.get("updates", {})
    if not isinstance(updates, dict) or not updates:
        raise HTTPException(status_code=400, detail="No updates provided")
    result = config_io.write_config_updates(updates)
    if not result["written"]:
        raise HTTPException(status_code=400, detail={"rejected": result["rejected"]})
    try:
        await _reload_engine()
        result["reloaded"] = True
    except Exception as e:  # reload failure must be surfaced, not hidden
        logger.error("engine_reload_failed", error=str(e))
        result["reloaded"] = False
        result["reload_error"] = str(e)
    return JSONResponse(result)


@router.post("/api/restart")
async def api_restart(_: bool = Depends(require_admin)) -> JSONResponse:
    engine = await _get_engine()
    engine.begin_shutdown()
    _schedule_exit()
    return JSONResponse({"restarting": True})
```

NOTE: the tests monkeypatch `_reload_engine`, `_get_engine`, and `_schedule_exit` at module level, so define them as module-level names (not closures).

- [ ] **Step 4:** Run the test (3 passed) and the full suite (only the 4 known failures).

- [ ] **Step 5: Commit**

```bash
git add src/admin/routes.py tests/test_admin_apply.py
git commit -m "feat(admin): config apply (engine reload) + graceful restart endpoint"
```

---

### Task 5: Log tail API

**Files:** Modify `src/admin/routes.py`; Create `tests/test_admin_logs.py`.

- [ ] **Step 1: Write the failing test.** Create `tests/test_admin_logs.py`:

```python
"""GET /admin/api/logs/tail returns recent log lines, filterable."""

import importlib

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _client(monkeypatch, tmp_path):
    from src.config import settings as s
    monkeypatch.setattr(s, "admin_password", "pw")
    monkeypatch.setattr(s, "log_dir", str(tmp_path))
    (tmp_path / "reranker.log").write_text(
        "\n".join(f"line {i} INFO hello" for i in range(100)) + "\n"
        + "line 100 ERROR boom\n"
    )
    import src.admin.routes as routes
    importlib.reload(routes)
    app = FastAPI(); app.include_router(routes.router)
    c = TestClient(app)
    c.post("/admin/login", data={"password": "pw"})
    return c


def test_tail_returns_last_n(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.get("/admin/api/logs/tail?lines=10")
    assert r.status_code == 200
    lines = r.json()["lines"]
    assert len(lines) == 10
    assert "line 100 ERROR boom" in lines[-1]


def test_tail_filters_by_query(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.get("/admin/api/logs/tail?lines=200&q=ERROR")
    lines = r.json()["lines"]
    assert all("ERROR" in ln for ln in lines)
    assert len(lines) == 1
```

- [ ] **Step 2: Run it.** Expected: FAIL (endpoint missing).

- [ ] **Step 3: Implement.** In `src/admin/routes.py` add:

```python
import glob


@router.get("/api/logs/tail")
async def api_logs_tail(_: bool = Depends(require_admin), lines: int = 200, q: str = "") -> JSONResponse:
    lines = max(1, min(lines, 5000))
    log_files = sorted(glob.glob(os.path.join(settings.log_dir, "*.log")))
    collected: list[str] = []
    for path in log_files:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                collected.extend(f.read().splitlines())
        except OSError:
            continue
    if q:
        collected = [ln for ln in collected if q in ln]
    return JSONResponse({"lines": collected[-lines:]})
```

- [ ] **Step 4:** Run the test (2 passed) and the full suite (only the 4 known failures).

- [ ] **Step 5: Commit**

```bash
git add src/admin/routes.py tests/test_admin_logs.py
git commit -m "feat(admin): log tail API (last-N + substring filter)"
```

---

### Task 6: Pages (Jinja2 + HTMX) + mount in main.py

**Files:** Modify `src/admin/routes.py`; Create `src/admin/templates/{base,login,dashboard,config,logs}.html`; Modify `src/main.py`; Create `tests/test_admin_pages.py`.

- [ ] **Step 1: Write the failing test.** Create `tests/test_admin_pages.py`:

```python
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
```

- [ ] **Step 2: Run it.** Expected: FAIL (pages missing).

- [ ] **Step 3: Implement templates + page routes.** Create `src/admin/templates/base.html`:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Reranker Admin</title>
  <script src="/admin/static/htmx.min.js"></script>
  <style>
    body{font-family:system-ui,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
    header{display:flex;gap:1rem;align-items:center;padding:.6rem 1rem;background:#1a1d24;border-bottom:1px solid #2a2e37}
    header a{color:#9ad;text-decoration:none;margin-right:.8rem}
    main{padding:1rem;max-width:1100px;margin:0 auto}
    .card{background:#171a21;border:1px solid #262b35;border-radius:8px;padding:1rem;margin-bottom:1rem}
    .bar{height:18px;background:#2a2e37;border-radius:4px;overflow:hidden}
    .bar>span{display:block;height:100%;background:linear-gradient(90deg,#4ad,#4fa)}
    table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:.3rem .5rem;border-bottom:1px solid #262b35;font-size:.9rem}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem}
    input,button,select,textarea{background:#0f1115;color:#e6e6e6;border:1px solid #2a2e37;border-radius:4px;padding:.4rem}
    button{background:#2563eb;border:none;cursor:pointer}
  </style>
</head>
<body>
  {% if authed %}
  <header>
    <strong>Reranker Admin</strong>
    <a href="/admin">Dashboard</a><a href="/admin/config">Config</a><a href="/admin/logs">Logs</a>
    <form method="post" action="/admin/logout" style="margin-left:auto"><button>Logout</button></form>
  </header>
  {% endif %}
  <main>{% block body %}{% endblock %}</main>
</body>
</html>
```

Create `src/admin/templates/login.html`:

```html
{% extends "base.html" %}
{% block body %}
<div class="card" style="max-width:360px;margin:3rem auto">
  <h2>Admin login</h2>
  {% if error %}<p style="color:#f87171">Invalid password.</p>{% endif %}
  <form method="post" action="/admin/login">
    <p><input type="password" name="password" placeholder="Password" autofocus style="width:100%"/></p>
    <button type="submit">Sign in</button>
  </form>
</div>
{% endblock %}
```

Create `src/admin/templates/dashboard.html`:

```html
{% extends "base.html" %}
{% block body %}
<h2>Dashboard</h2>
<div class="grid">
  <div class="card"><h3>GPU / Device Quota</h3>
    <div hx-get="/admin/partials/resources" hx-trigger="load, every 2s">Loading…</div>
  </div>
  <div class="card"><h3>Queue</h3>
    <div hx-get="/admin/partials/queue" hx-trigger="load, every 2s">Loading…</div>
  </div>
</div>
{% endblock %}
```

Create `src/admin/templates/config.html`:

```html
{% extends "base.html" %}
{% block body %}
<h2>Configuration</h2>
<div class="card" hx-get="/admin/partials/config" hx-trigger="load">Loading…</div>
{% endblock %}
```

Create `src/admin/templates/logs.html`:

```html
{% extends "base.html" %}
{% block body %}
<h2>Logs</h2>
<div class="card">
  <input id="q" placeholder="filter" hx-get="/admin/partials/logs" hx-trigger="keyup changed delay:400ms" hx-target="#logbox" hx-include="#q" name="q"/>
  <pre id="logbox" hx-get="/admin/partials/logs" hx-trigger="load, every 3s" hx-include="#q" style="max-height:60vh;overflow:auto"></pre>
</div>
{% endblock %}
```

In `src/admin/routes.py`, set up Jinja2 and add page + partial routes:

```python
import os as _os
from fastapi.templating import Jinja2Templates

_TEMPLATES = Jinja2Templates(directory=_os.path.join(_os.path.dirname(__file__), "templates"))


def _authed(request: Request) -> bool:
    if not settings.admin_password:
        return False
    return auth.verify_session_token(request.cookies.get(_COOKIE, ""), settings.admin_password)


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: int = 0):
    return _TEMPLATES.TemplateResponse("login.html", {"request": request, "error": error, "authed": False})


@router.get("", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    if not _authed(request):
        return RedirectResponse("/admin/login", status_code=303)
    return _TEMPLATES.TemplateResponse("dashboard.html", {"request": request, "authed": True})


@router.get("/config", response_class=HTMLResponse)
async def config_page(request: Request):
    if not _authed(request):
        return RedirectResponse("/admin/login", status_code=303)
    return _TEMPLATES.TemplateResponse("config.html", {"request": request, "authed": True})


@router.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request):
    if not _authed(request):
        return RedirectResponse("/admin/login", status_code=303)
    return _TEMPLATES.TemplateResponse("logs.html", {"request": request, "authed": True})


@router.get("/partials/resources", response_class=HTMLResponse)
async def partial_resources(_: bool = Depends(require_admin)):
    engine = await _get_engine()
    s = engine.get_stats()
    res = s.get("device_resources", {})
    pct = res.get("used_pct", 0.0)
    extra = ""
    if res.get("util_pct") is not None:
        extra = f"<p>Util {res.get('util_pct')}% &nbsp; {res.get('temp_c','?')}&deg;C &nbsp; {res.get('power_w','?')}W</p>"
    return HTMLResponse(
        f"<p>{res.get('device','?')} — {res.get('mem_used_mb',0)} / {res.get('mem_total_mb',0)} MB ({pct}%)</p>"
        f'<div class="bar"><span style="width:{min(pct,100)}%"></span></div>{extra}'
        f"<p>p50 {s.get('inference_latency_p50_ms','?')} ms &nbsp; p95 {s.get('inference_latency_p95_ms','?')} ms &nbsp; "
        f"{s.get('throughput_pairs_per_sec','?')} pairs/s</p>"
    )


@router.get("/partials/queue", response_class=HTMLResponse)
async def partial_queue(_: bool = Depends(require_admin)):
    engine = await _get_engine()
    snap = engine.get_queue_snapshot()
    run = "".join(f"<tr><td>{b['batch_id']}</td><td>{b['num_requests']}</td><td>{b['pairs']}</td><td>{b['elapsed_ms']}</td></tr>" for b in snap["running"])
    wait = "".join(f"<tr><td>{w['request_id']}</td><td>{w['num_docs']}</td><td>{w['waited_ms']}</td></tr>" for w in snap["waiting"])
    return HTMLResponse(
        f"<h4>Running ({len(snap['running'])})</h4><table><tr><th>batch</th><th>reqs</th><th>pairs</th><th>ms</th></tr>{run}</table>"
        f"<h4>Waiting ({len(snap['waiting'])})</h4><table><tr><th>request</th><th>docs</th><th>ms</th></tr>{wait}</table>"
    )


@router.get("/partials/config", response_class=HTMLResponse)
async def partial_config(_: bool = Depends(require_admin)):
    rows = config_io.get_config_snapshot()
    body = "".join(
        f"<tr><td>{r['name']}</td><td>{r['value']}</td><td>{r['source']}</td>"
        f"<td>{'restart' if r['needs_restart'] else 'hot'}</td></tr>" for r in rows
    )
    return HTMLResponse(f"<table><tr><th>setting</th><th>value</th><th>source</th><th>apply</th></tr>{body}</table>")


@router.get("/partials/logs", response_class=HTMLResponse)
async def partial_logs(_: bool = Depends(require_admin), q: str = ""):
    import glob, html
    log_files = sorted(glob.glob(_os.path.join(settings.log_dir, "*.log")))
    out: list[str] = []
    for path in log_files:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                out.extend(f.read().splitlines())
        except OSError:
            continue
    if q:
        out = [ln for ln in out if q in ln]
    return HTMLResponse(html.escape("\n".join(out[-300:])))
```

In `src/main.py` `create_app`, after the existing routers are included, mount the admin router and its static dir:

```python
    from src.admin.routes import router as admin_router
    from fastapi.staticfiles import StaticFiles
    import os as _os
    app.include_router(admin_router)
    _static = _os.path.join(_os.path.dirname(__file__), "admin", "static")
    app.mount("/admin/static", StaticFiles(directory=_static), name="admin-static")
```

- [ ] **Step 4:** Run the test (3 passed) and the full suite (only the 4 known failures). If `Jinja2Templates` import fails, ensure `jinja2` is installed in the venv (Task 0).

- [ ] **Step 5: Commit**

```bash
git add src/admin/routes.py src/admin/templates src/main.py tests/test_admin_pages.py
git commit -m "feat(admin): Jinja2+HTMX pages (dashboard/config/logs) mounted at /admin"
```

---

### Task 7: Docs + integration sweep

**Files:** Create `docs/admin-ui.md`; Modify `README.md`.

- [ ] **Step 1: Write `docs/admin-ui.md`** documenting: enabling the UI (`RERANKER_ADMIN_PASSWORD`), local-only binding, the dashboard/config/logs pages, config apply (hot vs restart), and the security model (session cookie, throttle, `/stats` `/info` gating). Keep it ~1 page.

- [ ] **Step 2: Link it** from the README documentation table: `| [Admin UI](docs/admin-ui.md) | Local password-gated dashboard: GPU quota, queue, config, logs |`.

- [ ] **Step 3: Full suite** — only the 4 known-failing tests fail.

- [ ] **Step 4: Manual smoke (optional, documented for the target box)** — `RERANKER_ADMIN_PASSWORD=secret ./run.sh`, open `http://localhost:8000/admin`, log in, confirm the dashboard polls live data.

- [ ] **Step 5: Commit**

```bash
git add docs/admin-ui.md README.md
git commit -m "docs(admin): admin UI runbook + README link"
```

---

## Self-Review

**Spec coverage (Part D + B3):** session-cookie auth + login (single password) → Task 2; dashboard (GPU quota gauge, queue running/waiting, throughput/latency) → Tasks 3, 6; config snapshot + edit + apply(reload)/restart → Tasks 1, 4; log viewer → Tasks 5, 6; mounted at /admin with vendored HTMX, offline → Tasks 0, 6; `/admin/*` guarded by `require_admin` → all API/page routes. Stack deviation (HTMX+CSS partials instead of Chart.js+SSE) documented at top.

**Placeholder scan:** none — every route, template, and test shown in full.

**Type consistency:** `require_admin` dependency used by every guarded route; `_get_engine`/`_reload_engine`/`_schedule_exit` defined module-level (Tasks 3/4) and monkeypatched in tests; `config_io.get_config_snapshot()`/`write_config_updates()` (Task 1) consumed by Tasks 4 and 6; consumes Phase 3B `engine.get_stats().device_resources` + `engine.get_queue_snapshot()` and Phase 3C `auth.*` + `settings.admin_password`.
