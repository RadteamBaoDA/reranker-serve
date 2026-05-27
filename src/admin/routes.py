"""Admin UI router: session login + guarded API/page routes. Mounted at /admin."""

from __future__ import annotations

import os as _os

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from src.config import settings, get_logger
from src.admin import auth, config_io

_TEMPLATES = Jinja2Templates(directory=_os.path.join(_os.path.dirname(__file__), "templates"))

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


def _authed(request: Request) -> bool:
    if not settings.admin_password:
        return False
    return auth.verify_session_token(request.cookies.get(_COOKIE, ""), settings.admin_password)


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


async def _reload_engine():
    """Re-create the engine so config changes (model/device/batch) take effect."""
    from src.engine import reset_async_engine, get_async_engine
    await reset_async_engine()
    await get_async_engine()


def _schedule_exit():
    """Ask the process to exit so supervisord respawns with new config."""
    import os
    import signal
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
    except Exception as e:
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


@router.get("/api/logs/tail")
async def api_logs_tail(_: bool = Depends(require_admin), lines: int = 200, q: str = "") -> JSONResponse:
    import glob
    import os
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


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: int = 0):
    return _TEMPLATES.TemplateResponse(request, "login.html", {"error": error, "authed": False})


@router.get("", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    if not _authed(request):
        return RedirectResponse("/admin/login", status_code=303)
    return _TEMPLATES.TemplateResponse(request, "dashboard.html", {"authed": True})


@router.get("/config", response_class=HTMLResponse)
async def config_page(request: Request):
    if not _authed(request):
        return RedirectResponse("/admin/login", status_code=303)
    return _TEMPLATES.TemplateResponse(request, "config.html", {"authed": True})


@router.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request):
    if not _authed(request):
        return RedirectResponse("/admin/login", status_code=303)
    return _TEMPLATES.TemplateResponse(request, "logs.html", {"authed": True})


# ---------------------------------------------------------------------------
# Partial routes (HTMX fragments)
# ---------------------------------------------------------------------------

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
    import glob
    import html
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
