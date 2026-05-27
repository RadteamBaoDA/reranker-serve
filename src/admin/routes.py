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


@router.get("/api/metrics/history")
async def api_metrics_history(
    _: bool = Depends(require_admin),
    window: float | None = None,
    since: float | None = None,
) -> JSONResponse:
    """Time-series perf samples for the dashboard charts.

    No params -> full retained buffer. ``window`` (seconds) seeds a recent
    window; ``since`` (epoch seconds) returns only newer samples for polling.
    """
    import time as _t
    from src.observability.metrics_history import get_history
    samples = get_history().get(window_seconds=window, since_ts=since)
    return JSONResponse({"samples": samples, "server_now": _t.time()})


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
    return _TEMPLATES.TemplateResponse(request, "dashboard.html", {"authed": True, "active": "dashboard"})


@router.get("/config", response_class=HTMLResponse)
async def config_page(request: Request):
    if not _authed(request):
        return RedirectResponse("/admin/login", status_code=303)
    return _TEMPLATES.TemplateResponse(request, "config.html", {"authed": True, "active": "config"})


@router.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request):
    if not _authed(request):
        return RedirectResponse("/admin/login", status_code=303)
    return _TEMPLATES.TemplateResponse(request, "logs.html", {"authed": True, "active": "logs"})


# ---------------------------------------------------------------------------
# Partial routes (HTMX fragments)
# ---------------------------------------------------------------------------

def _fmt(value, suffix="", dash="—"):
    """Render a number compactly, falling back to a dash when missing."""
    if value is None:
        return dash
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    out = f"{f:,.0f}" if abs(f) >= 100 else f"{f:,.1f}"
    return f"{out}{suffix}"


def _kpi(label, value, sub="", accent=""):
    cls = f"kpi {accent}".strip()
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="{cls}"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{sub_html}</div>'
    )


@router.get("/partials/stats", response_class=HTMLResponse)
async def partial_stats(_: bool = Depends(require_admin)):
    """Live KPI cards across the top of the dashboard."""
    engine = await _get_engine()
    s = engine.get_stats()
    res = s.get("device_resources", {}) or {}
    running = s.get("inflight_batches", 0)
    waiting = s.get("pending_requests", 0)
    mem_used, mem_total = res.get("mem_used_mb"), res.get("mem_total_mb")
    mem_pct = res.get("used_pct")
    has_gpu = res.get("util_pct") is not None

    if has_gpu:
        gpu_value = _fmt(res.get("util_pct"), "%")
        gpu_sub = f"{_fmt(res.get('temp_c'), '°C')} · {_fmt(res.get('power_w'), 'W')}"
    else:
        gpu_value = res.get("device", "cpu")
        gpu_sub = "no GPU telemetry"

    vram_sub = (
        f'{_fmt(mem_used,"")} / {_fmt(mem_total," MB")} '
        f'<div class="bar"><span style="width:{min(float(mem_pct or 0),100)}%"></span></div>'
        if mem_total else "—"
    )
    occ = float(s.get("batch_occupancy_pct") or 0)

    cards = "".join([
        _kpi("Latency p50", _fmt(s.get("inference_latency_p50_ms"), " ms"),
             f"p95 {_fmt(s.get('inference_latency_p95_ms'), ' ms')}", "accent-blue"),
        _kpi("Throughput", _fmt(s.get("requests_per_second"), " req/s"),
             f"{_fmt(s.get('throughput_pairs_per_sec'), ' pairs/s')}", "accent-green"),
        _kpi("GPU Util", gpu_value, gpu_sub, "accent-violet"),
        _kpi("VRAM", _fmt(mem_pct, "%"), vram_sub, "accent-amber"),
        _kpi("Queue", _fmt(running, "") + " running", f"{_fmt(waiting,'')} waiting", "accent-cyan"),
        _kpi("Batch occupancy", _fmt(occ, "%"),
             f'<div class="bar"><span style="width:{min(occ,100)}%"></span></div>', "accent-pink"),
    ])
    return HTMLResponse(f'<div class="kpi-grid">{cards}</div>')


@router.get("/partials/queue", response_class=HTMLResponse)
async def partial_queue(_: bool = Depends(require_admin)):
    engine = await _get_engine()
    snap = engine.get_queue_snapshot()
    run = "".join(
        f"<tr><td class='mono'>{b['batch_id']}</td><td>{b['num_requests']}</td>"
        f"<td>{b['pairs']}</td><td>{_fmt(b['elapsed_ms'],' ms')}</td></tr>"
        for b in snap["running"]
    ) or '<tr class="empty"><td colspan="4">idle — no batches running</td></tr>'
    wait = "".join(
        f"<tr><td class='mono'>{w['request_id']}</td><td>{w['num_docs']}</td>"
        f"<td>{_fmt(w['waited_ms'],' ms')}</td></tr>"
        for w in snap["waiting"]
    ) or '<tr class="empty"><td colspan="3">empty — nothing waiting</td></tr>'
    return HTMLResponse(
        f'<div class="table-head"><h4>Running</h4><span class="pill">{len(snap["running"])}</span></div>'
        f'<table class="data"><thead><tr><th>batch</th><th>reqs</th><th>pairs</th><th>elapsed</th></tr></thead>'
        f'<tbody>{run}</tbody></table>'
        f'<div class="table-head"><h4>Waiting</h4><span class="pill">{len(snap["waiting"])}</span></div>'
        f'<table class="data"><thead><tr><th>request</th><th>docs</th><th>waited</th></tr></thead>'
        f'<tbody>{wait}</tbody></table>'
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
