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
