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
