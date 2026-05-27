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
