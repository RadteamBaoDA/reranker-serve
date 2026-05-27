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
