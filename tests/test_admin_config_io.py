"""Config snapshot (value/source/needs_restart, secrets redacted) + YAML writeback."""

import textwrap

from src.admin import config_io


def test_secrets_are_redacted(monkeypatch):
    from src.config import settings as s
    monkeypatch.setattr(s, "api_key", "supersecret")
    snap = {row["name"]: row for row in config_io.get_config_snapshot()}
    assert snap["api_key"]["value"] == "***set***"
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
    assert data["inference"]["max_length"] == 256


def test_write_rejects_unknown_key(tmp_path):
    cfg = tmp_path / "config.yml"
    cfg.write_text("model:\n  name: x\n")
    result = config_io.write_config_updates({"not_a_setting": 1}, path=str(cfg))
    assert result["written"] is False
    assert "not_a_setting" in result["rejected"]


def test_write_coerces_string_to_int(tmp_path):
    cfg = tmp_path / "config.yml"
    cfg.write_text("model:\n  name: x\n")
    result = config_io.write_config_updates({"max_length": "256"}, path=str(cfg))
    assert result["written"] is True
    import yaml
    assert yaml.safe_load(cfg.read_text())["inference"]["max_length"] == 256
    assert isinstance(yaml.safe_load(cfg.read_text())["inference"]["max_length"], int)


def test_write_coerces_string_to_bool(tmp_path):
    cfg = tmp_path / "config.yml"
    cfg.write_text("model:\n  name: x\n")
    result = config_io.write_config_updates({"enable_docs": "false"}, path=str(cfg))
    assert result["written"] is True
    import yaml
    assert yaml.safe_load(cfg.read_text())["api"]["enable_docs"] is False


def test_write_rejects_uncoercible_value(tmp_path):
    cfg = tmp_path / "config.yml"
    cfg.write_text("model:\n  name: x\n")
    result = config_io.write_config_updates({"max_length": "notanint"}, path=str(cfg))
    assert result["written"] is False
    assert "max_length" in result["rejected"]
