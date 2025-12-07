"""Settings helpers should remain deterministic with mocks."""

import os
from unittest.mock import patch


def test_default_settings_values():
    from src.config.settings import Settings

    cfg = Settings()

    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8000
    assert cfg.model_name == "BAAI/bge-reranker-v2-m3"
    assert cfg.max_length == 512
    assert cfg.batch_size == 32


def test_env_overrides_apply():
    with patch.dict(os.environ, {
        "RERANKER_PORT": "9100",
        "RERANKER_MODEL_NAME": "custom/model",
        "RERANKER_MAX_LENGTH": "128",
    }):
        from src.config.settings import Settings

        cfg = Settings()
        assert cfg.port == 9100
        assert cfg.model_name == "custom/model"
        assert cfg.max_length == 128


def test_device_prefers_force_cpu():
    from src.config.settings import Settings

    cfg = Settings(force_cpu_only=True, device=None)
    assert cfg.get_device() == "cpu"


def test_model_load_path_prefers_local(tmp_path):
    from src.config.settings import Settings

    local = tmp_path / "model"
    local.mkdir()
    cfg = Settings(model_path=str(local), model_name="remote/model")

    assert cfg.get_model_load_path() == str(local)


def test_cors_parsing_handles_list_and_wildcard():
    from src.config.settings import Settings

    wildcard = Settings(cors_origins="*")
    assert wildcard.get_cors_origins_list() == ["*"]

    mixed = Settings(cors_origins="http://one.test, http://two.test")
    assert set(mixed.get_cors_origins_list()) == {"http://one.test", "http://two.test"}
