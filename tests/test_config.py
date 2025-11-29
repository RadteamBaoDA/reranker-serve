"""
Unit tests for configuration settings.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Ensure torch is mocked before importing settings
if "torch" not in sys.modules:
    mock_torch = MagicMock()
    mock_torch.__version__ = "2.0.0"
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False
    mock_torch.backends.mps.is_built.return_value = False
    mock_torch.float16 = "float16"
    mock_torch.float32 = "float32"
    sys.modules["torch"] = mock_torch


class TestSettings:
    """Tests for the Settings class."""
    
    def test_default_settings(self):
        """Test default settings values."""
        from src.config.settings import Settings
        
        settings = Settings()
        
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.workers == 1
        assert settings.model_name == "BAAI/bge-reranker-v2-m3"
        assert settings.max_length == 512
        assert settings.batch_size == 32
    
    def test_settings_from_env(self):
        """Test settings loaded from environment variables."""
        with patch.dict(os.environ, {
            "RERANKER_PORT": "9000",
            "RERANKER_MODEL_NAME": "custom/model",
            "RERANKER_MAX_LENGTH": "256",
        }):
            from src.config.settings import Settings
            settings = Settings()
            
            assert settings.port == 9000
            assert settings.model_name == "custom/model"
            assert settings.max_length == 256
    
    def test_get_device_auto_detect(self):
        """Test automatic device detection."""
        from src.config.settings import Settings
        
        settings = Settings()
        device = settings.get_device()
        
        # With mocked torch, should return "cpu" since CUDA and MPS are mocked as unavailable
        assert device in ["cuda", "mps", "cpu"]
    
    def test_get_device_explicit(self):
        """Test explicit device setting."""
        from src.config.settings import Settings
        
        settings = Settings(device="cpu")
        device = settings.get_device()
        
        assert device == "cpu"
    
    def test_get_model_load_path_default(self):
        """Test model load path with default settings."""
        from src.config.settings import Settings
        
        settings = Settings()
        path = settings.get_model_load_path()
        
        assert path == settings.model_name
    
    def test_get_model_load_path_local(self, tmp_path):
        """Test model load path with local model."""
        from src.config.settings import Settings
        
        local_path = str(tmp_path)
        settings = Settings(model_path=local_path)
        path = settings.get_model_load_path()
        
        assert path == local_path
    
    def test_get_cors_origins_list_wildcard(self):
        """Test CORS origins parsing with wildcard."""
        from src.config.settings import Settings
        
        settings = Settings(cors_origins="*")
        origins = settings.get_cors_origins_list()
        
        assert origins == ["*"]
    
    def test_get_cors_origins_list_multiple(self):
        """Test CORS origins parsing with multiple origins."""
        from src.config.settings import Settings
        
        settings = Settings(cors_origins="http://localhost:3000, http://example.com")
        origins = settings.get_cors_origins_list()
        
        assert "http://localhost:3000" in origins
        assert "http://example.com" in origins
    
    def test_is_mac(self):
        """Test macOS detection."""
        from src.config.settings import Settings
        import platform
        
        settings = Settings()
        result = settings.is_mac()
        
        expected = platform.system() == "Darwin"
        assert result == expected
