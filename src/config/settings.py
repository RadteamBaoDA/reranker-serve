"""
Configuration settings for the Reranker Service.
Supports environment variables, .env file, and YAML configuration files.
Priority: Environment Variables > config.yml > .env > defaults
"""

import os
import platform
import yaml
from typing import Optional, Literal, List, Dict, Any
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_prefix="RERANKER_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra environment variables
    )
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of uvicorn workers")
    reload: bool = Field(default=False, description="Enable auto-reload for development")
    
    # Model Configuration
    model_name: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Model name or path. Supports: BAAI/bge-reranker-*, Qwen3-reranker models"
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Local path to load model from disk. If set, will load from this path instead of downloading."
    )
    model_cache_dir: str = Field(
        default="./models",
        description="Directory to cache downloaded models"
    )
    use_offline_mode: bool = Field(
        default=False,
        description="If True, only load models from local path, no internet access"
    )
    
    # Model Inference Configuration
    max_length: int = Field(default=512, description="Maximum sequence length")
    batch_size: int = Field(default=32, description="Batch size for inference")
    normalize_scores: bool = Field(default=True, description="Normalize reranker scores to 0-1")
    
    # Async Engine / Concurrency Configuration (vLLM-inspired)
    enable_async_engine: bool = Field(
        default=True,
        description="Enable async engine for concurrent request handling"
    )
    max_concurrent_batches: int = Field(
        default=2,
        description="Maximum number of batches processing concurrently"
    )
    inference_threads: int = Field(
        default=1,
        description="Number of threads for model inference (increase for CPU-bound workloads)"
    )
    max_batch_size: int = Field(
        default=32,
        description="Maximum number of requests per batch"
    )
    max_batch_pairs: int = Field(
        default=1024,
        description="Maximum query-document pairs per batch"
    )
    batch_wait_timeout: float = Field(
        default=0.01,
        description="Time (seconds) to wait for batching requests together"
    )
    max_queue_size: int = Field(
        default=1000,
        description="Maximum number of requests in queue"
    )
    request_timeout: float = Field(
        default=60.0,
        description="Request timeout in seconds"
    )
    
    # Device Configuration
    device: Optional[str] = Field(
        default=None,
        description="Device to use: 'cuda', 'mps', 'cpu', or None for auto-detect"
    )
    force_cpu_only: bool = Field(
        default=False,
        description="Force CPU-only mode. Disables CUDA/MPS and sets CUDA_VISIBLE_DEVICES=''"
    )
    use_fp16: bool = Field(
        default=True,
        description="Use FP16 for faster inference (if supported)"
    )
    
    # MPS (Apple Silicon) Optimization
    mps_fallback_to_cpu: bool = Field(
        default=True,
        description="Fallback to CPU if MPS operation fails"
    )
    
    # API Configuration
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication (optional)"
    )
    enable_cors: bool = Field(default=True, description="Enable CORS")
    cors_origins: str = Field(
        default="*",
        description="Comma-separated list of allowed CORS origins"
    )
    
    # Load Balancer Configuration
    enable_load_balancer: bool = Field(
        default=False,
        description="Enable load balancing across multiple backends"
    )
    config_path: Optional[str] = Field(
        default=None,
        description="Path to LiteLLM-style YAML configuration file"
    )
    
    # HTTP Client Configuration
    trust_env: bool = Field(
        default=True,
        description="Trust environment variables for proxy settings (HTTP_PROXY, HTTPS_PROXY). Set to False to ignore proxy."
    )
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    json_logs: bool = Field(default=False, description="Output logs in JSON format")
    log_dir: str = Field(default="./logs", description="Directory to store log files")
    log_retention_days: int = Field(default=7, description="Number of days to keep log files")
    log_max_bytes: int = Field(default=10485760, description="Maximum size of log file in bytes before rotation (default: 10MB)")
    log_backup_count: int = Field(default=5, description="Number of backup log files to keep per day")
    
    def get_device(self) -> str:
        """Auto-detect the best available device."""
        # Force CPU mode if configured
        if self.force_cpu_only:
            return "cpu"
        
        if self.device:
            return self.device
        
        # Try CUDA first
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        
        # Try MPS (Apple Silicon)
        try:
            import torch
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return "mps"
        except (ImportError, AttributeError):
            pass
        
        return "cpu"
    
    def get_model_load_path(self) -> str:
        """Get the path to load the model from."""
        if self.model_path and os.path.exists(self.model_path):
            return self.model_path
        return self.model_name
    
    def is_mac(self) -> bool:
        """Check if running on macOS."""
        return platform.system() == "Darwin"
    
    def get_torch_dtype(self):
        """Get the appropriate torch dtype based on device and settings."""
        import torch
        
        device = self.get_device()
        
        # FP16 support varies by device
        if self.use_fp16:
            if device == "cuda":
                return torch.float16
            elif device == "mps":
                # MPS has limited FP16 support, use FP32 for stability
                return torch.float32
        
        return torch.float32
    
    def get_cors_origins_list(self) -> List[str]:
        """Parse CORS origins string into list."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]


def load_yaml_config(yaml_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        yaml_path: Path to YAML config file. If None, checks RERANKER_CONFIG_PATH env var.
        
    Returns:
        Dictionary with configuration values, empty dict if file not found.
    """
    # Determine config path
    if yaml_path is None:
        yaml_path = os.environ.get("RERANKER_CONFIG_PATH", "config.yml")
    
    config_file = Path(yaml_path)
    
    if not config_file.exists():
        return {}
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f) or {}
        
        # Flatten nested YAML structure to match environment variable names
        flat_config = {}
        
        # Map YAML sections to settings
        if 'server' in yaml_config:
            for key, value in yaml_config['server'].items():
                flat_config[key] = value
        
        if 'model' in yaml_config:
            model_cfg = yaml_config['model']
            flat_config['model_name'] = model_cfg.get('name')
            flat_config['model_path'] = model_cfg.get('path')
            flat_config['model_cache_dir'] = model_cfg.get('cache_dir')
            flat_config['use_offline_mode'] = model_cfg.get('use_offline_mode')
        
        if 'inference' in yaml_config:
            for key, value in yaml_config['inference'].items():
                flat_config[key] = value
        
        if 'device' in yaml_config:
            device_cfg = yaml_config['device']
            flat_config['device'] = device_cfg.get('name')
            flat_config['force_cpu_only'] = device_cfg.get('force_cpu_only')
            flat_config['use_fp16'] = device_cfg.get('use_fp16')
            flat_config['mps_fallback_to_cpu'] = device_cfg.get('mps_fallback_to_cpu')
        
        if 'api' in yaml_config:
            api_cfg = yaml_config['api']
            flat_config['api_key'] = api_cfg.get('key')
            flat_config['enable_cors'] = api_cfg.get('enable_cors')
            flat_config['cors_origins'] = api_cfg.get('cors_origins')
        
        if 'load_balancer' in yaml_config:
            lb_cfg = yaml_config['load_balancer']
            flat_config['enable_load_balancer'] = lb_cfg.get('enabled')
            flat_config['config_path'] = lb_cfg.get('config_path')
        
        if 'async_engine' in yaml_config:
            ae_cfg = yaml_config['async_engine']
            flat_config['enable_async_engine'] = ae_cfg.get('enabled')
            flat_config['max_concurrent_batches'] = ae_cfg.get('max_concurrent_batches')
            flat_config['inference_threads'] = ae_cfg.get('inference_threads')
            flat_config['max_batch_size'] = ae_cfg.get('max_batch_size')
            flat_config['max_batch_pairs'] = ae_cfg.get('max_batch_pairs')
            flat_config['batch_wait_timeout'] = ae_cfg.get('batch_wait_timeout')
            flat_config['max_queue_size'] = ae_cfg.get('max_queue_size')
            flat_config['request_timeout'] = ae_cfg.get('request_timeout')
        
        if 'http' in yaml_config:
            flat_config['trust_env'] = yaml_config['http'].get('trust_env')
        
        if 'logging' in yaml_config:
            log_cfg = yaml_config['logging']
            flat_config['log_level'] = log_cfg.get('level')
            flat_config['json_logs'] = log_cfg.get('json_logs')
            flat_config['log_dir'] = log_cfg.get('log_dir')
            flat_config['log_retention_days'] = log_cfg.get('retention_days')
            flat_config['log_max_bytes'] = log_cfg.get('max_bytes')
            flat_config['log_backup_count'] = log_cfg.get('backup_count')
        
        # Remove None values to allow defaults to take effect
        return {k: v for k, v in flat_config.items() if v is not None}
        
    except Exception as e:
        # If YAML loading fails, log and continue with defaults
        print(f"Warning: Failed to load YAML config from {yaml_path}: {e}")
        return {}


def create_settings() -> Settings:
    """
    Create Settings instance with proper configuration priority:
    1. Environment variables (highest priority)
    2. config.yml
    3. .env file
    4. Default values (lowest priority)
    """
    # Load YAML config first
    yaml_config = load_yaml_config()
    
    # Create settings with YAML config as overrides
    # Pydantic will handle the priority: env vars > init values > .env > defaults
    return Settings(**yaml_config)


# Global settings instance
settings = create_settings()
