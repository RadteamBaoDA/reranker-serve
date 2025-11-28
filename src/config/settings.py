"""
Configuration settings for the Reranker Service.
Supports environment variables and optimized settings for different platforms.
"""

import os
import platform
from typing import Optional, Literal
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
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
    
    # Device Configuration
    device: Optional[str] = Field(
        default=None,
        description="Device to use: 'cuda', 'mps', 'cpu', or None for auto-detect"
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
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    
    class Config:
        env_prefix = "RERANKER_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def get_device(self) -> str:
        """Auto-detect the best available device."""
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
    
    def get_cors_origins_list(self) -> list[str]:
        """Parse CORS origins string into list."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]


# Global settings instance
settings = Settings()
