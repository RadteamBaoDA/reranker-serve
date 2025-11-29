"""
LiteLLM-style YAML configuration loader for load balancer.
"""

import os
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator

from src.config import get_logger

logger = get_logger(__name__)


class ModelConfig(BaseModel):
    """Configuration for a single reranker model/backend."""
    
    model_name: str = Field(..., description="Model identifier or name")
    litellm_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="LiteLLM-compatible parameters"
    )
    
    # Backend configuration
    api_base: Optional[str] = Field(
        default=None, 
        description="Base URL for the reranker API backend"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication"
    )
    
    # Load balancing parameters
    weight: float = Field(
        default=1.0, 
        ge=0.0,
        description="Weight for weighted load balancing (higher = more traffic)"
    )
    priority: int = Field(
        default=0,
        description="Priority for failover (lower = higher priority)"
    )
    
    # Rate limiting
    rpm: Optional[int] = Field(
        default=None, 
        description="Requests per minute limit"
    )
    tpm: Optional[int] = Field(
        default=None,
        description="Tokens per minute limit"
    )
    max_parallel_requests: Optional[int] = Field(
        default=None,
        description="Maximum concurrent requests"
    )
    
    # Health check
    health_check_enabled: bool = Field(
        default=True,
        description="Enable health checks for this backend"
    )
    health_check_interval: int = Field(
        default=60,
        description="Health check interval in seconds"
    )
    
    # Timeout settings
    timeout: float = Field(
        default=30.0,
        description="Request timeout in seconds"
    )
    
    # Model-specific settings
    max_length: Optional[int] = Field(
        default=None,
        description="Maximum sequence length override"
    )
    batch_size: Optional[int] = Field(
        default=None,
        description="Batch size override"
    )
    
    @field_validator('api_base')
    @classmethod
    def validate_api_base(cls, v: Optional[str]) -> Optional[str]:
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError("api_base must start with http:// or https://")
        return v.rstrip('/') if v else v


class RouterSettings(BaseModel):
    """Settings for the load balancer router."""
    
    routing_strategy: Literal[
        "simple-shuffle",
        "least-busy", 
        "usage-based-routing",
        "latency-based-routing",
        "cost-based-routing",
        "weighted-random",
        "round-robin",
        "priority-failover"
    ] = Field(
        default="weighted-random",
        description="Load balancing strategy"
    )
    
    # Retry settings
    num_retries: int = Field(
        default=3,
        ge=0,
        description="Number of retries on failure"
    )
    retry_delay: float = Field(
        default=1.0,
        ge=0.0,
        description="Delay between retries in seconds"
    )
    retry_on_status_codes: List[int] = Field(
        default=[429, 500, 502, 503, 504],
        description="HTTP status codes to retry on"
    )
    
    # Fallback settings
    fallback_to_local: bool = Field(
        default=True,
        description="Fall back to local model if all backends fail"
    )
    
    # Timeout settings
    default_timeout: float = Field(
        default=30.0,
        description="Default request timeout"
    )
    
    # Caching
    enable_caching: bool = Field(
        default=False,
        description="Enable response caching"
    )
    cache_ttl: int = Field(
        default=3600,
        description="Cache TTL in seconds"
    )
    
    # Health check
    health_check_interval: int = Field(
        default=60,
        description="Global health check interval in seconds"
    )
    
    # Logging
    enable_request_logging: bool = Field(
        default=True,
        description="Log all requests"
    )


class LoadBalancerConfig(BaseModel):
    """Main load balancer configuration."""
    
    model_list: List[ModelConfig] = Field(
        default_factory=list,
        description="List of model configurations"
    )
    
    router_settings: RouterSettings = Field(
        default_factory=RouterSettings,
        description="Router settings"
    )
    
    # Environment variable overrides
    environment_variables: Dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables to set"
    )
    
    # General settings
    general_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="General settings"
    )
    
    def get_model_by_name(self, name: str) -> Optional[ModelConfig]:
        """Get a model configuration by name."""
        for model in self.model_list:
            if model.model_name == name:
                return model
        return None
    
    def get_healthy_models(self, healthy_backends: set) -> List[ModelConfig]:
        """Get list of healthy model configurations."""
        return [
            m for m in self.model_list 
            if m.model_name in healthy_backends or m.api_base is None
        ]


def load_config(config_path: Optional[str] = None) -> LoadBalancerConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file. If None, looks for:
            1. RERANKER_CONFIG_PATH environment variable
            2. ./reranker_config.yaml
            3. ./config.yaml
            
    Returns:
        LoadBalancerConfig instance
    """
    import yaml
    
    # Determine config path
    if config_path is None:
        config_path = os.environ.get("RERANKER_CONFIG_PATH")
    
    if config_path is None:
        # Check default locations
        default_paths = [
            "./reranker_config.yaml",
            "./reranker_config.yml", 
            "./config.yaml",
            "./config.yml",
            "./litellm_config.yaml",
        ]
        for path in default_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path is None or not os.path.exists(config_path):
        logger.info("No config file found, using default configuration")
        return LoadBalancerConfig()
    
    logger.info(f"Loading configuration from: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        if config_data is None:
            return LoadBalancerConfig()
        
        # Handle LiteLLM-style config format
        config_data = _normalize_config(config_data)
        
        # Set environment variables if specified
        env_vars = config_data.get('environment_variables', {})
        for key, value in env_vars.items():
            if value is not None:
                os.environ[key] = str(value)
        
        return LoadBalancerConfig(**config_data)
        
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML config: {e}")
        raise ValueError(f"Invalid YAML configuration: {e}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


def _normalize_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize configuration from various formats to our standard format.
    Supports LiteLLM-style configuration.
    """
    normalized = {}
    
    # Handle model_list
    if 'model_list' in config_data:
        models = []
        for model in config_data['model_list']:
            normalized_model = _normalize_model_config(model)
            models.append(normalized_model)
        normalized['model_list'] = models
    
    # Handle router_settings / litellm_settings
    router_settings = config_data.get('router_settings', {})
    litellm_settings = config_data.get('litellm_settings', {})
    
    # Merge settings with router_settings taking precedence
    merged_settings = {**litellm_settings, **router_settings}
    if merged_settings:
        normalized['router_settings'] = merged_settings
    
    # Handle environment_variables
    if 'environment_variables' in config_data:
        normalized['environment_variables'] = config_data['environment_variables']
    
    # Handle general_settings
    if 'general_settings' in config_data:
        normalized['general_settings'] = config_data['general_settings']
    
    return normalized


def _normalize_model_config(model: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a single model configuration."""
    normalized = {}
    
    # Handle model_name
    normalized['model_name'] = model.get('model_name') or model.get('model') or 'default'
    
    # Handle litellm_params
    litellm_params = model.get('litellm_params', {})
    normalized['litellm_params'] = litellm_params
    
    # Extract api_base from litellm_params or model config
    normalized['api_base'] = (
        model.get('api_base') or 
        litellm_params.get('api_base') or
        litellm_params.get('base_url')
    )
    
    # Extract api_key
    normalized['api_key'] = (
        model.get('api_key') or
        litellm_params.get('api_key')
    )
    
    # Copy other fields
    optional_fields = [
        'weight', 'priority', 'rpm', 'tpm', 'max_parallel_requests',
        'health_check_enabled', 'health_check_interval', 'timeout',
        'max_length', 'batch_size'
    ]
    
    for field in optional_fields:
        if field in model:
            normalized[field] = model[field]
    
    return normalized
