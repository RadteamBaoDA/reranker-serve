"""
Load Balancer module for reranker API.
Supports LiteLLM-style YAML configuration for routing to multiple backends.
"""

from .config import LoadBalancerConfig, ModelConfig, load_config
from .router import (
    LoadBalancerRouter,
    get_router,
    initialize_router,
    close_router,
)

__all__ = [
    "LoadBalancerConfig",
    "ModelConfig", 
    "load_config",
    "LoadBalancerRouter",
    "get_router",
    "initialize_router",
    "close_router",
]
