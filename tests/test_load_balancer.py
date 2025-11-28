"""Tests for load balancer module."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import tempfile
import yaml
from pathlib import Path

from src.load_balancer.config import (
    ModelConfig,
    RouterSettings,
    LoadBalancerConfig,
    load_config,
)
from src.load_balancer.router import (
    LoadBalancerRouter,
    BackendStats,
)


class TestModelConfig:
    """Test ModelConfig parsing."""
    
    def test_basic_config(self):
        """Test basic model config."""
        config = ModelConfig(
            model_name="test-model",
            api_base="http://localhost:8000",
        )
        assert config.model_name == "test-model"
        assert config.api_base == "http://localhost:8000"
        assert config.api_key is None
        assert config.weight == 1.0
        assert config.timeout == 30.0
    
    def test_full_config(self):
        """Test full model config with all options."""
        config = ModelConfig(
            model_name="production-reranker",
            api_base="http://prod.example.com:8000",
            api_key="secret-key",
            weight=2.0,
            timeout=60.0,
            priority=1,
        )
        assert config.model_name == "production-reranker"
        assert config.api_key == "secret-key"
        assert config.weight == 2.0
        assert config.timeout == 60.0
        assert config.priority == 1
    
    def test_api_base_validation(self):
        """Test api_base URL validation."""
        config = ModelConfig(
            model_name="test",
            api_base="http://example.com/",
        )
        # Trailing slash should be stripped
        assert config.api_base == "http://example.com"
        
    def test_invalid_api_base(self):
        """Test invalid api_base raises error."""
        with pytest.raises(ValueError, match="must start with http"):
            ModelConfig(
                model_name="test",
                api_base="ftp://invalid.com",
            )


class TestRouterSettings:
    """Test RouterSettings parsing."""
    
    def test_default_settings(self):
        """Test default router settings."""
        settings = RouterSettings()
        assert settings.routing_strategy == "weighted-random"
        assert settings.num_retries == 3
        assert settings.retry_delay == 1.0
        assert settings.fallback_to_local is True
        assert settings.default_timeout == 30.0
    
    def test_custom_settings(self):
        """Test custom router settings."""
        settings = RouterSettings(
            routing_strategy="round-robin",
            num_retries=5,
            retry_delay=2.0,
            fallback_to_local=False,
            default_timeout=60.0,
        )
        assert settings.routing_strategy == "round-robin"
        assert settings.num_retries == 5
        assert settings.retry_delay == 2.0
        assert settings.fallback_to_local is False
        assert settings.default_timeout == 60.0


class TestLoadBalancerConfig:
    """Test LoadBalancerConfig parsing."""
    
    def test_minimal_config(self):
        """Test minimal configuration."""
        config = LoadBalancerConfig(
            model_list=[
                ModelConfig(
                    model_name="model-1",
                    api_base="http://localhost:8001",
                )
            ]
        )
        assert len(config.model_list) == 1
        assert config.router_settings.routing_strategy == "weighted-random"
    
    def test_full_config(self):
        """Test full configuration."""
        config = LoadBalancerConfig(
            model_list=[
                ModelConfig(
                    model_name="model-1",
                    api_base="http://localhost:8001",
                    weight=2.0,
                ),
                ModelConfig(
                    model_name="model-2",
                    api_base="http://localhost:8002",
                    weight=1.0,
                ),
            ],
            router_settings=RouterSettings(
                routing_strategy="round-robin",
            ),
        )
        assert len(config.model_list) == 2
        assert config.router_settings.routing_strategy == "round-robin"
        assert config.model_list[0].weight == 2.0
    
    def test_get_model_by_name(self):
        """Test getting model by name."""
        config = LoadBalancerConfig(
            model_list=[
                ModelConfig(
                    model_name="target-model",
                    api_base="http://localhost:8001",
                ),
                ModelConfig(
                    model_name="other-model",
                    api_base="http://localhost:8002",
                ),
            ]
        )
        model = config.get_model_by_name("target-model")
        assert model is not None
        assert model.model_name == "target-model"
        
        # Non-existent model
        assert config.get_model_by_name("non-existent") is None


class TestLoadConfig:
    """Test YAML config loading."""
    
    def test_load_valid_config(self):
        """Test loading a valid YAML config."""
        config_data = {
            "model_list": [
                {
                    "model_name": "test-reranker",
                    "litellm_params": {
                        "api_base": "http://localhost:8000",
                        "api_key": "test-key",
                    },
                }
            ],
            "router_settings": {
                "routing_strategy": "round-robin",
            },
        }
        
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert config is not None
            assert len(config.model_list) == 1
            assert config.model_list[0].model_name == "test-reranker"
            assert config.model_list[0].api_base == "http://localhost:8000"
            assert config.model_list[0].api_key == "test-key"
            assert config.router_settings.routing_strategy == "round-robin"
        finally:
            Path(temp_path).unlink()
    
    def test_load_empty_config(self):
        """Test loading empty config returns defaults."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump({}, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert config is not None
            assert len(config.model_list) == 0
        finally:
            Path(temp_path).unlink()
    
    def test_load_invalid_yaml(self):
        """Test loading invalid YAML raises error."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Invalid YAML"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()


class TestBackendStats:
    """Test BackendStats dataclass."""
    
    def test_default_stats(self):
        """Test default stats values."""
        stats = BackendStats()
        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.failed_requests == 0
        assert stats.is_healthy is True
        assert stats.current_requests == 0
    
    def test_average_latency(self):
        """Test average latency calculation."""
        stats = BackendStats()
        assert stats.average_latency == 0.0
        
        stats.successful_requests = 10
        stats.total_latency = 5.0
        assert stats.average_latency == 0.5
    
    def test_success_rate(self):
        """Test success rate calculation."""
        stats = BackendStats()
        assert stats.success_rate == 1.0  # No requests = 100%
        
        stats.total_requests = 100
        stats.successful_requests = 80
        assert stats.success_rate == 0.8


class TestLoadBalancerRouter:
    """Test LoadBalancerRouter."""
    
    def test_router_initialization(self):
        """Test router initializes correctly."""
        config = LoadBalancerConfig(
            model_list=[
                ModelConfig(model_name="model-1", api_base="http://host1:8000"),
            ],
            router_settings=RouterSettings(routing_strategy="round-robin"),
        )
        
        router = LoadBalancerRouter(config)
        assert router.config == config
        assert len(router._stats) == 0  # Stats created on demand
    
    def test_get_available_backends(self):
        """Test getting available backends."""
        config = LoadBalancerConfig(
            model_list=[
                ModelConfig(model_name="model-1", api_base="http://host1:8000"),
                ModelConfig(model_name="model-2", api_base="http://host2:8000"),
            ],
        )
        
        router = LoadBalancerRouter(config)
        available = router.get_available_backends()
        assert len(available) == 2
    
    def test_unhealthy_backend_filtered(self):
        """Test unhealthy backends are filtered out."""
        config = LoadBalancerConfig(
            model_list=[
                ModelConfig(model_name="healthy", api_base="http://host1:8000"),
                ModelConfig(model_name="unhealthy", api_base="http://host2:8000"),
            ],
        )
        
        router = LoadBalancerRouter(config)
        # Mark one as unhealthy
        router._stats["unhealthy"].is_healthy = False
        
        available = router.get_available_backends()
        assert len(available) == 1
        assert available[0].model_name == "healthy"
    
    def test_weighted_random_select(self):
        """Test weighted random selection."""
        config = LoadBalancerConfig(
            model_list=[
                ModelConfig(model_name="high-weight", api_base="http://host1:8000", weight=100.0),
                ModelConfig(model_name="zero-weight", api_base="http://host2:8000", weight=0.0),
            ],
            router_settings=RouterSettings(routing_strategy="weighted-random"),
        )
        
        router = LoadBalancerRouter(config)
        backends = router.get_available_backends()
        
        # With weight 0, should never select zero-weight
        for _ in range(10):
            selected = router._weighted_random_select(backends)
            assert selected.model_name == "high-weight"
    
    def test_least_busy_select(self):
        """Test least busy selection."""
        config = LoadBalancerConfig(
            model_list=[
                ModelConfig(model_name="busy", api_base="http://host1:8000"),
                ModelConfig(model_name="idle", api_base="http://host2:8000"),
            ],
            router_settings=RouterSettings(routing_strategy="least-busy"),
        )
        
        router = LoadBalancerRouter(config)
        # Add some active requests to "busy"
        router._stats["busy"].current_requests = 10
        router._stats["idle"].current_requests = 0
        
        backends = router.get_available_backends()
        selected = router._least_busy_select(backends)
        assert selected.model_name == "idle"
    
    def test_priority_select(self):
        """Test priority-based selection."""
        config = LoadBalancerConfig(
            model_list=[
                ModelConfig(model_name="low-priority", api_base="http://host1:8000", priority=10),
                ModelConfig(model_name="high-priority", api_base="http://host2:8000", priority=1),
            ],
            router_settings=RouterSettings(routing_strategy="priority-failover"),
        )
        
        router = LoadBalancerRouter(config)
        backends = router.get_available_backends()
        selected = router._priority_select(backends)
        assert selected.model_name == "high-priority"
    
    def test_latency_based_select(self):
        """Test latency-based selection."""
        config = LoadBalancerConfig(
            model_list=[
                ModelConfig(model_name="slow", api_base="http://host1:8000"),
                ModelConfig(model_name="fast", api_base="http://host2:8000"),
            ],
            router_settings=RouterSettings(routing_strategy="latency-based-routing"),
        )
        
        router = LoadBalancerRouter(config)
        # Record latencies
        router._stats["slow"].total_latency = 10.0
        router._stats["slow"].successful_requests = 10
        router._stats["fast"].total_latency = 1.0
        router._stats["fast"].successful_requests = 10
        
        backends = router.get_available_backends()
        # Note: latency_based_select has randomness, but fast should generally be preferred
        selected = router._latency_based_select(backends)
        assert selected is not None
    
    def test_get_stats(self):
        """Test getting router statistics."""
        config = LoadBalancerConfig(
            model_list=[
                ModelConfig(model_name="model-1", api_base="http://host1:8000"),
            ],
        )
        
        router = LoadBalancerRouter(config)
        # Simulate some requests
        router._stats["model-1"].total_requests = 100
        router._stats["model-1"].successful_requests = 95
        router._stats["model-1"].failed_requests = 5
        router._stats["model-1"].total_latency = 10.0
        
        stats = router.get_stats()
        assert "model-1" in stats
        assert stats["model-1"]["total_requests"] == 100
        assert stats["model-1"]["successful_requests"] == 95
        assert stats["model-1"]["success_rate"] == 0.95
    
    def test_all_unhealthy_returns_empty(self):
        """Test returns empty list when all backends are unhealthy."""
        config = LoadBalancerConfig(
            model_list=[
                ModelConfig(model_name="model-1", api_base="http://host1:8000"),
                ModelConfig(model_name="model-2", api_base="http://host2:8000"),
            ],
        )
        
        router = LoadBalancerRouter(config)
        router._stats["model-1"].is_healthy = False
        router._stats["model-2"].is_healthy = False
        
        available = router.get_available_backends()
        assert len(available) == 0
    
    @pytest.mark.asyncio
    async def test_round_robin_select(self):
        """Test round-robin selection."""
        config = LoadBalancerConfig(
            model_list=[
                ModelConfig(model_name="model-1", api_base="http://host1:8000"),
                ModelConfig(model_name="model-2", api_base="http://host2:8000"),
                ModelConfig(model_name="model-3", api_base="http://host3:8000"),
            ],
            router_settings=RouterSettings(routing_strategy="round-robin"),
        )
        
        router = LoadBalancerRouter(config)
        backends = router.get_available_backends()
        
        # First cycle
        selected1 = await router._round_robin_select(backends)
        selected2 = await router._round_robin_select(backends)
        selected3 = await router._round_robin_select(backends)
        
        names = [selected1.model_name, selected2.model_name, selected3.model_name]
        assert "model-1" in names
        assert "model-2" in names
        assert "model-3" in names
