"""Lightweight coverage for load balancer config and router helpers."""

import pytest

from src.load_balancer.config import LoadBalancerConfig, ModelConfig, RouterSettings, load_config
from src.load_balancer.router import BackendStats, LoadBalancerRouter


def test_model_config_validation_and_defaults():
    cfg = ModelConfig(model_name="m1", api_base="http://example.com/")
    assert cfg.api_base == "http://example.com"
    assert cfg.weight == 1.0
    assert cfg.priority == 0

    with pytest.raises(ValueError):
        ModelConfig(model_name="bad", api_base="ftp://nope")


def test_load_config_handles_empty_yaml(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("{}", encoding="utf-8")

    cfg = load_config(str(cfg_path))
    assert isinstance(cfg, LoadBalancerConfig)
    assert cfg.model_list == []


def test_router_filters_unhealthy_backends():
    cfg = LoadBalancerConfig(
        model_list=[
            ModelConfig(model_name="healthy", api_base="http://h"),
            ModelConfig(model_name="sick", api_base="http://s"),
        ]
    )
    router = LoadBalancerRouter(cfg)
    router._stats["sick"].is_healthy = False

    available = router.get_available_backends()
    assert len(available) == 1
    assert available[0].model_name == "healthy"


@pytest.mark.asyncio
async def test_priority_strategy_prefers_lower_value():
    cfg = LoadBalancerConfig(
        model_list=[
            ModelConfig(model_name="primary", api_base="http://a", priority=0),
            ModelConfig(model_name="secondary", api_base="http://b", priority=5),
        ],
        router_settings=RouterSettings(routing_strategy="priority-failover"),
    )
    router = LoadBalancerRouter(cfg)
    selected = await router.select_backend()
    assert selected.model_name == "primary"


@pytest.mark.asyncio
async def test_round_robin_cycles_through_backends():
    cfg = LoadBalancerConfig(
        model_list=[
            ModelConfig(model_name="a", api_base="http://a"),
            ModelConfig(model_name="b", api_base="http://b"),
        ],
        router_settings=RouterSettings(routing_strategy="round-robin"),
    )
    router = LoadBalancerRouter(cfg)
    available = router.get_available_backends()

    first = await router._round_robin_select(available)
    second = await router._round_robin_select(available)
    assert {first.model_name, second.model_name} == {"a", "b"}


def test_backend_stats_helpers():
    stats = BackendStats()
    assert stats.average_latency == 0.0
    assert stats.success_rate == 1.0

    stats.total_latency = 2.0
    stats.successful_requests = 4
    stats.total_requests = 5
    assert stats.average_latency == 0.5
    assert stats.success_rate == pytest.approx(0.8)
