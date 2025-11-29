"""
Load Balancer Router for distributing requests across multiple reranker backends.
"""

import time
import random
import asyncio
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from contextlib import asynccontextmanager

import httpx

from src.config import get_logger
from .config import LoadBalancerConfig, ModelConfig, load_config

logger = get_logger(__name__)


@dataclass
class BackendStats:
    """Statistics for a backend."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency: float = 0.0
    last_request_time: float = 0.0
    last_failure_time: float = 0.0
    consecutive_failures: int = 0
    is_healthy: bool = True
    current_requests: int = 0
    
    @property
    def average_latency(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency / self.successful_requests
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests


class LoadBalancerRouter:
    """
    Router for load balancing requests across multiple reranker backends.
    
    Supports multiple routing strategies:
    - weighted-random: Random selection weighted by model weights
    - round-robin: Sequential rotation through backends
    - least-busy: Route to backend with fewest active requests
    - latency-based-routing: Route to backend with lowest average latency
    - priority-failover: Use highest priority backend, failover on error
    """
    
    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        """Initialize the router with configuration."""
        self.config = config or load_config()
        self._stats: Dict[str, BackendStats] = defaultdict(BackendStats)
        self._round_robin_index = 0
        self._lock = asyncio.Lock()
        self._http_client: Optional[httpx.AsyncClient] = None
        self._health_check_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize async resources."""
        # Configure httpx client with proxy bypass for internal requests
        # trust_env=False prevents httpx from using system proxy settings
        # This is important for internal load balancer communication
        self._http_client = httpx.AsyncClient(
            timeout=self.config.router_settings.default_timeout,
            follow_redirects=True,
            trust_env=False,  # Ignore proxy environment variables
        )
        
        logger.info(
            "http_client_initialized",
            timeout=self.config.router_settings.default_timeout,
            proxy_bypass=True,
        )
        
        # Start health check task
        if any(m.health_check_enabled for m in self.config.model_list):
            self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def close(self):
        """Close async resources."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._http_client:
            await self._http_client.aclose()
    
    @asynccontextmanager
    async def lifespan(self):
        """Context manager for router lifecycle."""
        await self.initialize()
        try:
            yield self
        finally:
            await self.close()
    
    def get_available_backends(self) -> List[ModelConfig]:
        """Get list of healthy, available backends."""
        available = []
        for model in self.config.model_list:
            stats = self._stats[model.model_name]
            
            # Check if healthy
            if not stats.is_healthy:
                continue
            
            # Check rate limits
            if model.rpm and stats.total_requests > 0:
                elapsed = time.time() - stats.last_request_time
                if elapsed < 60 and stats.total_requests >= model.rpm:
                    continue
            
            # Check parallel request limit
            if model.max_parallel_requests:
                if stats.current_requests >= model.max_parallel_requests:
                    continue
            
            available.append(model)
        
        return available
    
    async def select_backend(self) -> Optional[ModelConfig]:
        """Select a backend based on the routing strategy."""
        available = self.get_available_backends()
        
        if not available:
            logger.warning("no_available_backends")
            return None
        
        strategy = self.config.router_settings.routing_strategy
        
        if strategy == "weighted-random":
            return self._weighted_random_select(available)
        elif strategy == "round-robin":
            return await self._round_robin_select(available)
        elif strategy == "least-busy":
            return self._least_busy_select(available)
        elif strategy == "latency-based-routing":
            return self._latency_based_select(available)
        elif strategy == "priority-failover":
            return self._priority_select(available)
        elif strategy == "simple-shuffle":
            return random.choice(available)
        else:
            # Default to weighted random
            return self._weighted_random_select(available)
    
    def _weighted_random_select(self, backends: List[ModelConfig]) -> ModelConfig:
        """Select backend using weighted random selection."""
        weights = [b.weight for b in backends]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return random.choice(backends)
        
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for backend, weight in zip(backends, weights):
            cumulative += weight
            if r <= cumulative:
                return backend
        
        return backends[-1]
    
    async def _round_robin_select(self, backends: List[ModelConfig]) -> ModelConfig:
        """Select backend using round-robin."""
        async with self._lock:
            backend = backends[self._round_robin_index % len(backends)]
            self._round_robin_index += 1
            return backend
    
    def _least_busy_select(self, backends: List[ModelConfig]) -> ModelConfig:
        """Select backend with fewest active requests."""
        return min(
            backends,
            key=lambda b: self._stats[b.model_name].current_requests
        )
    
    def _latency_based_select(self, backends: List[ModelConfig]) -> ModelConfig:
        """Select backend with lowest average latency."""
        # Add some randomness to avoid thundering herd
        candidates = sorted(
            backends,
            key=lambda b: self._stats[b.model_name].average_latency
        )[:3]
        
        return random.choice(candidates) if candidates else backends[0]
    
    def _priority_select(self, backends: List[ModelConfig]) -> ModelConfig:
        """Select highest priority (lowest priority number) backend."""
        return min(backends, key=lambda b: b.priority)
    
    async def route_request(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        return_documents: bool = True,
        **kwargs
    ) -> Tuple[Optional[Dict[str, Any]], Optional[ModelConfig]]:
        """
        Route a rerank request to an appropriate backend.
        
        Returns:
            Tuple of (response_data, selected_backend) or (None, None) on failure
        """
        retries = self.config.router_settings.num_retries
        last_error = None
        tried_backends = set()
        
        for attempt in range(retries + 1):
            # Select a backend
            backend = await self.select_backend()
            
            if backend is None:
                break
            
            # Skip if we already tried this backend
            if backend.model_name in tried_backends and len(self.config.model_list) > 1:
                continue
            
            tried_backends.add(backend.model_name)
            
            try:
                response = await self._send_request(
                    backend=backend,
                    query=query,
                    documents=documents,
                    top_n=top_n,
                    return_documents=return_documents,
                    **kwargs
                )
                
                return response, backend
                
            except Exception as e:
                last_error = e
                logger.warning(
                    "backend_request_failed",
                    backend=backend.model_name,
                    attempt=attempt + 1,
                    max_attempts=retries + 1,
                    error=str(e),
                )
                
                # Mark backend as potentially unhealthy
                stats = self._stats[backend.model_name]
                stats.consecutive_failures += 1
                stats.last_failure_time = time.time()
                
                if stats.consecutive_failures >= 3:
                    stats.is_healthy = False
                    logger.warning("backend_marked_unhealthy", backend=backend.model_name)
                
                # Wait before retry
                if attempt < retries:
                    await asyncio.sleep(self.config.router_settings.retry_delay)
        
        # All retries failed
        if last_error:
            logger.error("all_backends_failed", last_error=str(last_error))
        
        return None, None
    
    async def _send_request(
        self,
        backend: ModelConfig,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        return_documents: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Send a rerank request to a specific backend."""
        stats = self._stats[backend.model_name]
        stats.current_requests += 1
        stats.total_requests += 1
        stats.last_request_time = time.time()
        
        start_time = time.time()
        
        try:
            # If no api_base, this is a local model request
            if not backend.api_base:
                return await self._local_rerank(
                    backend, query, documents, top_n, return_documents
                )
            
            # Build request
            url = f"{backend.api_base}/rerank"
            headers = {"Content-Type": "application/json"}
            
            if backend.api_key:
                headers["Authorization"] = f"Bearer {backend.api_key}"
            
            payload = {
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "return_documents": return_documents,
            }
            
            # Send request
            response = await self._http_client.post(
                url,
                json=payload,
                headers=headers,
                timeout=backend.timeout,
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Update stats
            latency = time.time() - start_time
            stats.successful_requests += 1
            stats.total_latency += latency
            stats.consecutive_failures = 0
            stats.is_healthy = True
            
            if self.config.router_settings.enable_request_logging:
                logger.info(
                    "backend_request_completed",
                    backend=backend.model_name,
                    latency_ms=round(latency * 1000, 2),
                )
            
            return result
            
        except httpx.HTTPStatusError as e:
            stats.failed_requests += 1
            raise RuntimeError(f"HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            stats.failed_requests += 1
            raise
        finally:
            stats.current_requests -= 1
    
    async def _local_rerank(
        self,
        backend: ModelConfig,
        query: str,
        documents: List[str],
        top_n: Optional[int],
        return_documents: bool,
    ) -> Dict[str, Any]:
        """Handle reranking with local model."""
        from src.models import get_reranker_model
        
        model = get_reranker_model()
        
        results = model.rerank(
            query=query,
            documents=documents,
            top_k=top_n,
            return_documents=return_documents,
        )
        
        return {
            "results": results,
            "model": backend.model_name,
        }
    
    async def _health_check_loop(self):
        """Background task to check backend health."""
        while True:
            try:
                interval = self.config.router_settings.health_check_interval
                await asyncio.sleep(interval)
                
                for model in self.config.model_list:
                    if not model.health_check_enabled:
                        continue
                    
                    if not model.api_base:
                        # Local model, always healthy
                        self._stats[model.model_name].is_healthy = True
                        continue
                    
                    try:
                        await self._check_backend_health(model)
                    except Exception as e:
                        logger.warning("health_check_failed", backend=model.model_name, error=str(e))
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("health_check_loop_error", error=str(e))
    
    async def _check_backend_health(self, backend: ModelConfig):
        """Check health of a single backend."""
        url = f"{backend.api_base}/health"
        headers = {}
        
        if backend.api_key:
            headers["Authorization"] = f"Bearer {backend.api_key}"
        
        try:
            response = await self._http_client.get(
                url,
                headers=headers,
                timeout=10.0,
            )
            
            is_healthy = response.status_code == 200
            stats = self._stats[backend.model_name]
            
            if is_healthy and not stats.is_healthy:
                logger.info("backend_now_healthy", backend=backend.model_name)
            
            stats.is_healthy = is_healthy
            
        except Exception:
            self._stats[backend.model_name].is_healthy = False
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all backends."""
        return {
            name: {
                "total_requests": stats.total_requests,
                "successful_requests": stats.successful_requests,
                "failed_requests": stats.failed_requests,
                "average_latency": stats.average_latency,
                "success_rate": stats.success_rate,
                "is_healthy": stats.is_healthy,
                "current_requests": stats.current_requests,
            }
            for name, stats in self._stats.items()
        }


# Global router instance
_router_instance: Optional[LoadBalancerRouter] = None


def get_router(config: Optional[LoadBalancerConfig] = None) -> LoadBalancerRouter:
    """Get or create the global router instance."""
    global _router_instance
    
    if _router_instance is None:
        _router_instance = LoadBalancerRouter(config)
    
    return _router_instance


async def initialize_router(config: Optional[LoadBalancerConfig] = None):
    """Initialize the global router."""
    global _router_instance
    _router_instance = LoadBalancerRouter(config)
    await _router_instance.initialize()
    return _router_instance


async def close_router():
    """Close the global router."""
    global _router_instance
    if _router_instance:
        await _router_instance.close()
        _router_instance = None
