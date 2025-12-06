"""
Health check endpoints for the reranker service.
"""

from typing import Any, Dict, Optional
from fastapi import APIRouter
from pydantic import BaseModel

from src.config import settings


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model: str
    device: str
    version: str
    engine_mode: str


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    model_path: str | None
    device: str
    max_length: int
    batch_size: int
    use_fp16: bool
    offline_mode: bool
    # Async engine settings
    async_engine_enabled: bool
    max_concurrent_batches: int
    inference_threads: int
    max_queue_size: int


class EngineStatsResponse(BaseModel):
    """Engine statistics response."""
    engine_mode: str
    running: bool
    model_loaded: bool
    stats: Dict[str, Any]


health_router = APIRouter(tags=["Health"])


@health_router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Check the health status of the service.
    """
    from src import __version__
    
    return HealthResponse(
        status="healthy",
        model=settings.model_name,
        device=settings.get_device(),
        version=__version__,
        engine_mode="async" if settings.enable_async_engine else "sync",
    )


@health_router.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    """
    Root endpoint - returns health status.
    """
    return await health_check()


@health_router.get("/info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """
    Get detailed model information.
    """
    return ModelInfoResponse(
        model_name=settings.model_name,
        model_path=settings.model_path,
        device=settings.get_device(),
        max_length=settings.max_length,
        batch_size=settings.batch_size,
        use_fp16=settings.use_fp16,
        offline_mode=settings.use_offline_mode,
        async_engine_enabled=settings.enable_async_engine,
        max_concurrent_batches=settings.max_concurrent_batches,
        inference_threads=settings.inference_threads,
        max_queue_size=settings.max_queue_size,
    )


@health_router.get("/stats", response_model=EngineStatsResponse)
async def engine_stats() -> EngineStatsResponse:
    """
    Get engine statistics including request queue metrics.
    """
    try:
        if settings.enable_async_engine:
            from src.engine import get_async_engine
            engine = await get_async_engine()
            return EngineStatsResponse(
                engine_mode="async",
                running=engine.is_running,
                model_loaded=engine.is_loaded,
                stats=engine.get_stats(),
            )
        else:
            from src.models import get_reranker_model
            model = get_reranker_model()
            return EngineStatsResponse(
                engine_mode="sync",
                running=True,
                model_loaded=model.is_loaded,
                stats={
                    "model": settings.model_name,
                    "device": settings.get_device(),
                },
            )
    except Exception as e:
        return EngineStatsResponse(
            engine_mode="async" if settings.enable_async_engine else "sync",
            running=False,
            model_loaded=False,
            stats={"error": str(e)},
        )


@health_router.get("/ready")
async def readiness_check():
    """
    Kubernetes-style readiness probe.
    Checks if the model is loaded and ready to serve requests.
    """
    try:
        if settings.enable_async_engine:
            from src.engine import get_async_engine
            engine = await get_async_engine()
            if engine.is_running and engine.is_loaded:
                return {"status": "ready", "model_loaded": True, "engine_mode": "async"}
            else:
                return {"status": "not_ready", "model_loaded": engine.is_loaded, "engine_mode": "async"}
        else:
            from src.models import get_reranker_model
            model = get_reranker_model()
            if model.is_loaded:
                return {"status": "ready", "model_loaded": True, "engine_mode": "sync"}
            else:
                return {"status": "not_ready", "model_loaded": False, "engine_mode": "sync"}
    except Exception as e:
        return {"status": "not_ready", "error": str(e)}


@health_router.get("/live")
async def liveness_check():
    """
    Kubernetes-style liveness probe.
    Simply checks if the service is running.
    """
    return {"status": "alive"}
