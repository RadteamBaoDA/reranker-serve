"""
Health check endpoints for the reranker service.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from src.config import settings


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model: str
    device: str
    version: str


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    model_path: str | None
    device: str
    max_length: int
    batch_size: int
    use_fp16: bool
    offline_mode: bool


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
    )


@health_router.get("/ready")
async def readiness_check():
    """
    Kubernetes-style readiness probe.
    Checks if the model is loaded and ready to serve requests.
    """
    try:
        from src.models import get_reranker_model
        model = get_reranker_model()
        
        if model.is_loaded:
            return {"status": "ready", "model_loaded": True}
        else:
            return {"status": "not_ready", "model_loaded": False}
    except Exception as e:
        return {"status": "not_ready", "error": str(e)}


@health_router.get("/live")
async def liveness_check():
    """
    Kubernetes-style liveness probe.
    Simply checks if the service is running.
    """
    return {"status": "alive"}
