"""
Main FastAPI application for the Reranker Service.
"""

import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from src import __version__
from src.config import settings, configure_logging, get_logger, bind_request_context, clear_request_context


# Configure structured logging
configure_logging(
    log_level=settings.log_level,
    json_logs=os.environ.get("RERANKER_JSON_LOGS", "false").lower() == "true",
)
logger = get_logger(__name__)


# Configure proxy bypass for internal requests
# This ensures httpx and other HTTP clients don't use proxy for local/internal requests
def configure_proxy_bypass() -> None:
    """
    Configure environment to bypass proxy for internal/local requests.
    This is important for load balancer communication between services.
    """
    # Common internal hosts that should bypass proxy
    no_proxy_hosts = [
        "localhost",
        "127.0.0.1",
        "::1",
        "0.0.0.0",
        ".local",
        ".internal",
    ]
    
    # Get existing NO_PROXY value
    existing_no_proxy = os.environ.get("NO_PROXY", os.environ.get("no_proxy", ""))
    
    # Combine with our internal hosts
    all_no_proxy = set(existing_no_proxy.split(",")) if existing_no_proxy else set()
    all_no_proxy.update(no_proxy_hosts)
    all_no_proxy.discard("")  # Remove empty strings
    
    # Set both uppercase and lowercase versions
    no_proxy_str = ",".join(sorted(all_no_proxy))
    os.environ["NO_PROXY"] = no_proxy_str
    os.environ["no_proxy"] = no_proxy_str
    
    logger.debug("proxy_bypass_configured", no_proxy=no_proxy_str)


# Configure proxy bypass on module load
configure_proxy_bypass()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware to bind request context for structured logging."""
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        
        # Bind request context for logging
        bind_request_context(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else None,
        )
        
        try:
            response = await call_next(request)
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            clear_request_context()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Loads the model on startup and cleans up on shutdown.
    """
    # Startup
    logger.info(
        "starting_reranker_service",
        model=settings.model_name,
        device=settings.get_device(),
        load_balancer_enabled=settings.enable_load_balancer,
        version=__version__,
    )
    
    # Initialize load balancer if enabled
    lb_router = None
    if settings.enable_load_balancer:
        try:
            from src.load_balancer import load_config, initialize_router
            config = load_config(settings.config_path)
            lb_router = await initialize_router(config)
            logger.info(
                "load_balancer_initialized",
                backend_count=len(config.model_list),
                routing_strategy=config.router_settings.routing_strategy,
            )
        except Exception as e:
            logger.error("load_balancer_init_failed", error=str(e))
            logger.info("falling_back_to_local_model")
    
    # Pre-load the model (async engine or sync model)
    try:
        if settings.enable_async_engine:
            from src.engine import get_async_engine
            engine = await get_async_engine()
            logger.info(
                "async_engine_started",
                model=settings.model_name,
                device=settings.get_device(),
                max_concurrent_batches=settings.max_concurrent_batches,
            )
        else:
            from src.models import get_reranker_model
            get_reranker_model()
            logger.info("sync_model_loaded_successfully")
    except Exception as e:
        logger.error("model_load_failed", error=str(e))
        # Don't fail startup, model will be loaded on first request
    
    yield
    
    # Shutdown
    logger.info("shutting_down_reranker_service")
    
    # Close load balancer
    if lb_router:
        try:
            from src.load_balancer import close_router
            await close_router()
            logger.info("load_balancer_closed")
        except Exception as e:
            logger.error("load_balancer_close_error", error=str(e))
    
    # Stop async engine or unload sync model
    try:
        if settings.enable_async_engine:
            from src.engine import reset_async_engine
            await reset_async_engine()
            logger.info("async_engine_stopped")
        else:
            from src.models.reranker import reset_reranker_model
            reset_reranker_model()
            logger.info("sync_model_unloaded")
    except Exception as e:
        logger.error("shutdown_error", error=str(e))


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    from src.api import router, health_router
    
    app = FastAPI(
        title="Reranker Service",
        description="""
        A high-performance reranker service using Sentence Transformer models.
        
        ## Architecture (vLLM-inspired)
        - **Async Engine**: Concurrent request handling with automatic batching
        - **Request Queue**: Priority-based scheduling with configurable batch size
        - **Thread Pool**: Non-blocking model inference for high throughput
        
        ## Supported Models
        - **BAAI/bge-reranker-v2-m3** - Multilingual reranker
        - **BAAI/bge-reranker-large** - English reranker
        - **Qwen3-reranker** - Qwen-based reranker models
        
        ## API Compatibility
        - Native API at `/rerank`
        - Cohere-compatible API at `/v1/rerank`
        - Jina AI-compatible API at `/api/v1/rerank`
        
        ## Load Balancer
        - Load-balanced API at `/lb/rerank`, `/lb/v1/rerank`, `/lb/api/v1/rerank`
        - Supports multiple backends with LiteLLM-style YAML configuration
        - Routing strategies: weighted-random, round-robin, least-busy, latency-based, priority-failover
        
        ## Features
        - **Concurrent Request Handling**: Handle multiple requests simultaneously
        - **Automatic Batching**: Requests are batched for optimal GPU utilization
        - Offline model loading support
        - Apple Silicon (MPS) optimization
        - CUDA acceleration
        - Automatic device detection
        - Load balancing across multiple backends
        
        ## Monitoring
        - `/stats` - Engine statistics and request queue metrics
        - `/ready` - Kubernetes readiness probe
        - `/live` - Kubernetes liveness probe
        """,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    
    # Configure CORS
    if settings.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.get_cors_origins_list(),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Add request context middleware for structured logging
    app.add_middleware(RequestContextMiddleware)
    
    # Include routers
    app.include_router(health_router)
    app.include_router(router)
    
    # Include load balancer routes if enabled
    if settings.enable_load_balancer:
        from src.api.lb_routes import router as lb_router
        app.include_router(lb_router)
    
    return app


# Create the application instance
app = create_app()


def run_server():
    """
    Run the server using uvicorn.
    
    For production with multiple workers, the async engine is initialized
    per-worker with proper isolation. Each worker has its own:
    - Model instance
    - Request queue
    - Thread pool for inference
    
    For maximum performance:
    - Use workers=1 with async engine (handles concurrency internally)
    - Or use workers>1 for multi-process scaling (each has own GPU memory)
    """
    import uvicorn
    
    # Log configuration
    logger.info(
        "starting_uvicorn_server",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        async_engine=settings.enable_async_engine,
        max_concurrent_batches=settings.max_concurrent_batches,
    )
    
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
        # Enable for production
        access_log=settings.log_level.upper() == "DEBUG",
        # Timeout settings
        timeout_keep_alive=30,
    )


if __name__ == "__main__":
    run_server()
