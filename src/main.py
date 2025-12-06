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
    """Middleware to bind request context for structured logging and request tracing."""
    
    async def dispatch(self, request: Request, call_next):
        import time
        
        # Generate request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start_time = time.time()
        
        # Bind request context for logging
        bind_request_context(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else None,
        )
        
        # Log incoming request details
        logger.debug(
            "http_request_start",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            query_string=str(request.query_params) if request.query_params else None,
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            content_length=request.headers.get("content-length"),
            content_type=request.headers.get("content-type"),
        )
        
        try:
            response = await call_next(request)
            
            # Calculate response time
            elapsed = time.time() - start_time
            elapsed_ms = round(elapsed * 1000, 2)
            
            # Log response details
            logger.debug(
                "http_request_complete",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                elapsed_ms=elapsed_ms,
                response_content_type=response.headers.get("content-type"),
            )
            
            # Add timing headers to response
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time-Ms"] = str(elapsed_ms)
            
            return response
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.debug(
                "http_request_error",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                error=str(e),
                error_type=type(e).__name__,
                elapsed_ms=round(elapsed * 1000, 2),
            )
            raise
        finally:
            clear_request_context()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Loads the model on startup and cleans up on shutdown.
    """
    import time
    startup_start = time.time()
    
    # Startup
    logger.info(
        "starting_reranker_service",
        model=settings.model_name,
        device=settings.get_device(),
        load_balancer_enabled=settings.enable_load_balancer,
        version=__version__,
    )
    
    logger.debug(
        "startup_config_details",
        async_engine_enabled=settings.enable_async_engine,
        max_concurrent_batches=settings.max_concurrent_batches,
        inference_threads=settings.inference_threads,
        max_batch_size=settings.max_batch_size,
        max_batch_pairs=settings.max_batch_pairs,
        batch_wait_timeout=settings.batch_wait_timeout,
        max_queue_size=settings.max_queue_size,
        request_timeout=settings.request_timeout,
        log_level=settings.log_level,
    )
    
    # Initialize load balancer if enabled
    lb_router = None
    if settings.enable_load_balancer:
        logger.debug("load_balancer_init_start", config_path=settings.config_path)
        try:
            from src.load_balancer import load_config, initialize_router
            config = load_config(settings.config_path)
            lb_router = await initialize_router(config)
            logger.info(
                "load_balancer_initialized",
                backend_count=len(config.model_list),
                routing_strategy=config.router_settings.routing_strategy,
            )
            logger.debug(
                "load_balancer_config_details",
                backends=[b.model_name for b in config.model_list],
                health_check_interval=config.router_settings.health_check_interval,
            )
        except Exception as e:
            logger.error("load_balancer_init_failed", error=str(e))
            logger.debug("load_balancer_init_failed_details", error_type=type(e).__name__)
            logger.info("falling_back_to_local_model")
    else:
        logger.debug("load_balancer_disabled")
    
    # Pre-load the model (async engine or sync model)
    try:
        if settings.enable_async_engine:
            logger.debug("async_engine_loading_start")
            from src.engine import get_async_engine
            engine = await get_async_engine()
            logger.info(
                "async_engine_started",
                model=settings.model_name,
                device=settings.get_device(),
                max_concurrent_batches=settings.max_concurrent_batches,
            )
            logger.debug(
                "async_engine_ready",
                engine_running=engine.is_running,
            )
        else:
            logger.debug("sync_model_loading_start")
            from src.models import get_reranker_model
            get_reranker_model()
            logger.info("sync_model_loaded_successfully")
            logger.debug("sync_model_ready")
    except Exception as e:
        logger.error("model_load_failed", error=str(e))
        logger.debug("model_load_failed_details", error_type=type(e).__name__)
        # Don't fail startup, model will be loaded on first request
    
    startup_elapsed = time.time() - startup_start
    logger.debug(
        "startup_complete",
        elapsed_seconds=round(startup_elapsed, 2),
    )
    
    yield
    
    # Shutdown
    import time
    shutdown_start = time.time()
    logger.info("shutting_down_reranker_service")
    logger.debug("shutdown_start")
    
    # Close load balancer
    if lb_router:
        try:
            logger.debug("load_balancer_closing_start")
            from src.load_balancer import close_router
            await close_router()
            logger.info("load_balancer_closed")
            logger.debug("load_balancer_closed_success")
        except Exception as e:
            logger.error("load_balancer_close_error", error=str(e))
            logger.debug("load_balancer_close_error_details", error_type=type(e).__name__)
    
    # Stop async engine or unload sync model
    try:
        if settings.enable_async_engine:
            logger.debug("async_engine_stopping_start")
            from src.engine import reset_async_engine
            await reset_async_engine()
            logger.info("async_engine_stopped")
            logger.debug("async_engine_stopped_success")
        else:
            logger.debug("sync_model_unloading_start")
            from src.models.reranker import reset_reranker_model
            reset_reranker_model()
            logger.info("sync_model_unloaded")
            logger.debug("sync_model_unloaded_success")
    except Exception as e:
        logger.error("shutdown_error", error=str(e))
        logger.debug("shutdown_error_details", error_type=type(e).__name__)
    
    shutdown_elapsed = time.time() - shutdown_start
    logger.debug(
        "shutdown_complete",
        elapsed_seconds=round(shutdown_elapsed, 2),
    )


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    from src.api import router, health_router
    
    logger.debug("create_app_start")
    
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
        logger.debug(
            "cors_enabled",
            origins=settings.get_cors_origins_list(),
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.get_cors_origins_list(),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        logger.debug("cors_disabled")
    
    # Add request context middleware for structured logging
    logger.debug("adding_request_context_middleware")
    app.add_middleware(RequestContextMiddleware)
    
    # Include routers
    logger.debug("including_routers")
    app.include_router(health_router)
    app.include_router(router)
    
    # Include load balancer routes if enabled
    if settings.enable_load_balancer:
        from src.api.lb_routes import router as lb_router
        logger.debug("including_load_balancer_routes")
        app.include_router(lb_router)
    
    logger.debug("create_app_complete")
    
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
    
    logger.debug(
        "uvicorn_config_details",
        reload=settings.reload,
        log_level=settings.log_level.lower(),
        access_log_enabled=settings.log_level.upper() == "DEBUG",
        timeout_keep_alive=30,
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
