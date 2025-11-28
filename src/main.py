"""
Main FastAPI application for the Reranker Service.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src import __version__
from src.config import settings
from src.api import router, health_router


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Loads the model on startup and cleans up on shutdown.
    """
    # Startup
    logger.info("Starting Reranker Service...")
    logger.info(f"Model: {settings.model_name}")
    logger.info(f"Device: {settings.get_device()}")
    
    # Pre-load the model
    try:
        from src.models import get_reranker_model
        get_reranker_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        # Don't fail startup, model will be loaded on first request
    
    yield
    
    # Shutdown
    logger.info("Shutting down Reranker Service...")
    try:
        from src.models.reranker import reset_reranker_model
        reset_reranker_model()
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Reranker Service",
        description="""
        A high-performance reranker service using Sentence Transformer models.
        
        ## Supported Models
        - **BAAI/bge-reranker-v2-m3** - Multilingual reranker
        - **BAAI/bge-reranker-large** - English reranker
        - **Qwen3-reranker** - Qwen-based reranker models
        
        ## API Compatibility
        - Native API at `/rerank`
        - Cohere-compatible API at `/v1/rerank`
        - Jina AI-compatible API at `/api/v1/rerank`
        
        ## Features
        - Offline model loading support
        - Apple Silicon (MPS) optimization
        - CUDA acceleration
        - Automatic device detection
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
    
    # Include routers
    app.include_router(health_router)
    app.include_router(router)
    
    return app


# Create the application instance
app = create_app()


def run_server():
    """Run the server using uvicorn."""
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run_server()
