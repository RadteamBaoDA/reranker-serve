"""
Load balancer API routes for distributed reranking.
"""

import uuid
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, Header

from src.config import settings
from src.schemas import (
    RerankRequest,
    RerankResponse,
    RerankResult,
    Document,
    CohereRerankRequest,
    CohereRerankResponse,
    CohereRerankResult,
    JinaRerankRequest,
    JinaRerankResponse,
    JinaRerankResult,
    JinaUsage,
)
from src.load_balancer import get_router, LoadBalancerRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/lb", tags=["Load Balancer"])


def get_lb_router() -> LoadBalancerRouter:
    """Dependency to get the load balancer router."""
    return get_router()


def verify_api_key(authorization: str = Header(default=None)) -> bool:
    """Verify API key if configured."""
    if not settings.api_key:
        return True
    
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    token = authorization
    if authorization.startswith("Bearer "):
        token = authorization[7:]
    
    if token != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True


@router.post("/rerank", response_model=RerankResponse, tags=["Load Balancer"])
async def lb_rerank(
    request: RerankRequest,
    lb_router: LoadBalancerRouter = Depends(get_lb_router),
    _: bool = Depends(verify_api_key),
) -> RerankResponse:
    """
    Load-balanced rerank endpoint.
    
    Routes requests to available backends based on configured strategy.
    """
    try:
        response, backend = await lb_router.route_request(
            query=request.query,
            documents=request.documents,
            top_n=request.top_n,
            return_documents=request.return_documents,
        )
        
        if response is None:
            raise HTTPException(
                status_code=503,
                detail="All backends are unavailable"
            )
        
        # Convert response to standard format
        results = response.get("results", [])
        rerank_results = [
            RerankResult(
                index=r.get("index", i),
                relevance_score=r.get("relevance_score", 0.0),
                document=Document(text=r["document"]["text"]) if r.get("document") else None,
            )
            for i, r in enumerate(results)
        ]
        
        return RerankResponse(
            results=rerank_results,
            model=backend.model_name if backend else settings.model_name,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Load balancer rerank error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/rerank", response_model=CohereRerankResponse, tags=["Load Balancer"])
async def lb_cohere_rerank(
    request: CohereRerankRequest,
    lb_router: LoadBalancerRouter = Depends(get_lb_router),
    _: bool = Depends(verify_api_key),
) -> CohereRerankResponse:
    """
    Load-balanced Cohere-compatible rerank endpoint.
    """
    try:
        # Extract document texts
        documents = []
        for doc in request.documents:
            if isinstance(doc, str):
                documents.append(doc)
            elif isinstance(doc, dict):
                documents.append(doc.get("text") or doc.get("content") or "")
            else:
                documents.append(str(doc))
        
        response, backend = await lb_router.route_request(
            query=request.query,
            documents=documents,
            top_n=request.top_n,
            return_documents=request.return_documents,
        )
        
        if response is None:
            raise HTTPException(
                status_code=503,
                detail="All backends are unavailable"
            )
        
        results = response.get("results", [])
        cohere_results = [
            CohereRerankResult(
                index=r.get("index", i),
                relevance_score=r.get("relevance_score", 0.0),
                document={"text": r["document"]["text"]} if r.get("document") else None,
            )
            for i, r in enumerate(results)
        ]
        
        return CohereRerankResponse(
            id=str(uuid.uuid4()),
            results=cohere_results,
            meta={
                "api_version": {"version": "1"},
                "billed_units": {"search_units": 1},
                "backend": backend.model_name if backend else "local",
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Load balancer Cohere rerank error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/rerank", response_model=JinaRerankResponse, tags=["Load Balancer"])
async def lb_jina_rerank(
    request: JinaRerankRequest,
    lb_router: LoadBalancerRouter = Depends(get_lb_router),
    _: bool = Depends(verify_api_key),
) -> JinaRerankResponse:
    """
    Load-balanced Jina AI-compatible rerank endpoint.
    """
    try:
        # Extract document texts
        documents = []
        for doc in request.documents:
            if isinstance(doc, str):
                documents.append(doc)
            elif isinstance(doc, dict):
                documents.append(doc.get("text") or doc.get("content") or "")
            else:
                documents.append(str(doc))
        
        response, backend = await lb_router.route_request(
            query=request.query,
            documents=documents,
            top_n=request.top_n,
            return_documents=request.return_documents,
        )
        
        if response is None:
            raise HTTPException(
                status_code=503,
                detail="All backends are unavailable"
            )
        
        results = response.get("results", [])
        jina_results = [
            JinaRerankResult(
                index=r.get("index", i),
                relevance_score=r.get("relevance_score", 0.0),
                document=Document(text=r["document"]["text"]) if r.get("document") else None,
            )
            for i, r in enumerate(results)
        ]
        
        # Estimate token usage
        total_chars = len(request.query) + sum(len(d) for d in documents)
        estimated_tokens = total_chars // 4
        
        return JinaRerankResponse(
            model=backend.model_name if backend else settings.model_name,
            usage=JinaUsage(
                total_tokens=estimated_tokens,
                prompt_tokens=estimated_tokens,
            ),
            results=jina_results,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Load balancer Jina rerank error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", tags=["Load Balancer"])
async def lb_stats(
    lb_router: LoadBalancerRouter = Depends(get_lb_router),
    _: bool = Depends(verify_api_key),
):
    """
    Get load balancer statistics for all backends.
    """
    return {
        "backends": lb_router.get_stats(),
        "routing_strategy": lb_router.config.router_settings.routing_strategy,
        "total_backends": len(lb_router.config.model_list),
        "available_backends": len(lb_router.get_available_backends()),
    }


@router.get("/backends", tags=["Load Balancer"])
async def lb_backends(
    lb_router: LoadBalancerRouter = Depends(get_lb_router),
    _: bool = Depends(verify_api_key),
):
    """
    Get list of configured backends.
    """
    return {
        "backends": [
            {
                "model_name": m.model_name,
                "api_base": m.api_base,
                "weight": m.weight,
                "priority": m.priority,
                "is_healthy": lb_router._stats[m.model_name].is_healthy,
                "is_local": m.api_base is None,
            }
            for m in lb_router.config.model_list
        ]
    }
