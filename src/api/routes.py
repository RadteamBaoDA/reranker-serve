"""
API routes for reranker service.
Compatible with Jina AI and Cohere API formats.
"""

import uuid
from typing import List, Union

from fastapi import APIRouter, HTTPException, Depends, Header

from src.config import settings, get_logger
from src.models import get_reranker_model, RerankerModel
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

logger = get_logger(__name__)

router = APIRouter()


def verify_api_key(authorization: str = Header(default=None)) -> bool:
    """Verify API key if configured."""
    if not settings.api_key:
        return True
    
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    # Support both "Bearer <key>" and just "<key>"
    token = authorization
    if authorization.startswith("Bearer "):
        token = authorization[7:]
    
    if token != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True


def extract_document_texts(documents: Union[List[str], List[dict]]) -> List[str]:
    """Extract text from documents (handles both string and dict formats)."""
    texts = []
    for doc in documents:
        if isinstance(doc, str):
            texts.append(doc)
        elif isinstance(doc, dict):
            # Support both 'text' and 'content' keys
            text = doc.get("text") or doc.get("content") or ""
            texts.append(text)
        else:
            texts.append(str(doc))
    return texts


@router.post("/rerank", response_model=RerankResponse, tags=["Rerank"])
async def rerank(
    request: RerankRequest,
    _: bool = Depends(verify_api_key),
) -> RerankResponse:
    """
    Rerank documents based on relevance to a query.
    
    This is the native API format for the reranker service.
    """
    try:
        model = get_reranker_model()
        
        results = model.rerank(
            query=request.query,
            documents=request.documents,
            top_k=request.top_n,
            return_documents=request.return_documents,
        )
        
        # Convert to response format
        rerank_results = [
            RerankResult(
                index=r["index"],
                relevance_score=r["relevance_score"],
                document=Document(text=r["document"]["text"]) if r.get("document") else None,
            )
            for r in results
        ]
        
        return RerankResponse(
            results=rerank_results,
            model=settings.model_name,
        )
        
    except Exception as e:
        logger.error(f"Rerank error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/rerank", response_model=CohereRerankResponse, tags=["Cohere Compatible"])
@router.post("/rerank/v1", response_model=CohereRerankResponse, tags=["Cohere Compatible"])
async def cohere_rerank(
    request: CohereRerankRequest,
    _: bool = Depends(verify_api_key),
) -> CohereRerankResponse:
    """
    Cohere-compatible rerank endpoint.
    
    Compatible with Cohere's /v1/rerank API format.
    """
    try:
        model = get_reranker_model()
        
        # Extract document texts
        documents = extract_document_texts(request.documents)
        
        results = model.rerank(
            query=request.query,
            documents=documents,
            top_k=request.top_n,
            return_documents=request.return_documents,
        )
        
        # Convert to Cohere response format
        cohere_results = [
            CohereRerankResult(
                index=r["index"],
                relevance_score=r["relevance_score"],
                document={"text": r["document"]["text"]} if r.get("document") else None,
            )
            for r in results
        ]
        
        return CohereRerankResponse(
            id=str(uuid.uuid4()),
            results=cohere_results,
            meta={
                "api_version": {"version": "1"},
                "billed_units": {"search_units": 1}
            }
        )
        
    except Exception as e:
        logger.error(f"Cohere rerank error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/rerank", response_model=JinaRerankResponse, tags=["Jina AI Compatible"])
async def jina_rerank(
    request: JinaRerankRequest,
    _: bool = Depends(verify_api_key),
) -> JinaRerankResponse:
    """
    Jina AI-compatible rerank endpoint.
    
    Compatible with Jina AI's reranker API format.
    """
    try:
        model = get_reranker_model()
        
        # Extract document texts
        documents = extract_document_texts(request.documents)
        
        results = model.rerank(
            query=request.query,
            documents=documents,
            top_k=request.top_n,
            return_documents=request.return_documents,
        )
        
        # Convert to Jina response format
        jina_results = [
            JinaRerankResult(
                index=r["index"],
                relevance_score=r["relevance_score"],
                document=Document(text=r["document"]["text"]) if r.get("document") else None,
            )
            for r in results
        ]
        
        # Estimate token usage (rough approximation)
        total_chars = len(request.query) + sum(len(d) for d in documents)
        estimated_tokens = total_chars // 4  # Rough estimate
        
        return JinaRerankResponse(
            model=settings.model_name,
            usage=JinaUsage(
                total_tokens=estimated_tokens,
                prompt_tokens=estimated_tokens,
            ),
            results=jina_results,
        )
        
    except Exception as e:
        logger.error(f"Jina rerank error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
