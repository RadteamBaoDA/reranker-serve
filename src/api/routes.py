"""
API routes for reranker service.
Compatible with Jina AI and Cohere API formats.
Optimized for async concurrent request handling.
"""

import uuid
from typing import List, Union, Optional

from fastapi import APIRouter, HTTPException, Depends, Header

from src.config import settings, get_logger
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
    HuggingFaceRerankRequest,
    HuggingFaceRerankResponse,
    HuggingFaceRerankResult,
)

logger = get_logger(__name__)

router = APIRouter()


async def get_rerank_results(
    query: str,
    documents: List[str],
    top_k: int = None,
    return_documents: bool = True,
    request_id: str = None,
) -> List[dict]:
    """
    Get rerank results using async engine or fallback to sync model.
    Provides a unified interface for both modes.
    """
    import time
    start_time = time.time()
    
    logger.debug(
        "get_rerank_results_start",
        request_id=request_id,
        query_length=len(query),
        num_documents=len(documents),
        top_k=top_k,
        return_documents=return_documents,
        async_engine_enabled=settings.enable_async_engine,
    )
    
    if settings.enable_async_engine:
        from src.engine import get_async_engine
        logger.debug(
            "get_rerank_results_using_async_engine",
            request_id=request_id,
        )
        engine = await get_async_engine()
        results = await engine.rerank(
            query=query,
            documents=documents,
            top_k=top_k,
            return_documents=return_documents,
            request_id=request_id,
        )
    else:
        # Fallback to sync model for compatibility
        from src.models import get_reranker_model
        logger.debug(
            "get_rerank_results_using_sync_model",
            request_id=request_id,
        )
        model = get_reranker_model()
        results = model.rerank(
            query=query,
            documents=documents,
            top_k=top_k,
            return_documents=return_documents,
        )
    
    elapsed = time.time() - start_time
    logger.debug(
        "get_rerank_results_complete",
        request_id=request_id,
        num_results=len(results),
        elapsed_ms=round(elapsed * 1000, 2),
        top_score=results[0]["relevance_score"] if results else None,
    )
    
    return results


def verify_api_key(authorization: str = Header(default=None)) -> bool:
    """Verify API key if configured."""
    logger.debug(
        "verify_api_key_start",
        has_authorization=authorization is not None,
        api_key_configured=bool(settings.api_key),
    )
    
    if not settings.api_key:
        logger.debug("verify_api_key_skip_no_key_configured")
        return True
    
    if not authorization:
        logger.debug("verify_api_key_missing_header")
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    # Support both "Bearer <key>" and just "<key>"
    token = authorization
    if authorization.startswith("Bearer "):
        token = authorization[7:]
    
    if token != settings.api_key:
        logger.debug("verify_api_key_invalid")
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    logger.debug("verify_api_key_success")
    return True


def extract_document_text(doc: Union[str, dict, None]) -> Optional[str]:
    """Safely extract text from a document object."""
    if doc is None:
        return None
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        return doc.get("text") or doc.get("content") or ""
    return str(doc)


def extract_document_texts(documents: Union[List[str], List[dict]]) -> List[str]:
    """Extract text from documents (handles both string and dict formats)."""
    logger.debug(
        "extract_document_texts_start",
        num_documents=len(documents),
        first_doc_type=type(documents[0]).__name__ if documents else None,
    )
    
    texts = []
    string_count = 0
    dict_count = 0
    other_count = 0
    
    for doc in documents:
        if isinstance(doc, str):
            texts.append(doc)
            string_count += 1
        elif isinstance(doc, dict):
            # Support both 'text' and 'content' keys
            text = doc.get("text") or doc.get("content") or ""
            texts.append(text)
            dict_count += 1
        else:
            texts.append(str(doc))
            other_count += 1
    
    logger.debug(
        "extract_document_texts_complete",
        total_extracted=len(texts),
        string_docs=string_count,
        dict_docs=dict_count,
        other_docs=other_count,
        avg_text_length=sum(len(t) for t in texts) / len(texts) if texts else 0,
    )
    
    return texts


@router.post("/rerank", response_model=RerankResponse, tags=["Rerank"])
async def rerank(
    request: RerankRequest,
    _: bool = Depends(verify_api_key),
) -> RerankResponse:
    """
    Rerank documents based on relevance to a query.
    
    Supports both native format (documents field) and HuggingFace format (texts field).
    Supports concurrent request handling with automatic batching.
    
    Native format:
        {"query": "...", "documents": [...], "top_n": 5}
    
    HuggingFace format:
        {"query": "...", "texts": [...], "top_k": 5}
    """
    import time
    endpoint_start = time.time()
    request_id = str(uuid.uuid4())
    
    # Get documents list (works for both formats)
    documents = request.get_documents()
    format_type = "native" if request.documents is not None else "huggingface"
    
    logger.debug(
        "rerank_endpoint_start",
        request_id=request_id,
        endpoint="/rerank",
        format=format_type,
        query_length=len(request.query),
        num_documents=len(documents),
        top_n=request.top_n,
        return_documents=request.return_documents,
    )
    
    try:
        results = await get_rerank_results(
            query=request.query,
            documents=documents,
            top_k=request.top_n,
            return_documents=request.return_documents,
            request_id=request_id,
        )
        
        # Convert to response format
        rerank_results = [
            RerankResult(
                index=r["index"],
                relevance_score=r["relevance_score"],
                document=Document(text=extract_document_text(r.get("document"))) if r.get("document") else None,
            )
            for r in results
        ]
        
        response = RerankResponse(
            results=rerank_results,
            model=settings.model_name,
        )
        
        elapsed = time.time() - endpoint_start
        logger.debug(
            "rerank_endpoint_success",
            request_id=request_id,
            num_results=len(rerank_results),
            elapsed_ms=round(elapsed * 1000, 2),
        )
        
        return response
        
    except Exception as e:
        elapsed = time.time() - endpoint_start
        logger.debug(
            "rerank_endpoint_error",
            request_id=request_id,
            error=str(e),
            error_type=type(e).__name__,
            elapsed_ms=round(elapsed * 1000, 2),
        )
        logger.error(f"Rerank error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/rerank", response_model=CohereRerankResponse, tags=["Cohere Compatible"])
@router.post("/v2/rerank", response_model=CohereRerankResponse, tags=["Cohere Compatible v2"])
@router.post("/rerank/v1", response_model=CohereRerankResponse, tags=["Cohere Compatible"])
async def cohere_rerank(
    request: CohereRerankRequest,
    _: bool = Depends(verify_api_key),
) -> CohereRerankResponse:
    """
    Cohere-compatible rerank endpoint.
    
    Compatible with Cohere's /v1/rerank API format.
    Supports concurrent request handling with automatic batching.
    """
    import time
    endpoint_start = time.time()
    request_id = str(uuid.uuid4())
    
    logger.debug(
        "cohere_rerank_endpoint_start",
        request_id=request_id,
        endpoint="/v1/rerank or /v2/rerank",
        query_length=len(request.query),
        num_documents=len(request.documents),
        top_n=request.top_n,
        return_documents=request.return_documents,
        model=request.model,
    )
    
    try:
        # Extract document texts
        documents = extract_document_texts(request.documents)
        
        results = await get_rerank_results(
            query=request.query,
            documents=documents,
            top_k=request.top_n,
            return_documents=request.return_documents,
            request_id=request_id,
        )
        
        # Convert to Cohere response format
        cohere_results = [
            CohereRerankResult(
                index=r["index"],
                relevance_score=r["relevance_score"],
                document={"text": extract_document_text(r.get("document"))} if r.get("document") else None,
            )
            for r in results
        ]
        
        response = CohereRerankResponse(
            id=request_id,
            results=cohere_results,
            meta={
                "api_version": {"version": "1"},
                "billed_units": {"search_units": 1}
            }
        )
        
        elapsed = time.time() - endpoint_start
        logger.debug(
            "cohere_rerank_endpoint_success",
            request_id=request_id,
            num_results=len(cohere_results),
            elapsed_ms=round(elapsed * 1000, 2),
        )
        
        return response
        
    except Exception as e:
        elapsed = time.time() - endpoint_start
        logger.debug(
            "cohere_rerank_endpoint_error",
            request_id=request_id,
            error=str(e),
            error_type=type(e).__name__,
            elapsed_ms=round(elapsed * 1000, 2),
        )
        logger.error(f"Cohere rerank error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/rerank", response_model=JinaRerankResponse, tags=["Jina AI Compatible v2"])
@router.post("/api/v1/rerank", response_model=JinaRerankResponse, tags=["Jina AI Compatible"])
async def jina_rerank(
    request: JinaRerankRequest,
    _: bool = Depends(verify_api_key),
) -> JinaRerankResponse:
    """
    Jina AI-compatible rerank endpoint.
    
    Compatible with Jina AI's reranker API format.
    Supports concurrent request handling with automatic batching.
    """
    import time
    endpoint_start = time.time()
    request_id = str(uuid.uuid4())
    
    logger.debug(
        "jina_rerank_endpoint_start",
        request_id=request_id,
        endpoint="/v1/rerank or /api/v1/rerank",
        query_length=len(request.query),
        num_documents=len(request.documents),
        top_n=request.top_n,
        return_documents=request.return_documents,
        model=request.model,
    )
    
    try:
        # Extract document texts
        documents = extract_document_texts(request.documents)
        
        results = await get_rerank_results(
            query=request.query,
            documents=documents,
            top_k=request.top_n,
            return_documents=request.return_documents,
            request_id=request_id,
        )
        
        # Convert to Jina response format
        jina_results = [
            JinaRerankResult(
                index=r["index"],
                relevance_score=r["relevance_score"],
                document=Document(text=extract_document_text(r.get("document"))) if r.get("document") else None,
            )
            for r in results
        ]
        
        # Estimate token usage (rough approximation)
        total_chars = len(request.query) + sum(len(d) for d in documents)
        estimated_tokens = total_chars // 4  # Rough estimate
        
        response = JinaRerankResponse(
            model=settings.model_name,
            usage=JinaUsage(
                total_tokens=estimated_tokens,
                prompt_tokens=estimated_tokens,
            ),
            results=jina_results,
        )
        
        elapsed = time.time() - endpoint_start
        logger.debug(
            "jina_rerank_endpoint_success",
            request_id=request_id,
            num_results=len(jina_results),
            estimated_tokens=estimated_tokens,
            elapsed_ms=round(elapsed * 1000, 2),
        )
        
        return response
        
    except Exception as e:
        elapsed = time.time() - endpoint_start
        logger.debug(
            "jina_rerank_endpoint_error",
            request_id=request_id,
            error=str(e),
            error_type=type(e).__name__,
            elapsed_ms=round(elapsed * 1000, 2),
        )
        logger.error(f"Jina rerank error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/reranking", response_model=HuggingFaceRerankResponse, tags=["HuggingFace Compatible"])
@router.post("/reranking", response_model=HuggingFaceRerankResponse, tags=["HuggingFace Compatible"])
async def huggingface_rerank(
    request: HuggingFaceRerankRequest,
    _: bool = Depends(verify_api_key),
) -> HuggingFaceRerankResponse:
    """
    HuggingFace-compatible rerank endpoint.
    
    Compatible with HuggingFace's Inference API rerank format.
    Uses 'texts' field instead of 'documents' for compatibility.
    Supports concurrent request handling with automatic batching.
    """
    import time
    endpoint_start = time.time()
    request_id = str(uuid.uuid4())
    
    logger.debug(
        "huggingface_rerank_endpoint_start",
        request_id=request_id,
        endpoint="/reranking or /v1/reranking",
        query_length=len(request.query),
        num_texts=len(request.texts),
        top_k=request.top_k,
        return_texts=request.return_texts,
        model=request.model,
    )
    
    try:
        results = await get_rerank_results(
            query=request.query,
            documents=request.texts,  # Map texts to documents
            top_k=request.top_k,
            return_documents=request.return_texts,
            request_id=request_id,
        )
        
        # Convert to HuggingFace response format
        hf_results = [
            HuggingFaceRerankResult(
                index=r["index"],
                score=r["relevance_score"],
                text=extract_document_text(r.get("document")),
            )
            for r in results
        ]
        
        response = HuggingFaceRerankResponse(
            results=hf_results,
            model=settings.model_name,
        )
        
        elapsed = time.time() - endpoint_start
        logger.debug(
            "huggingface_rerank_endpoint_success",
            request_id=request_id,
            num_results=len(hf_results),
            elapsed_ms=round(elapsed * 1000, 2),
        )
        
        return response
        
    except Exception as e:
        elapsed = time.time() - endpoint_start
        logger.debug(
            "huggingface_rerank_endpoint_error",
            request_id=request_id,
            error=str(e),
            error_type=type(e).__name__,
            elapsed_ms=round(elapsed * 1000, 2),
        )
        logger.error(f"HuggingFace rerank error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
