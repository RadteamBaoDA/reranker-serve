"""
API Request/Response schemas for reranker endpoints.
Compatible with Jina AI and Cohere API formats.
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field


# Common schemas

class Document(BaseModel):
    """Document object for reranking."""
    text: str = Field(..., description="The document text")


# Generic Rerank API (our native format)

class RerankRequest(BaseModel):
    """Native rerank request format."""
    query: str = Field(..., description="The search query")
    documents: List[str] = Field(..., description="List of documents to rerank")
    top_n: Optional[int] = Field(default=None, description="Number of top results to return")
    return_documents: bool = Field(default=True, description="Whether to return document text")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is deep learning?",
                "documents": [
                    "Deep learning is a subset of machine learning.",
                    "The weather is nice today.",
                    "Neural networks are used in deep learning."
                ],
                "top_n": 2,
                "return_documents": True
            }
        }


class RerankResult(BaseModel):
    """Single rerank result."""
    index: int = Field(..., description="Original document index")
    relevance_score: float = Field(..., description="Relevance score")
    document: Optional[Document] = Field(default=None, description="Document object")


class RerankResponse(BaseModel):
    """Native rerank response format."""
    results: List[RerankResult] = Field(..., description="Reranked results")
    model: str = Field(..., description="Model used for reranking")


# Cohere-compatible API

class CohereRerankRequest(BaseModel):
    """Cohere-compatible rerank request."""
    query: str = Field(..., description="The search query")
    documents: Union[List[str], List[dict]] = Field(..., description="Documents to rerank")
    top_n: Optional[int] = Field(default=None, description="Number of results to return")
    return_documents: bool = Field(default=False, description="Return document text")
    model: Optional[str] = Field(default=None, description="Model to use (ignored, uses configured model)")
    max_chunks_per_doc: Optional[int] = Field(default=None, description="Max chunks (ignored)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is deep learning?",
                "documents": [
                    "Deep learning is a subset of machine learning.",
                    "The weather is nice today."
                ],
                "top_n": 2
            }
        }


class CohereRerankResult(BaseModel):
    """Cohere-compatible result format."""
    index: int
    relevance_score: float
    document: Optional[dict] = None


class CohereRerankResponse(BaseModel):
    """Cohere-compatible rerank response."""
    id: str = Field(..., description="Request ID")
    results: List[CohereRerankResult]
    meta: dict = Field(default_factory=dict)


# Jina AI-compatible API

class JinaRerankRequest(BaseModel):
    """Jina AI-compatible rerank request."""
    query: str = Field(..., description="The search query")
    documents: Union[List[str], List[dict]] = Field(..., description="Documents to rerank")
    model: Optional[str] = Field(default=None, description="Model to use")
    top_n: Optional[int] = Field(default=None, description="Number of results")
    return_documents: bool = Field(default=True, description="Return documents")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "documents": [
                    {"text": "Machine learning is a branch of AI."},
                    {"text": "I like pizza."}
                ],
                "top_n": 1
            }
        }


class JinaRerankResult(BaseModel):
    """Jina AI-compatible result format."""
    index: int
    relevance_score: float
    document: Optional[Document] = None


class JinaUsage(BaseModel):
    """Jina AI usage information."""
    total_tokens: int = 0
    prompt_tokens: int = 0


class JinaRerankResponse(BaseModel):
    """Jina AI-compatible rerank response."""
    model: str
    usage: JinaUsage
    results: List[JinaRerankResult]
