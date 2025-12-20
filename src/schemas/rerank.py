"""
API Request/Response schemas for reranker endpoints.
Compatible with Jina AI and Cohere API formats.
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


# Common schemas

class Document(BaseModel):
    """Document object for reranking."""
    text: str = Field(..., description="The document text")


# Generic Rerank API (our native format)

class RerankRequest(BaseModel):
    """Native rerank request format - supports both native and HuggingFace formats."""
    query: str = Field(..., description="The search query")
    documents: Optional[List[str]] = Field(default=None, description="List of documents to rerank (native format)")
    texts: Optional[List[str]] = Field(default=None, description="List of texts to rerank (HuggingFace format)")
    top_n: Optional[int] = Field(default=None, description="Number of top results to return", alias="top_k")
    return_documents: bool = Field(default=True, description="Whether to return document text", alias="return_texts")
    
    model_config = ConfigDict(
        populate_by_name=True,  # Allow both field name and alias
        json_schema_extra={
            "examples": [
                {
                    "query": "What is deep learning?",
                    "documents": [
                        "Deep learning is a subset of machine learning.",
                        "The weather is nice today.",
                        "Neural networks are used in deep learning."
                    ],
                    "top_n": 2,
                    "return_documents": True
                },
                {
                    "query": "What is deep learning?",
                    "texts": [
                        "Deep learning is a subset of machine learning.",
                        "The weather is nice today."
                    ],
                    "top_k": 2,
                    "return_texts": True
                }
            ]
        }
    )
    
    @model_validator(mode='after')
    def validate_documents_or_texts(self):
        """Ensure either documents or texts is provided, not both or neither."""
        if self.documents is None and self.texts is None:
            raise ValueError("Either 'documents' or 'texts' field must be provided")
        if self.documents is not None and self.texts is not None:
            raise ValueError("Cannot provide both 'documents' and 'texts' fields")
        return self
    
    def get_documents(self) -> List[str]:
        """Get the document list regardless of which field was used."""
        return self.documents if self.documents is not None else self.texts


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
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What is deep learning?",
                "documents": [
                    "Deep learning is a subset of machine learning.",
                    "The weather is nice today."
                ],
                "top_n": 2
            }
        }
    )


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
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What is machine learning?",
                "documents": [
                    {"text": "Machine learning is a branch of AI."},
                    {"text": "I like pizza."}
                ],
                "top_n": 1
            }
        }
    )


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


# HuggingFace-compatible API

class HuggingFaceRerankRequest(BaseModel):
    """HuggingFace-compatible rerank request."""
    query: str = Field(..., description="The search query")
    texts: List[str] = Field(..., description="List of texts/documents to rerank")
    top_k: Optional[int] = Field(default=None, description="Number of top results to return", alias="top_n")
    return_texts: bool = Field(default=True, description="Return text in response", alias="return_documents")
    model: Optional[str] = Field(default=None, description="Model to use (ignored, uses configured model)")
    truncate: bool = Field(default=True, description="Truncate inputs if too long")
    
    model_config = ConfigDict(
        populate_by_name=True,  # Allow both field name and alias
        json_schema_extra={
            "example": {
                "query": "What is deep learning?",
                "texts": [
                    "Deep learning is a subset of machine learning.",
                    "The weather is nice today."
                ],
                "top_k": 2
            }
        }
    )


class HuggingFaceRerankResult(BaseModel):
    """HuggingFace-compatible result format."""
    index: int = Field(..., description="Document index")
    score: float = Field(..., description="Relevance score")
    text: Optional[str] = Field(default=None, description="Document text")


class HuggingFaceRerankResponse(BaseModel):
    """HuggingFace-compatible rerank response."""
    results: List[HuggingFaceRerankResult] = Field(..., description="Ranked results")
    model: str = Field(..., description="Model used")
