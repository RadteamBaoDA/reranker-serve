"""
HTTP client for making requests to remote reranker backends.
"""

from typing import Optional, List, Dict, Any, Union

import httpx

from src.config import get_logger
from .config import ModelConfig

logger = get_logger(__name__)


class RerankerClient:
    """
    HTTP client for communicating with remote reranker backends.
    Supports multiple API formats (native, Cohere, Jina).
    """
    
    def __init__(
        self,
        backend: ModelConfig,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        """
        Initialize the client.
        
        Args:
            backend: Backend configuration
            http_client: Optional shared HTTP client
        """
        self.backend = backend
        self._http_client = http_client
        self._owns_client = http_client is None
    
    async def __aenter__(self):
        if self._owns_client:
            self._http_client = httpx.AsyncClient(
                timeout=self.backend.timeout,
                follow_redirects=True,
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._owns_client and self._http_client:
            await self._http_client.aclose()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.backend.api_key:
            headers["Authorization"] = f"Bearer {self.backend.api_key}"
        return headers
    
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        return_documents: bool = True,
        api_format: str = "native",
    ) -> Dict[str, Any]:
        """
        Send a rerank request to the backend.
        
        Args:
            query: Search query
            documents: List of documents to rerank
            top_n: Number of top results to return
            return_documents: Whether to include document text in response
            api_format: API format to use ('native', 'cohere', 'jina')
            
        Returns:
            Rerank response as dictionary
        """
        if api_format == "cohere":
            return await self._rerank_cohere(query, documents, top_n, return_documents)
        elif api_format == "jina":
            return await self._rerank_jina(query, documents, top_n, return_documents)
        else:
            return await self._rerank_native(query, documents, top_n, return_documents)
    
    async def _rerank_native(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int],
        return_documents: bool,
    ) -> Dict[str, Any]:
        """Send request using native API format."""
        url = f"{self.backend.api_base}/rerank"
        
        payload = {
            "query": query,
            "documents": documents,
            "return_documents": return_documents,
        }
        if top_n is not None:
            payload["top_n"] = top_n
        
        response = await self._http_client.post(
            url,
            json=payload,
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return response.json()
    
    async def _rerank_cohere(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int],
        return_documents: bool,
    ) -> Dict[str, Any]:
        """Send request using Cohere API format."""
        url = f"{self.backend.api_base}/v1/rerank"
        
        payload = {
            "query": query,
            "documents": documents,
            "return_documents": return_documents,
        }
        if top_n is not None:
            payload["top_n"] = top_n
        
        response = await self._http_client.post(
            url,
            json=payload,
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return response.json()
    
    async def _rerank_jina(
        self,
        query: str,
        documents: Union[List[str], List[Dict[str, str]]],
        top_n: Optional[int],
        return_documents: bool,
    ) -> Dict[str, Any]:
        """Send request using Jina AI API format."""
        url = f"{self.backend.api_base}/api/v1/rerank"
        
        # Jina expects documents as list of dicts with 'text' key
        if documents and isinstance(documents[0], str):
            documents = [{"text": doc} for doc in documents]
        
        payload = {
            "query": query,
            "documents": documents,
            "return_documents": return_documents,
        }
        if top_n is not None:
            payload["top_n"] = top_n
        
        response = await self._http_client.post(
            url,
            json=payload,
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return response.json()
    
    async def health_check(self) -> bool:
        """Check if the backend is healthy."""
        try:
            url = f"{self.backend.api_base}/health"
            response = await self._http_client.get(
                url,
                headers=self._get_headers(),
                timeout=10.0,
            )
            return response.status_code == 200
        except Exception:
            return False
