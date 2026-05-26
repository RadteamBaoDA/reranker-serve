"""
LiteLLM CustomLLM provider that proxies litellm.rerank() to this service's
native /rerank endpoint.

Wire-up (in litellm config.yaml):

    model_list:
      - model_name: local-reranker
        litellm_params:
          model: reranker_custom/local
          api_base: http://localhost:8000
          api_key: "optional-bearer-token"

    litellm_settings:
      custom_provider_map:
        - provider: reranker_custom
          custom_handler: src.integrations.litellm_provider.reranker_provider
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

import httpx


def _require_litellm():
    try:
        from litellm import CustomLLM
        from litellm.types.utils import RerankResponse, RerankResponseResult
    except ImportError as exc:
        raise ImportError(
            "litellm is required to use RerankerLiteLLMProvider. "
            "Install it with: pip install litellm"
        ) from exc
    return CustomLLM, RerankResponse, RerankResponseResult


CustomLLM, RerankResponse, RerankResponseResult = _require_litellm()


class RerankerLiteLLMProvider(CustomLLM):
    """Maps litellm.rerank() onto the native /rerank endpoint."""

    DEFAULT_TIMEOUT = 60.0

    def rerank(
        self,
        model: str,
        query: str,
        documents: List[Union[str, dict]],
        top_n: Optional[int] = None,
        return_documents: bool = False,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> RerankResponse:
        base = (api_base or "http://localhost:8000").rstrip("/")
        headers = {"Content-Type": "application/json"}
        if api_key and api_key != "dummy":
            headers["Authorization"] = f"Bearer {api_key}"

        payload: dict[str, Any] = {
            "query": query,
            "documents": documents,
            "return_documents": return_documents,
        }
        if top_n is not None:
            payload["top_n"] = top_n

        with httpx.Client(timeout=timeout or self.DEFAULT_TIMEOUT) as client:
            response = client.post(f"{base}/rerank", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        results = [
            RerankResponseResult(
                index=item["index"],
                relevance_score=item["relevance_score"],
                document=item.get("document"),
            )
            for item in data.get("results", [])
        ]
        return RerankResponse(
            id=data.get("id", ""),
            results=results,
            meta=data.get("meta", {}),
        )

    async def arerank(
        self,
        model: str,
        query: str,
        documents: List[Union[str, dict]],
        top_n: Optional[int] = None,
        return_documents: bool = False,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> RerankResponse:
        base = (api_base or "http://localhost:8000").rstrip("/")
        headers = {"Content-Type": "application/json"}
        if api_key and api_key != "dummy":
            headers["Authorization"] = f"Bearer {api_key}"

        payload: dict[str, Any] = {
            "query": query,
            "documents": documents,
            "return_documents": return_documents,
        }
        if top_n is not None:
            payload["top_n"] = top_n

        async with httpx.AsyncClient(timeout=timeout or self.DEFAULT_TIMEOUT) as client:
            response = await client.post(f"{base}/rerank", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        results = [
            RerankResponseResult(
                index=item["index"],
                relevance_score=item["relevance_score"],
                document=item.get("document"),
            )
            for item in data.get("results", [])
        ]
        return RerankResponse(
            id=data.get("id", ""),
            results=results,
            meta=data.get("meta", {}),
        )


reranker_provider = RerankerLiteLLMProvider()
