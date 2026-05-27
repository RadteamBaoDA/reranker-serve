import logging
from typing import List, Dict, Any

from src.engine.handlers.base import BaseHandler
from src.engine.request_queue import BatchedRequest
from src.models.qwen3_reranker import Qwen3Reranker

logger = logging.getLogger(__name__)

class QwenRerankerHandler(BaseHandler):
    """
    Handler for Qwen3-Reranker models.
    """
    
    def load_model(self) -> None:
        logger.info(f"Loading Qwen3-Reranker model: {self.model_path}")
        
        self.model = Qwen3Reranker(
            model_name_or_path=self.model_path,
            device=self.device,
            max_length=self.max_length,
            use_fp16=self.use_fp16,
        )
        self.model.load()
        logger.info("Qwen3-Reranker model loaded successfully")

    def predict(self, batch: BatchedRequest) -> List[List[Dict[str, Any]]]:
        if not batch.requests:
            return []

        # Flatten every (query, doc) pair across all requests into one list,
        # remembering which request each pair belongs to.
        flat_pairs: List[tuple] = []
        spans: List[tuple] = []  # (start, end) index range per request
        for request in batch.requests:
            start = len(flat_pairs)
            flat_pairs.extend((request.query, doc) for doc in request.documents)
            spans.append((start, len(flat_pairs)))

        # ONE batched scoring call for the whole BatchedRequest.
        scores = self.model.score_pairs(flat_pairs) if flat_pairs else []

        # Scatter scores back to each request, then sort + top_k per request.
        all_results: List[List[Dict[str, Any]]] = []
        for request, (start, end) in zip(batch.requests, spans):
            request_scores = scores[start:end]
            results: List[Dict[str, Any]] = []
            for idx, score in enumerate(request_scores):
                result = {"index": idx, "relevance_score": float(score)}
                if request.return_documents:
                    result["document"] = {"text": request.documents[idx]}
                results.append(result)

            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            if request.top_k is not None and request.top_k > 0:
                results = results[:request.top_k]
            all_results.append(results)

        return all_results
