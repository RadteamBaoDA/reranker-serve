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
        all_results = []
        
        for request in batch.requests:
            # Use Qwen3Reranker's rerank method directly
            results = self.model.rerank(
                query=request.query,
                documents=request.documents,
                top_k=request.top_k,
                return_documents=request.return_documents,
            )
            all_results.append(results)
            
        return all_results
