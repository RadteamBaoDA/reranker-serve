import logging
import numpy as np
import torch
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

from src.config import settings
from src.engine.handlers.base import BaseHandler
from src.engine.request_queue import BatchedRequest

logger = logging.getLogger(__name__)

class CrossEncoderHandler(BaseHandler):
    """
    Handler for standard SentenceTransformers CrossEncoder models.
    """
    
    def load_model(self) -> None:
        logger.info(f"Loading standard CrossEncoder model: {self.model_path}")
        
        model_kwargs = {
            "max_length": self.max_length,
            "device": self.device,
        }
        
        if "qwen" in self.model_path.lower():
            model_kwargs["trust_remote_code"] = True
            
        # Load model
        self.model = CrossEncoder(self.model_path, **model_kwargs)
        
        # Fix padding token logic
        self._fix_tokenizer()
        
        # Apply dtype optimizations
        if self.device != "cpu" and hasattr(self.model, 'model'):
            dtype = settings.get_torch_dtype()
            if dtype == torch.float16 and self.device == "cuda":
                self.model.model = self.model.model.half()
        
        logger.info("CrossEncoder model loaded successfully")

    def _fix_tokenizer(self):
        """Ensure tokenizer has a padding token."""
        tokenizer = getattr(self.model, 'tokenizer', None)
        if tokenizer is not None:
            if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                    logger.info("Set padding token to EOS token")
                elif tokenizer.unk_token is not None:
                    tokenizer.pad_token = tokenizer.unk_token
                    tokenizer.pad_token_id = tokenizer.unk_token_id
                    logger.info("Set padding token to UNK token")
                else:
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    logger.info("Added [PAD] token")
                
                if not hasattr(tokenizer, 'padding_side') or tokenizer.padding_side is None:
                    tokenizer.padding_side = 'right'

    def predict(self, batch: BatchedRequest) -> List[List[Dict[str, Any]]]:
        all_results = []
        
        for request in batch.requests:
            pairs = [[request.query, doc] for doc in request.documents]
            
            # Run inference
            scores = self.model.predict(
                pairs,
                batch_size=settings.batch_size,
                show_progress_bar=False,
            )
            
            # Normalize if configured
            if settings.normalize_scores:
                scores_array = np.array(scores)
                scores = (1 / (1 + np.exp(-scores_array))).tolist()
            
            # Build results
            results = []
            for idx, score in enumerate(scores):
                result = {
                    "index": idx,
                    "relevance_score": float(score),
                }
                if request.return_documents:
                    result["document"] = {"text": request.documents[idx]}
                results.append(result)
            
            # Sort by score
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Apply top_k
            if request.top_k is not None and request.top_k > 0:
                results = results[:request.top_k]
                
            all_results.append(results)
            
        return all_results
