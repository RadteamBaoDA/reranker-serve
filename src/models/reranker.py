"""
Reranker Model Service using Sentence Transformers.
Supports BAAI/bge-reranker and Qwen3-reranker models.
"""

import os
import logging
from typing import Optional, List, Tuple, Union
from functools import lru_cache

import torch
from sentence_transformers import CrossEncoder

from src.config import settings

logger = logging.getLogger(__name__)


class RerankerModel:
    """
    Reranker model wrapper for Sentence Transformer CrossEncoder models.
    Optimized for different platforms including Apple Silicon (MPS).
    """
    
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        max_length: Optional[int] = None,
        use_fp16: Optional[bool] = None,
    ):
        """
        Initialize the reranker model.
        
        Args:
            model_name_or_path: Model name or local path
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto)
            max_length: Maximum sequence length
            use_fp16: Whether to use FP16 precision
        """
        self.model_name_or_path = model_name_or_path or settings.get_model_load_path()
        self.device = device or settings.get_device()
        self.max_length = max_length or settings.max_length
        self.use_fp16 = use_fp16 if use_fp16 is not None else settings.use_fp16
        
        self._model: Optional[CrossEncoder] = None
        self._setup_environment()
    
    def _setup_environment(self):
        """Set up environment variables for optimal performance."""
        # Set offline mode if configured
        if settings.use_offline_mode:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        # Set cache directory (use absolute path for consistency)
        if settings.model_cache_dir:
            cache_dir = os.path.abspath(settings.model_cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            os.environ["HF_HOME"] = cache_dir
            os.environ["HF_HUB_CACHE"] = cache_dir
            os.environ["TRANSFORMERS_CACHE"] = cache_dir
            os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir
        
        # MPS-specific optimizations for Apple Silicon
        if self.device == "mps":
            # Enable MPS fallback for unsupported operations
            if settings.mps_fallback_to_cpu:
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            
            # Optimize memory for MPS
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    
    def _is_valid_local_model(self, path: str) -> bool:
        """Check if the path contains a valid local model."""
        if not path or not os.path.isdir(path):
            return False
        
        # Check for common model files that indicate a valid model directory
        model_files = [
            "config.json",
            "pytorch_model.bin",
            "model.safetensors",
            "tf_model.h5",
        ]
        
        for model_file in model_files:
            if os.path.exists(os.path.join(path, model_file)):
                return True
        
        # Also check for sharded models
        for f in os.listdir(path):
            if f.startswith("pytorch_model-") and f.endswith(".bin"):
                return True
            if f.startswith("model-") and f.endswith(".safetensors"):
                return True
        
        return False
    
    def _get_cache_path(self, model_name: str) -> str:
        """Get the cache path for a model."""
        # Convert model name to cache directory format
        cache_dir = os.path.abspath(settings.model_cache_dir)
        
        # HuggingFace hub uses models--org--name format
        model_dir_name = model_name.replace("/", "--")
        hf_cache_path = os.path.join(cache_dir, f"models--{model_dir_name}")
        
        # Also check simple format (org_name)
        simple_cache_path = os.path.join(cache_dir, model_name.replace("/", "_").replace("\\", "_"))
        
        # Return HF format if it exists, otherwise simple format
        if os.path.isdir(hf_cache_path):
            # Look for snapshots directory in HF cache
            snapshots_dir = os.path.join(hf_cache_path, "snapshots")
            if os.path.isdir(snapshots_dir):
                # Get the latest snapshot
                snapshots = os.listdir(snapshots_dir)
                if snapshots:
                    return os.path.join(snapshots_dir, snapshots[0])
        
        return simple_cache_path
    
    def load(self) -> "RerankerModel":
        """Load the model into memory."""
        if self._model is not None:
            return self
        
        logger.info(f"Loading reranker model: {self.model_name_or_path}")
        logger.info(f"Device: {self.device}, Max Length: {self.max_length}")
        
        try:
            # Determine model loading kwargs
            model_kwargs = {
                "max_length": self.max_length,
                "device": self.device,
            }
            
            # Add trust_remote_code for Qwen models
            if "qwen" in self.model_name_or_path.lower():
                model_kwargs["trust_remote_code"] = True
            
            # Determine the model source
            model_source = None
            
            # Priority 1: Check explicit model_path setting
            if settings.model_path:
                if self._is_valid_local_model(settings.model_path):
                    model_source = settings.model_path
                    logger.info(f"Using model from configured path: {model_source}")
                else:
                    logger.warning(f"Configured model_path not found or invalid: {settings.model_path}")
            
            # Priority 2: Check if model exists in cache directory
            if model_source is None:
                cache_path = self._get_cache_path(settings.model_name)
                if self._is_valid_local_model(cache_path):
                    model_source = cache_path
                    logger.info(f"Using cached model from: {model_source}")
            
            # Priority 3: Check if model_name_or_path is a local path
            if model_source is None and os.path.isdir(self.model_name_or_path):
                if self._is_valid_local_model(self.model_name_or_path):
                    model_source = self.model_name_or_path
                    logger.info(f"Using model from path: {model_source}")
            
            # Priority 4: Download from Hugging Face
            if model_source is None:
                if settings.use_offline_mode:
                    raise RuntimeError(
                        f"Model not found locally and offline mode is enabled. "
                        f"Please download the model first using download_model.sh or "
                        f"set RERANKER_MODEL_PATH to a valid model directory. "
                        f"Checked paths: {settings.model_path}, {self._get_cache_path(settings.model_name)}"
                    )
                
                # Download model from Hugging Face to configured cache directory
                logger.info(f"Model not found locally, downloading from Hugging Face: {settings.model_name}")
                
                # Ensure cache directory exists (use absolute path)
                cache_dir = os.path.abspath(settings.model_cache_dir)
                os.makedirs(cache_dir, exist_ok=True)
                
                # Use model name directly - CrossEncoder will handle download
                model_source = settings.model_name
                
                # Pass cache_folder to CrossEncoder to download to configured path
                model_kwargs["cache_folder"] = cache_dir
                logger.info(f"Downloading model to cache: {cache_dir}")
            
            # Load the model
            logger.info(f"Loading model from: {model_source}")
            self._model = CrossEncoder(
                model_source,
                **model_kwargs
            )
            
            # Move to device and set dtype
            if self.device != "cpu" and hasattr(self._model, 'model'):
                dtype = settings.get_torch_dtype()
                if dtype == torch.float16 and self.device == "cuda":
                    self._model.model = self._model.model.half()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        return self
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        return_documents: bool = True,
    ) -> List[dict]:
        """
        Rerank documents based on relevance to the query.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top results to return (None for all)
            return_documents: Whether to include document text in results
            
        Returns:
            List of dicts with 'index', 'score', and optionally 'document'
        """
        if self._model is None:
            self.load()
        
        if not documents:
            return []
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get scores with batch processing
        try:
            scores = self._model.predict(
                pairs,
                batch_size=settings.batch_size,
                show_progress_bar=False,
            )
        except Exception as e:
            # Fallback for MPS issues
            if self.device == "mps" and settings.mps_fallback_to_cpu:
                logger.warning(f"MPS inference failed, falling back to CPU: {e}")
                self.device = "cpu"
                self._model = None
                self.load()
                scores = self._model.predict(
                    pairs,
                    batch_size=settings.batch_size,
                    show_progress_bar=False,
                )
            else:
                raise
        
        # Normalize scores if configured
        if settings.normalize_scores:
            scores = self._normalize_scores(scores)
        
        # Create results with original indices
        results = []
        for idx, score in enumerate(scores):
            result = {
                "index": idx,
                "relevance_score": float(score),
            }
            if return_documents:
                result["document"] = {"text": documents[idx]}
            results.append(result)
        
        # Sort by score descending
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Apply top_k
        if top_k is not None and top_k > 0:
            results = results[:top_k]
        
        return results
    
    def _normalize_scores(self, scores) -> List[float]:
        """Normalize scores using sigmoid function."""
        import numpy as np
        
        # Apply sigmoid to convert logits to probabilities
        normalized = 1 / (1 + np.exp(-scores))
        return normalized.tolist()
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None
    
    def unload(self):
        """Unload the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            
            # Clear CUDA cache if applicable
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            logger.info("Model unloaded")


# Singleton instance
_reranker_instance: Optional[RerankerModel] = None


def get_reranker_model() -> RerankerModel:
    """Get or create the singleton reranker model instance."""
    global _reranker_instance
    
    if _reranker_instance is None:
        _reranker_instance = RerankerModel()
        _reranker_instance.load()
    
    return _reranker_instance


def reset_reranker_model():
    """Reset the singleton instance (useful for testing)."""
    global _reranker_instance
    
    if _reranker_instance is not None:
        _reranker_instance.unload()
        _reranker_instance = None
