"""
Qwen3 Reranker Model Implementation.
Based on the official Qwen3-Reranker architecture from Hugging Face.

Qwen3-Reranker uses a different approach than standard CrossEncoder:
- It's a Causal LM (decoder-only) model
- Uses special prompt format with instruction, query, and document
- Computes relevance scores from "yes"/"no" token logits
"""

import os
from typing import Optional, List, Dict, Any
import torch

from src.config import settings, get_logger

logger = get_logger(__name__)


class Qwen3Reranker:
    """
    Qwen3 Reranker model wrapper.
    
    This implementation follows the official Qwen3-Reranker usage pattern
    from https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
    """
    
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        max_length: Optional[int] = None,
        use_fp16: Optional[bool] = None,
    ):
        """
        Initialize the Qwen3 reranker model.
        
        Args:
            model_name_or_path: Model name or local path
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto)
            max_length: Maximum sequence length (default: 8192 for Qwen3)
            use_fp16: Whether to use FP16 precision
        """
        self.model_name_or_path = model_name_or_path or settings.get_model_load_path()
        self.device = device or settings.get_device()
        # Qwen3-Reranker supports up to 32k, but 8192 is recommended
        self.max_length = max_length or min(settings.max_length, 8192)
        self.use_fp16 = use_fp16 if use_fp16 is not None else settings.use_fp16
        
        self._model = None
        self._tokenizer = None
        self._token_true_id = None
        self._token_false_id = None
        self._prefix = None
        self._suffix = None
        
        # Default instruction for reranking
        self.default_instruction = "Given a web search query, retrieve relevant passages that answer the query"
        
        self._setup_environment()
        
        logger.debug(
            "qwen3_reranker_init",
            model_name_or_path=self.model_name_or_path,
            device=self.device,
            max_length=self.max_length,
            use_fp16=self.use_fp16,
        )
    
    def _setup_environment(self):
        """Set up environment variables for optimal performance."""
        # Force CPU-only mode if configured
        if settings.force_cpu_only:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            logger.debug("cpu_only_mode_enabled")
        
        # Set offline mode if configured
        if settings.use_offline_mode:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        # Set cache directory (HF_HOME is the unified cache location in transformers v5+)
        if settings.model_cache_dir:
            cache_dir = os.path.abspath(settings.model_cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            os.environ["HF_HOME"] = cache_dir
    
    def load(self) -> "Qwen3Reranker":
        """Load the model and tokenizer."""
        if self._model is not None:
            return self
        
        logger.info(f"Loading Qwen3 reranker model: {self.model_name_or_path}")
        logger.info(f"Device: {self.device}, Max Length: {self.max_length}")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Determine model source
            model_source = self._get_model_source()
            logger.info(f"Loading model from: {model_source}")
            
            # Load tokenizer with left padding (required for batch processing)
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                padding_side='left',
                trust_remote_code=True,
            )
            
            # Ensure pad token is set
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            
            # Load model
            model_kwargs = {
                "trust_remote_code": True,
            }
            
            # Set dtype based on device and settings
            if self.use_fp16 and self.device == "cuda":
                model_kwargs["torch_dtype"] = torch.float16
            elif self.device != "cpu":
                model_kwargs["torch_dtype"] = torch.bfloat16
            
            # Check if flash attention is available (only for CUDA)
            use_flash_attn = False
            if self.device == "cuda":
                try:
                    import flash_attn
                    use_flash_attn = True
                    logger.debug("flash_attention_available", version=getattr(flash_attn, "__version__", "unknown"))
                except ImportError:
                    logger.debug("flash_attention_not_available", using="sdpa")
            
            if use_flash_attn:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            else:
                # Use SDPA (scaled dot-product attention) as fallback - available in PyTorch 2.0+
                model_kwargs["attn_implementation"] = "sdpa"
            
            self._model = AutoModelForCausalLM.from_pretrained(
                model_source,
                **model_kwargs
            )
            
            # Move to device
            if self.device == "cuda":
                self._model = self._model.cuda()
            elif self.device == "mps":
                self._model = self._model.to("mps")
            
            self._model.eval()
            
            # Get token IDs for "yes" and "no"
            self._token_true_id = self._tokenizer.convert_tokens_to_ids("yes")
            self._token_false_id = self._tokenizer.convert_tokens_to_ids("no")
            
            # Prepare prefix and suffix strings for prompt construction
            self._prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
            self._suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            
            logger.info("Qwen3 reranker model loaded successfully")
            logger.debug(
                "qwen3_model_loaded",
                token_true_id=self._token_true_id,
                token_false_id=self._token_false_id,
            )
            
        except Exception as e:
            logger.error(f"Failed to load Qwen3 reranker model: {e}")
            raise
        
        return self
    
    def _get_model_source(self) -> str:
        """Determine the model source path."""
        # Check explicit model_path
        if settings.model_path and os.path.isdir(settings.model_path):
            return settings.model_path
        
        # Check cache
        cache_dir = os.path.abspath(settings.model_cache_dir)
        model_dir_name = settings.model_name.replace("/", "--")
        hf_cache_path = os.path.join(cache_dir, f"models--{model_dir_name}")
        
        if os.path.isdir(hf_cache_path):
            snapshots_dir = os.path.join(hf_cache_path, "snapshots")
            if os.path.isdir(snapshots_dir):
                snapshots = os.listdir(snapshots_dir)
                if snapshots:
                    return os.path.join(snapshots_dir, snapshots[0])
        
        # Check model_name_or_path as local path
        if os.path.isdir(self.model_name_or_path):
            return self.model_name_or_path
        
        # Download from HuggingFace
        if settings.use_offline_mode:
            raise RuntimeError(
                f"Model not found locally and offline mode is enabled."
            )
        
        return settings.model_name
    
    def _format_instruction(
        self,
        query: str,
        doc: str,
        instruction: Optional[str] = None,
    ) -> str:
        """Format the input according to Qwen3-Reranker prompt template."""
        if instruction is None:
            instruction = self.default_instruction
        
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
    
    def _build_full_prompt(self, content: str) -> str:
        """Build the full prompt with prefix and suffix."""
        return f"{self._prefix}{content}{self._suffix}"
    
    def _process_inputs(self, pairs: List[str]) -> Dict[str, torch.Tensor]:
        """Process input pairs into model inputs with proper tokenization.
        
        Uses a single tokenizer __call__ for better performance with fast tokenizers.
        """
        # Build full prompts with prefix and suffix
        full_prompts = [self._build_full_prompt(pair) for pair in pairs]
        
        # Use single tokenizer call with padding and truncation
        inputs = self._tokenizer(
            full_prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Move to device
        for key in inputs:
            inputs[key] = inputs[key].to(self._model.device)
        
        return inputs
    
    @torch.no_grad()
    def _compute_logits(self, inputs: Dict[str, torch.Tensor]) -> List[float]:
        """Compute relevance scores from model logits with MPS fallback."""
        try:
            # Get model outputs
            outputs = self._model(**inputs)
            
            # Get logits for the last token position
            batch_scores = outputs.logits[:, -1, :]
            
            # Extract scores for "yes" and "no" tokens
            true_vector = batch_scores[:, self._token_true_id]
            false_vector = batch_scores[:, self._token_false_id]
            
            # Stack and apply log_softmax
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            
            # Get probability of "yes" token
            scores = batch_scores[:, 1].exp().tolist()
            
            return scores
            
        except RuntimeError as e:
            error_msg = str(e)
            # Handle MPS tensor size limitations
            if "MPSGraph" in error_msg or "INT_MAX" in error_msg:
                logger.warning(
                    "mps_tensor_too_large_fallback_to_cpu",
                    error=error_msg,
                    device=self.device,
                )
                # Move inputs to CPU and retry
                cpu_inputs = {k: v.cpu() for k, v in inputs.items()}
                self._model = self._model.cpu()
                self.device = "cpu"
                
                outputs = self._model(**cpu_inputs)
                batch_scores = outputs.logits[:, -1, :]
                true_vector = batch_scores[:, self._token_true_id]
                false_vector = batch_scores[:, self._token_false_id]
                batch_scores = torch.stack([false_vector, true_vector], dim=1)
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                scores = batch_scores[:, 1].exp().tolist()
                
                logger.info("successfully_completed_inference_on_cpu_after_mps_fallback")
                return scores
            else:
                # Re-raise other runtime errors
                raise
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        return_documents: bool = True,
        instruction: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to the query.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top results to return (None for all)
            return_documents: Whether to include document text in results
            instruction: Custom instruction for the task (optional)
            
        Returns:
            List of dicts with 'index', 'relevance_score', and optionally 'document'
        """
        if self._model is None:
            self.load()
        
        if not documents:
            return []
        
        logger.debug(
            "qwen3_rerank_start",
            query_length=len(query),
            num_documents=len(documents),
            top_k=top_k,
            instruction=instruction[:50] if instruction else None,
        )
        
        # Format all query-document pairs
        pairs = [
            self._format_instruction(query, doc, instruction)
            for doc in documents
        ]
        
        # Process in batches if needed
        batch_size = settings.batch_size
        all_scores = []
        
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            inputs = self._process_inputs(batch_pairs)
            batch_scores = self._compute_logits(inputs)
            all_scores.extend(batch_scores)
        
        # Create results with original indices
        results = []
        for idx, score in enumerate(all_scores):
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
        
        logger.debug(
            "qwen3_rerank_complete",
            num_results=len(results),
            top_score=results[0]["relevance_score"] if results else None,
        )
        
        return results
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None
    
    @property
    def tokenizer(self):
        """Get the tokenizer (for compatibility)."""
        return self._tokenizer
    
    def unload(self):
        """Unload the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            del self._tokenizer
            self._tokenizer = None
            
            # Clear CUDA cache if applicable
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            logger.info("Qwen3 reranker model unloaded")


def is_qwen3_reranker(model_name: str) -> bool:
    """Check if the model is a Qwen3-Reranker model."""
    model_lower = model_name.lower()
    return "qwen3" in model_lower and "reranker" in model_lower
