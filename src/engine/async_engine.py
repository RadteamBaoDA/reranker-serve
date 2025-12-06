"""
Async Reranker Engine.
Inspired by vLLM's AsyncLLM for high-performance concurrent inference.

Key features:
- Async request handling with concurrent batching
- Background batch processing loop
- Thread pool for CPU-bound model inference
- Request priority scheduling
- Graceful shutdown
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

import torch

from src.config import settings, get_logger
from src.engine.request_queue import (
    RequestQueue,
    RerankRequest,
    RerankResult,
    BatchedRequest,
    RequestStatus,
)

logger = get_logger(__name__)


class AsyncRerankerEngine:
    """
    High-performance async reranker engine with batching.
    
    Architecture (inspired by vLLM):
    1. API layer adds requests to async queue with futures
    2. Background processor loop batches requests
    3. Model inference runs in thread pool (non-blocking)
    4. Results are distributed back to waiting futures
    """
    
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        max_length: Optional[int] = None,
        use_fp16: Optional[bool] = None,
        # Concurrency settings
        max_concurrent_batches: int = 2,
        inference_threads: int = 1,
        max_batch_size: int = 32,
        max_batch_pairs: int = 1024,
        batch_wait_timeout: float = 0.01,
        max_queue_size: int = 1000,
        request_timeout: float = 60.0,
    ):
        # Model settings
        self.model_name_or_path = model_name_or_path or settings.get_model_load_path()
        self.device = device or settings.get_device()
        self.max_length = max_length or settings.max_length
        self.use_fp16 = use_fp16 if use_fp16 is not None else settings.use_fp16
        
        # Concurrency settings
        self.max_concurrent_batches = max_concurrent_batches
        self.inference_threads = inference_threads
        
        # Request queue
        self.request_queue = RequestQueue(
            max_batch_size=max_batch_size,
            max_batch_pairs=max_batch_pairs,
            batch_wait_timeout=batch_wait_timeout,
            max_queue_size=max_queue_size,
            request_timeout=request_timeout,
        )
        
        # Thread pool for blocking inference
        self._executor = ThreadPoolExecutor(
            max_workers=inference_threads,
            thread_name_prefix="reranker-inference",
        )
        
        # Model (loaded lazily)
        self._model = None
        self._model_lock = asyncio.Lock()
        
        # Background processor
        self._processor_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Semaphore for concurrent batch processing
        self._batch_semaphore = asyncio.Semaphore(max_concurrent_batches)
        
        # Metrics
        self._total_requests = 0
        self._total_inference_time = 0.0
        self._start_time = time.time()
        
        logger.info(
            "async_engine_created",
            model=self.model_name_or_path,
            device=self.device,
            max_concurrent_batches=max_concurrent_batches,
            inference_threads=inference_threads,
        )
    
    async def start(self) -> "AsyncRerankerEngine":
        """Start the engine and background processor."""
        if self._running:
            logger.debug("engine_already_running", engine_id=id(self))
            return self
        
        logger.info("starting_async_engine")
        logger.debug(
            "engine_config",
            model=self.model_name_or_path,
            device=self.device,
            max_length=self.max_length,
            use_fp16=self.use_fp16,
            max_concurrent_batches=self.max_concurrent_batches,
            inference_threads=self.inference_threads,
        )
        
        # Load model
        await self._load_model()
        
        # Start background processor
        self._running = True
        self._processor_task = asyncio.create_task(
            self._batch_processor_loop(),
            name="reranker-batch-processor",
        )
        
        logger.info("async_engine_started")
        logger.debug("background_processor_task_created", task_name="reranker-batch-processor")
        return self
    
    async def stop(self) -> None:
        """Stop the engine gracefully."""
        if not self._running:
            return
        
        logger.info("stopping_async_engine")
        
        # Signal shutdown
        self._running = False
        self.request_queue.shutdown()
        
        # Wait for processor to finish
        if self._processor_task:
            try:
                await asyncio.wait_for(self._processor_task, timeout=10.0)
            except asyncio.TimeoutError:
                self._processor_task.cancel()
                try:
                    await self._processor_task
                except asyncio.CancelledError:
                    pass
        
        # Shutdown thread pool
        self._executor.shutdown(wait=True)
        
        # Unload model
        self._unload_model()
        
        logger.info("async_engine_stopped")
    
    async def _load_model(self) -> None:
        """Load the model (thread-safe)."""
        logger.debug("acquiring_model_lock")
        async with self._model_lock:
            if self._model is not None:
                logger.debug("model_already_loaded", model=self.model_name_or_path)
                return
            
            logger.info(f"Loading reranker model: {self.model_name_or_path}")
            logger.debug(
                "model_load_start",
                model_path=self.model_name_or_path,
                device=self.device,
                max_length=self.max_length,
            )
            
            load_start_time = time.time()
            
            # Run blocking model load in thread pool
            loop = asyncio.get_running_loop()
            self._model = await loop.run_in_executor(
                self._executor,
                self._load_model_sync,
            )
            
            load_duration = time.time() - load_start_time
            logger.info("Model loaded successfully")
            logger.debug(
                "model_load_complete",
                load_time_seconds=load_duration,
                model_type=type(self._model).__name__,
            )
    
    def _load_model_sync(self):
        """Synchronous model loading (runs in thread pool)."""
        from sentence_transformers import CrossEncoder
        import os
        
        # Force CPU-only mode if configured
        if settings.force_cpu_only:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            os.environ["SENTENCE_TRANSFORMERS_DEVICE"] = "cpu"
            logger.debug(
                "cpu_only_mode_enforced",
                cuda_visible_devices="",
                sentence_transformers_device="cpu",
            )
        
        # Set up environment
        if settings.use_offline_mode:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        if settings.model_cache_dir:
            cache_dir = os.path.abspath(settings.model_cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            os.environ["HF_HOME"] = cache_dir
            os.environ["TRANSFORMERS_CACHE"] = cache_dir
            os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir
        
        # MPS optimizations
        if self.device == "mps":
            if settings.mps_fallback_to_cpu:
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        
        # Prepare model kwargs
        model_kwargs = {
            "max_length": self.max_length,
            "device": self.device,
        }
        
        if "qwen" in self.model_name_or_path.lower():
            model_kwargs["trust_remote_code"] = True
        
        # Determine model source
        model_source = self._get_model_source()
        
        # Load model
        model = CrossEncoder(model_source, **model_kwargs)
        
        # Apply dtype optimizations
        if self.device != "cpu" and hasattr(model, 'model'):
            dtype = settings.get_torch_dtype()
            if dtype == torch.float16 and self.device == "cuda":
                model.model = model.model.half()
        
        return model
    
    def _get_model_source(self) -> str:
        """Determine the model source path."""
        import os
        
        # Check explicit model_path
        if settings.model_path and self._is_valid_local_model(settings.model_path):
            return settings.model_path
        
        # Check cache
        cache_path = self._get_cache_path(settings.model_name)
        if self._is_valid_local_model(cache_path):
            return cache_path
        
        # Check model_name_or_path as local path
        if os.path.isdir(self.model_name_or_path):
            if self._is_valid_local_model(self.model_name_or_path):
                return self.model_name_or_path
        
        # Download from HuggingFace
        if settings.use_offline_mode:
            raise RuntimeError(
                f"Model not found locally and offline mode is enabled."
            )
        
        return settings.model_name
    
    def _is_valid_local_model(self, path: str) -> bool:
        """Check if path contains a valid model."""
        import os
        if not path or not os.path.isdir(path):
            return False
        
        model_files = ["config.json", "pytorch_model.bin", "model.safetensors"]
        for f in model_files:
            if os.path.exists(os.path.join(path, f)):
                return True
        
        for f in os.listdir(path):
            if f.startswith("pytorch_model-") or f.startswith("model-"):
                return True
        
        return False
    
    def _get_cache_path(self, model_name: str) -> str:
        """Get cache path for a model."""
        import os
        cache_dir = os.path.abspath(settings.model_cache_dir)
        model_dir_name = model_name.replace("/", "--")
        hf_cache_path = os.path.join(cache_dir, f"models--{model_dir_name}")
        
        if os.path.isdir(hf_cache_path):
            snapshots_dir = os.path.join(hf_cache_path, "snapshots")
            if os.path.isdir(snapshots_dir):
                snapshots = os.listdir(snapshots_dir)
                if snapshots:
                    return os.path.join(snapshots_dir, snapshots[0])
        
        return os.path.join(
            cache_dir,
            model_name.replace("/", "_").replace("\\", "_")
        )
    
    def _unload_model(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
    
    async def _batch_processor_loop(self) -> None:
        """
        Background loop that processes batches.
        
        This runs continuously, pulling batches from the queue
        and processing them in the thread pool.
        """
        logger.info("batch_processor_started")
        logger.debug(
            "batch_processor_config",
            max_concurrent_batches=self.max_concurrent_batches,
            queue_size=self.request_queue.max_queue_size,
        )
        
        loop_iteration = 0
        while self._running or self.request_queue.active_count > 0:
            loop_iteration += 1
            try:
                logger.debug(
                    "batch_processor_loop_iteration",
                    iteration=loop_iteration,
                    running=self._running,
                    active_requests=self.request_queue.active_count,
                    pending_requests=self.request_queue.pending_count,
                )
                
                # Get a batch (with timeout for shutdown check)
                batch = await self.request_queue.get_batch()
                
                if batch is None:
                    logger.debug("no_batch_available", iteration=loop_iteration)
                    continue
                
                logger.debug(
                    "batch_acquired",
                    batch_id=batch.batch_id,
                    num_requests=len(batch.requests),
                    waiting_for_semaphore=True,
                )
                
                # Process batch with concurrency limit
                async with self._batch_semaphore:
                    logger.debug("semaphore_acquired", batch_id=batch.batch_id)
                    await self._process_batch(batch)
                    
            except asyncio.CancelledError:
                logger.info("batch_processor_cancelled")
                logger.debug("batch_processor_cancelled_at_iteration", iteration=loop_iteration)
                break
            except Exception as e:
                logger.error(f"batch_processor_error: {e}")
                logger.debug(
                    "batch_processor_exception",
                    iteration=loop_iteration,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    exc_info=True,
                )
                await asyncio.sleep(0.1)  # Prevent tight loop on error
        
        logger.info("batch_processor_stopped")
        logger.debug("batch_processor_total_iterations", total_iterations=loop_iteration)
    
    async def _process_batch(self, batch: BatchedRequest) -> None:
        """Process a batch of requests."""
        start_time = time.time()
        
        logger.debug(
            "processing_batch_start",
            batch_id=batch.batch_id,
            num_requests=len(batch.requests),
            total_pairs=batch.total_pairs,
            request_ids=[req.request_id for req in batch.requests],
        )
        
        try:
            # Update request status
            for req in batch.requests:
                req.status = RequestStatus.PROCESSING
                logger.debug(
                    "request_status_updated",
                    request_id=req.request_id,
                    status="PROCESSING",
                )
            
            # Run inference in thread pool
            logger.debug("submitting_to_thread_pool", batch_id=batch.batch_id)
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(
                self._executor,
                self._inference_batch_sync,
                batch,
            )
            
            # Distribute results
            processing_time = time.time() - start_time
            self._total_inference_time += processing_time
            
            logger.debug(
                "distributing_results",
                batch_id=batch.batch_id,
                num_results=len(results),
                processing_time=processing_time,
            )
            
            for request, result_data in zip(batch.requests, results):
                result = RerankResult(
                    request_id=request.request_id,
                    results=result_data,
                    processing_time=processing_time / len(batch.requests),
                )
                logger.debug(
                    "completing_request",
                    request_id=request.request_id,
                    num_results=len(result_data),
                    top_score=result_data[0]["relevance_score"] if result_data else None,
                )
                self.request_queue.complete_request(request.request_id, result)
            
            logger.debug(
                "batch_completed",
                batch_id=batch.batch_id,
                processing_time=processing_time,
                avg_time_per_request=processing_time / len(batch.requests),
            )
            
        except Exception as e:
            logger.error(f"batch_processing_failed: {e}")
            logger.debug(
                "batch_processing_exception",
                batch_id=batch.batch_id,
                error_type=type(e).__name__,
                error_message=str(e),
                request_ids=[req.request_id for req in batch.requests],
                exc_info=True,
            )
            for req in batch.requests:
                logger.debug("failing_request", request_id=req.request_id, error=str(e))
                self.request_queue.fail_request(
                    req.request_id,
                    str(e),
                )
    
    def _inference_batch_sync(
        self,
        batch: BatchedRequest,
    ) -> List[List[Dict[str, Any]]]:
        """
        Synchronous batch inference (runs in thread pool).
        
        Processes all requests in the batch together for efficiency.
        """
        import numpy as np
        import threading
        
        logger.debug(
            "inference_batch_sync_start",
            batch_id=batch.batch_id,
            thread_id=threading.current_thread().name,
            num_requests=len(batch.requests),
        )
        
        all_results = []
        
        # Process each request (could be further optimized with mega-batching)
        for req_idx, request in enumerate(batch.requests):
            # Create query-document pairs
            pairs = [[request.query, doc] for doc in request.documents]
            
            logger.debug(
                "processing_request_in_batch",
                batch_id=batch.batch_id,
                request_id=request.request_id,
                request_index=req_idx,
                num_pairs=len(pairs),
                query_preview=request.query[:100] if request.query else None,
            )
            
            try:
                # Get scores
                inference_start = time.time()
                scores = self._model.predict(
                    pairs,
                    batch_size=settings.batch_size,
                    show_progress_bar=False,
                )
                inference_time = time.time() - inference_start
                
                logger.debug(
                    "model_predict_complete",
                    request_id=request.request_id,
                    num_scores=len(scores),
                    inference_time=inference_time,
                    raw_scores_sample=list(scores[:3]) if len(scores) > 0 else [],
                )
                
                # Normalize if configured
                if settings.normalize_scores:
                    scores_array = np.array(scores)
                    scores = (1 / (1 + np.exp(-scores_array))).tolist()
                    logger.debug(
                        "scores_normalized",
                        request_id=request.request_id,
                        normalized_scores_sample=scores[:3] if len(scores) > 0 else [],
                    )
                
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
                    logger.debug(
                        "top_k_applied",
                        request_id=request.request_id,
                        top_k=request.top_k,
                        results_count=len(results),
                    )
                
                all_results.append(results)
                
                logger.debug(
                    "request_inference_complete",
                    request_id=request.request_id,
                    num_results=len(results),
                    top_score=results[0]["relevance_score"] if results else None,
                    bottom_score=results[-1]["relevance_score"] if results else None,
                )
                
            except Exception as e:
                logger.debug(
                    "inference_exception",
                    request_id=request.request_id,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    device=self.device,
                )
                # Fallback for MPS issues
                if self.device == "mps" and settings.mps_fallback_to_cpu:
                    logger.warning(f"MPS failed, falling back to CPU: {e}")
                    self.device = "cpu"
                    self._model = None
                    # Re-raise to trigger retry
                    raise
                else:
                    raise
        
        logger.debug(
            "inference_batch_sync_complete",
            batch_id=batch.batch_id,
            total_results=len(all_results),
        )
        
        return all_results
    
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        return_documents: bool = True,
        request_id: Optional[str] = None,
        priority: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Async rerank interface.
        
        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Number of top results
            return_documents: Include document text
            request_id: Optional request ID
            priority: Request priority (higher = more urgent)
            
        Returns:
            List of rerank results
        """
        rerank_start_time = time.time()
        
        logger.debug(
            "rerank_request_received",
            request_id=request_id,
            query_length=len(query),
            query_preview=query[:100] if query else None,
            num_documents=len(documents),
            top_k=top_k,
            return_documents=return_documents,
            priority=priority,
        )
        
        if not self._running:
            logger.debug("rerank_rejected_engine_not_running", request_id=request_id)
            raise RuntimeError("Engine is not running")
        
        if not documents:
            logger.debug("rerank_empty_documents", request_id=request_id)
            return []
        
        # Create request
        request = RerankRequest(
            request_id=request_id or "",
            query=query,
            documents=documents,
            top_k=top_k,
            return_documents=return_documents,
            priority=priority,
        )
        
        logger.debug(
            "rerank_request_created",
            request_id=request.request_id,
            arrival_time=request.arrival_time,
        )
        
        self._total_requests += 1
        
        # Add to queue and wait for result
        logger.debug("adding_request_to_queue", request_id=request.request_id)
        result_future = await self.request_queue.add_request(request)
        logger.debug("request_added_to_queue", request_id=request.request_id)
        
        try:
            logger.debug("waiting_for_result", request_id=request.request_id)
            result: RerankResult = await result_future
            
            total_time = time.time() - rerank_start_time
            logger.debug(
                "rerank_request_complete",
                request_id=request.request_id,
                total_time=total_time,
                processing_time=result.processing_time,
                queue_wait_time=total_time - result.processing_time,
                num_results=len(result.results),
            )
            
            return result.results
        except asyncio.CancelledError:
            logger.debug("rerank_request_cancelled", request_id=request.request_id)
            await self.request_queue.cancel_request(request.request_id)
            raise
    
    @property
    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._running
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        uptime = time.time() - self._start_time
        return {
            "running": self._running,
            "model_loaded": self.is_loaded,
            "model": self.model_name_or_path,
            "device": self.device,
            "uptime_seconds": uptime,
            "total_requests": self._total_requests,
            "total_inference_time": self._total_inference_time,
            "avg_inference_time": (
                self._total_inference_time / self._total_requests
                if self._total_requests > 0 else 0
            ),
            "requests_per_second": (
                self._total_requests / uptime if uptime > 0 else 0
            ),
            **self.request_queue.get_stats(),
        }


# Singleton instance
_engine_instance: Optional[AsyncRerankerEngine] = None
_engine_lock = asyncio.Lock()


async def get_async_engine() -> AsyncRerankerEngine:
    """Get or create the singleton async engine instance."""
    global _engine_instance
    
    if _engine_instance is None:
        async with _engine_lock:
            if _engine_instance is None:
                _engine_instance = AsyncRerankerEngine(
                    max_concurrent_batches=settings.max_concurrent_batches,
                    inference_threads=settings.inference_threads,
                    max_batch_size=settings.max_batch_size,
                    max_batch_pairs=settings.max_batch_pairs,
                    batch_wait_timeout=settings.batch_wait_timeout,
                    max_queue_size=settings.max_queue_size,
                    request_timeout=settings.request_timeout,
                )
                await _engine_instance.start()
    
    return _engine_instance


async def reset_async_engine() -> None:
    """Reset the singleton instance."""
    global _engine_instance
    
    if _engine_instance is not None:
        async with _engine_lock:
            if _engine_instance is not None:
                await _engine_instance.stop()
                _engine_instance = None
