"""
Async Reranker Engine with pluggable model handlers.
Supports concurrent batching, thread-pooled inference, and model-specific
handlers (e.g., CrossEncoder, Qwen3).
"""

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from src.config import settings, get_logger
from src.engine.request_queue import (
    RequestQueue,
    RerankRequest,
    RerankResult,
    BatchedRequest,
    RequestStatus,
)
from src.engine.handlers.base import BaseHandler
from src.engine.handlers.factory import get_handler

logger = get_logger(__name__)


class AsyncRerankerEngine:
    """High-performance async reranker engine with batching."""

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        max_length: Optional[int] = None,
        use_fp16: Optional[bool] = None,
        # Concurrency settings
        max_concurrent_batches: int = None,
        inference_threads: int = None,
        max_batch_size: int = None,
        max_batch_pairs: int = None,
        batch_wait_timeout: Optional[float] = None,
        max_queue_size: Optional[int] = None,
        request_timeout: Optional[float] = None,
    ):
        self.model_name_or_path = model_name_or_path or settings.get_model_load_path()
        self.device = device or settings.get_device()
        self.max_length = max_length or settings.max_length
        self.use_fp16 = use_fp16 if use_fp16 is not None else settings.use_fp16

        self.max_concurrent_batches = max_concurrent_batches or settings.max_concurrent_batches
        self.inference_threads = inference_threads or settings.inference_threads

        self.request_queue = RequestQueue(
            max_batch_size=max_batch_size or settings.max_batch_size,
            max_batch_pairs=max_batch_pairs or settings.max_batch_pairs,
            batch_wait_timeout=batch_wait_timeout if batch_wait_timeout is not None else settings.batch_wait_timeout,
            max_queue_size=max_queue_size or settings.max_queue_size,
            request_timeout=request_timeout if request_timeout is not None else settings.request_timeout,
        )

        self._executor = ThreadPoolExecutor(
            max_workers=self.inference_threads,
            thread_name_prefix="reranker-inference",
        )

        self._handler: Optional[BaseHandler] = None
        self._model_lock = asyncio.Lock()

        self._processor_task: Optional[asyncio.Task] = None
        self._running = False
        self._batch_semaphore = asyncio.Semaphore(self.max_concurrent_batches)

        self._total_requests = 0
        self._total_inference_time = 0.0
        self._start_time = time.time()

        logger.info(
            "async_engine_created",
            model=self.model_name_or_path,
            device=self.device,
            max_concurrent_batches=self.max_concurrent_batches,
            inference_threads=self.inference_threads,
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

        await self._load_model()

        self._running = True
        self._processor_task = asyncio.create_task(
            self._batch_processor_loop(),
            name="reranker-batch-processor",
        )

        logger.info("async_engine_started")
        return self

    async def stop(self) -> None:
        """Stop the engine gracefully."""
        if not self._running:
            return

        logger.info("stopping_async_engine")
        self._running = False
        self.request_queue.shutdown()

        if self._processor_task:
            try:
                await asyncio.wait_for(self._processor_task, timeout=10.0)
            except asyncio.TimeoutError:
                self._processor_task.cancel()
                try:
                    await self._processor_task
                except asyncio.CancelledError:
                    pass

        self._executor.shutdown(wait=True)
        self._unload_model()
        logger.info("async_engine_stopped")

    async def _load_model(self) -> None:
        """Load the model (thread-safe)."""
        logger.debug("acquiring_model_lock")
        async with self._model_lock:
            if self._handler is not None:
                logger.debug("model_already_loaded", model=self.model_name_or_path)
                return

            logger.info(f"Loading reranker model: {self.model_name_or_path}")
            load_start_time = time.time()

            loop = asyncio.get_running_loop()
            self._handler = await loop.run_in_executor(
                self._executor,
                self._load_model_sync,
            )

            load_duration = time.time() - load_start_time
            logger.info("Model loaded successfully")
            logger.debug(
                "model_load_complete",
                load_time_seconds=load_duration,
                model_type=type(self._handler).__name__,
            )

    def _prepare_environment(self) -> None:
        """Set environment variables before model load."""
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

        if settings.use_offline_mode:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        if settings.model_cache_dir:
            cache_dir = os.path.abspath(settings.model_cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            os.environ["HF_HOME"] = cache_dir
            os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir

        if self.device == "mps":
            if settings.mps_fallback_to_cpu:
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    def _load_model_sync(self) -> BaseHandler:
        """Blocking model load executed in a thread pool."""
        self._prepare_environment()
        model_source = self._get_model_source()
        handler = get_handler(
            model_path=model_source,
            device=self.device,
            max_length=self.max_length,
            use_fp16=self.use_fp16,
        )
        handler.load_model()
        return handler

    def _get_model_source(self) -> str:
        """Determine path or identifier to load the model from."""
        if settings.model_path and self._is_valid_local_model(settings.model_path):
            return settings.model_path

        cache_path = self._get_cache_path(settings.model_name)
        if self._is_valid_local_model(cache_path):
            return cache_path

        if os.path.isdir(self.model_name_or_path) and self._is_valid_local_model(self.model_name_or_path):
            return self.model_name_or_path

        if settings.use_offline_mode:
            raise RuntimeError("Model not found locally and offline mode is enabled.")

        return settings.model_name

    def _is_valid_local_model(self, path: str) -> bool:
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
            model_name.replace("/", "_").replace("\\", "_"),
        )

    def _unload_model(self) -> None:
        if self._handler is not None:
            self._handler.unload()
            self._handler = None

    async def _batch_processor_loop(self) -> None:
        logger.info("batch_processor_started")
        loop_iteration = 0

        while self._running or self.request_queue.active_count > 0:
            loop_iteration += 1
            try:
                batch = await self.request_queue.get_batch()

                if batch is None:
                    logger.debug("no_batch_available", iteration=loop_iteration)
                    continue

                async with self._batch_semaphore:
                    await self._process_batch(batch)

            except asyncio.CancelledError:
                logger.info("batch_processor_cancelled")
                break
            except Exception as e:
                logger.error("batch_processor_error", error=str(e))
                await asyncio.sleep(0.1)

        logger.info("batch_processor_stopped")

    async def _process_batch(self, batch: BatchedRequest) -> None:
        start_time = time.time()

        try:
            for req in batch.requests:
                req.status = RequestStatus.PROCESSING

            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(
                self._executor,
                self._inference_batch_sync,
                batch,
            )

            processing_time = time.time() - start_time
            self._total_inference_time += processing_time

            for request, result_data in zip(batch.requests, results):
                result = RerankResult(
                    request_id=request.request_id,
                    results=result_data,
                    processing_time=processing_time / len(batch.requests),
                )
                self.request_queue.complete_request(request.request_id, result)

        except Exception as e:
            logger.error("batch_processing_failed", error=str(e))
            for req in batch.requests:
                self.request_queue.fail_request(req.request_id, str(e))

    def _inference_batch_sync(self, batch: BatchedRequest) -> List[List[Dict[str, Any]]]:
        if not self._handler:
            raise RuntimeError("Model handler is not loaded")

        return self._handler.predict(batch)

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        return_documents: bool = True,
        request_id: Optional[str] = None,
        priority: int = 0,
    ) -> List[Dict[str, Any]]:
        rerank_start_time = time.time()

        if not self._running:
            raise RuntimeError("Engine is not running")

        if not documents:
            return []

        request = RerankRequest(
            request_id=request_id or "",
            query=query,
            documents=documents,
            top_k=top_k,
            return_documents=return_documents,
            priority=priority,
        )

        self._total_requests += 1

        result_future = await self.request_queue.add_request(request)

        try:
            result: RerankResult = await result_future
            total_time = time.time() - rerank_start_time
            logger.debug(
                "rerank_request_complete",
                request_id=request.request_id,
                total_time=total_time,
                processing_time=result.processing_time,
                num_results=len(result.results),
            )
            return result.results
        except asyncio.CancelledError:
            await self.request_queue.cancel_request(request.request_id)
            raise

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_loaded(self) -> bool:
        return self._handler is not None

    def get_stats(self) -> Dict[str, Any]:
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
    """Get or create the singleton async engine instance (auto-starts)."""
    global _engine_instance

    if _engine_instance is None:
        async with _engine_lock:
            if _engine_instance is None:
                _engine_instance = AsyncRerankerEngine()
                await _engine_instance.start()
    elif not _engine_instance.is_running:
        async with _engine_lock:
            if not _engine_instance.is_running:
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
