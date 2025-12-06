"""
Request Queue and Batching System.
Inspired by vLLM's AsyncMicrobatchTokenizer for efficient request batching.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from concurrent.futures import Future
from enum import Enum

from src.config import get_logger

logger = get_logger(__name__)


class RequestStatus(Enum):
    """Status of a rerank request."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RerankRequest:
    """
    Represents a single rerank request in the queue.
    """
    request_id: str
    query: str
    documents: List[str]
    top_k: Optional[int] = None
    return_documents: bool = True
    arrival_time: float = field(default_factory=time.time)
    status: RequestStatus = RequestStatus.PENDING
    priority: int = 0  # Higher = more priority
    
    # Result handling
    result_future: Optional[asyncio.Future] = None
    
    def __post_init__(self):
        if not self.request_id:
            self.request_id = str(uuid.uuid4())


@dataclass
class RerankResult:
    """
    Result of a rerank request.
    """
    request_id: str
    results: List[Dict[str, Any]]
    processing_time: float
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class BatchedRequest:
    """
    A batch of requests to be processed together.
    """
    batch_id: str
    requests: List[RerankRequest]
    created_time: float = field(default_factory=time.time)
    
    @property
    def total_pairs(self) -> int:
        """Total number of query-document pairs in the batch."""
        return sum(len(r.documents) for r in self.requests)
    
    def __post_init__(self):
        if not self.batch_id:
            self.batch_id = str(uuid.uuid4())


class RequestQueue:
    """
    Async request queue with batching support.
    
    Features:
    - Priority-based scheduling
    - Dynamic batching based on batch_wait_timeout
    - Max batch size limits
    - Request timeout handling
    """
    
    def __init__(
        self,
        max_batch_size: int = 32,
        max_batch_pairs: int = 1024,  # Max query-doc pairs per batch
        batch_wait_timeout: float = 0.01,  # 10ms wait for batching
        max_queue_size: int = 1000,
        request_timeout: float = 60.0,
    ):
        self.max_batch_size = max_batch_size
        self.max_batch_pairs = max_batch_pairs
        self.batch_wait_timeout = batch_wait_timeout
        self.max_queue_size = max_queue_size
        self.request_timeout = request_timeout
        
        # Async queue for pending requests
        self._queue: asyncio.Queue[RerankRequest] = asyncio.Queue(maxsize=max_queue_size)
        
        # Track active requests
        self._active_requests: Dict[str, RerankRequest] = {}
        
        # Metrics
        self._total_requests = 0
        self._total_batches = 0
        self._total_processing_time = 0.0
        
        # Shutdown flag
        self._shutdown = False
        
        logger.info(
            "request_queue_initialized",
            max_batch_size=max_batch_size,
            max_batch_pairs=max_batch_pairs,
            batch_wait_timeout=batch_wait_timeout,
        )
        logger.debug(
            "request_queue_config_details",
            max_queue_size=max_queue_size,
            request_timeout=request_timeout,
        )
    
    async def add_request(self, request: RerankRequest) -> asyncio.Future:
        """
        Add a request to the queue and return a future for the result.
        """
        logger.debug(
            "add_request_start",
            request_id=request.request_id,
            query_length=len(request.query),
            num_documents=len(request.documents),
            top_k=request.top_k,
            return_documents=request.return_documents,
            priority=request.priority,
        )
        
        if self._shutdown:
            logger.debug(
                "add_request_rejected_shutdown",
                request_id=request.request_id,
            )
            raise RuntimeError("Queue is shutting down")
        
        # Create result future
        loop = asyncio.get_running_loop()
        request.result_future = loop.create_future()
        
        # Track request
        self._active_requests[request.request_id] = request
        self._total_requests += 1
        
        logger.debug(
            "request_tracked",
            request_id=request.request_id,
            total_active=len(self._active_requests),
            total_requests_count=self._total_requests,
        )
        
        # Add to queue
        try:
            queue_size_before = self._queue.qsize()
            await asyncio.wait_for(
                self._queue.put(request),
                timeout=self.request_timeout
            )
            logger.debug(
                "request_queued_success",
                request_id=request.request_id,
                queue_size_before=queue_size_before,
                queue_size_after=self._queue.qsize(),
            )
        except asyncio.TimeoutError:
            del self._active_requests[request.request_id]
            logger.debug(
                "request_queue_timeout",
                request_id=request.request_id,
                queue_size=self._queue.qsize(),
                timeout=self.request_timeout,
            )
            raise RuntimeError("Queue is full, request timed out")
        
        logger.debug(
            "request_added",
            request_id=request.request_id,
            documents=len(request.documents),
            queue_size=self._queue.qsize(),
        )
        
        return request.result_future
    
    async def get_batch(self) -> Optional[BatchedRequest]:
        """
        Get a batch of requests for processing.
        
        Uses dynamic batching: waits up to batch_wait_timeout to accumulate
        requests, then returns whatever is available.
        """
        logger.debug(
            "get_batch_start",
            queue_size=self._queue.qsize(),
            shutdown=self._shutdown,
        )
        
        if self._shutdown and self._queue.empty():
            logger.debug("get_batch_exit_shutdown_empty")
            return None
        
        requests: List[RerankRequest] = []
        total_pairs = 0
        deadline = asyncio.get_event_loop().time() + self.batch_wait_timeout
        
        # Get first request (blocking)
        try:
            wait_timeout = 1.0 if not self._shutdown else 0.1
            logger.debug(
                "get_batch_waiting_first",
                timeout=wait_timeout,
            )
            first_request = await asyncio.wait_for(
                self._queue.get(),
                timeout=wait_timeout
            )
            requests.append(first_request)
            total_pairs += len(first_request.documents)
            logger.debug(
                "get_batch_first_request",
                request_id=first_request.request_id,
                documents=len(first_request.documents),
            )
        except asyncio.TimeoutError:
            logger.debug("get_batch_timeout_no_requests")
            return None
        
        # Try to batch more requests within timeout
        logger.debug(
            "get_batch_accumulation_start",
            deadline=deadline,
            max_batch_size=self.max_batch_size,
            max_batch_pairs=self.max_batch_pairs,
        )
        
        while (
            len(requests) < self.max_batch_size
            and total_pairs < self.max_batch_pairs
        ):
            remaining_time = deadline - asyncio.get_event_loop().time()
            if remaining_time <= 0:
                logger.debug(
                    "get_batch_deadline_reached",
                    requests_count=len(requests),
                    total_pairs=total_pairs,
                )
                break
            
            try:
                request = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=remaining_time
                )
                
                # Check if adding this would exceed limits
                if total_pairs + len(request.documents) > self.max_batch_pairs:
                    # Put it back for next batch
                    await self._queue.put(request)
                    logger.debug(
                        "get_batch_request_deferred",
                        request_id=request.request_id,
                        request_pairs=len(request.documents),
                        current_total=total_pairs,
                        would_exceed=total_pairs + len(request.documents),
                    )
                    break
                
                requests.append(request)
                total_pairs += len(request.documents)
                logger.debug(
                    "get_batch_request_added",
                    request_id=request.request_id,
                    batch_size=len(requests),
                    total_pairs=total_pairs,
                )
                
            except asyncio.TimeoutError:
                logger.debug(
                    "get_batch_accumulation_timeout",
                    requests_count=len(requests),
                    remaining_time=remaining_time,
                )
                break
        
        if not requests:
            logger.debug("get_batch_no_requests_collected")
            return None
        
        self._total_batches += 1
        batch = BatchedRequest(
            batch_id=f"batch-{self._total_batches}",
            requests=requests,
        )
        
        logger.debug(
            "batch_created",
            batch_id=batch.batch_id,
            num_requests=len(requests),
            total_pairs=total_pairs,
            request_ids=[r.request_id for r in requests],
            avg_docs_per_request=total_pairs / len(requests) if requests else 0,
        )
        
        return batch
    
    def complete_request(
        self,
        request_id: str,
        result: RerankResult,
    ) -> None:
        """Complete a request with its result."""
        logger.debug(
            "complete_request_start",
            request_id=request_id,
            success=result.success,
            processing_time=result.processing_time,
            num_results=len(result.results),
            error=result.error,
        )
        
        request = self._active_requests.pop(request_id, None)
        if request and request.result_future and not request.result_future.done():
            if result.error:
                logger.debug(
                    "complete_request_set_exception",
                    request_id=request_id,
                    error=result.error,
                )
                request.result_future.set_exception(
                    RuntimeError(result.error)
                )
            else:
                logger.debug(
                    "complete_request_set_result",
                    request_id=request_id,
                    num_results=len(result.results),
                )
                request.result_future.set_result(result)
            request.status = RequestStatus.COMPLETED
            logger.debug(
                "complete_request_done",
                request_id=request_id,
                remaining_active=len(self._active_requests),
            )
        else:
            logger.debug(
                "complete_request_not_found_or_done",
                request_id=request_id,
                request_exists=request is not None,
                future_exists=request.result_future is not None if request else False,
            )
    
    def fail_request(self, request_id: str, error: str) -> None:
        """Mark a request as failed."""
        logger.debug(
            "fail_request",
            request_id=request_id,
            error=error,
        )
        result = RerankResult(
            request_id=request_id,
            results=[],
            processing_time=0.0,
            error=error,
        )
        self.complete_request(request_id, result)
    
    async def cancel_request(self, request_id: str) -> bool:
        """Cancel a pending request."""
        logger.debug(
            "cancel_request_start",
            request_id=request_id,
        )
        request = self._active_requests.get(request_id)
        if request:
            request.status = RequestStatus.CANCELLED
            if request.result_future and not request.result_future.done():
                request.result_future.cancel()
            del self._active_requests[request_id]
            logger.debug(
                "cancel_request_success",
                request_id=request_id,
            )
            return True
        logger.debug(
            "cancel_request_not_found",
            request_id=request_id,
        )
        return False
    
    def shutdown(self) -> None:
        """Signal shutdown of the queue."""
        logger.debug(
            "queue_shutdown_start",
            active_requests=len(self._active_requests),
            pending_in_queue=self._queue.qsize(),
        )
        self._shutdown = True
        
        # Cancel all pending requests
        pending_ids = list(self._active_requests.keys())
        for request_id in pending_ids:
            self.fail_request(request_id, "Queue shutdown")
        
        logger.debug(
            "queue_shutdown_complete",
            cancelled_count=len(pending_ids),
        )
    
    @property
    def pending_count(self) -> int:
        """Number of pending requests."""
        return self._queue.qsize()
    
    @property
    def active_count(self) -> int:
        """Number of active (queued + processing) requests."""
        return len(self._active_requests)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        stats = {
            "pending_requests": self.pending_count,
            "active_requests": self.active_count,
            "total_requests": self._total_requests,
            "total_batches": self._total_batches,
            "avg_batch_size": (
                self._total_requests / self._total_batches
                if self._total_batches > 0 else 0
            ),
        }
        logger.debug(
            "queue_stats_retrieved",
            **stats,
        )
        return stats
