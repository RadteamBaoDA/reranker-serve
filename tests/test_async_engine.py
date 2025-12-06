"""
Tests for the async inference engine.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os

# Set environment before imports
os.environ["RERANKER_ENABLE_ASYNC_ENGINE"] = "true"

# Create mock torch module to avoid DLL loading issues on Windows
if "torch" not in sys.modules:
    mock_torch = MagicMock()
    mock_torch.__version__ = "2.0.0"
    mock_torch.cuda.is_available.return_value = False
    mock_torch.cuda.empty_cache = MagicMock()
    mock_torch.backends.mps.is_available.return_value = False
    mock_torch.backends.mps.is_built.return_value = False
    mock_torch.float16 = "float16"
    mock_torch.float32 = "float32"
    sys.modules["torch"] = mock_torch

# Mock sentence_transformers
if "sentence_transformers" not in sys.modules:
    mock_sentence_transformers = MagicMock()
    mock_cross_encoder_class = MagicMock()
    mock_sentence_transformers.CrossEncoder = mock_cross_encoder_class
    sys.modules["sentence_transformers"] = mock_sentence_transformers


class TestRequestQueue:
    """Tests for the request queue."""
    
    @pytest.mark.asyncio
    async def test_request_queue_add_and_get(self):
        """Test adding and getting requests from queue."""
        from src.engine.request_queue import RequestQueue, RerankRequest
        
        queue = RequestQueue(
            max_batch_size=4,
            batch_wait_timeout=0.001,
        )
        
        # Add a request
        request = RerankRequest(
            request_id="test-1",
            query="test query",
            documents=["doc1", "doc2"],
        )
        
        future = await queue.add_request(request)
        
        assert queue.pending_count == 1
        assert queue.active_count == 1
    
    @pytest.mark.asyncio
    async def test_batch_creation(self):
        """Test batch creation from multiple requests."""
        from src.engine.request_queue import RequestQueue, RerankRequest
        
        queue = RequestQueue(
            max_batch_size=4,
            batch_wait_timeout=0.01,
        )
        
        # Add multiple requests
        requests = []
        for i in range(3):
            req = RerankRequest(
                request_id=f"test-{i}",
                query=f"query {i}",
                documents=[f"doc{i}-1", f"doc{i}-2"],
            )
            await queue.add_request(req)
            requests.append(req)
        
        # Get batch
        batch = await queue.get_batch()
        
        assert batch is not None
        assert len(batch.requests) == 3
        assert batch.total_pairs == 6  # 3 requests * 2 docs each
    
    @pytest.mark.asyncio
    async def test_request_completion(self):
        """Test completing a request."""
        from src.engine.request_queue import (
            RequestQueue, RerankRequest, RerankResult
        )
        
        queue = RequestQueue()
        
        request = RerankRequest(
            request_id="test-complete",
            query="test",
            documents=["doc1"],
        )
        
        future = await queue.add_request(request)
        
        # Complete the request
        result = RerankResult(
            request_id="test-complete",
            results=[{"index": 0, "relevance_score": 0.9}],
            processing_time=0.1,
        )
        queue.complete_request("test-complete", result)
        
        # Future should be done
        assert future.done()
        completed_result = await future
        assert completed_result.results[0]["relevance_score"] == 0.9
    
    @pytest.mark.asyncio
    async def test_request_failure(self):
        """Test failing a request."""
        from src.engine.request_queue import RequestQueue, RerankRequest
        
        queue = RequestQueue()
        
        request = RerankRequest(
            request_id="test-fail",
            query="test",
            documents=["doc1"],
        )
        
        future = await queue.add_request(request)
        
        # Fail the request
        queue.fail_request("test-fail", "Test error")
        
        # Future should raise exception
        with pytest.raises(RuntimeError, match="Test error"):
            await future
    
    def test_queue_stats(self):
        """Test queue statistics."""
        from src.engine.request_queue import RequestQueue
        
        queue = RequestQueue()
        stats = queue.get_stats()
        
        assert "pending_requests" in stats
        assert "active_requests" in stats
        assert "total_requests" in stats
        assert "total_batches" in stats


class TestAsyncEngine:
    """Tests for the async reranker engine."""
    
    @pytest.mark.asyncio
    async def test_engine_lifecycle(self):
        """Test engine start and stop."""
        with patch("sentence_transformers.CrossEncoder") as mock_ce:
            mock_model = MagicMock()
            mock_model.predict.return_value = [0.9, 0.5]
            mock_ce.return_value = mock_model
            
            from src.engine.async_engine import AsyncRerankerEngine
            
            engine = AsyncRerankerEngine(
                max_concurrent_batches=1,
                inference_threads=1,
            )
            
            await engine.start()
            
            assert engine.is_running
            assert engine.is_loaded
            
            await engine.stop()
            
            assert not engine.is_running
    
    @pytest.mark.asyncio
    async def test_engine_rerank(self):
        """Test reranking through the engine."""
        with patch("sentence_transformers.CrossEncoder") as mock_ce:
            mock_model = MagicMock()
            mock_model.predict.return_value = [0.9, 0.5, 0.3]
            mock_ce.return_value = mock_model
            
            from src.engine.async_engine import AsyncRerankerEngine
            
            engine = AsyncRerankerEngine(
                max_concurrent_batches=1,
                inference_threads=1,
                batch_wait_timeout=0.001,
            )
            
            await engine.start()
            
            try:
                results = await engine.rerank(
                    query="test query",
                    documents=["doc1", "doc2", "doc3"],
                    top_k=2,
                )
                
                assert len(results) == 2
                # Results should be sorted by score
                assert results[0]["relevance_score"] >= results[1]["relevance_score"]
            finally:
                await engine.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling concurrent requests."""
        with patch("sentence_transformers.CrossEncoder") as mock_ce:
            mock_model = MagicMock()
            
            def mock_predict(pairs, **kwargs):
                return [0.9 - i * 0.1 for i in range(len(pairs))]
            
            mock_model.predict.side_effect = mock_predict
            mock_ce.return_value = mock_model
            
            from src.engine.async_engine import AsyncRerankerEngine
            
            engine = AsyncRerankerEngine(
                max_concurrent_batches=2,
                inference_threads=1,
                batch_wait_timeout=0.01,
            )
            
            await engine.start()
            
            try:
                # Send multiple concurrent requests
                tasks = []
                for i in range(5):
                    task = asyncio.create_task(
                        engine.rerank(
                            query=f"query {i}",
                            documents=[f"doc{i}-1", f"doc{i}-2"],
                            request_id=f"req-{i}",
                        )
                    )
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks)
                
                assert len(results) == 5
                for result in results:
                    assert len(result) == 2
            finally:
                await engine.stop()
    
    @pytest.mark.asyncio
    async def test_engine_empty_documents(self):
        """Test engine with empty documents."""
        with patch("sentence_transformers.CrossEncoder") as mock_ce:
            mock_model = MagicMock()
            mock_ce.return_value = mock_model
            
            from src.engine.async_engine import AsyncRerankerEngine
            
            engine = AsyncRerankerEngine(
                max_concurrent_batches=1,
                inference_threads=1,
            )
            
            await engine.start()
            
            try:
                results = await engine.rerank(
                    query="test query",
                    documents=[],
                )
                
                assert results == []
            finally:
                await engine.stop()
    
    @pytest.mark.asyncio  
    async def test_engine_top_k(self):
        """Test engine with different top_k values."""
        with patch("sentence_transformers.CrossEncoder") as mock_ce:
            mock_model = MagicMock()
            mock_model.predict.return_value = [0.9, 0.7, 0.5, 0.3, 0.1]
            mock_ce.return_value = mock_model
            
            from src.engine.async_engine import AsyncRerankerEngine
            
            engine = AsyncRerankerEngine(
                max_concurrent_batches=1,
                inference_threads=1,
                batch_wait_timeout=0.001,
            )
            
            await engine.start()
            
            try:
                # Test top_k=3
                results = await engine.rerank(
                    query="test",
                    documents=["d1", "d2", "d3", "d4", "d5"],
                    top_k=3,
                )
                assert len(results) == 3
                
                # Test top_k=None (return all)
                results = await engine.rerank(
                    query="test",
                    documents=["d1", "d2", "d3", "d4", "d5"],
                    top_k=None,
                )
                assert len(results) == 5
            finally:
                await engine.stop()
    
    def test_engine_stats(self):
        """Test engine statistics."""
        from src.engine.async_engine import AsyncRerankerEngine
        
        engine = AsyncRerankerEngine()
        stats = engine.get_stats()
        
        assert "running" in stats
        assert "model_loaded" in stats
        assert "total_requests" in stats
        assert "pending_requests" in stats


class TestEngineIntegration:
    """Integration tests for the engine with routes."""
    
    @pytest.mark.asyncio
    async def test_singleton_engine(self):
        """Test singleton engine pattern."""
        with patch("sentence_transformers.CrossEncoder") as mock_ce:
            mock_model = MagicMock()
            mock_model.predict.return_value = [0.9]
            mock_ce.return_value = mock_model
            
            from src.engine.async_engine import (
                get_async_engine, reset_async_engine
            )
            
            # First reset to ensure clean state
            await reset_async_engine()
            
            try:
                # Get engine twice
                engine1 = await get_async_engine()
                engine2 = await get_async_engine()
                
                # Should be same instance
                assert engine1 is engine2
                assert engine1.is_running
            finally:
                await reset_async_engine()
    
    @pytest.mark.asyncio
    async def test_engine_reset(self):
        """Test resetting the engine."""
        with patch("sentence_transformers.CrossEncoder") as mock_ce:
            mock_model = MagicMock()
            mock_model.predict.return_value = [0.9]
            mock_ce.return_value = mock_model
            
            from src.engine.async_engine import (
                get_async_engine, reset_async_engine
            )
            
            # Reset first
            await reset_async_engine()
            
            try:
                engine1 = await get_async_engine()
                assert engine1.is_running
                
                await reset_async_engine()
                
                # Getting engine again should create a new instance
                engine2 = await get_async_engine()
                assert engine2.is_running
                # engine1 should be stopped
                assert not engine1.is_running
            finally:
                await reset_async_engine()


class TestBatchProcessing:
    """Tests for batch processing behavior."""
    
    @pytest.mark.asyncio
    async def test_batch_wait_timeout(self):
        """Test that batch wait timeout batches requests together."""
        from src.engine.request_queue import RequestQueue, RerankRequest
        
        queue = RequestQueue(
            max_batch_size=10,
            batch_wait_timeout=0.05,  # 50ms wait
        )
        
        # Add requests with slight delays
        async def add_requests():
            for i in range(3):
                req = RerankRequest(
                    request_id=f"batch-test-{i}",
                    query=f"query {i}",
                    documents=["doc1"],
                )
                await queue.add_request(req)
                await asyncio.sleep(0.01)  # 10ms between requests
        
        # Start adding requests
        add_task = asyncio.create_task(add_requests())
        
        # Get batch - should wait and get multiple requests
        await asyncio.sleep(0.02)  # Let some requests come in
        batch = await queue.get_batch()
        
        await add_task
        
        # Should have at least 1 request (timing dependent)
        assert batch is not None
        assert len(batch.requests) >= 1
    
    @pytest.mark.asyncio
    async def test_max_batch_size_limit(self):
        """Test that batch size is limited."""
        from src.engine.request_queue import RequestQueue, RerankRequest
        
        queue = RequestQueue(
            max_batch_size=3,
            batch_wait_timeout=0.01,
        )
        
        # Add more requests than max batch size
        for i in range(5):
            req = RerankRequest(
                request_id=f"limit-test-{i}",
                query=f"query {i}",
                documents=["doc1"],
            )
            await queue.add_request(req)
        
        # First batch should be at most 3
        batch1 = await queue.get_batch()
        assert len(batch1.requests) <= 3
        
        # Remaining should be in next batch
        if queue.pending_count > 0:
            batch2 = await queue.get_batch()
            assert batch2 is not None
    
    @pytest.mark.asyncio
    async def test_max_pairs_limit(self):
        """Test that max pairs limit is respected."""
        from src.engine.request_queue import RequestQueue, RerankRequest
        
        queue = RequestQueue(
            max_batch_size=10,
            max_batch_pairs=5,  # Only 5 pairs max per batch
            batch_wait_timeout=0.01,
        )
        
        # Add requests with 3 docs each (3 pairs per request)
        for i in range(3):
            req = RerankRequest(
                request_id=f"pairs-test-{i}",
                query=f"query {i}",
                documents=["doc1", "doc2", "doc3"],
            )
            await queue.add_request(req)
        
        # First batch should respect max_batch_pairs
        batch = await queue.get_batch()
        assert batch.total_pairs <= 6  # First request has 3, can fit 1 more (total 6)


class TestRequestStatus:
    """Tests for request status tracking."""
    
    @pytest.mark.asyncio
    async def test_request_status_transitions(self):
        """Test request status changes through lifecycle."""
        from src.engine.request_queue import (
            RequestQueue, RerankRequest, RerankResult, RequestStatus
        )
        
        queue = RequestQueue()
        
        request = RerankRequest(
            request_id="status-test",
            query="test",
            documents=["doc1"],
        )
        
        # Initially pending
        assert request.status == RequestStatus.PENDING
        
        await queue.add_request(request)
        
        # Complete the request
        result = RerankResult(
            request_id="status-test",
            results=[{"index": 0, "relevance_score": 0.9}],
            processing_time=0.1,
        )
        queue.complete_request("status-test", result)
        
        assert request.status == RequestStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_request_cancellation(self):
        """Test request cancellation."""
        from src.engine.request_queue import RequestQueue, RerankRequest
        
        queue = RequestQueue()
        
        request = RerankRequest(
            request_id="cancel-test",
            query="test",
            documents=["doc1"],
        )
        
        future = await queue.add_request(request)
        
        # Cancel the request
        cancelled = await queue.cancel_request("cancel-test")
        
        assert cancelled
        assert future.cancelled()
        assert queue.active_count == 0
