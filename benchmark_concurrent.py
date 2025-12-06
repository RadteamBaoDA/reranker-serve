#!/usr/bin/env python3
"""
Benchmark script for testing concurrent request handling.
Tests the async engine performance with parallel requests.

Usage:
    python benchmark_concurrent.py --url http://localhost:8000 --requests 100 --concurrency 10
"""

import argparse
import asyncio
import time
import statistics
from typing import List, Dict, Any

import httpx


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    query: str,
    documents: List[str],
    request_id: int,
) -> Dict[str, Any]:
    """Send a single rerank request."""
    start_time = time.perf_counter()
    
    try:
        response = await client.post(
            f"{url}/rerank",
            json={
                "query": query,
                "documents": documents,
                "top_n": 3,
            },
            timeout=60.0,
        )
        
        latency = time.perf_counter() - start_time
        
        return {
            "request_id": request_id,
            "status_code": response.status_code,
            "latency": latency,
            "success": response.status_code == 200,
            "error": None if response.status_code == 200 else response.text,
        }
    except Exception as e:
        latency = time.perf_counter() - start_time
        return {
            "request_id": request_id,
            "status_code": 0,
            "latency": latency,
            "success": False,
            "error": str(e),
        }


async def run_benchmark(
    url: str,
    num_requests: int,
    concurrency: int,
    documents_per_request: int,
) -> Dict[str, Any]:
    """Run the benchmark with specified parameters."""
    
    # Sample query and documents
    query = "What is machine learning?"
    base_documents = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Deep learning uses neural networks with multiple layers to learn representations.",
        "The weather is sunny today with clear skies.",
        "Natural language processing helps computers understand human language.",
        "Python is a popular programming language for data science.",
        "Transformers have revolutionized NLP with attention mechanisms.",
        "Random forests are ensemble learning methods for classification.",
        "Gradient descent is an optimization algorithm used in machine learning.",
        "Convolutional neural networks are commonly used for image recognition.",
        "Reinforcement learning trains agents through reward-based feedback.",
    ]
    
    documents = base_documents[:documents_per_request]
    
    print(f"\n🚀 Starting benchmark:")
    print(f"   URL: {url}")
    print(f"   Total Requests: {num_requests}")
    print(f"   Concurrency: {concurrency}")
    print(f"   Documents per request: {documents_per_request}")
    print()
    
    # Check server is up
    async with httpx.AsyncClient() as client:
        try:
            health = await client.get(f"{url}/health", timeout=5.0)
            if health.status_code != 200:
                print(f"❌ Server health check failed: {health.status_code}")
                return {}
            
            health_data = health.json()
            print(f"✅ Server is healthy")
            print(f"   Model: {health_data.get('model', 'unknown')}")
            print(f"   Device: {health_data.get('device', 'unknown')}")
            print(f"   Engine Mode: {health_data.get('engine_mode', 'unknown')}")
            print()
        except Exception as e:
            print(f"❌ Failed to connect to server: {e}")
            return {}
    
    # Run benchmark
    start_time = time.perf_counter()
    semaphore = asyncio.Semaphore(concurrency)
    
    async def bounded_request(client, request_id):
        async with semaphore:
            return await send_request(client, url, query, documents, request_id)
    
    async with httpx.AsyncClient() as client:
        tasks = [
            bounded_request(client, i)
            for i in range(num_requests)
        ]
        
        results = await asyncio.gather(*tasks)
    
    total_time = time.perf_counter() - start_time
    
    # Analyze results
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    latencies = [r["latency"] for r in successful]
    
    stats = {
        "total_requests": num_requests,
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "total_time": total_time,
        "requests_per_second": num_requests / total_time,
        "success_rate": len(successful) / num_requests * 100,
    }
    
    if latencies:
        stats.update({
            "avg_latency": statistics.mean(latencies),
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "median_latency": statistics.median(latencies),
            "p95_latency": (
                sorted(latencies)[int(len(latencies) * 0.95)]
                if len(latencies) > 1 else latencies[0]
            ),
            "p99_latency": (
                sorted(latencies)[int(len(latencies) * 0.99)]
                if len(latencies) > 1 else latencies[0]
            ),
        })
    
    # Print results
    print("=" * 60)
    print("📊 BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\n🎯 Throughput:")
    print(f"   Requests/second: {stats['requests_per_second']:.2f}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"\n✅ Success Rate: {stats['success_rate']:.1f}%")
    print(f"   Successful: {len(successful)}")
    print(f"   Failed: {len(failed)}")
    
    if latencies:
        print(f"\n⏱️  Latency (seconds):")
        print(f"   Average: {stats['avg_latency']:.4f}")
        print(f"   Median:  {stats['median_latency']:.4f}")
        print(f"   Min:     {stats['min_latency']:.4f}")
        print(f"   Max:     {stats['max_latency']:.4f}")
        print(f"   P95:     {stats['p95_latency']:.4f}")
        print(f"   P99:     {stats['p99_latency']:.4f}")
    
    if failed:
        print(f"\n❌ Sample errors:")
        for r in failed[:3]:
            print(f"   Request {r['request_id']}: {r['error'][:100]}")
    
    print("\n" + "=" * 60)
    
    # Get engine stats if available
    try:
        async with httpx.AsyncClient() as client:
            stats_response = await client.get(f"{url}/stats", timeout=5.0)
            if stats_response.status_code == 200:
                engine_stats = stats_response.json()
                print("\n📈 Engine Statistics:")
                engine_data = engine_stats.get("stats", {})
                print(f"   Total Requests Processed: {engine_data.get('total_requests', 'N/A')}")
                print(f"   Pending Requests: {engine_data.get('pending_requests', 'N/A')}")
                print(f"   Total Batches: {engine_data.get('total_batches', 'N/A')}")
                print(f"   Avg Batch Size: {engine_data.get('avg_batch_size', 'N/A'):.2f}" if isinstance(engine_data.get('avg_batch_size'), (int, float)) else f"   Avg Batch Size: N/A")
    except Exception:
        pass
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark concurrent request handling"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the reranker service",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Total number of requests to send",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent requests",
    )
    parser.add_argument(
        "--documents",
        type=int,
        default=5,
        help="Number of documents per request",
    )
    
    args = parser.parse_args()
    
    asyncio.run(run_benchmark(
        url=args.url,
        num_requests=args.requests,
        concurrency=args.concurrency,
        documents_per_request=args.documents,
    ))


if __name__ == "__main__":
    main()
