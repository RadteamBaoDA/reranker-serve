#!/usr/bin/env python3
"""
Benchmark script for testing concurrent request handling.

Sends N parallel rerank requests, optionally logging each request's start
and completion in detail. While the benchmark runs, a background task polls
the server's /stats endpoint and prints batch occupancy / inflight / queue
depth at a configurable interval, so you can watch server-side batching
behavior in real time.

Usage:
    python benchmark_concurrent.py --url http://localhost:8000 \
        --requests 100 --concurrency 10 --verbose --stats-interval 2
"""

import argparse
import asyncio
import statistics
import time
from typing import Any, Dict, List, Optional

import httpx


# ---------- formatting helpers ----------

def _ts(start: float) -> str:
    return f"t={time.perf_counter() - start:6.2f}s"


def _fmt(v: Any, spec: str = "{:.2f}") -> str:
    return spec.format(v) if isinstance(v, (int, float)) else str(v)


# ---------- request / response ----------

async def send_request(
    client: httpx.AsyncClient,
    url: str,
    query: str,
    documents: List[str],
    request_id: int,
    inflight_counter: Optional[List[int]],
    verbose: bool,
    bench_start: float,
) -> Dict[str, Any]:
    """Send a single rerank request, optionally logging start/finish."""
    if inflight_counter is not None:
        inflight_counter[0] += 1
        inflight_now = inflight_counter[0]
    else:
        inflight_now = -1

    if verbose:
        print(
            f"[{_ts(bench_start)}] REQ #{request_id:04d} start  | "
            f"docs={len(documents)} | inflight={inflight_now}"
        )

    start = time.perf_counter()
    try:
        response = await client.post(
            f"{url}/rerank",
            json={"query": query, "documents": documents, "top_n": 3},
            timeout=60.0,
        )
        latency = time.perf_counter() - start
        ok = response.status_code == 200
        body: Optional[Dict[str, Any]] = None
        if ok:
            try:
                body = response.json()
            except Exception:
                body = None

        top_score = None
        if ok and body and isinstance(body.get("results"), list) and body["results"]:
            top_score = body["results"][0].get("relevance_score")

        retry_after = response.headers.get("Retry-After")
        if verbose:
            extra = ""
            if top_score is not None:
                extra += f" | top_score={top_score:.4f}"
            if retry_after is not None:
                extra += f" | retry_after={retry_after}"
            print(
                f"[{_ts(bench_start)}] REQ #{request_id:04d} done   | "
                f"status={response.status_code} | latency={latency * 1000:7.1f}ms"
                f"{extra}"
            )

        return {
            "request_id": request_id,
            "status_code": response.status_code,
            "latency": latency,
            "success": ok,
            "top_score": top_score,
            "retry_after": retry_after,
            "error": None if ok else response.text[:200],
        }
    except Exception as e:
        latency = time.perf_counter() - start
        if verbose:
            print(
                f"[{_ts(bench_start)}] REQ #{request_id:04d} FAIL   | "
                f"latency={latency * 1000:7.1f}ms | error={str(e)[:100]}"
            )
        return {
            "request_id": request_id,
            "status_code": 0,
            "latency": latency,
            "success": False,
            "top_score": None,
            "retry_after": None,
            "error": str(e),
        }
    finally:
        if inflight_counter is not None:
            inflight_counter[0] -= 1


# ---------- background stats poller ----------

async def stats_poller(
    url: str,
    interval: float,
    bench_start: float,
    stop_event: asyncio.Event,
) -> None:
    """Poll /stats every `interval` seconds while the benchmark runs."""
    async with httpx.AsyncClient() as client:
        while not stop_event.is_set():
            try:
                resp = await client.get(f"{url}/stats", timeout=2.0)
                if resp.status_code == 200:
                    payload = resp.json()
                    s = payload.get("stats", payload)
                    print(
                        f"[{_ts(bench_start)}] STATS | "
                        f"batches={s.get('total_batches', '?')} "
                        f"avg_batch={_fmt(s.get('avg_batch_size'))} "
                        f"occupancy={_fmt(s.get('batch_occupancy_pct'))}% "
                        f"pending={s.get('pending_requests', '?')} "
                        f"inflight_batches={s.get('inflight_batches', '?')} "
                        f"queue_wait_p95={_fmt(s.get('queue_wait_p95_ms'))}ms "
                        f"inference_p95={_fmt(s.get('inference_latency_p95_ms'))}ms"
                    )
            except Exception:
                # /stats may not exist (old server); silent skip.
                pass
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue


# ---------- run ----------

async def run_benchmark(
    url: str,
    num_requests: int,
    concurrency: int,
    documents_per_request: int,
    verbose: bool,
    stats_interval: float,
) -> Dict[str, Any]:
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
    print(f"   Verbose: {verbose}")
    print(f"   Stats poll interval: {stats_interval}s" if stats_interval > 0
          else f"   Stats poll: disabled")
    print()

    # Health check
    async with httpx.AsyncClient() as client:
        try:
            health = await client.get(f"{url}/health", timeout=5.0)
            if health.status_code != 200:
                print(f"❌ Server health check failed: {health.status_code}")
                return {}
            hd = health.json()
            print(f"✅ Server is healthy")
            print(f"   Model: {hd.get('model', 'unknown')}")
            print(f"   Device: {hd.get('device', 'unknown')}")
            print(f"   Engine Mode: {hd.get('engine_mode', 'unknown')}")
            print()
        except Exception as e:
            print(f"❌ Failed to connect to server: {e}")
            return {}

    bench_start = time.perf_counter()
    semaphore = asyncio.Semaphore(concurrency)
    inflight_counter: List[int] = [0]
    stop_event = asyncio.Event()

    async def bounded_request(client, request_id):
        async with semaphore:
            return await send_request(
                client, url, query, documents, request_id,
                inflight_counter, verbose, bench_start,
            )

    poller_task: Optional[asyncio.Task] = None
    if stats_interval > 0:
        poller_task = asyncio.create_task(
            stats_poller(url, stats_interval, bench_start, stop_event)
        )

    async with httpx.AsyncClient() as client:
        tasks = [bounded_request(client, i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)

    total_time = time.perf_counter() - bench_start

    stop_event.set()
    if poller_task is not None:
        try:
            await asyncio.wait_for(poller_task, timeout=stats_interval + 1)
        except asyncio.TimeoutError:
            poller_task.cancel()

    # Aggregate
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    latencies = [r["latency"] for r in successful]

    status_breakdown: Dict[int, int] = {}
    for r in results:
        status_breakdown[r["status_code"]] = status_breakdown.get(r["status_code"], 0) + 1

    stats = {
        "total_requests": num_requests,
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "total_time": total_time,
        "requests_per_second": num_requests / total_time,
        "success_rate": len(successful) / num_requests * 100,
        "status_breakdown": status_breakdown,
    }
    if latencies:
        sorted_lat = sorted(latencies)
        stats.update({
            "avg_latency": statistics.mean(latencies),
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "median_latency": statistics.median(latencies),
            "p95_latency": sorted_lat[int(len(sorted_lat) * 0.95)] if len(sorted_lat) > 1 else sorted_lat[0],
            "p99_latency": sorted_lat[int(len(sorted_lat) * 0.99)] if len(sorted_lat) > 1 else sorted_lat[0],
        })

    # Print
    print()
    print("=" * 60)
    print("📊 BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\n🎯 Throughput:")
    print(f"   Requests/second: {stats['requests_per_second']:.2f}")
    print(f"   Total time:      {total_time:.2f}s")
    print(f"\n✅ Success Rate:  {stats['success_rate']:.1f}%")
    print(f"   Successful:     {len(successful)}")
    print(f"   Failed:         {len(failed)}")
    print(f"   Status codes:   {dict(sorted(status_breakdown.items()))}")
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
        for r in failed[:5]:
            print(f"   Request {r['request_id']} (status={r['status_code']}): {r['error'][:120]}")
    print("\n" + "=" * 60)

    # Final engine stats
    try:
        async with httpx.AsyncClient() as client:
            sr = await client.get(f"{url}/stats", timeout=5.0)
            if sr.status_code == 200:
                payload = sr.json()
                ed = payload.get("stats", payload)

                def _show(label: str, key: str, spec: str = "{:.2f}"):
                    v = ed.get(key)
                    if v is None:
                        return
                    if isinstance(v, (int, float)):
                        print(f"   {label}: {spec.format(v)}")
                    else:
                        print(f"   {label}: {v}")

                print("\n📈 Final Engine Statistics:")
                _show("Total Requests Processed", "total_requests", "{}")
                _show("Pending Requests", "pending_requests", "{}")
                _show("Total Batches", "total_batches", "{}")
                _show("Avg Batch Size", "avg_batch_size")
                _show("Batch Occupancy %", "batch_occupancy_pct")
                _show("Queue Wait p50 (ms)", "queue_wait_p50_ms")
                _show("Queue Wait p95 (ms)", "queue_wait_p95_ms")
                _show("Inference Latency p50 (ms)", "inference_latency_p50_ms")
                _show("Inference Latency p95 (ms)", "inference_latency_p95_ms")
                _show("Throughput (pairs/sec)", "throughput_pairs_per_sec")
                _show("In-flight Batches", "inflight_batches", "{}")
                _show("Semaphore Available", "semaphore_available", "{}")
    except Exception:
        pass

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark concurrent reranker requests",
    )
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Base URL of the reranker service")
    parser.add_argument("--requests", type=int, default=100,
                        help="Total number of requests to send")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Maximum concurrent in-flight requests")
    parser.add_argument("--documents", type=int, default=5,
                        help="Number of documents per request")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Log every request start and completion")
    parser.add_argument("--stats-interval", type=float, default=2.0,
                        help="Poll /stats every N seconds during the run (0 to disable)")
    args = parser.parse_args()

    # Auto-verbose for small runs where the noise is helpful
    verbose = args.verbose or (args.requests <= 20 and args.concurrency <= 8)

    asyncio.run(run_benchmark(
        url=args.url,
        num_requests=args.requests,
        concurrency=args.concurrency,
        documents_per_request=args.documents,
        verbose=verbose,
        stats_interval=args.stats_interval,
    ))


if __name__ == "__main__":
    main()
