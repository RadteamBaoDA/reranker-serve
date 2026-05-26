# Concurrency Model

This service serves many concurrent rerank requests on a single GPU through
**dynamic batch coalescing** — not continuous batching.

## Why not continuous batching?

vLLM-style continuous batching lets new requests join an already-running batch
mid-flight, which matters for autoregressive generation where each token is its
own step. A reranker is a **single forward pass** per `(query, document)` pair:
once a batch enters the GPU, it runs to completion and there is nothing to
"join." The right primitive for a reranker is:

- accumulate as many requests as possible within a short latency budget,
- launch one forward pass with the whole batch,
- begin accumulating the next batch in parallel with the running pass.

That is exactly what this service does.

## The pipeline

```
client request
    -> RequestQueue (async)
    -> get_batch() — accumulates up to (max_batch_size, max_batch_pairs)
       within batch_wait_timeout
    -> asyncio.create_task(_process_batch)  ── fire-and-forget
    -> _process_batch acquires batch_semaphore
    -> run_in_executor (ThreadPoolExecutor) submits the forward pass
       to the inference thread (one per GPU is correct)
    -> handler.predict() runs on the GPU
    -> RerankResult flows back through the future
```

The accumulation step (`get_batch`) and the inference step (`run_in_executor`)
are decoupled. While one batch is on the GPU, the next batch is already being
formed in the queue. The `batch_semaphore` keeps the GPU from being
oversubscribed.

## Tuning knobs

All exposed as env vars (`RERANKER_<NAME>`) and in `config.yml` under
`async_engine:`.

| Knob | Default | What it does |
|---|---|---|
| `max_concurrent_batches` | 1 | Number of batches allowed inside `_process_batch` at once. **Keep at 1 for a single GPU.** Raise for multi-GPU workers or pure-CPU multi-thread inference. |
| `inference_threads` | 1 | Worker threads in the executor. PyTorch+CUDA serializes on the device, so 1 is correct for GPU. Raise on CPU to use more cores. |
| `max_batch_size` | 32 | Max number of *requests* per batch. |
| `max_batch_pairs` | 1024 | Max number of *(query, doc) pairs* per batch — the real GPU work cap. |
| `batch_wait_timeout` | 0.005 (5 ms) | How long `get_batch` waits to accumulate more requests after the first one arrives. Smaller = lower added latency, larger = bigger batches. |
| `max_queue_size` | 1000 | Backpressure cap before new requests are rejected. |
| `request_timeout` | 60.0 | Per-request hard timeout. |

### Picking values

- **Latency-sensitive RAG (1-5 docs/request, low traffic):** keep
  `batch_wait_timeout` at 5 ms. Each request adds at most 5 ms of queue wait.
- **Throughput-bound batch jobs:** raise `batch_wait_timeout` to 0.02-0.05 (20-50
  ms). Larger batches amortize GPU launch overhead.
- **Many concurrent clients (8-32):** the defaults are sized for this; check
  `batch_occupancy_pct` at `/stats` and raise `max_batch_size` if you're
  saturating it.

## Observability — `/stats`

`/stats` exposes everything you need to know whether you are queue-bound,
GPU-bound, or under-batched.

| Field | Meaning | What to do if it's bad |
|---|---|---|
| `batch_occupancy_pct` | Average batch size as % of `max_batch_size` | Below ~50%: traffic is too thin to amortize; consider raising `batch_wait_timeout`. Above ~95%: you may be batch-size-limited; raise `max_batch_size`. |
| `queue_wait_p50_ms` / `p95_ms` | Time requests sit in the queue before inference | If p95 grows but inference latency is flat, you're behind the curve — raise `max_batch_pairs` or scale out. |
| `inference_latency_p50_ms` / `p95_ms` | Per-batch GPU time | Should be roughly constant for a given batch size. If it climbs, check VRAM pressure or thermals. |
| `throughput_pairs_per_sec` | `(query, doc)` pairs processed per second | Your primary throughput metric. |
| `inflight_batches` | How many batches are currently inside `_process_batch` | With `max_concurrent_batches=1`, this is 0 or 1. |
| `semaphore_available` | Free semaphore slots | If consistently 0, the GPU is saturated and the queue is filling. |

## How this differs from vLLM / Ollama / LMStudio

| | vLLM | Ollama | This service |
|---|---|---|---|
| Workload | LLM generation | LLM generation | Reranker scoring |
| Per-request work | Many tokens, autoregressive | Many tokens, autoregressive | One forward pass |
| Batching strategy | Continuous (mid-flight join) | Sequential queue + minor batching | Dynamic coalescing (window-based) |
| Right for rerankers? | No — overkill, wrong primitive | No — under-batched | Yes |

If you ever switch to a reranker that does multi-step scoring (e.g. an LLM-style
listwise reranker that decodes a list of doc IDs), continuous batching becomes
relevant. Today's cross-encoder + Qwen3 "yes/no" approach is single-step.
