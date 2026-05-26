# API Reference

The Reranker Service provides multiple API endpoints compatible with different platforms.

## Endpoints Overview

| Endpoint | Description | Compatibility |
|----------|-------------|---------------|
| `POST /rerank` | Native API | Reranker Service |
| `POST /reranking` | HuggingFace-compatible | HuggingFace Inference API |
| `POST /v1/reranking` | HuggingFace-compatible | HuggingFace Inference API |
| `POST /v1/rerank` | Cohere-compatible | Cohere API |
| `POST /api/v1/rerank` | Jina-compatible | Jina AI API |
| `GET /health` | Health check | - |
| `GET /info` | Model + device profile + available devices | - |
| `GET /stats` | Live engine metrics (queue, batch occupancy, latency p50/p95) | - |
| `GET /ready`, `GET /live` | Kubernetes probes | - |
| `GET /docs` | API documentation | OpenAPI/Swagger |

---

## Native API

```http
POST /rerank
```

**Supports both native and HuggingFace formats with automatic response format matching!**

The endpoint automatically detects your request format and returns the corresponding response format:
- Request with `documents` → Native response format
- Request with `texts` → HuggingFace response format

This ensures clients get the expected format and prevents errors like "'str' object has no attribute get".

### Native Format Request → Native Response

**Request:**
```json
{
  "query": "What is deep learning?",
  "documents": [
    "Deep learning is a subset of machine learning.",
    "The weather is nice today."
  ],
  "top_n": 2,
  "return_documents": true
}
```

**Response:**
```json
{
  "results": [
    {
      "index": 0,
      "relevance_score": 0.95,
      "document": {"text": "Deep learning is a subset of machine learning."}
    }
  ],
  "model": "..."
}
```

### HuggingFace Format Request → HuggingFace Response

**Request:**
```json
{
  "query": "What is deep learning?",
  "texts": [
    "Deep learning is a subset of machine learning.",
    "The weather is nice today."
  ],
  "top_k": 2,
  "return_texts": true
}
```

**Response:**
```json
{
  "results": [
    {
      "index": 0,
      "score": 0.95,
      "text": "Deep learning is a subset of machine learning."
    }
  ],
  "model": "..."
}
```

### Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | The query to rank documents against |
| `documents` | array | Either this or `texts` | List of documents (native format) → Returns native response |
| `texts` | array | Either this or `documents` | List of texts (HuggingFace format) → Returns HuggingFace response |
| `top_n` | integer | No | Number of top results (alias: `top_k`) |
| `top_k` | integer | No | Number of top results (alias: `top_n`) |
| `return_documents` | boolean | No | Include text in response (alias: `return_texts`) |
| `return_texts` | boolean | No | Include text in response (alias: `return_documents`) |
| `prefer_device` | `"cuda"`, `"mps"`, `"cpu"` | No | Reject the request with HTTP 400 unless this matches the device the server is serving on. Useful for client-side device routing across multiple workers. |

**Important:** 
- You must provide either `documents` OR `texts`, not both or neither
- The response format automatically matches your request format:
  - `documents` field → Native response with `relevance_score` and `document` object  
  - `texts` field → HuggingFace response with `score` and `text` string

### Response

```json
{
  "results": [
    {
      "index": 0,
      "relevance_score": 0.95,
      "document": "Deep learning is a subset of machine learning."
    },
    {
      "index": 1,
      "relevance_score": 0.12,
      "document": "The weather is nice today."
    }
  ]
}
```

---

## HuggingFace-Compatible API

```http
POST /reranking
POST /v1/reranking
```

### Request

```json
{
  "query": "What is deep learning?",
  "texts": [
    "Deep learning is a subset of machine learning.",
    "The weather is nice today."
  ],
  "top_k": 2,
  "return_texts": true
}
```

### Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | The query to rank documents against |
| `texts` | array | Yes | List of texts to rerank (HuggingFace format) |
| `top_k` | integer | No | Number of top results to return (alias: `top_n`) |
| `return_texts` | boolean | No | Include text in response (default: true) |
| `model` | string | No | Model name (ignored, uses configured model) |
| `truncate` | boolean | No | Truncate long inputs (default: true) |

### Response

```json
{
  "model": "Qwen/Qwen3-Reranker-0.6B",
  "results": [
    {
      "index": 0,
      "score": 0.95,
      "text": "Deep learning is a subset of machine learning."
    },
    {
      "index": 1,
      "score": 0.12,
      "text": "The weather is nice today."
    }
  ]
}
```

**Key Differences from Native API:**
- Uses `texts` field instead of `documents`
- Uses `score` field instead of `relevance_score` in results
- Uses `top_k` instead of `top_n` (both accepted via aliases)
- Uses `return_texts` instead of `return_documents` (both accepted)

---

## Cohere-Compatible API

```http
POST /v1/rerank
```

### Request

```json
{
  "query": "What is deep learning?",
  "documents": ["Document 1", "Document 2"],
  "top_n": 2
}
```

### Response

```json
{
  "results": [
    {
      "index": 0,
      "relevance_score": 0.95
    }
  ]
}
```

---

## Jina AI-Compatible API

```http
POST /api/v1/rerank
```

### Request

```json
{
  "query": "What is deep learning?",
  "documents": [
    {"text": "Document 1"},
    {"text": "Document 2"}
  ],
  "top_n": 2
}
```

### Response

```json
{
  "results": [
    {
      "index": 0,
      "relevance_score": 0.95,
      "document": {"text": "Document 1"}
    }
  ]
}
```

---

## Health Check

```http
GET /health
```

### Response

```json
{
  "status": "healthy",
  "model": "BAAI/bge-reranker-v2-m3",
  "device": "cuda"
}
```

---

## Model Info

```http
GET /info
```

Returns the loaded model, the device actually serving inference (after any
MPS→CPU fallback), every device the running torch install can reach, and the
startup probe results.

### Response

```json
{
  "model_name": "BAAI/bge-reranker-v2-m3",
  "device": "cuda",
  "available_devices": ["cuda", "cpu"],
  "device_profile": {
    "device": "cuda",
    "probes": [
      {"batch_size": 1,  "pairs": 4,   "elapsed_ms": 12.5, "ms_per_pair": 3.13},
      {"batch_size": 8,  "pairs": 32,  "elapsed_ms": 41.0, "ms_per_pair": 1.28},
      {"batch_size": 32, "pairs": 128, "elapsed_ms": 132.4, "ms_per_pair": 1.03}
    ],
    "suggested_batch_size": 32,
    "user_pinned_batch_size": false
  },
  "max_length": 512,
  "batch_size": 32,
  "use_fp16": true,
  "async_engine_enabled": true,
  "max_concurrent_batches": 1,
  "inference_threads": 1,
  "max_queue_size": 1000
}
```

`device_profile` is `null` until the engine has started. Set
`RERANKER_ENABLE_DEVICE_PROBE=false` to skip the warmup.

---

## Engine Stats

```http
GET /stats
```

Live concurrency + latency snapshot. Use this to decide whether to widen the
batch window, raise `max_batch_size`, or add a worker.

### Response

```json
{
  "engine_mode": "async",
  "running": true,
  "model_loaded": true,
  "stats": {
    "total_requests": 1284,
    "total_batches": 312,
    "avg_batch_size": 4.1,
    "batch_occupancy_pct": 12.8,
    "queue_wait_p50_ms": 3.2,
    "queue_wait_p95_ms": 18.7,
    "inference_latency_p50_ms": 41.0,
    "inference_latency_p95_ms": 102.3,
    "throughput_pairs_per_sec": 614.2,
    "inflight_batches": 0,
    "semaphore_available": 1,
    "max_concurrent_batches": 1,
    "device_profile": { "...": "(same shape as /info)" },
    "pending_requests": 0,
    "active_requests": 0
  }
}
```

See [Concurrency](concurrency.md) for what each number means and which knob to turn.

---

## Authentication

If `RERANKER_API_KEY` is configured, include the API key in requests:

```bash
curl -X POST http://localhost:8000/rerank \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "documents": ["doc1", "doc2"]}'
```
