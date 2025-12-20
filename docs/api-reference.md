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

## Authentication

If `RERANKER_API_KEY` is configured, include the API key in requests:

```bash
curl -X POST http://localhost:8000/rerank \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "documents": ["doc1", "doc2"]}'
```
