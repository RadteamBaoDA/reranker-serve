# HuggingFace-Compatible Rerank API Test

## Quick Test Commands

### Using curl (HuggingFace format with 'texts' field)

```bash
curl -X POST "http://localhost:8000/reranking" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is deep learning?",
    "texts": [
      "Deep learning is a subset of machine learning.",
      "The weather is nice today."
    ],
    "top_k": 2
  }'
```

### Using Python requests

```python
import requests

response = requests.post(
    "http://localhost:8000/reranking",
    json={
        "query": "What is deep learning?",
        "texts": [
            "Deep learning is a subset of machine learning.",
            "The weather is nice today."
        ],
        "top_k": 2
    }
)

print(response.json())
```

## Endpoint Information

### HuggingFace-Compatible Endpoints
- `POST /reranking` - HuggingFace format
- `POST /v1/reranking` - HuggingFace format with version prefix

### Request Format (HuggingFace)
```json
{
  "query": "search query",
  "texts": ["text1", "text2", "text3"],
  "top_k": 3,
  "return_texts": true
}
```

**Note**: Uses `texts` field (HuggingFace format), not `documents`

### Response Format
```json
{
  "model": "Qwen/Qwen3-Reranker-0.6B",
  "results": [
    {
      "index": 0,
      "score": 0.95,
      "text": "text content (if return_texts=true)"
    }
  ]
}
```

**Note**: Uses `score` field (HuggingFace format), not `relevance_score`

## Backward Compatibility

The original endpoints still work:
- `POST /rerank` - Original format (uses `documents` field)
- `POST /v1/rerank` - Cohere format
- `POST /api/v1/rerank` - Jina format

## Run the Test Script

```bash
# Make sure the server is running first
python -m src.main

# In another terminal, run the test
python test_huggingface_api.py
```
