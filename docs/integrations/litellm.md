# LiteLLM Proxy Integration

> **TL;DR:** a ready-to-use config lives at [`examples/litellm.config.yaml`](../../examples/litellm.config.yaml). Run `litellm --config examples/litellm.config.yaml --port 4000` and you're done. The rest of this page explains what's in it.

Two paths are supported. Pick one — they are not stacked.

## Path A — Cohere-compatible (recommended; zero code)

The reranker's `/v1/rerank` endpoint speaks Cohere format. Point LiteLLM at it as a Cohere provider.

`litellm.config.yaml`:

```yaml
model_list:
  - model_name: reranker
    litellm_params:
      model: cohere/rerank-english-v3.0
      api_base: http://localhost:8000
      api_key: "dummy"

litellm_settings:
  drop_params: true
```

`api_key` only needs a real value if you set `RERANKER_API_KEY` on the reranker — LiteLLM's Cohere transport always sends an `Authorization` header. `drop_params: true` lets LiteLLM silently drop fields our schema does not accept.

Call from Python:

```python
import litellm

response = litellm.rerank(
    model="reranker",
    query="What is deep learning?",
    documents=[
        "Deep learning is a subset of machine learning.",
        "The weather is nice today.",
        "Neural networks are used in deep learning.",
    ],
    top_n=2,
)
for r in response.results:
    print(r.index, r.relevance_score)
```

Run the proxy:

```bash
litellm --config litellm.config.yaml --port 4000
curl -X POST http://localhost:4000/rerank \
  -H "Content-Type: application/json" \
  -d '{"model":"reranker","query":"q","documents":["a","b","c"],"top_n":2}'
```

## Path B — Custom provider (use this for our native endpoint)

Use when you want LiteLLM to hit our native `/rerank` (not `/v1/rerank`) — for example to take advantage of richer responses or to bypass LiteLLM's Cohere parser.

`litellm.config.yaml`:

```yaml
model_list:
  - model_name: local-reranker
    litellm_params:
      model: reranker_custom/local
      api_base: http://localhost:8000
      api_key: "optional-bearer-token"

litellm_settings:
  custom_provider_map:
    - provider: reranker_custom
      custom_handler: src.integrations.litellm_provider.reranker_provider
```

The handler ships in this repo at `src/integrations/litellm_provider.py`. Requires `litellm` installed in the LiteLLM proxy's environment and `PYTHONPATH` pointing at this repo (or pip-install this repo).

Call from Python:

```python
import litellm

response = litellm.rerank(
    model="local-reranker",
    query="What is deep learning?",
    documents=["Deep learning is a subset of machine learning.", "Pizza."],
    top_n=1,
    return_documents=True,
)
```

The provider exposes both sync `rerank()` and async `arerank()`, so LiteLLM's async paths work transparently.

## Auth

If `RERANKER_API_KEY` is set on the reranker service, callers must send `Authorization: Bearer <key>`. Both LiteLLM paths above honor that automatically — set `api_key` in `litellm_params` to the same value.

## Verification

Both paths should produce equivalent results for the same input. Differences in `results[].document` shape come from the underlying endpoint (`/v1/rerank` returns Cohere-style `{"text": "..."}`, native `/rerank` returns the same).
