# Phase 3A — Performance Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make concurrent rerank requests collapse into a few padded GPU forward passes instead of one-pass-per-request, with device-aware precision/kernels and VRAM-aware batch sizing, so Qwen3-Reranker-4B serves ~50 concurrent interactive users at p95 < 800 ms on a 16 GB Ada GPU (and runs faster on MPS/CPU too).

**Architecture:** Introduce a `score_pairs()` primitive on the model that takes a flat list of `(query, document)` pairs, length-buckets them, runs padded forward passes, and returns one score per pair in input order. The handler flattens all pairs across every request in a `BatchedRequest` into one `score_pairs()` call and scatters the scores back. Model-load kwargs (dtype, attention impl, quantization) are chosen by pure, unit-tested helpers gated on the active device. The startup device probe is extended to suggest a VRAM-safe `max_batch_pairs`.

**Tech Stack:** Python 3.11, PyTorch, transformers (`AutoModelForCausalLM`), sentence-transformers (`CrossEncoder`), pytest + pytest-asyncio. Optional: `torchao` (FP8 on Ada), `psutil` (CPU memory probe).

**Reference spec:** `docs/superpowers/specs/2026-05-27-perf-telemetry-admin-ui-design.md` (Part A).

**Branch:** `feature/phase3a-performance-core` (worktree recommended via `superpowers:using-git-worktrees`).

---

### Task 0: Branch, settings fields, config defaults, dependencies

**Files:**
- Modify: `src/config/settings.py`
- Modify: `config.yml`
- Modify: `.env.example`
- Modify: `requirements.txt`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create the branch / worktree**

```bash
git worktree add ../reranker-serve-phase3a feature/phase3a-performance-core
cd ../reranker-serve-phase3a
```

- [ ] **Step 2: Add settings fields**

In `src/config/settings.py`, change the `max_length` default and add four new fields. Find:

```python
    max_length: int = Field(default=512, description="Maximum sequence length")
```

Replace with:

```python
    max_length: int = Field(default=256, description="Maximum sequence length for the rerank prompt (lower = faster/less VRAM; rerankers rarely need long context)")
```

Then, immediately after the `use_fp16` field (around line 134), add:

```python
    # Quantization / precision lever (device-gated, opt-in)
    quantization: Literal["none", "fp8", "int8"] = Field(
        default="none",
        description="none = device default precision; fp8 = Ada CUDA FP8; int8 = CPU dynamic int8. Unsupported value for the active device uses the default precision with a warning."
    )

    # CPU performance
    cpu_num_threads: Optional[int] = Field(
        default=None,
        description="torch intra-op thread count on CPU. None = os.cpu_count()."
    )

    # VRAM-aware batch sizing
    device_mem_safety_margin: float = Field(
        default=0.15,
        description="Fraction of device memory the batch-size probe keeps free (0.15 = keep 15% headroom)."
    )
```

- [ ] **Step 3: Map the new keys in the YAML loader**

In `src/config/settings.py`, find the `if 'inference' in yaml_config:` block. It currently does:

```python
        if 'inference' in yaml_config:
            for key, value in yaml_config['inference'].items():
                flat_config[key] = value
```

Leave it as-is (it already forwards `max_length`, `batch_size`, `normalize_scores`, and will forward `quantization` if placed under `inference`). Then find the `if 'device' in yaml_config:` block and add `cpu_num_threads` + `device_mem_safety_margin` + `quantization` mapping at its end:

```python
        if 'device' in yaml_config:
            device_cfg = yaml_config['device']
            flat_config['device'] = device_cfg.get('name')
            flat_config['force_cpu_only'] = device_cfg.get('force_cpu_only')
            flat_config['use_fp16'] = device_cfg.get('use_fp16')
            flat_config['mps_fallback_to_cpu'] = device_cfg.get('mps_fallback_to_cpu')
            flat_config['quantization'] = device_cfg.get('quantization')
            flat_config['cpu_num_threads'] = device_cfg.get('cpu_num_threads')
            flat_config['device_mem_safety_margin'] = device_cfg.get('device_mem_safety_margin')
```

(The trailing `return {k: v for k, v in flat_config.items() if v is not None}` already drops unset keys, so defaults still apply.)

- [ ] **Step 4: Update `config.yml`**

In `config.yml`, set the model to 4B and lower `max_length`. Change:

```yaml
  name: Qwen/Qwen3-Reranker-8B
  path: ./models/Qwen3-Reranker-8B
```

to:

```yaml
  name: Qwen/Qwen3-Reranker-4B
  path: ./models/Qwen3-Reranker-4B
```

Change `inference.max_length: 512` to `inference.max_length: 256`. In the `device:` section add:

```yaml
  # Quantization lever: none | fp8 (Ada CUDA) | int8 (CPU). Opt-in.
  quantization: none
  # On CUDA, prefer bf16 for Qwen3 on Ada: set use_fp16 to false.
  use_fp16: false
  # torch intra-op threads on CPU (null = all cores)
  cpu_num_threads: null
  # Keep this fraction of device memory free when probing batch size
  device_mem_safety_margin: 0.15
```

- [ ] **Step 5: Document the new keys in `.env.example`**

Append to `.env.example`:

```
# Quantization lever (device-gated, opt-in): none | fp8 | int8
RERANKER_QUANTIZATION=none
# torch intra-op threads on CPU (unset = all cores)
# RERANKER_CPU_NUM_THREADS=8
# VRAM/memory headroom kept free by the batch-size probe (0.0-0.5)
RERANKER_DEVICE_MEM_SAFETY_MARGIN=0.15
```

- [ ] **Step 6: Add optional dependencies**

Append to `requirements.txt`:

```
# CPU memory probe for VRAM-aware batch sizing
psutil>=5.9.0
```

In `pyproject.toml`, under `[project.optional-dependencies]`, add:

```toml
quant = [
    "torchao>=0.7.0",
]
```

- [ ] **Step 7: Confirm settings still load**

Run: `python -c "from src.config import settings; print(settings.max_length, settings.quantization, settings.device_mem_safety_margin)"`
Expected: `256 none 0.15`

- [ ] **Step 8: Commit**

```bash
git add src/config/settings.py config.yml .env.example requirements.txt pyproject.toml
git commit -m "feat(config): perf settings (max_length=256, quantization, cpu threads, mem margin)"
```

---

### Task 1: `score_pairs()` primitive on `Qwen3Reranker`

**Files:**
- Modify: `src/models/qwen3_reranker.py`
- Test: `tests/test_score_pairs.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_score_pairs.py`:

```python
"""score_pairs(): flat (query, doc) scoring, scores returned in input order."""

import pytest

from src.models.qwen3_reranker import Qwen3Reranker


def _make_reranker(monkeypatch):
    """Build a Qwen3Reranker without loading real weights; stub the GPU bits."""
    r = Qwen3Reranker.__new__(Qwen3Reranker)
    r._model = object()          # non-None so score_pairs won't try to load
    r._tokenizer = object()
    r.default_instruction = "inst"
    r.max_length = 256
    r.device = "cpu"

    # _process_inputs just passes the formatted prompts straight through;
    # _compute_logits scores by prompt length so we can assert ordering.
    monkeypatch.setattr(r, "_process_inputs", lambda prompts: prompts)
    monkeypatch.setattr(
        r, "_compute_logits", lambda prompts: [float(len(p)) for p in prompts]
    )
    return r


def test_score_pairs_returns_one_score_per_pair_in_input_order(monkeypatch):
    r = _make_reranker(monkeypatch)
    pairs = [("q", "short"), ("q", "a much longer document here"), ("q", "mid len")]
    scores = r.score_pairs(pairs)

    assert len(scores) == 3
    # Score is the formatted-prompt length; the longest doc must score highest,
    # and crucially the result is in INPUT order, not bucketed order.
    assert scores[1] == max(scores)
    assert scores[0] < scores[2] < scores[1]


def test_score_pairs_empty_returns_empty(monkeypatch):
    r = _make_reranker(monkeypatch)
    assert r.score_pairs([]) == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_score_pairs.py -v`
Expected: FAIL with `AttributeError: 'Qwen3Reranker' object has no attribute 'score_pairs'`.

- [ ] **Step 3: Implement `score_pairs()` and delegate `rerank()` to it**

In `src/models/qwen3_reranker.py`, add this method to the `Qwen3Reranker` class (place it just above `rerank`):

```python
    def score_pairs(
        self,
        pairs: List[tuple],
        instruction: Optional[str] = None,
    ) -> List[float]:
        """Score a flat list of (query, document) pairs.

        Returns one relevance score per pair, in the SAME order as the input.
        Pairs are length-bucketed (sorted by formatted-prompt length) before
        padding so a short pair is never padded up to a long pair's length.
        """
        if self._model is None:
            self.load()
        if not pairs:
            return []

        formatted = [
            self._format_instruction(query, doc, instruction)
            for (query, doc) in pairs
        ]

        # Length bucketing: sort indices by formatted length (cheap char-length
        # proxy for token count), batch similar lengths, scatter back to input order.
        order = sorted(range(len(formatted)), key=lambda i: len(formatted[i]))
        scores: List[float] = [0.0] * len(formatted)

        batch_size = settings.batch_size
        for start in range(0, len(order), batch_size):
            idx_chunk = order[start:start + batch_size]
            prompt_chunk = [formatted[i] for i in idx_chunk]
            inputs = self._process_inputs(prompt_chunk)
            chunk_scores = self._compute_logits(inputs)
            for local_i, global_i in enumerate(idx_chunk):
                scores[global_i] = float(chunk_scores[local_i])

        return scores
```

Then replace the body of `rerank()` (the loop that builds `pairs` and iterates `batch_size`) so it delegates. Find:

```python
        # Format all query-document pairs
        pairs = [
            self._format_instruction(query, doc, instruction)
            for doc in documents
        ]

        # Process in batches if needed
        batch_size = settings.batch_size
        all_scores = []

        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            inputs = self._process_inputs(batch_pairs)
            batch_scores = self._compute_logits(inputs)
            all_scores.extend(batch_scores)
```

Replace with:

```python
        # Delegate to the shared primitive (handles bucketing + batching).
        all_scores = self.score_pairs(
            [(query, doc) for doc in documents],
            instruction=instruction,
        )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_score_pairs.py -v`
Expected: PASS for both tests.

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `pytest tests/ -x`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/models/qwen3_reranker.py tests/test_score_pairs.py
git commit -m "feat(qwen): score_pairs() primitive with length bucketing; rerank() delegates"
```

---

### Task 2: Flat cross-request batching in `QwenRerankerHandler`

**Files:**
- Modify: `src/engine/handlers/qwen.py`
- Test: `tests/test_flat_batching.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_flat_batching.py`:

```python
"""Handler flattens all pairs across requests into ONE score_pairs() call."""

from src.engine.handlers.qwen import QwenRerankerHandler
from src.engine.request_queue import BatchedRequest, RerankRequest


class _SpyModel:
    """score_pairs scores each pair as len(query)+len(doc); records call count."""
    def __init__(self):
        self.calls = 0
        self.last_pairs = None

    def score_pairs(self, pairs, instruction=None):
        self.calls += 1
        self.last_pairs = list(pairs)
        return [float(len(q) + len(d)) for (q, d) in pairs]


def _handler_with_spy():
    h = QwenRerankerHandler.__new__(QwenRerankerHandler)
    h.model = _SpyModel()
    return h


def test_predict_makes_single_score_pairs_call_across_requests():
    h = _handler_with_spy()
    batch = BatchedRequest(
        batch_id="b1",
        requests=[
            RerankRequest(request_id="r1", query="qq", documents=["a", "bb"]),
            RerankRequest(request_id="r2", query="q", documents=["ccc"]),
        ],
    )

    results = h.predict(batch)

    # One flat call covering all 3 pairs (2 + 1), not one call per request.
    assert h.model.calls == 1
    assert len(h.model.last_pairs) == 3
    assert len(results) == 2
    assert len(results[0]) == 2 and len(results[1]) == 1


def test_predict_scatters_scores_back_to_correct_request_and_sorts():
    h = _handler_with_spy()
    batch = BatchedRequest(
        batch_id="b2",
        requests=[
            RerankRequest(request_id="r1", query="q", documents=["x", "longer"]),
        ],
    )
    results = h.predict(batch)
    r1 = results[0]
    # score = len(q)+len(doc): "longer"(6)+1=7 beats "x"(1)+1=2, so sorted desc.
    assert r1[0]["index"] == 1
    assert r1[0]["relevance_score"] == 7.0
    assert r1[1]["index"] == 0


def test_predict_respects_top_k_per_request():
    h = _handler_with_spy()
    batch = BatchedRequest(
        batch_id="b3",
        requests=[
            RerankRequest(request_id="r1", query="q", documents=["a", "bb", "ccc"], top_k=1),
        ],
    )
    results = h.predict(batch)
    assert len(results[0]) == 1
    assert results[0][0]["index"] == 2  # "ccc" highest


def test_predict_empty_batch_returns_empty():
    h = _handler_with_spy()
    batch = BatchedRequest(batch_id="b4", requests=[])
    assert h.predict(batch) == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_flat_batching.py -v`
Expected: FAIL — current `predict()` calls `self.model.rerank()` per request, so `score_pairs` is never called and `h.model.calls` stays 0 / `AttributeError`.

- [ ] **Step 3: Rewrite `QwenRerankerHandler.predict()` for flat batching**

Replace the entire body of `predict()` in `src/engine/handlers/qwen.py` with:

```python
    def predict(self, batch: BatchedRequest) -> List[List[Dict[str, Any]]]:
        if not batch.requests:
            return []

        # Flatten every (query, doc) pair across all requests into one list,
        # remembering which request each pair belongs to.
        flat_pairs: List[tuple] = []
        spans: List[tuple] = []  # (start, end) index range per request
        for request in batch.requests:
            start = len(flat_pairs)
            flat_pairs.extend((request.query, doc) for doc in request.documents)
            spans.append((start, len(flat_pairs)))

        # ONE batched scoring call for the whole BatchedRequest.
        scores = self.model.score_pairs(flat_pairs) if flat_pairs else []

        # Scatter scores back to each request, then sort + top_k per request.
        all_results: List[List[Dict[str, Any]]] = []
        for request, (start, end) in zip(batch.requests, spans):
            request_scores = scores[start:end]
            results: List[Dict[str, Any]] = []
            for idx, score in enumerate(request_scores):
                result = {"index": idx, "relevance_score": float(score)}
                if request.return_documents:
                    result["document"] = {"text": request.documents[idx]}
                results.append(result)

            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            if request.top_k is not None and request.top_k > 0:
                results = results[:request.top_k]
            all_results.append(results)

        return all_results
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_flat_batching.py -v`
Expected: PASS for all four tests.

- [ ] **Step 5: Run the full suite**

Run: `pytest tests/ -x`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/engine/handlers/qwen.py tests/test_flat_batching.py
git commit -m "feat(engine): true cross-request batching in QwenRerankerHandler"
```

---

### Task 3: Flat cross-request batching in `CrossEncoderHandler`

**Files:**
- Modify: `src/engine/handlers/cross_encoder.py`
- Test: `tests/test_flat_batching_cross_encoder.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_flat_batching_cross_encoder.py`:

```python
"""CrossEncoder handler also flattens pairs across requests into one predict()."""

import numpy as np

from src.engine.handlers.cross_encoder import CrossEncoderHandler
from src.engine.request_queue import BatchedRequest, RerankRequest


class _SpyCrossEncoder:
    def __init__(self):
        self.calls = 0
        self.last_pairs = None

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        self.calls += 1
        self.last_pairs = list(pairs)
        # score = len(query) + len(doc)
        return np.array([len(p[0]) + len(p[1]) for p in pairs], dtype=float)


def _handler_with_spy():
    h = CrossEncoderHandler.__new__(CrossEncoderHandler)
    h.model = _SpyCrossEncoder()
    h.device = "cpu"
    return h


def test_cross_encoder_single_predict_call_across_requests(monkeypatch):
    from src.config import settings as s
    monkeypatch.setattr(s, "normalize_scores", False)

    h = _handler_with_spy()
    batch = BatchedRequest(
        batch_id="b1",
        requests=[
            RerankRequest(request_id="r1", query="q", documents=["a", "bb"]),
            RerankRequest(request_id="r2", query="qq", documents=["ccc"]),
        ],
    )
    results = h.predict(batch)
    assert h.model.calls == 1
    assert len(h.model.last_pairs) == 3
    assert len(results) == 2
    assert len(results[0]) == 2 and len(results[1]) == 1


def test_cross_encoder_scatter_and_sort(monkeypatch):
    from src.config import settings as s
    monkeypatch.setattr(s, "normalize_scores", False)
    h = _handler_with_spy()
    batch = BatchedRequest(
        batch_id="b2",
        requests=[RerankRequest(request_id="r1", query="q", documents=["x", "longer"])],
    )
    results = h.predict(batch)
    assert results[0][0]["index"] == 1  # "longer" wins
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_flat_batching_cross_encoder.py -v`
Expected: FAIL — current `predict()` loops per request, calling `model.predict` N times (`h.model.calls == 2`).

- [ ] **Step 3: Rewrite `CrossEncoderHandler.predict()` for flat batching**

Replace the entire body of `predict()` in `src/engine/handlers/cross_encoder.py` with:

```python
    def predict(self, batch: BatchedRequest) -> List[List[Dict[str, Any]]]:
        if not batch.requests:
            return []

        # Flatten all pairs across requests, tracking per-request spans.
        flat_pairs: List[list] = []
        spans: List[tuple] = []
        for request in batch.requests:
            start = len(flat_pairs)
            flat_pairs.extend([request.query, doc] for doc in request.documents)
            spans.append((start, len(flat_pairs)))

        if not flat_pairs:
            return [[] for _ in batch.requests]

        # ONE batched inference call for the whole BatchedRequest, with the
        # existing MPS->CPU recovery path preserved.
        try:
            scores = self.model.predict(
                flat_pairs,
                batch_size=settings.batch_size,
                show_progress_bar=False,
            )
        except RuntimeError as e:
            error_msg = str(e)
            mps_kernel_error = self.device == "mps" and (
                "MPSGraph" in error_msg or "INT_MAX" in error_msg or "MPS" in error_msg
            )
            if mps_kernel_error and settings.mps_fallback_to_cpu:
                logger.warning(f"MPS inference failed, falling back to CPU: {error_msg}")
                from src.observability import get_observer
                get_observer().on_mps_fallback()
                self.device = "cpu"
                self.model = None
                self.load_model()
                scores = self.model.predict(
                    flat_pairs, batch_size=settings.batch_size, show_progress_bar=False
                )
                logger.info("Successfully completed inference on CPU after MPS fallback")
            else:
                raise

        if settings.normalize_scores:
            scores_array = np.array(scores)
            scores = (1 / (1 + np.exp(-scores_array))).tolist()
        else:
            scores = list(scores)

        all_results: List[List[Dict[str, Any]]] = []
        for request, (start, end) in zip(batch.requests, spans):
            request_scores = scores[start:end]
            results: List[Dict[str, Any]] = []
            for idx, score in enumerate(request_scores):
                result = {"index": idx, "relevance_score": float(score)}
                if request.return_documents:
                    result["document"] = {"text": request.documents[idx]}
                results.append(result)
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            if request.top_k is not None and request.top_k > 0:
                results = results[:request.top_k]
            all_results.append(results)

        return all_results
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_flat_batching_cross_encoder.py -v`
Expected: PASS for both tests.

- [ ] **Step 5: Run the full suite**

Run: `pytest tests/ -x`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/engine/handlers/cross_encoder.py tests/test_flat_batching_cross_encoder.py
git commit -m "feat(engine): cross-request batching in CrossEncoderHandler"
```

---

### Task 4: Device-aware model-load kwargs (dtype + attention)

**Files:**
- Modify: `src/models/qwen3_reranker.py`
- Test: `tests/test_load_kwargs.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_load_kwargs.py`:

```python
"""Pure selector for dtype + attention implementation by device."""

import torch

from src.models.qwen3_reranker import build_load_kwargs


def test_cuda_bf16_when_fp16_disabled():
    kw = build_load_kwargs(device="cuda", use_fp16=False, has_flash_attn=False)
    assert kw["torch_dtype"] == torch.bfloat16
    assert kw["attn_implementation"] == "sdpa"


def test_cuda_fp16_when_enabled():
    kw = build_load_kwargs(device="cuda", use_fp16=True, has_flash_attn=False)
    assert kw["torch_dtype"] == torch.float16


def test_cuda_uses_flash_attn_when_available():
    kw = build_load_kwargs(device="cuda", use_fp16=False, has_flash_attn=True)
    assert kw["attn_implementation"] == "flash_attention_2"


def test_mps_is_fp32_sdpa():
    kw = build_load_kwargs(device="mps", use_fp16=True, has_flash_attn=False)
    assert kw["torch_dtype"] == torch.float32
    assert kw["attn_implementation"] == "sdpa"


def test_cpu_is_fp32_sdpa():
    kw = build_load_kwargs(device="cpu", use_fp16=True, has_flash_attn=True)
    assert kw["torch_dtype"] == torch.float32
    assert kw["attn_implementation"] == "sdpa"


def test_always_trusts_remote_code():
    kw = build_load_kwargs(device="cpu", use_fp16=False, has_flash_attn=False)
    assert kw["trust_remote_code"] is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_load_kwargs.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_load_kwargs'`.

- [ ] **Step 3: Implement the pure selector and use it in `load()`**

In `src/models/qwen3_reranker.py`, add this module-level function near the top (after the imports, before the class):

```python
def build_load_kwargs(device: str, use_fp16: bool, has_flash_attn: bool) -> dict:
    """Choose model-load kwargs (dtype + attention) for the active device.

    CUDA: fp16 if use_fp16 else bf16; flash-attn-2 when available, else SDPA.
    MPS:  fp32 (kernel stability), SDPA.
    CPU:  fp32, SDPA.
    """
    kwargs: dict = {"trust_remote_code": True}

    if device == "cuda":
        kwargs["torch_dtype"] = torch.float16 if use_fp16 else torch.bfloat16
        kwargs["attn_implementation"] = (
            "flash_attention_2" if has_flash_attn else "sdpa"
        )
    else:
        # MPS and CPU both run fp32 with SDPA.
        kwargs["torch_dtype"] = torch.float32
        kwargs["attn_implementation"] = "sdpa"

    return kwargs
```

Then, in `Qwen3Reranker.load()`, replace the inline dtype/attention block. Find the section from `model_kwargs = { "trust_remote_code": True, }` through the `if use_flash_attn:` / `else:` assignment of `attn_implementation`, and replace it all with:

```python
            # Detect flash-attention-2 (CUDA only).
            has_flash_attn = False
            if self.device == "cuda":
                try:
                    import flash_attn  # noqa: F401
                    has_flash_attn = True
                    logger.debug("flash_attention_available")
                except ImportError:
                    logger.debug("flash_attention_not_available", using="sdpa")

            model_kwargs = build_load_kwargs(
                device=self.device,
                use_fp16=self.use_fp16,
                has_flash_attn=has_flash_attn,
            )
```

(Leave the subsequent `AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)` call unchanged.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_load_kwargs.py -v`
Expected: PASS for all six tests.

- [ ] **Step 5: Run the full suite**

Run: `pytest tests/ -x`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/models/qwen3_reranker.py tests/test_load_kwargs.py
git commit -m "feat(qwen): device-aware load kwargs (bf16/fp32, flash-attn/SDPA)"
```

---

### Task 5: Quantization lever (FP8 on Ada, int8 on CPU)

**Files:**
- Modify: `src/models/qwen3_reranker.py`
- Test: `tests/test_quantization.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_quantization.py`:

```python
"""Quantization selection: FP8 config on Ada CUDA, int8 post-load on CPU, else no-op."""

import pytest

from src.models.qwen3_reranker import build_quantization_config, maybe_quantize_int8


def test_no_quantization_returns_none():
    assert build_quantization_config(device="cuda", quantization="none") is None
    assert build_quantization_config(device="cpu", quantization="none") is None


def test_fp8_only_on_cuda():
    # fp8 requested on cpu/mps must be ignored (None), never crash.
    assert build_quantization_config(device="cpu", quantization="fp8") is None
    assert build_quantization_config(device="mps", quantization="fp8") is None


def test_int8_is_noop_in_load_config():
    # int8 is applied post-load, not via from_pretrained config.
    assert build_quantization_config(device="cpu", quantization="int8") is None


def test_maybe_quantize_int8_only_on_cpu():
    sentinel = object()
    # Non-int8 or non-cpu returns the model unchanged.
    assert maybe_quantize_int8(sentinel, device="cuda", quantization="int8") is sentinel
    assert maybe_quantize_int8(sentinel, device="cpu", quantization="none") is sentinel


def test_maybe_quantize_int8_applies_dynamic_quant_on_cpu():
    import torch
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
    out = maybe_quantize_int8(model, device="cpu", quantization="int8")
    # quantize_dynamic swaps Linear for a DynamicQuantizedLinear.
    assert any("DynamicQuantizedLinear" in type(m).__name__ for m in out.modules())


def test_fp8_on_cuda_builds_config():
    pytest.importorskip("torchao")
    cfg = build_quantization_config(device="cuda", quantization="fp8")
    assert cfg is not None
    assert "TorchAo" in type(cfg).__name__
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_quantization.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_quantization_config'`.

- [ ] **Step 3: Implement the quantization helpers and wire them into `load()`**

In `src/models/qwen3_reranker.py`, add these module-level functions next to `build_load_kwargs`:

```python
def build_quantization_config(device: str, quantization: str):
    """Return a transformers quantization_config for from_pretrained, or None.

    FP8 is CUDA-only (Ada+). int8 is applied post-load (see maybe_quantize_int8),
    so it returns None here. Any unsupported combination returns None.
    """
    q = (quantization or "none").lower()
    if q == "fp8" and device == "cuda":
        # Requires `torchao` and a recent transformers. e4m3 dynamic FP8.
        from transformers import TorchAoConfig
        return TorchAoConfig("float8_dynamic_activation_float8_weight")
    return None


def maybe_quantize_int8(model, device: str, quantization: str):
    """Apply CPU dynamic int8 quantization to Linear layers, else return model as-is."""
    if (quantization or "none").lower() == "int8" and device == "cpu":
        import torch
        return torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    return model
```

Then wire them into `Qwen3Reranker.load()`. After the `model_kwargs = build_load_kwargs(...)` call from Task 4, add:

```python
            quant_config = build_quantization_config(self.device, settings.quantization)
            if quant_config is not None:
                model_kwargs["quantization_config"] = quant_config
                logger.info("quantization_enabled", mode=settings.quantization, device=self.device)
```

After the model is created, moved to device, and switched to inference mode (just below the existing `self._model` eval-mode line, `self._model.eval()`), add:

```python
            self._model = maybe_quantize_int8(self._model, self.device, settings.quantization)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_quantization.py -v`
Expected: PASS (the FP8 test is skipped if `torchao` is not installed).

- [ ] **Step 5: Run the full suite**

Run: `pytest tests/ -x`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/models/qwen3_reranker.py tests/test_quantization.py
git commit -m "feat(qwen): opt-in quantization (FP8 on Ada CUDA, int8 on CPU)"
```

---

### Task 6: CPU thread configuration

**Files:**
- Modify: `src/engine/async_engine.py`
- Test: `tests/test_cpu_threads.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_cpu_threads.py`:

```python
"""CPU intra-op thread count is resolved from settings (None = all cores)."""

from src.engine.async_engine import resolve_cpu_threads


def test_explicit_thread_count_wins():
    assert resolve_cpu_threads(device="cpu", configured=6, cpu_count=12) == 6


def test_none_uses_cpu_count():
    assert resolve_cpu_threads(device="cpu", configured=None, cpu_count=12) == 12


def test_non_cpu_device_returns_none():
    # GPU paths must not touch torch.set_num_threads.
    assert resolve_cpu_threads(device="cuda", configured=8, cpu_count=12) is None
    assert resolve_cpu_threads(device="mps", configured=None, cpu_count=12) is None


def test_zero_or_negative_configured_falls_back_to_cpu_count():
    assert resolve_cpu_threads(device="cpu", configured=0, cpu_count=12) == 12
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_cpu_threads.py -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_cpu_threads'`.

- [ ] **Step 3: Implement `resolve_cpu_threads` and apply it at engine start**

In `src/engine/async_engine.py`, add this module-level function (near the top, after imports):

```python
def resolve_cpu_threads(device: str, configured: Optional[int], cpu_count: int) -> Optional[int]:
    """Thread count for torch intra-op parallelism on CPU. None for GPU devices."""
    if device != "cpu":
        return None
    if configured and configured > 0:
        return configured
    return cpu_count
```

Then, in `AsyncRerankerEngine.start()`, immediately after `await self._load_model()`, add:

```python
        import os
        import torch
        threads = resolve_cpu_threads(
            self.device, settings.cpu_num_threads, os.cpu_count() or 1
        )
        if threads is not None:
            torch.set_num_threads(threads)
            logger.info("cpu_threads_configured", threads=threads)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_cpu_threads.py -v`
Expected: PASS for all four tests.

- [ ] **Step 5: Run the full suite**

Run: `pytest tests/ -x`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/engine/async_engine.py tests/test_cpu_threads.py
git commit -m "feat(engine): configure torch CPU intra-op threads at startup"
```

---

### Task 7: VRAM-aware batch-size suggestion in the device probe

**Files:**
- Modify: `src/engine/device_probe.py`
- Test: `tests/test_device_probe_memory.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_device_probe_memory.py`:

```python
"""The probe suggests the largest pairs-per-batch that keeps a memory margin."""

from src.engine.device_probe import suggest_max_batch_pairs


def test_picks_largest_candidate_within_margin():
    # free_fraction_after(pairs) returns the fraction of device memory that
    # would remain free after a batch of `pairs`. Margin 0.15 => need >= 0.15.
    free_after = {64: 0.60, 128: 0.40, 256: 0.20, 512: 0.05}
    chosen = suggest_max_batch_pairs(
        candidates=[64, 128, 256, 512],
        free_fraction_after=lambda p: free_after[p],
        safety_margin=0.15,
    )
    assert chosen == 256  # 512 would drop below the 15% margin


def test_falls_back_to_smallest_when_all_exceed_margin():
    chosen = suggest_max_batch_pairs(
        candidates=[64, 128],
        free_fraction_after=lambda p: 0.05,  # everything blows the margin
        safety_margin=0.15,
    )
    assert chosen == 64  # never return nothing; smallest is the safest attempt


def test_all_fit_returns_largest():
    chosen = suggest_max_batch_pairs(
        candidates=[64, 128, 256],
        free_fraction_after=lambda p: 0.5,
        safety_margin=0.15,
    )
    assert chosen == 256
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_device_probe_memory.py -v`
Expected: FAIL with `ImportError: cannot import name 'suggest_max_batch_pairs'`.

- [ ] **Step 3: Implement `suggest_max_batch_pairs`**

In `src/engine/device_probe.py`, add this module-level function:

```python
from typing import Callable


def suggest_max_batch_pairs(
    candidates: List[int],
    free_fraction_after: Callable[[int], float],
    safety_margin: float,
) -> int:
    """Largest candidate batch (in pairs) that still leaves >= safety_margin
    of device memory free. If none qualifies, return the smallest candidate."""
    ordered = sorted(candidates)
    chosen = ordered[0]
    for pairs in ordered:
        if free_fraction_after(pairs) >= safety_margin:
            chosen = pairs
        else:
            break
    return chosen
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_device_probe_memory.py -v`
Expected: PASS for all three tests.

- [ ] **Step 5: Commit**

```bash
git add src/engine/device_probe.py tests/test_device_probe_memory.py
git commit -m "feat(probe): VRAM-aware max_batch_pairs suggestion helper"
```

---

### Task 8: Run the device memory probe at startup and expose the suggestion

**Files:**
- Modify: `src/engine/device_probe.py`
- Modify: `src/engine/async_engine.py`
- Test: `tests/test_device_probe.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_device_probe.py`:

```python
def test_device_profile_carries_suggested_max_batch_pairs():
    from src.engine.device_probe import DeviceProfile, ProbeResult

    profile = DeviceProfile(
        device="cpu",
        probes=[ProbeResult(batch_size=1, pairs=4, elapsed_ms=10.0)],
        suggested_batch_size=1,
        user_pinned_batch_size=False,
        suggested_max_batch_pairs=256,
    )
    assert profile.to_dict()["suggested_max_batch_pairs"] == 256
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_device_probe.py::test_device_profile_carries_suggested_max_batch_pairs -v`
Expected: FAIL — `DeviceProfile.__init__` has no `suggested_max_batch_pairs`.

- [ ] **Step 3: Add the field to `DeviceProfile`**

In `src/engine/device_probe.py`, update the dataclass and `to_dict()`:

```python
@dataclass
class DeviceProfile:
    device: str
    probes: List[ProbeResult]
    suggested_batch_size: int
    user_pinned_batch_size: bool
    suggested_max_batch_pairs: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "probes": [
                {
                    "batch_size": p.batch_size,
                    "pairs": p.pairs,
                    "elapsed_ms": round(p.elapsed_ms, 2),
                    "ms_per_pair": round(p.ms_per_pair, 3),
                }
                for p in self.probes
            ],
            "suggested_batch_size": self.suggested_batch_size,
            "user_pinned_batch_size": self.user_pinned_batch_size,
            "suggested_max_batch_pairs": self.suggested_max_batch_pairs,
        }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_device_probe.py::test_device_profile_carries_suggested_max_batch_pairs -v`
Expected: PASS.

- [ ] **Step 5: Populate the suggestion during `run_device_probe`**

In `src/engine/device_probe.py`, add a helper that reads the free device-memory fraction:

```python
def _device_free_fraction(device: str) -> float:
    """Fraction of the active device's memory currently free (0.0-1.0)."""
    try:
        if device == "cuda":
            import torch
            free, total = torch.cuda.mem_get_info()
            return free / total if total else 0.0
        if device == "mps":
            import torch
            total = torch.mps.recommended_max_memory()
            used = torch.mps.current_allocated_memory()
            return (total - used) / total if total else 0.0
        import psutil
        vm = psutil.virtual_memory()
        return vm.available / vm.total if vm.total else 0.0
    except Exception:
        return 1.0  # unknown -> do not constrain batch size
```

Then, at the end of `run_device_probe`, just before building the `DeviceProfile`, compute the suggestion by re-running probe batches and measuring residual free memory:

```python
    from src.config import settings as _settings

    pair_candidates = [64, 128, 256, 512, 1024]

    def _free_after(pairs: int) -> float:
        n_requests = max(1, pairs // PROBE_DOCS_PER_REQUEST)
        probe_batch = BatchedRequest(
            batch_id=f"mem-probe-{pairs}",
            requests=[
                RerankRequest(
                    request_id=f"mem-{uuid.uuid4().hex[:6]}",
                    query=PROBE_QUERY,
                    documents=[PROBE_DOC] * PROBE_DOCS_PER_REQUEST,
                    return_documents=False,
                )
                for _ in range(n_requests)
            ],
        )
        try:
            handler.predict(probe_batch)
        except Exception:
            return 0.0  # did not fit / errored -> treat as over budget
        return _device_free_fraction(device)

    suggested_pairs = suggest_max_batch_pairs(
        candidates=pair_candidates,
        free_fraction_after=_free_after,
        safety_margin=_settings.device_mem_safety_margin,
    )
```

Update the `DeviceProfile(...)` construction to pass `suggested_max_batch_pairs=suggested_pairs`.

- [ ] **Step 6: Use the suggestion in the engine when the operator has not pinned `max_batch_pairs`**

In `src/engine/async_engine.py`, in `start()`, after the `self.device_profile = await loop.run_in_executor(...)` assignment, add:

```python
            if (
                self.device_profile is not None
                and self.device_profile.suggested_max_batch_pairs is not None
                and "RERANKER_MAX_BATCH_PAIRS" not in os.environ
            ):
                suggested = self.device_profile.suggested_max_batch_pairs
                self.request_queue.max_batch_pairs = suggested
                logger.info("max_batch_pairs_autotuned", value=suggested)
```

- [ ] **Step 7: Run the full suite**

Run: `pytest tests/ -x`
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add src/engine/device_probe.py src/engine/async_engine.py tests/test_device_probe.py
git commit -m "feat(engine): autotune max_batch_pairs from device memory probe at startup"
```

---

### Task 9: Benchmark validation + integration sweep

**Files:**
- None (validation only)

- [ ] **Step 1: Confirm the full suite is green**

Run: `pytest tests/ -x`
Expected: every test passes.

- [ ] **Step 2: Baseline benchmark (record before merging)**

With the model available locally, run the existing concurrent benchmark and record throughput + p95:

```bash
python benchmark_concurrent.py --requests 1000 --concurrency 50 2>&1 | tee /tmp/bench_after.txt
```

Expected: compared to a pre-Task-2 run (`git stash`/checkout main), materially higher pairs/sec and lower p95 at concurrency 50. Capture both numbers in the PR description.

- [ ] **Step 3: Verify device behavior on the target box**

```bash
curl -s http://localhost:8000/info | python -m json.tool
```

Expected: `device_profile.suggested_max_batch_pairs` is populated and non-null; `device` matches the deployment (`cuda`).

- [ ] **Step 4: FP8 smoke test (Ada only, optional lever)**

```bash
RERANKER_QUANTIZATION=fp8 RERANKER_MODEL_NAME=Qwen/Qwen3-Reranker-4B ./run.sh &
sleep 30
curl -s -X POST http://localhost:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{"query":"what is deep learning","documents":["deep learning is a subset of ML","the weather is nice"],"top_n":2}'
kill %1
```

Expected: 200 with sensible ranking (ML doc ranked first); logs show `quantization_enabled mode=fp8`. If the installed `torchao`/transformers FP8 API is unavailable, the log shows the import error; rerun with `RERANKER_QUANTIZATION=none` and record the installed `torch`/`transformers`/`torchao` versions in the PR.

- [ ] **Step 5: Open the PR**

```bash
gh pr create \
  --title "Phase 3A: cross-request batching + device-aware perf for Qwen3-Reranker-4B" \
  --body "Implements Part A of docs/superpowers/specs/2026-05-27-perf-telemetry-admin-ui-design.md. Includes before/after benchmark at 50 concurrent."
```

---

## Self-Review

**1. Spec coverage (Part A):**
- A1 score_pairs primitive + flat cross-request batching → Tasks 1, 2, 3
- A2 length bucketing → Task 1 (bucketing lives inside `score_pairs`)
- A3 VRAM-aware batch sizing → Tasks 7, 8
- A4 device-specific precision/kernels (bf16/fp32, flash-attn/SDPA, FP8/int8, CPU threads, max_length=256) → Tasks 0, 4, 5, 6
- A5 expected outcome / validation → Task 9

**2. Placeholder scan:** No "TBD/TODO/handle edge cases" language. Every code step shows complete code. The only runtime-version-gated item is the FP8 path, whose code is concrete (`TorchAoConfig("float8_dynamic_activation_float8_weight")`) and whose hardware behavior is validated in Task 9 Step 4.

**3. Type/signature consistency:**
- `score_pairs(pairs, instruction=None) -> List[float]` defined Task 1, called by the handler Task 2 and by `rerank()` Task 1.
- `build_load_kwargs(device, use_fp16, has_flash_attn)` defined and used Task 4.
- `build_quantization_config(device, quantization)` / `maybe_quantize_int8(model, device, quantization)` defined and wired Task 5.
- `resolve_cpu_threads(device, configured, cpu_count)` defined and used Task 6.
- `suggest_max_batch_pairs(candidates, free_fraction_after, safety_margin)` defined Task 7, used Task 8.
- `DeviceProfile.suggested_max_batch_pairs` added Task 8 and consumed in `async_engine.start()` same task.
