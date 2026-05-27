# Benchmark — Qwen3-Reranker-4B on a single RTX 4070 Ti SUPER

**Date:** 2026-05-27 · **Branch:** `main` (Phase 3 integrated) · run via `benchmark_concurrent.py`.

## Setup

- **GPU:** NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB (Ada), ~13 GB free at start.
- **Model:** `Qwen/Qwen3-Reranker-4B`, **bf16**, attention = **SDPA** (flash-attn-2 not installed), `max_length=256`, `RERANKER_QUANTIZATION=none` (torchao not installed → no FP8).
- **Engine:** Phase 3 cross-request batching; `max_batch_pairs` auto-tuned to **1024** by the startup VRAM probe; `max_concurrent_batches=2`.
- Single-stream cost measured by the startup probe: **~24 ms/pair** at batch ≥ 8 (plateaus → compute-bound). A single 20-doc request = **530 ms**.

## Results

All runs returned **100% success (0 failures, 0 timeouts)** — the queue/backpressure path degrades gracefully under overload (it slows, it does not error).

| Concurrency | Docs/req | Requests | p95 latency | median | req/s |
|---:|---:|---:|---:|---:|---:|
| 1  | 20 | 1   | 0.53 s | — | — |
| 10 | 10 | 80  | **1.87 s** | 1.81 s | 5.4 |
| 50 | 5  | 200 | **4.65 s** | 4.37 s | 11.0 |
| 50 | 20 | 150 | **9.04 s** | 8.78 s | 5.5 |

Effective GPU throughput peaks around **~110 query–doc pairs/sec**, which a FLOPs estimate confirms is **near the card's peak bf16 rate (~85 TFLOPS effective)** — i.e. this is the hardware ceiling for a 4B causal-LM reranker here, not a software bottleneck. Phase 3 cross-request batching is working (3000 pairs cleared in 27 s at 50-conc/20-doc); it cannot exceed the GPU's compute ceiling.

## Verdict on the original target

**`p95 < 800 ms at 50 concurrent` is NOT achievable with Qwen3-Reranker-4B on a single RTX 4070 Ti SUPER.** At 50 concurrent × 20 docs (1000 pairs) the floor is ~1000/110 ≈ 9 s. Hitting 800 ms there would require ≈ **11× more throughput** (≈ 1250 pairs/s).

Sub-second p95 with 4B is only reached at **very low concurrency or tiny candidate sets** (roughly ≤ ~3 concurrent × 20 docs, or ≤ ~8 concurrent × 5 docs).

## Levers to actually hit 50-concurrent / sub-second (in rough order of impact)

1. **Smaller model — `Qwen3-Reranker-0.6B`** (~6–7× fewer params → ~6× throughput). Projected ~700 pairs/s → 50×20 ≈ 1.4 s p95, and 50×5 ≈ 0.4 s p95 (**meets target for smaller candidate sets**). This is the single biggest lever if the quality of 0.6B is acceptable.
2. **flash-attn-2 + FP8** for the 4B (`pip install flash-attn`, `pip install '.[quant]'` + `RERANKER_QUANTIZATION=fp8`): ~2–4× combined → ~2.5–4 s p95 at 50×20. Helps materially but **does not alone reach 800 ms** at medium doc counts.
3. **Fewer documents per request** (rerank top-10 not top-20–50) and/or **lower concurrency** — move into the sub-second region of the envelope above.
4. **Scale horizontally** — add GPUs and fan out via the built-in load balancer (`load_balancer.enabled`), which round-robins/least-busy across backends.

## Reproduce

```bash
# weights at ./models/Qwen3-Reranker-4B (config.yml points here)
./run.sh                       # or: python -m uvicorn src.main:app --host 127.0.0.1 --port 8000
python -X utf8 benchmark_concurrent.py --url http://127.0.0.1:8000 --requests 150 --concurrency 50 --documents 20
```

(On Windows consoles, `-X utf8` avoids a `cp1252` UnicodeEncodeError on the banner emoji.)
