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
