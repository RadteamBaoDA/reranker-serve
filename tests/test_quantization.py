"""Quantization selection: FP8 config on Ada CUDA, int8 post-load on CPU, else no-op."""

import sys
import pytest

# ---------------------------------------------------------------------------
# The global conftest replaces sys.modules["torch"] with a MagicMock so that
# the bulk of the test suite stays fast and offline.  The tests in *this*
# module need the real torch (quantize_dynamic on a real nn.Module).
#
# Strategy: at collection time, temporarily pop the stub, import the real
# torch (which registers itself and all sub-packages), stash the real module
# object, then put the stub back.  An autouse fixture around each test swaps
# them in/out for the duration of the test.
# ---------------------------------------------------------------------------

def _acquire_real_torch():
    stub = sys.modules.pop("torch", None)
    try:
        import torch as _real  # noqa: F401 – loads the real extension
        real = sys.modules["torch"]
    finally:
        if stub is not None:
            sys.modules["torch"] = stub
        else:
            sys.modules.pop("torch", None)
    return real


_REAL_TORCH = _acquire_real_torch()


@pytest.fixture(autouse=True)
def _use_real_torch():
    """Install real torch for the test, restore the conftest stub afterward."""
    stub = sys.modules.get("torch")
    sys.modules["torch"] = _REAL_TORCH

    import src.models.qwen3_reranker as _qmod
    _saved_torch = _qmod.torch
    _qmod.torch = _REAL_TORCH

    yield

    sys.modules["torch"] = stub
    _qmod.torch = _saved_torch


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
    # quantize_dynamic swaps each Linear for a dynamic quantized Linear,
    # whose module path lives under torch.ao.nn.quantized.dynamic.
    assert any("quantized.dynamic" in type(m).__module__ for m in out.modules())


def test_fp8_on_cuda_builds_config():
    pytest.importorskip("torchao")
    cfg = build_quantization_config(device="cuda", quantization="fp8")
    assert cfg is not None
    assert "TorchAo" in type(cfg).__name__
