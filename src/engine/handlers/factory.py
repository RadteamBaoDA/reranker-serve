from typing import Optional

from src.engine.handlers.base import BaseHandler
from src.engine.handlers.cross_encoder import CrossEncoderHandler
from src.engine.handlers.qwen import QwenRerankerHandler


def get_handler(model_path: str, device: str, max_length: int, use_fp16: bool) -> BaseHandler:
    """Return appropriate handler based on model name/path."""
    lowered = model_path.lower()
    if "qwen3" in lowered and "reranker" in lowered:
        return QwenRerankerHandler(model_path, device, max_length, use_fp16)
    return CrossEncoderHandler(model_path, device, max_length, use_fp16)
