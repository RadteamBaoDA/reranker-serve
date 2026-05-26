"""OpenTelemetry initialization. Lazy-imported; only touched when enable_otel=True."""

from __future__ import annotations

from typing import Optional


_in_memory_exporter = None  # for tests


def init_otel(use_in_memory_exporter: bool = False):
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    existing = trace.get_tracer_provider()
    if isinstance(existing, SdkTracerProvider):
        # Already initialized in this process — re-use rather than crash on set.
        return existing

    resource = Resource.create({})  # service.name comes from OTEL_SERVICE_NAME
    provider = SdkTracerProvider(resource=resource)

    if use_in_memory_exporter:
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )
        global _in_memory_exporter
        _in_memory_exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(_in_memory_exporter))
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

    trace.set_tracer_provider(provider)
    return provider


def instrument_fastapi(app) -> None:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    FastAPIInstrumentor().instrument_app(app)


def get_tracer():
    from opentelemetry import trace
    return trace.get_tracer("reranker-serve")


def get_in_memory_exporter():
    """Test-only accessor."""
    return _in_memory_exporter
