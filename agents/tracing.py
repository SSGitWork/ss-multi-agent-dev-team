"""
Distributed tracing with Phoenix/Arize via OpenTelemetry.

Every pipeline run creates a single root span. Each agent invocation
is a child span with custom metadata: agent_name, token_count,
tool_calls, duration_ms.

If Phoenix is not available, tracing degrades gracefully to no-ops.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

from agents.config import get_settings

logger = logging.getLogger(__name__)

# Lazy initialization of OpenTelemetry tracer
_tracer = None
_initialized = False


def _init_tracing() -> None:
    """Initialize Phoenix tracing. Called once on first use."""
    global _tracer, _initialized
    if _initialized:
        return
    _initialized = True

    settings = get_settings()
    if not settings.phoenix.enabled:
        logger.info("Phoenix tracing is disabled.")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({
            "service.name": settings.phoenix.project_name,
        })
        provider = TracerProvider(resource=resource)

        # Try Phoenix OTLP exporter
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            phoenix_exporter = OTLPSpanExporter(
                endpoint=f"{settings.phoenix.endpoint}/v1/traces"
            )
            provider.add_span_processor(BatchSpanProcessor(phoenix_exporter))
            logger.info("Phoenix OTLP exporter configured at %s", settings.phoenix.endpoint)
        except Exception as exc:
            logger.warning("Phoenix OTLP exporter unavailable: %s. Using console.", exc)
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(settings.phoenix.project_name)
        logger.info("OpenTelemetry tracing initialized.")

    except ImportError as exc:
        logger.warning("OpenTelemetry not available: %s. Tracing disabled.", exc)
    except Exception as exc:
        logger.warning("Tracing initialization failed: %s. Continuing without tracing.", exc)


def get_tracer():
    """Get the OpenTelemetry tracer (lazy init)."""
    _init_tracing()
    return _tracer


# Span context managers
@contextmanager
def root_span(name: str, attributes: Optional[Dict[str, Any]] = None) -> Generator:
    """Create a root span for the entire pipeline run."""
    tracer = get_tracer()
    if tracer is None:
        yield _NoOpSpan()
        return

    with tracer.start_as_current_span(name, attributes=attributes or {}) as span:
        span.set_attribute("span.type", "root")
        start = time.time()
        try:
            yield span
        finally:
            duration_ms = (time.time() - start) * 1000
            span.set_attribute("duration_ms", duration_ms)


@contextmanager
def agent_span(
    agent_name: str,
    parent_span=None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Generator:
    """Create a child span for an agent invocation."""
    tracer = get_tracer()
    if tracer is None:
        yield _NoOpSpan()
        return

    attrs = {"agent_name": agent_name, "span.type": "agent"}
    if attributes:
        attrs.update(attributes)

    with tracer.start_as_current_span(f"agent.{agent_name}", attributes=attrs) as span:
        start = time.time()
        try:
            yield span
        finally:
            duration_ms = (time.time() - start) * 1000
            span.set_attribute("duration_ms", duration_ms)


def record_span_metadata(span, **kwargs) -> None:
    """Record custom metadata on a span (safe no-op if span is None)."""
    if span is None or isinstance(span, _NoOpSpan):
        return
    for key, value in kwargs.items():
        try:
            if isinstance(value, (str, int, float, bool)):
                span.set_attribute(key, value)
            else:
                span.set_attribute(key, str(value))
        except Exception:
            pass


class _NoOpSpan:
    """Fallback span when tracing is disabled."""
    def set_attribute(self, key: str, value: Any) -> None:
        """Attach an attribute to the current tracing span."""
        pass

    def add_event(self, name: str, attributes: Optional[Dict] = None) -> None:
        """Record an event on the current tracing span."""
        pass
