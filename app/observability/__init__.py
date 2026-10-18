"""Observability: `get_tracer()` is the one accessor the app uses."""

import logging

from app.observability.redact import EXTERNAL_REDACTED, Redactor
from app.observability.tracing import (
    LangfuseTracer,
    Observation,
    Tracer,
    current_task,
    external_scope,
    external_seen,
    llm_task,
    mark_external,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EXTERNAL_REDACTED",
    "LangfuseTracer",
    "Observation",
    "Redactor",
    "Tracer",
    "build_tracer",
    "current_task",
    "external_scope",
    "external_seen",
    "get_tracer",
    "llm_task",
    "mark_external",
    "set_tracer",
]

_tracer: Tracer | None = None


def build_tracer() -> Tracer:
    """Langfuse when both keys are set (and the package is installed); otherwise the no-op."""
    from app.config import get_settings

    settings = get_settings()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return Tracer()
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_base_url,
            environment=settings.langfuse_environment or settings.app_env,
            sample_rate=settings.langfuse_sample_rate,
        )
        return LangfuseTracer(client, Redactor(settings.langfuse_capture_content))
    except Exception:
        logger.warning("Langfuse could not be started; tracing is off.", exc_info=True)
        return Tracer()


def get_tracer() -> Tracer:
    global _tracer
    if _tracer is None:
        _tracer = build_tracer()
    return _tracer


def set_tracer(tracer: Tracer | None) -> None:
    """Install a tracer (tests, or None to rebuild from settings on next use)."""
    global _tracer
    _tracer = tracer