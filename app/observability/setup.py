"""Alias: `from app.observability.setup import get_tracer` is the same as importing it from
`app.observability` (one shared tracer, defined in the package's __init__.py)."""

from app.observability import build_tracer, get_tracer, set_tracer

__all__ = ["build_tracer", "get_tracer", "set_tracer"]