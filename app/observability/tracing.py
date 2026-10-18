"""Tracing, with Langfuse behind a thin layer of our own.

The rest of the app only knows `Tracer`. The default tracer does nothing, so tracing costs nothing and
changes nothing until LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set. Tracing can NEVER
break a request: every Langfuse call is wrapped, and a failure only logs a warning.

    with tracer.trace("agent.chat", user_id="u1", session_id="u1:2026-10-18", input=message) as t:
        ...
        t.update(output=reply)
    with tracer.span("tool:log_meal", as_type="tool", input=args) as s: ...
    with tracer.generation("agent", model="gpt-4o-mini", input=messages) as g:
        g.update(output=text, usage={"input": 10, "output": 5})
    tracer.event("reminder.sent", metadata={...})
"""

import contextvars
import logging
from contextlib import contextmanager
from typing import Any, Iterator

from app.observability.redact import Redactor

logger = logging.getLogger(__name__)

# What the model call is FOR ("agent", "plan.workout", "plan.meals", "photo.estimate"): it names the
# generation in the trace so cost can be broken down by feature.
_task: contextvars.ContextVar[str] = contextvars.ContextVar("llm_task", default="llm")


@contextmanager
def llm_task(name: str) -> Iterator[None]:
    token = _task.set(name)
    try:
        yield
    finally:
        _task.reset(token)


def current_task() -> str:
    return _task.get()


# Set once email or calendar content has entered the conversation: from then on tool results and
# model text in traces are replaced by a placeholder (third parties wrote that text).
_external: contextvars.ContextVar[bool] = contextvars.ContextVar("external_content", default=False)


@contextmanager
def external_scope(initial: bool = False) -> Iterator[None]:
    token = _external.set(initial)
    try:
        yield
    finally:
        _external.reset(token)


def mark_external() -> None:
    _external.set(True)


def external_seen() -> bool:
    return _external.get()


class Observation:
    """What a `with` block yields. Does nothing here; Langfuse's version records."""

    def update(
        self,
        *,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
        usage: dict[str, int] | None = None,
        model: str | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        return None


class Tracer:
    """The no-op tracer (the default). Subclass and override to record."""

    enabled = False

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> Iterator[Observation]:
        yield Observation()

    @contextmanager
    def span(
        self,
        name: str,
        *,
        as_type: str = "span",
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Observation]:
        yield Observation()

    @contextmanager
    def generation(
        self,
        name: str,
        *,
        model: str,
        input: Any = None,
        params: dict[str, Any] | None = None,
    ) -> Iterator[Observation]:
        yield Observation()

    def event(
        self, name: str, *, metadata: dict[str, Any] | None = None, level: str | None = None
    ) -> None:
        return None

    def flush(self) -> None:
        return None

    def check(self) -> str:
        return "Tracing is off (set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to turn it on)."


class _LangfuseObservation(Observation):
    def __init__(self, obs: Any, redactor: Redactor) -> None:
        self._obs = obs
        self._redactor = redactor

    def update(
        self,
        *,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
        usage: dict[str, int] | None = None,
        model: str | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        try:
            kwargs: dict[str, Any] = {}
            if output is not None:
                kwargs["output"] = self._redactor.content(output)
            if metadata:
                kwargs["metadata"] = self._redactor.metadata(metadata)
            if usage:
                kwargs["usage_details"] = {k: int(v) for k, v in usage.items()}
            if model:
                kwargs["model"] = model
            if level:
                kwargs["level"] = level
            if status_message:
                kwargs["status_message"] = status_message[:300]
            if kwargs:
                self._obs.update(**kwargs)
        except Exception:
            logger.warning("Langfuse update failed", exc_info=True)


class LangfuseTracer(Tracer):
    enabled = True

    def __init__(self, client: Any, redactor: Redactor, propagate: Any = None) -> None:
        self._client = client
        self._redactor = redactor
        if propagate is None:
            from langfuse import propagate_attributes

            propagate = propagate_attributes
        self._propagate = propagate

    # A context manager that can never raise into the caller's code path.
    @contextmanager
    def _observe(
        self,
        name: str,
        as_type: str,
        input: Any,
        metadata: dict[str, Any] | None,
        extra: dict[str, Any] | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> Iterator[Observation]:
        stack: list[Any] = []
        obs: Observation = Observation()
        try:
            if attrs:
                prop = self._propagate(**attrs)
                prop.__enter__()
                stack.append(prop)
            kwargs: dict[str, Any] = {"name": name, "as_type": as_type}
            if input is not None:
                kwargs["input"] = self._redactor.content(input)
            if metadata:
                kwargs["metadata"] = self._redactor.metadata(metadata)
            kwargs.update(extra or {})
            cm = self._client.start_as_current_observation(**kwargs)
            raw = cm.__enter__()
            stack.append(cm)
            obs = _LangfuseObservation(raw, self._redactor)
        except Exception:
            logger.warning("Langfuse could not start %s", name, exc_info=True)
            for cm in reversed(stack):
                try:
                    cm.__exit__(None, None, None)
                except Exception:
                    pass
            stack = []
            obs = Observation()

        try:
            yield obs
        except BaseException as exc:
            obs.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
            self._close(stack, type(exc), exc)
            raise
        else:
            self._close(stack, None, None)

    @staticmethod
    def _close(stack: list[Any], exc_type: Any, exc: Any) -> None:
        for cm in reversed(stack):
            try:
                cm.__exit__(exc_type, exc, exc.__traceback__ if exc else None)
            except Exception:
                logger.warning("Langfuse could not close an observation", exc_info=True)

    def trace(
        self,
        name: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> Any:
        attrs: dict[str, Any] = {"trace_name": name}
        if user_id:
            attrs["user_id"] = str(user_id)[:200]
        if session_id:
            attrs["session_id"] = str(session_id)[:200]
        if tags:
            attrs["tags"] = tags
        return self._observe(name, "span", input, metadata, attrs=attrs)

    def span(
        self,
        name: str,
        *,
        as_type: str = "span",
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        return self._observe(name, as_type, input, metadata)

    def generation(
        self,
        name: str,
        *,
        model: str,
        input: Any = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        extra: dict[str, Any] = {"model": model}
        if params:
            extra["model_parameters"] = {
                k: v for k, v in params.items() if isinstance(v, (str, int, float, bool)) and v is not None
            }
        return self._observe(name, "generation", input, None, extra=extra)

    def event(
        self, name: str, *, metadata: dict[str, Any] | None = None, level: str | None = None
    ) -> None:
        try:
            kwargs: dict[str, Any] = {"name": name}
            if metadata:
                kwargs["metadata"] = self._redactor.metadata(metadata)
            if level:
                kwargs["level"] = level
            self._client.create_event(**kwargs)
        except Exception:
            logger.warning("Langfuse event failed", exc_info=True)

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception:
            logger.warning("Langfuse flush failed", exc_info=True)

    def check(self) -> str:
        try:
            ok = self._client.auth_check()
        except Exception as exc:
            return f"Langfuse is configured but not reachable ({type(exc).__name__}). Traces will be retried."
        return "Langfuse tracing is on." if ok else "Langfuse rejected the keys: check LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_HOST."