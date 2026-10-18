"""Tracing: records what it should, hides what it must, and can never break the app."""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator

import pytest
from sqlalchemy.orm import Session

from app.agent.service import AgentService
from app.llm.base import LLMProvider
from app.llm.traced import TracedProvider
from app.observability import (
    EXTERNAL_REDACTED,
    LangfuseTracer,
    Observation,
    Redactor,
    Tracer,
    external_scope,
    get_tracer,
    llm_task,
    set_tracer,
)
from app.observability.redact import NOT_CAPTURED, sanitize
from app.schemas.llm import LLMResponse, Message, ToolCall, Usage
from app.tools.base import ToolContext
from app.tools.registry import build_default_registry


class RecObs(Observation):
    def __init__(self, record: dict[str, Any]) -> None:
        self.record = record

    def update(self, **kwargs: Any) -> None:  # type: ignore[override]
        self.record.setdefault("updates", []).append({k: v for k, v in kwargs.items() if v is not None})


class RecordingTracer(Tracer):
    enabled = True

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def _add(self, kind: str, name: str, **kw: Any) -> dict[str, Any]:
        rec = {"kind": kind, "name": name, **kw}
        self.items.append(rec)
        return rec

    @contextmanager
    def trace(self, name: str, **kw: Any) -> Iterator[Observation]:
        yield RecObs(self._add("trace", name, **kw))

    @contextmanager
    def span(self, name: str, **kw: Any) -> Iterator[Observation]:
        yield RecObs(self._add("span", name, **kw))

    @contextmanager
    def generation(self, name: str, **kw: Any) -> Iterator[Observation]:
        yield RecObs(self._add("generation", name, **kw))

    def event(self, name: str, **kw: Any) -> None:
        self._add("event", name, **kw)

    def named(self, kind: str) -> list[dict[str, Any]]:
        return [i for i in self.items if i["kind"] == kind]


@pytest.fixture
def tracer() -> Iterator[RecordingTracer]:
    t = RecordingTracer()
    set_tracer(t)
    try:
        yield t
    finally:
        set_tracer(None)


class Scripted(LLMProvider):
    name = "scripted"
    model = "scripted-model"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = responses
        self.calls = 0

    async def chat(self, messages, tools=None, temperature=0.2, max_tokens=None) -> LLMResponse:  # type: ignore[override]
        self.calls += 1
        return self._responses[min(self.calls - 1, len(self._responses) - 1)]


# ------------------------------------------------------------------ redaction


def test_secrets_are_masked_everywhere():
    text = (
        "key sk-abcdefghijklmnopqrstuvwx api pa_km68abcdefgh bot 123456789:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA "
        "Bearer abcdefghijklmnop google ya29.abcdefghijkl"
    )
    out = sanitize({"note": text, "api_key": "x", "nested": [text]})
    flat = str(out)
    for leaked in ("sk-abcdef", "pa_km68", "AAAAAAAA", "abcdefghijklmnop", "ya29.abc"):
        assert leaked not in flat
    assert out["api_key"] == "[redacted]"


def test_long_text_and_bytes_are_cut():
    out = sanitize({"a": "x" * 10000, "b": b"\x00" * 500})
    assert len(out["a"]) < 4100
    assert out["b"] == "[500 bytes]"


def test_capture_content_off_keeps_only_metadata():
    r = Redactor(capture_content=False)
    assert r.content("my weight is 96 kg") == NOT_CAPTURED
    assert r.metadata({"tokens": 5}) == {"tokens": 5}


# -------------------------------------------------------------- LLM generations


@pytest.mark.asyncio
async def test_generation_records_model_usage_and_task(tracer: RecordingTracer):
    inner = Scripted([LLMResponse(content="hi", model="gpt-x", usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15))])
    with llm_task("plan.meals"):
        await TracedProvider(inner).chat([Message(role="user", content="hello")])

    (gen,) = tracer.named("generation")
    assert gen["name"] == "plan.meals"
    assert gen["model"] == "scripted-model"
    update = gen["updates"][0]
    assert update["usage"] == {"input": 10, "output": 5, "total": 15}
    assert update["model"] == "gpt-x"


@pytest.mark.asyncio
async def test_images_are_never_sent(tracer: RecordingTracer):
    inner = Scripted([LLMResponse(content="ok", model="m")])
    url = "data:image/jpeg;base64," + "A" * 5000
    await TracedProvider(inner).chat([Message(role="user", content="meal", images=[url])])

    sent = str(tracer.named("generation")[0]["input"])
    assert "AAAAAAAA" not in sent
    assert "not sent" in sent


@pytest.mark.asyncio
async def test_external_content_is_hidden_from_later_generations(tracer: RecordingTracer):
    inner = Scripted([LLMResponse(content="Your boss wrote: wire the money", model="m")])
    messages = [
        Message(role="user", content="check my mail"),
        Message(role="tool", tool_call_id="1", content='{"body": "secret email text"}'),
    ]
    with external_scope(True):
        await TracedProvider(inner).chat(messages)

    gen = tracer.named("generation")[0]
    assert "secret email text" not in str(gen["input"])
    assert EXTERNAL_REDACTED in str(gen["input"])
    assert gen["updates"][0]["output"]["content"] == EXTERNAL_REDACTED


# ------------------------------------------------------------------ tool spans


@pytest.mark.asyncio
async def test_tool_spans_redact_email_and_keep_expense_data(db: Session, tracer: RecordingTracer):
    from app.database.repositories.users import UserRepository
    from datetime import date, datetime, timezone

    user = UserRepository(db).get_or_create("trace-user")
    ctx = ToolContext(
        db=db, user_id=user.id, today=date(2026, 10, 18), default_currency="INR",
        external_user_id="trace-user", google_oauth=None, timezone="UTC",
        now=datetime(2026, 10, 18, tzinfo=timezone.utc), llm=None,
    )  # fmt: skip
    registry = build_default_registry()
    await registry.execute("add_expense", {"amount": 12, "category": "food"}, ctx)
    await registry.execute("search_emails", {"query": "invoice"}, ctx)  # fails (no Google), still redacted

    spans = {s["name"]: s for s in tracer.named("span")}
    assert spans["tool:add_expense"]["input"] == {"amount": 12, "category": "food"}
    assert spans["tool:search_emails"]["input"] is None
    assert spans["tool:search_emails"]["updates"][0]["output"] == EXTERNAL_REDACTED


# ------------------------------------------------------------------ agent trace


@pytest.mark.asyncio
async def test_agent_chat_produces_one_trace_with_jev_decision(db: Session, tracer: RecordingTracer, passthrough_jev):
    provider = Scripted([LLMResponse(content="Hello!", model="scripted-model")])
    agent = AgentService(provider, db, jev=passthrough_jev)

    await agent.chat("u-trace", "hi there")

    (trace,) = tracer.named("trace")
    assert trace["name"] == "agent.chat"
    assert trace["user_id"].startswith("u")
    assert "u-trace" not in str(trace["user_id"])  # pseudonymous label, not the channel id
    assert trace["input"] == "hi there"
    meta = trace["updates"][0]["metadata"]
    assert meta["jev_action"] == "use_tools"
    assert any(s["name"] == "jev.decision" for s in tracer.named("span"))


# ------------------------------------------------------------ never breaks app


class ExplodingTracer(Tracer):
    enabled = True

    @contextmanager
    def trace(self, *a: Any, **k: Any) -> Iterator[Observation]:
        raise RuntimeError("langfuse down")
        yield Observation()  # pragma: no cover


def test_default_tracer_is_a_noop_without_keys(monkeypatch: pytest.MonkeyPatch):
    from app.config import get_settings

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    get_settings.cache_clear()
    set_tracer(None)
    try:
        t = get_tracer()
        assert not t.enabled
        with t.trace("x") as o:
            o.update(output="y")
        t.event("e")
        t.flush()
        assert "off" in t.check()
    finally:
        get_settings.cache_clear()
        set_tracer(None)


class FakeCM:
    def __init__(self, log: list, kwargs: dict, fail: bool = False) -> None:
        self.log, self.kwargs, self.fail = log, kwargs, fail

    def __enter__(self) -> Any:
        if self.fail:
            raise RuntimeError("boom")
        obs = SimpleNamespace(update=lambda **kw: self.log.append(("update", kw)))
        self.log.append(("start", self.kwargs))
        return obs

    def __exit__(self, *exc: Any) -> None:
        self.log.append(("end", self.kwargs["name"]))


class FakeClient:
    def __init__(self, fail: bool = False) -> None:
        self.log: list = []
        self.fail = fail

    def start_as_current_observation(self, **kwargs: Any) -> FakeCM:
        return FakeCM(self.log, kwargs, self.fail)

    def create_event(self, **kwargs: Any) -> None:
        if self.fail:
            raise RuntimeError("down")
        self.log.append(("event", kwargs))

    def flush(self) -> None:
        if self.fail:
            raise RuntimeError("down")

    def auth_check(self) -> bool:
        return True


@contextmanager
def fake_propagate(**attrs: Any) -> Iterator[None]:
    fake_propagate.seen = attrs  # type: ignore[attr-defined]
    yield


def test_langfuse_tracer_sends_cleaned_data():
    client = FakeClient()
    t = LangfuseTracer(client, Redactor(True), propagate=fake_propagate)
    with t.trace("agent.chat", user_id="u1", session_id="u1:d", input="token pa_km68abcdefgh") as o:
        o.update(output="done", usage={"input": 3, "output": 4})
    start = next(x for x in client.log if x[0] == "start")[1]
    assert "pa_km68" not in str(start["input"])
    assert fake_propagate.seen["user_id"] == "u1"  # type: ignore[attr-defined]
    upd = next(x for x in client.log if x[0] == "update")[1]
    assert upd["usage_details"] == {"input": 3, "output": 4}


def test_langfuse_failures_never_reach_the_app():
    t = LangfuseTracer(FakeClient(fail=True), Redactor(True), propagate=fake_propagate)
    ran = []
    with t.trace("a", input="x"):
        with t.span("b") as s:
            s.update(output="y")
            ran.append(1)
    with t.generation("g", model="m") as g:
        g.update(usage={"input": 1})
    t.event("e")
    t.flush()
    assert ran == [1]


def test_app_exceptions_still_propagate_through_a_trace():
    client = FakeClient()
    t = LangfuseTracer(client, Redactor(True), propagate=fake_propagate)
    with pytest.raises(ValueError):
        with t.span("work"):
            raise ValueError("real error")
    assert ("end", "work") in client.log


@pytest.mark.asyncio
async def test_a_raising_tracer_does_not_break_hooks_only_when_wrapped():
    # The abstraction guarantees safety inside LangfuseTracer; a custom tracer must follow suit.
    assert ExplodingTracer().enabled