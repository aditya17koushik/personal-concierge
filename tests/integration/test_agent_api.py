from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.agent.service import AgentService
from app.api.agent import get_agent_service
from app.llm.base import LLMError, LLMProvider
from app.main import app
from app.schemas.llm import LLMResponse, Message


class FakeProvider(LLMProvider):
    name = "fake"
    model = "fake-model"

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.received: list[Message] = []

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if self.error:
            raise self.error
        self.received = messages
        return LLMResponse(content="Hello from fake", model="fake-model")


def make_client(provider: FakeProvider, db: Session) -> TestClient:
    app.dependency_overrides[get_agent_service] = lambda: AgentService(provider, db)
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


def test_chat_success(db: Session):
    provider = FakeProvider()
    client = make_client(provider, db)

    res = client.post("/agent/chat", json={"user_id": "u1", "message": "  hi  "})

    assert res.status_code == 200
    assert res.json() == {
        "reply": "Hello from fake",
        "provider": "fake",
        "model": "fake-model",
        "tools_used": [],
    }
    assert provider.received[0].role == "system"
    assert provider.received[1].content == "hi"


@pytest.mark.parametrize(
    "payload",
    [
        {"user_id": "u1", "message": ""},
        {"user_id": "u1", "message": "   "},
        {"user_id": "", "message": "hi"},
        {"message": "hi"},
    ],
)
def test_chat_validation(payload: dict[str, str], db: Session):
    client = make_client(FakeProvider(), db)
    assert client.post("/agent/chat", json=payload).status_code == 422


def test_chat_llm_failure_returns_502(db: Session):
    client = make_client(FakeProvider(error=LLMError("boom")), db)

    res = client.post("/agent/chat", json={"user_id": "u1", "message": "hi"})

    assert res.status_code == 502
    assert "boom" not in res.text