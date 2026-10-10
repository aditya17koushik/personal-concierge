import os

os.environ.setdefault("DATABASE_URL", "sqlite://")
os.environ.setdefault("APP_SECRET_KEY", "test-secret-key-not-for-production")
os.environ.setdefault("GOOGLE_CLIENT_ID", "test-client-id")
os.environ.setdefault("GOOGLE_CLIENT_SECRET", "test-client-secret")
os.environ.setdefault(
    "GOOGLE_REDIRECT_URI", "http://127.0.0.1:8000/auth/google/callback"
)

from typing import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import app.database.models  # noqa: F401  (registers tables on Base)
from app.database.session import Base
from app.decisions.schemas import Decision
from app.tools.registry import ToolRegistry


@pytest.fixture
def db() -> Iterator[Session]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


class PassthroughJev:
    """Test stand-in for Jev: allows every tool and skips the classification
    LLM call, so tests of tools/agent flow keep their scripted LLM sequences."""

    async def decide(self, message: str, registry: ToolRegistry) -> Decision:
        return Decision(
            intents=sorted(registry.domains()),
            action="use_tools",
            risk="low",
            reason="passthrough (tests)",
            allowed_tools=[t.name for t in registry.tools if not t.requires_approval],
        )


@pytest.fixture
def passthrough_jev() -> PassthroughJev:
    return PassthroughJev()