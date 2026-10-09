from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterator
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.auth import get_google_oauth_service
from app.config import get_settings
from app.database.models import GoogleCredential
from app.database.repositories.google_credentials import GoogleCredentialRepository
from app.database.repositories.users import UserRepository
from app.integrations.google.oauth import (
    CALENDAR_READONLY,
    GMAIL_READONLY,
    TOKEN_URL,
    USERINFO_URL,
    GoogleAuthError,
    GoogleNotConnectedError,
    GoogleOAuthService,
)
from app.main import app
from app.security.crypto import decrypt

FULL_SCOPE = f"openid email {GMAIL_READONLY} {CALENDAR_READONLY}"


def make_handler(
    calls: list[tuple[str, dict[str, list[str]]]],
    auth_code_response: dict[str, Any] | None = None,
    refresh_status: int = 200,
) -> Callable[[httpx.Request], httpx.Response]:
    code_response = auth_code_response or {
        "access_token": "access-1",
        "refresh_token": "refresh-1",
        "expires_in": 3600,
        "scope": FULL_SCOPE,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        body = parse_qs(request.content.decode()) if request.content else {}
        calls.append((url, body))

        if url.startswith(TOKEN_URL):
            if body.get("grant_type") == ["refresh_token"]:
                if refresh_status != 200:
                    return httpx.Response(refresh_status, json={"error": "invalid_grant"})
                return httpx.Response(
                    200, json={"access_token": "access-2", "expires_in": 3600}
                )
            return httpx.Response(200, json=code_response)
        if url.startswith(USERINFO_URL):
            return httpx.Response(200, json={"email": "me@example.com"})
        return httpx.Response(200)  # revoke

    return handler


def make_service(db: Session, handler: Callable[[httpx.Request], httpx.Response]):
    return GoogleOAuthService(db, get_settings(), transport=httpx.MockTransport(handler))


def make_client(db: Session, handler: Callable[[httpx.Request], httpx.Response]) -> TestClient:
    app.dependency_overrides[get_google_oauth_service] = lambda: make_service(db, handler)
    return TestClient(app, follow_redirects=False)


@pytest.fixture(autouse=True)
def clear_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


def get_state(client: TestClient, user_id: str = "me") -> str:
    res = client.get("/auth/google/login", params={"user_id": user_id})
    return parse_qs(urlparse(res.headers["location"]).query)["state"][0]


def test_login_redirects_to_google_with_expected_params(db: Session):
    client = make_client(db, make_handler([]))

    res = client.get("/auth/google/login", params={"user_id": "me"})

    assert res.status_code in (302, 307)
    url = urlparse(res.headers["location"])
    query = parse_qs(url.query)
    assert url.netloc == "accounts.google.com"
    assert query["client_id"] == ["test-client-id"]
    assert query["access_type"] == ["offline"]
    assert query["prompt"] == ["consent"]
    assert GMAIL_READONLY in query["scope"][0]
    assert "state" in query


def test_callback_stores_encrypted_tokens(db: Session):
    calls: list[tuple[str, dict[str, list[str]]]] = []
    client = make_client(db, make_handler(calls))
    state = get_state(client)

    res = client.get("/auth/google/callback", params={"code": "abc", "state": state})

    assert res.status_code == 200
    assert res.json()["email"] == "me@example.com"

    cred = db.query(GoogleCredential).one()
    assert cred.access_token_enc != "access-1"  # not stored in plain text
    assert "access-1" not in cred.access_token_enc
    assert decrypt(cred.access_token_enc) == "access-1"
    assert decrypt(cred.refresh_token_enc) == "refresh-1"

    status = client.get("/auth/google/status", params={"user_id": "me"}).json()
    assert status["connected"] is True
    assert status["email"] == "me@example.com"
    assert client.get("/auth/google/status", params={"user_id": "nobody"}).json() == {
        "connected": False
    }


def test_callback_rejects_bad_state(db: Session):
    client = make_client(db, make_handler([]))
    res = client.get("/auth/google/callback", params={"code": "abc", "state": "garbage"})
    assert res.status_code == 400


def test_callback_user_denied(db: Session):
    client = make_client(db, make_handler([]))
    res = client.get("/auth/google/callback", params={"error": "access_denied"})
    assert res.status_code == 400


def test_callback_missing_scopes_stores_nothing(db: Session):
    handler = make_handler(
        [],
        auth_code_response={
            "access_token": "a",
            "refresh_token": "r",
            "expires_in": 3600,
            "scope": "openid email",
        },
    )
    client = make_client(db, handler)
    state = get_state(client)

    res = client.get("/auth/google/callback", params={"code": "abc", "state": state})

    assert res.status_code == 400
    assert db.query(GoogleCredential).count() == 0


def test_callback_without_refresh_token_on_first_connect(db: Session):
    handler = make_handler(
        [],
        auth_code_response={
            "access_token": "a",
            "expires_in": 3600,
            "scope": FULL_SCOPE,
        },
    )
    client = make_client(db, handler)
    state = get_state(client)

    res = client.get("/auth/google/callback", params={"code": "abc", "state": state})

    assert res.status_code == 400
    assert db.query(GoogleCredential).count() == 0


def test_disconnect_deletes_credentials(db: Session):
    client = make_client(db, make_handler([]))
    state = get_state(client)
    client.get("/auth/google/callback", params={"code": "abc", "state": state})

    res = client.delete("/auth/google", params={"user_id": "me"})

    assert res.json() == {"disconnected": True}
    assert db.query(GoogleCredential).count() == 0
    assert client.delete("/auth/google", params={"user_id": "me"}).json() == {
        "disconnected": False
    }


async def connect(db: Session, calls: list[tuple[str, dict[str, list[str]]]], **kw: Any):
    service = make_service(db, make_handler(calls, **kw))
    state = service.build_login_url("me").split("state=")[1].split("&")[0]
    from urllib.parse import unquote

    cred = await service.handle_callback("abc", unquote(state))
    return service, cred


@pytest.mark.asyncio
async def test_get_access_token_uses_cached_token_when_fresh(db: Session):
    calls: list[tuple[str, dict[str, list[str]]]] = []
    service, cred = await connect(db, calls)
    calls.clear()

    token = await service.get_access_token(cred.user_id)

    assert token == "access-1"
    assert calls == []  # no network call


@pytest.mark.asyncio
async def test_get_access_token_refreshes_when_expired(db: Session):
    calls: list[tuple[str, dict[str, list[str]]]] = []
    service, cred = await connect(db, calls)
    cred.expires_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    db.commit()

    token = await service.get_access_token(cred.user_id)

    assert token == "access-2"
    refreshed = GoogleCredentialRepository(db).get(cred.user_id)
    assert refreshed is not None
    assert decrypt(refreshed.access_token_enc) == "access-2"
    assert decrypt(refreshed.refresh_token_enc) == "refresh-1"  # kept


@pytest.mark.asyncio
async def test_refresh_failure_raises_reconnect_error(db: Session):
    calls: list[tuple[str, dict[str, list[str]]]] = []
    service, cred = await connect(db, calls, refresh_status=400)
    cred.expires_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    db.commit()

    with pytest.raises(GoogleAuthError):
        await service.get_access_token(cred.user_id)


@pytest.mark.asyncio
async def test_not_connected_user(db: Session):
    user = UserRepository(db).get_or_create("lonely")
    service = make_service(db, make_handler([]))

    with pytest.raises(GoogleNotConnectedError):
        await service.get_access_token(user.id)