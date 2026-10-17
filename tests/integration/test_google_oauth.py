from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterator
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.auth import get_google_oauth_service, get_login_link_service
from app.api.deps import CurrentUser, get_current_user
from app.config import get_settings
from app.database.models import GoogleCredential, GoogleLoginLink
from app.database.repositories.google_credentials import GoogleCredentialRepository
from app.database.repositories.users import UserRepository
from app.integrations.google.links import LoginLinkService
from app.integrations.google.oauth import (
    CALENDAR_EVENTS,
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

FULL_SCOPE = f"openid email {GMAIL_READONLY} {CALENDAR_READONLY} {CALENDAR_EVENTS}"


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


def make_client(
    db: Session,
    handler: Callable[[httpx.Request], httpx.Response],
    as_user: str | None = "me",
) -> TestClient:
    """as_user=None leaves authentication real (no token => 401)."""
    app.dependency_overrides[get_google_oauth_service] = lambda: make_service(db, handler)
    app.dependency_overrides[get_login_link_service] = lambda: LoginLinkService(db, get_settings())
    if as_user is not None:
        app.dependency_overrides[get_current_user] = lambda: CurrentUser(0, as_user)
    return TestClient(app, follow_redirects=False)


@pytest.fixture(autouse=True)
def clear_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


def new_code(db: Session, external_id: str = "me") -> str:
    user = UserRepository(db).get_or_create(external_id)
    code, _expires = LoginLinkService(db, get_settings()).create(user.id)
    return code


def get_state(client: TestClient, db: Session, external_id: str = "me") -> str:
    res = client.get("/auth/google/login", params={"code": new_code(db, external_id)})
    return parse_qs(urlparse(res.headers["location"]).query)["state"][0]


def test_login_redirects_to_google_with_expected_params(db: Session):
    client = make_client(db, make_handler([]))

    res = client.get("/auth/google/login", params={"code": new_code(db)})

    assert res.status_code in (302, 307)
    assert res.headers["cache-control"] == "no-store"
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
    state = get_state(client, db)

    res = client.get("/auth/google/callback", params={"code": "abc", "state": state})

    assert res.status_code == 200
    assert res.json()["email"] == "me@example.com"

    cred = db.query(GoogleCredential).one()
    assert cred.access_token_enc != "access-1"  # not stored in plain text
    assert "access-1" not in cred.access_token_enc
    assert decrypt(cred.access_token_enc) == "access-1"
    assert decrypt(cred.refresh_token_enc) == "refresh-1"

    status = client.get("/auth/google/status").json()
    assert status["connected"] is True
    assert status["email"] == "me@example.com"
    assert make_service(db, make_handler([])).get_status("nobody") == {"connected": False}


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
    state = get_state(client, db)

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
    state = get_state(client, db)

    res = client.get("/auth/google/callback", params={"code": "abc", "state": state})

    assert res.status_code == 400
    assert db.query(GoogleCredential).count() == 0


def test_disconnect_deletes_credentials(db: Session):
    client = make_client(db, make_handler([]))
    state = get_state(client, db)
    client.get("/auth/google/callback", params={"code": "abc", "state": state})

    res = client.delete("/auth/google")

    assert res.json() == {"disconnected": True}
    assert db.query(GoogleCredential).count() == 0
    assert client.delete("/auth/google").json() == {
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


# ----------------------------------------------- single-use login links + auth


def test_login_link_works_once_only(db: Session):
    client = make_client(db, make_handler([]))
    code = new_code(db)

    first = client.get("/auth/google/login", params={"code": code})
    second = client.get("/auth/google/login", params={"code": code})

    assert first.status_code in (302, 307)
    assert second.status_code == 400
    assert "already used" in second.json()["detail"]


def test_login_link_expires(db: Session):
    client = make_client(db, make_handler([]))
    code = new_code(db)
    link = db.query(GoogleLoginLink).one()
    link.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    db.commit()

    res = client.get("/auth/google/login", params={"code": code})

    assert res.status_code == 400


@pytest.mark.parametrize("code", ["x" * 20, "short", "A" * 200])
def test_unknown_or_malformed_login_codes_are_rejected(db: Session, code: str):
    client = make_client(db, make_handler([]))
    assert client.get("/auth/google/login", params={"code": code}).status_code in (400, 422)


def test_login_codes_are_stored_hashed(db: Session):
    code = new_code(db)
    stored = db.query(GoogleLoginLink).one()
    assert code not in stored.code_hash and len(stored.code_hash) == 64


def test_the_login_link_binds_google_to_the_link_owner(db: Session):
    client = make_client(db, make_handler([]))
    bob_state = get_state(client, db, "bob")

    client.get("/auth/google/callback", params={"code": "abc", "state": bob_state})

    from app.database.models import User

    cred = db.query(GoogleCredential).one()
    owner = db.get(User, cred.user_id)
    assert owner is not None
    assert owner.external_id == "bob"


def test_creating_a_link_requires_a_token_and_returns_a_working_url(db: Session):
    anonymous = make_client(db, make_handler([]), as_user=None)
    assert anonymous.post("/auth/google/link").status_code == 401

    client = make_client(db, make_handler([]))
    user = UserRepository(db).get_or_create("me")
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(user.id, "me")
    body = client.post("/auth/google/link").json()

    assert "/auth/google/login?code=" in body["url"]
    assert body["expires_in_minutes"] == 10
    path = body["url"].split("127.0.0.1:8000", 1)[1]
    assert client.get(path).status_code in (302, 307)


def test_google_endpoints_reject_anonymous_callers(db: Session):
    client = make_client(db, make_handler([]), as_user=None)

    assert client.get("/auth/google/status").status_code == 401
    assert client.delete("/auth/google").status_code == 401