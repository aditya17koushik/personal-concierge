import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

import httpx
from sqlalchemy.orm import Session

from app.config import Settings
from app.database.models import GoogleCredential
from app.database.repositories.google_credentials import GoogleCredentialRepository
from app.database.repositories.users import UserRepository
from app.security.crypto import CryptoError, decrypt, encrypt

logger = logging.getLogger(__name__)

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"

GMAIL_READONLY = "https://www.googleapis.com/auth/gmail.readonly"
CALENDAR_READONLY = "https://www.googleapis.com/auth/calendar.readonly"

# Read-only for now. Write scopes get added after Jev + approvals exist.
SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    GMAIL_READONLY,
    CALENDAR_READONLY,
]
REQUIRED_SCOPES = {GMAIL_READONLY, CALENDAR_READONLY}

STATE_TTL_SECONDS = 600
STATE_PURPOSE = "oauth-state"
EXPIRY_SKEW = timedelta(seconds=60)


class GoogleAuthError(Exception):
    """Safe-to-show error message about the Google connection."""


class GoogleConfigError(GoogleAuthError):
    """Google OAuth settings are missing."""


class GoogleNotConnectedError(GoogleAuthError):
    """The user has not connected a Google account."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    # SQLite returns naive datetimes; Postgres returns aware ones.
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


class GoogleOAuthService:
    def __init__(
        self,
        db: Session,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._db = db
        self._settings = settings
        self._transport = transport
        self._creds = GoogleCredentialRepository(db)

    @property
    def http_transport(self) -> httpx.AsyncBaseTransport | None:
        """Shared with Google API clients (Gmail, Calendar); mocked in tests."""
        return self._transport

    # ------------------------------------------------------------- login

    def build_login_url(self, external_user_id: str) -> str:
        self._require_config()

        state = encrypt(
            json.dumps({"u": external_user_id}), purpose=STATE_PURPOSE
        )
        params = {
            "client_id": self._settings.google_client_id,
            "redirect_uri": self._settings.google_redirect_uri,
            "response_type": "code",
            "scope": " ".join(SCOPES),
            "access_type": "offline",  # we need a refresh token
            "prompt": "consent",  # make Google return it every time
            "state": state,
        }
        return f"{AUTH_URL}?{urlencode(params)}"

    # ---------------------------------------------------------- callback

    async def handle_callback(self, code: str, state: str) -> GoogleCredential:
        self._require_config()
        external_id = self._read_state(state)

        tokens = await self._post_token(
            {
                "code": code,
                "client_id": self._settings.google_client_id,
                "client_secret": self._settings.google_client_secret,
                "redirect_uri": self._settings.google_redirect_uri,
                "grant_type": "authorization_code",
            },
            failure_message="Google rejected the authorization code. Please try logging in again.",
        )

        access_token = tokens.get("access_token")
        if not access_token:
            raise GoogleAuthError("Google did not return an access token.")

        granted = set(str(tokens.get("scope", "")).split())
        missing = REQUIRED_SCOPES - granted
        if missing:
            raise GoogleAuthError(
                "Some requested permissions were not granted. "
                "Please log in again and allow all of them."
            )

        user = await asyncio.to_thread(
            UserRepository(self._db).get_or_create, external_id
        )
        existing = await asyncio.to_thread(self._creds.get, user.id)

        refresh_token = tokens.get("refresh_token")
        if not refresh_token and existing:
            refresh_token = decrypt(existing.refresh_token_enc)
        if not refresh_token:
            raise GoogleAuthError(
                "Google did not return a refresh token. Remove this app from "
                "your Google account's third-party access, then try again."
            )

        email = await self._fetch_email(access_token)

        return await asyncio.to_thread(
            self._creds.upsert,
            user_id=user.id,
            google_email=email,
            access_token_enc=encrypt(access_token),
            refresh_token_enc=encrypt(refresh_token),
            expires_at=_now() + timedelta(seconds=int(tokens.get("expires_in", 3600))),
            scopes=" ".join(sorted(granted)),
        )

    # ------------------------------------------- token access (Steps 8/9)

    async def get_access_token(self, user_id: int) -> str:
        """Return a valid access token for the user, refreshing if needed."""
        cred = await asyncio.to_thread(self._creds.get, user_id)
        if cred is None:
            raise GoogleNotConnectedError("Google account is not connected.")

        if _as_utc(cred.expires_at) - EXPIRY_SKEW > _now():
            return decrypt(cred.access_token_enc)

        tokens = await self._post_token(
            {
                "client_id": self._settings.google_client_id,
                "client_secret": self._settings.google_client_secret,
                "refresh_token": decrypt(cred.refresh_token_enc),
                "grant_type": "refresh_token",
            },
            failure_message=(
                "Google access expired or was revoked. "
                "Please reconnect your Google account."
            ),
        )

        access_token = tokens.get("access_token")
        if not access_token:
            raise GoogleAuthError("Google did not return an access token.")

        new_refresh = tokens.get("refresh_token")
        await asyncio.to_thread(
            self._creds.update_access_token,
            cred,
            access_token_enc=encrypt(access_token),
            expires_at=_now() + timedelta(seconds=int(tokens.get("expires_in", 3600))),
            refresh_token_enc=encrypt(new_refresh) if new_refresh else None,
        )
        return access_token

    # ------------------------------------------------- status / disconnect

    def get_status(self, external_user_id: str) -> dict[str, Any]:
        user = UserRepository(self._db).get(external_user_id)
        cred = self._creds.get(user.id) if user else None
        if cred is None:
            return {"connected": False}
        return {
            "connected": True,
            "email": cred.google_email,
            "scopes": cred.scopes.split(),
        }

    async def disconnect(self, external_user_id: str) -> bool:
        user = await asyncio.to_thread(UserRepository(self._db).get, external_user_id)
        cred = await asyncio.to_thread(self._creds.get, user.id) if user else None
        if user is None or cred is None:
            return False

        # Best effort: tell Google to revoke, but always delete our copy.
        try:
            refresh_token = decrypt(cred.refresh_token_enc)
            async with self._client() as client:
                await client.post(REVOKE_URL, data={"token": refresh_token})
        except (CryptoError, httpx.HTTPError):
            logger.warning("Could not revoke Google token (continuing)")

        return await asyncio.to_thread(self._creds.delete, user.id)

    # ----------------------------------------------------------- helpers

    def _require_config(self) -> None:
        s = self._settings
        if not (s.google_client_id and s.google_client_secret and s.google_redirect_uri):
            raise GoogleConfigError("Google OAuth is not configured on the server.")
        if not s.app_secret_key:
            raise GoogleConfigError("APP_SECRET_KEY is not set on the server.")

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self._transport, timeout=15.0)

    @staticmethod
    def _read_state(state: str) -> str:
        try:
            payload = json.loads(
                decrypt(state, purpose=STATE_PURPOSE, ttl=STATE_TTL_SECONDS)
            )
            external_id = payload["u"]
        except (CryptoError, KeyError, ValueError, TypeError) as exc:
            raise GoogleAuthError(
                "Login session is invalid or expired. Please start again."
            ) from exc
        if not isinstance(external_id, str) or not external_id:
            raise GoogleAuthError("Login session is invalid. Please start again.")
        return external_id

    async def _post_token(
        self, data: dict[str, str], failure_message: str
    ) -> dict[str, Any]:
        try:
            async with self._client() as client:
                response = await client.post(TOKEN_URL, data=data)
        except httpx.HTTPError as exc:
            logger.error("Could not reach Google token endpoint: %s", exc)
            raise GoogleAuthError("Could not reach Google. Please try again.") from exc

        if response.status_code != 200:
            # Log Google's reason (e.g. invalid_grant) but never the secrets.
            logger.warning(
                "Google token endpoint returned %s: %s",
                response.status_code,
                response.text[:300],
            )
            raise GoogleAuthError(failure_message)

        return response.json()

    async def _fetch_email(self, access_token: str) -> str | None:
        try:
            async with self._client() as client:
                response = await client.get(
                    USERINFO_URL,
                    headers={"Authorization": f"Bearer {access_token}"},
                )
        except httpx.HTTPError:
            return None
        if response.status_code != 200:
            return None
        email = response.json().get("email")
        return email if isinstance(email, str) else None