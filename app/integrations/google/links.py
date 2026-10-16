import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Callable

from sqlalchemy.orm import Session

from app.config import Settings
from app.database.repositories.google_login_links import GoogleLoginLinkRepository
from app.integrations.google.oauth import GoogleAuthError


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _hash(code: str) -> str:
    return hashlib.sha256(code.strip().encode()).hexdigest()


INVALID = "This login link is invalid, expired or was already used. Request a new one."


class LoginLinkService:
    """Single-use, short-lived links that start the Google login for ONE user.

    Links are only ever created for an already-verified user (an authenticated
    API call, or a verified Telegram account), never from a client-supplied id.
    """

    def __init__(
        self,
        db: Session,
        settings: Settings,
        now_fn: Callable[[], datetime] = _now,
    ) -> None:
        self._repo = GoogleLoginLinkRepository(db)
        self._ttl = timedelta(minutes=settings.google_login_link_ttl_minutes)
        self._now = now_fn

    def create(self, user_id: int) -> tuple[str, datetime]:
        code = secrets.token_urlsafe(24)
        expires_at = self._now() + self._ttl
        self._repo.create(_hash(code), user_id, expires_at)
        return code, expires_at

    def consume(self, code: str) -> str:
        """Use up a link; returns the owner's external id or raises GoogleAuthError."""
        code_hash = _hash(code)
        found = self._repo.get_with_user(code_hash)
        if found is None:
            raise GoogleAuthError(INVALID)

        link, external_id = found
        if link.used_at is not None or _as_utc(link.expires_at) <= self._now():
            raise GoogleAuthError(INVALID)
        if not self._repo.claim(code_hash, self._now()):  # lost a race: already used
            raise GoogleAuthError(INVALID)
        return external_id