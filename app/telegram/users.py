import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

from sqlalchemy.orm import Session

from app.config import Settings
from app.database.models import User
from app.database.repositories.telegram import TelegramRepository
from app.database.repositories.users import UserRepository


class InviteError(Exception):
    """A safe-to-show reason an invite could not be used."""

    def __init__(self, message: str, counts_as_failure: bool = True) -> None:
        super().__init__(message)
        self.message = message
        self.counts_as_failure = counts_as_failure


@dataclass(frozen=True)
class LinkedUser:
    """Plain values only, so nothing ORM-bound crosses thread boundaries."""

    user_id: int
    external_id: str
    telegram_id: int
    is_admin: bool


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def hash_code(code: str) -> str:
    return hashlib.sha256(code.strip().encode()).hexdigest()


class TelegramUsers:
    """Who may use the bot, and which internal user each Telegram account is."""

    def __init__(
        self,
        db: Session,
        settings: Settings,
        now_fn: Callable[[], datetime] = _now,
    ) -> None:
        self._db = db
        self._settings = settings
        self._repo = TelegramRepository(db)
        self._users = UserRepository(db)
        self._now = now_fn

    def is_admin(self, telegram_id: int) -> bool:
        admin_id = self._settings.telegram_admin_id
        return admin_id != 0 and telegram_id == admin_id

    def resolve(self, telegram_id: int, display_name: str | None) -> LinkedUser | None:
        """The linked user for this Telegram account, or None if not allowed.

        The configured admin is linked automatically to the existing internal
        user (default "me"), so existing data carries over.
        """
        link = self._repo.get_link(telegram_id)
        if link is None:
            if not self.is_admin(telegram_id):
                return None
            admin_user = self._users.get_or_create(self._settings.telegram_admin_external_id)
            link = self._repo.create_link(telegram_id, admin_user.id, display_name)

        user = self._db.get(User, link.user_id)
        if user is None:  # pragma: no cover - FK guarantees this
            return None
        return LinkedUser(user.id, user.external_id, telegram_id, self.is_admin(telegram_id))

    # ---------------------------------------------------------------- invites

    def create_invite(self, admin: LinkedUser) -> tuple[str, datetime]:
        if not admin.is_admin:
            raise PermissionError("admin only")
        code = secrets.token_urlsafe(9)
        expires_at = self._now() + timedelta(hours=self._settings.telegram_invite_ttl_hours)
        self._repo.create_invite(hash_code(code), admin.user_id, expires_at)
        return code, expires_at

    def redeem_invite(
        self, code: str, telegram_id: int, display_name: str | None
    ) -> LinkedUser:
        code_hash = hash_code(code)
        invite = self._repo.get_invite(code_hash)
        if (
            invite is None
            or invite.used_at is not None
            or _as_utc(invite.expires_at) <= self._now()
        ):
            raise InviteError("That invite code is invalid or has expired.")

        if self._repo.count_links() >= self._settings.telegram_max_users:
            raise InviteError(
                "This assistant has reached its user limit.", counts_as_failure=False
            )

        if not self._repo.claim_invite(code_hash, self._now(), telegram_id):
            raise InviteError("That invite code is invalid or has expired.")

        user = self._users.get_or_create(f"tg:{telegram_id}")
        self._repo.create_link(telegram_id, user.id, display_name)
        return LinkedUser(user.id, user.external_id, telegram_id, self.is_admin(telegram_id))

    # ------------------------------------------------------------- management

    def list_users(self) -> list[tuple[int, str, str | None]]:
        return self._repo.list_links()

    def remove(self, admin: LinkedUser, telegram_id: int) -> bool:
        if not admin.is_admin:
            raise PermissionError("admin only")
        if self.is_admin(telegram_id):
            raise ValueError("You can't remove yourself.")
        return self._repo.delete_link(telegram_id)