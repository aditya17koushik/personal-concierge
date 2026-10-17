from datetime import datetime

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.models import TelegramInvite, TelegramLink, User


class TelegramRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    # ------------------------------------------------------------------ links

    def get_link(self, telegram_id: int) -> TelegramLink | None:
        return self._db.scalar(
            select(TelegramLink).where(TelegramLink.telegram_user_id == telegram_id)
        )

    def get_link_for_user(self, user_id: int) -> TelegramLink | None:
        return self._db.scalar(select(TelegramLink).where(TelegramLink.user_id == user_id))

    def create_link(
        self, telegram_id: int, user_id: int, display_name: str | None
    ) -> TelegramLink:
        link = TelegramLink(
            telegram_user_id=telegram_id,
            user_id=user_id,
            display_name=(display_name or None) and display_name[:100],
        )
        self._db.add(link)
        try:
            self._db.commit()
        except IntegrityError:
            # created concurrently by another update for the same person
            self._db.rollback()
            existing = self.get_link(telegram_id)
            if existing is None:
                raise
            return existing
        self._db.refresh(link)
        return link

    def delete_link(self, telegram_id: int) -> bool:
        link = self.get_link(telegram_id)
        if link is None:
            return False
        self._db.delete(link)
        self._db.commit()
        return True

    def count_links(self) -> int:
        return int(self._db.scalar(select(func.count(TelegramLink.id))) or 0)

    def list_links(self) -> list[tuple[int, str, str | None]]:
        """(telegram_id, internal external_id, display_name), oldest first."""
        rows = self._db.execute(
            select(TelegramLink.telegram_user_id, User.external_id, TelegramLink.display_name)
            .join(User, User.id == TelegramLink.user_id)
            .order_by(TelegramLink.id)
        )
        return [(tg, ext, name) for tg, ext, name in rows]

    # ---------------------------------------------------------------- invites

    def create_invite(
        self, code_hash: str, created_by_user_id: int, expires_at: datetime
    ) -> TelegramInvite:
        invite = TelegramInvite(
            code_hash=code_hash,
            created_by_user_id=created_by_user_id,
            expires_at=expires_at,
        )
        self._db.add(invite)
        self._db.commit()
        self._db.refresh(invite)
        return invite

    def get_invite(self, code_hash: str) -> TelegramInvite | None:
        return self._db.scalar(
            select(TelegramInvite).where(TelegramInvite.code_hash == code_hash)
        )

    def claim_invite(self, code_hash: str, used_at: datetime, telegram_id: int) -> bool:
        """Atomically mark an unused invite as used. Exactly one caller wins."""
        result = self._db.execute(
            update(TelegramInvite)
            .where(TelegramInvite.code_hash == code_hash, TelegramInvite.used_at.is_(None))
            .values(used_at=used_at, used_by_telegram_id=telegram_id)
        )
        self._db.commit()
        return result.rowcount == 1  # type: ignore[attr-defined]