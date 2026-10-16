from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.database.models import GoogleLoginLink, User


class GoogleLoginLinkRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def create(self, code_hash: str, user_id: int, expires_at: datetime) -> GoogleLoginLink:
        link = GoogleLoginLink(code_hash=code_hash, user_id=user_id, expires_at=expires_at)
        self._db.add(link)
        self._db.commit()
        self._db.refresh(link)
        return link

    def get_with_user(self, code_hash: str) -> tuple[GoogleLoginLink, str] | None:
        """The link and its owner's external id."""
        row = self._db.execute(
            select(GoogleLoginLink, User.external_id)
            .join(User, User.id == GoogleLoginLink.user_id)
            .where(GoogleLoginLink.code_hash == code_hash)
        ).first()
        return (row[0], row[1]) if row else None

    def claim(self, code_hash: str, used_at: datetime) -> bool:
        """Atomically mark an unused link as used. Exactly one caller wins."""
        result = self._db.execute(
            update(GoogleLoginLink)
            .where(GoogleLoginLink.code_hash == code_hash, GoogleLoginLink.used_at.is_(None))
            .values(used_at=used_at)
        )
        self._db.commit()
        return result.rowcount == 1  # type: ignore[attr-defined]