from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.database.models import ConversationMessage


class ConversationRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def add_turn(
        self,
        user_id: int,
        user_text: str,
        assistant_text: str,
        external_content: bool,
        now: datetime,
    ) -> None:
        """Store both halves of a turn in one transaction (user first)."""
        self._db.add_all(
            [
                ConversationMessage(
                    user_id=user_id, role="user", content=user_text,
                    external_content=False, created_at=now,
                ),
                ConversationMessage(
                    user_id=user_id, role="assistant", content=assistant_text,
                    external_content=external_content, created_at=now,
                ),
            ]
        )
        self._db.commit()

    def recent(self, user_id: int, since: datetime, limit: int) -> list[ConversationMessage]:
        """The newest `limit` messages since `since`, returned oldest first."""
        rows = self._db.scalars(
            select(ConversationMessage)
            .where(ConversationMessage.user_id == user_id, ConversationMessage.created_at >= since)
            .order_by(ConversationMessage.id.desc())
            .limit(limit)
        )
        return list(reversed(list(rows)))

    def prune(self, user_id: int, older_than: datetime, keep: int) -> None:
        """Drop rows past the retention window, then cap the total kept."""
        self._db.execute(
            delete(ConversationMessage).where(
                ConversationMessage.user_id == user_id,
                ConversationMessage.created_at < older_than,
            )
        )
        newest = (
            select(ConversationMessage.id)
            .where(ConversationMessage.user_id == user_id)
            .order_by(ConversationMessage.id.desc())
            .limit(keep)
        )
        self._db.execute(
            delete(ConversationMessage).where(
                ConversationMessage.user_id == user_id,
                ConversationMessage.id.not_in(newest),
            )
        )
        self._db.commit()

    def clear(self, user_id: int) -> int:
        result = self._db.execute(
            delete(ConversationMessage).where(ConversationMessage.user_id == user_id)
        )
        self._db.commit()
        return int(result.rowcount or 0)  # type: ignore[attr-defined]