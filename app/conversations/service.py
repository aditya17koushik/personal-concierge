from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.database.repositories.conversation import ConversationRepository

MAX_STORED_CHARS = 4000
KEEP_AT_MOST = 200  # per user, whatever the retention window says


@dataclass(frozen=True)
class HistoryMessage:
    role: str  # user | assistant
    content: str
    external: bool  # an assistant reply that may quote email/calendar text


class ConversationService:
    """Short-term memory: the last few turns, within a time window and a size budget.

    Only final texts are kept. The window and budget keep every request cheap
    and stop old, unrelated chatter from steering a new question.
    """

    def __init__(
        self,
        db: Session,
        ttl_minutes: int = 360,
        max_messages: int = 12,
        max_chars: int = 6000,
        retention_days: int = 7,
    ) -> None:
        self._repo = ConversationRepository(db)
        self._ttl = timedelta(minutes=ttl_minutes)
        self._max_messages = max_messages
        self._max_chars = max_chars
        self._retention = timedelta(days=retention_days)

    @property
    def enabled(self) -> bool:
        return self._max_messages > 0

    def recent(self, user_id: int, now: datetime) -> list[HistoryMessage]:
        if not self.enabled:
            return []
        rows = self._repo.recent(user_id, now - self._ttl, self._max_messages)
        messages = [HistoryMessage(r.role, r.content, bool(r.external_content)) for r in rows]

        # Stay inside the size budget by dropping the OLDEST first.
        while messages and sum(len(m.content) for m in messages) > self._max_chars:
            messages.pop(0)
        # Always begin with a user message so the transcript reads naturally.
        while messages and messages[0].role != "user":
            messages.pop(0)
        return messages

    def add_turn(
        self,
        user_id: int,
        user_text: str,
        assistant_text: str,
        external: bool,
        now: datetime,
    ) -> None:
        if not self.enabled:
            return
        self._repo.add_turn(
            user_id,
            user_text[:MAX_STORED_CHARS],
            assistant_text[:MAX_STORED_CHARS],
            external,
            now,
        )
        self._repo.prune(user_id, now - self._retention, KEEP_AT_MOST)

    def clear(self, user_id: int) -> int:
        return self._repo.clear(user_id)