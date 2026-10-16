from datetime import date

from sqlalchemy.orm import Session

from app.database.repositories.usage import UsageRepository


class RateLimitExceeded(Exception):
    """The user used up today's message allowance."""

    def __init__(self, limit: int, retry_after_seconds: int) -> None:
        super().__init__(f"daily limit of {limit} messages reached")
        self.limit = limit
        self.retry_after_seconds = retry_after_seconds


class QuotaService:
    """Per-user daily message cap. Checked BEFORE any paid model call.
    A limit of 0 disables it."""

    def __init__(self, db: Session, daily_limit: int) -> None:
        self._repo = UsageRepository(db)
        self._limit = daily_limit

    def consume(self, user_id: int, day: date, retry_after_seconds: int) -> None:
        if self._limit <= 0:
            return
        if not self._repo.try_consume(user_id, day, self._limit):
            raise RateLimitExceeded(self._limit, retry_after_seconds)

    def used(self, user_id: int, day: date) -> int:
        return self._repo.count(user_id, day)