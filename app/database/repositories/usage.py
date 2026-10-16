from datetime import date

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.models import UsageCounter


class UsageRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def try_consume(self, user_id: int, day: date, limit: int) -> bool:
        """Count one message unless the user is already at `limit` for `day`.

        The check and the increment are ONE conditional UPDATE, so concurrent
        requests can never push a user past the limit.
        """
        for _ in range(2):
            result = self._db.execute(
                update(UsageCounter)
                .where(
                    UsageCounter.user_id == user_id,
                    UsageCounter.day == day,
                    UsageCounter.messages < limit,
                )
                .values(messages=UsageCounter.messages + 1)
            )
            if result.rowcount == 1:  # type: ignore[attr-defined]
                self._db.commit()
                return True

            exists = self._db.scalar(
                select(UsageCounter.id).where(
                    UsageCounter.user_id == user_id, UsageCounter.day == day
                )
            )
            if exists is not None:
                self._db.rollback()
                return False  # row exists and is at the limit

            try:
                self._db.add(UsageCounter(user_id=user_id, day=day, messages=1))
                self._db.commit()
                return True
            except IntegrityError:
                self._db.rollback()  # lost a race to create the row; retry the update
        return False

    def count(self, user_id: int, day: date) -> int:
        value = self._db.scalar(
            select(UsageCounter.messages).where(
                UsageCounter.user_id == user_id, UsageCounter.day == day
            )
        )
        return int(value or 0)