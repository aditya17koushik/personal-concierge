from datetime import date

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.models import CoachSettings


class CoachSettingsRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def get(self, user_id: int) -> CoachSettings | None:
        return self._db.get(CoachSettings, user_id)

    def set(
        self,
        user_id: int,
        *,
        enabled: bool | None = None,
        checkin_time: str | None = None,
        weekly_day: int | None = None,
    ) -> CoachSettings:
        row = self.get(user_id)
        if row is None:
            row = CoachSettings(user_id=user_id, enabled=False, checkin_time="20:00", weekly_day=6)
            self._db.add(row)
            try:
                self._db.flush()
            except IntegrityError:  # created concurrently
                self._db.rollback()
                row = self.get(user_id)
                if row is None:
                    raise
        if enabled is not None:
            row.enabled = enabled
        if checkin_time is not None:
            row.checkin_time = checkin_time
        if weekly_day is not None:
            row.weekly_day = weekly_day
        self._db.commit()
        self._db.refresh(row)
        return row

    def enabled(self) -> list[CoachSettings]:
        return list(self._db.scalars(select(CoachSettings).where(CoachSettings.enabled.is_(True))))

    def claim_send(self, user_id: int, day: date) -> bool:
        """Claim today's coach message: True only for the caller that gets to send it."""
        result = self._db.execute(
            update(CoachSettings)
            .where(
                CoachSettings.user_id == user_id,
                (CoachSettings.last_sent_on.is_(None)) | (CoachSettings.last_sent_on < day),
            )
            .values(last_sent_on=day)
        )
        self._db.commit()
        return result.rowcount == 1  # type: ignore[attr-defined]

    def release_send(self, user_id: int, previous: date | None) -> None:
        """Sending failed: let the next poll try again."""
        self._db.execute(
            update(CoachSettings)
            .where(CoachSettings.user_id == user_id)
            .values(last_sent_on=previous)
        )
        self._db.commit()