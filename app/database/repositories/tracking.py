from datetime import date

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.models import TrackingEvent, TrackingSettings

SLOT_COLUMNS = {
    "breakfast": "breakfast_time",
    "lunch": "lunch_time",
    "dinner": "dinner_time",
    "workout": "workout_time",
}


class TrackingRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def get(self, user_id: int) -> TrackingSettings | None:
        return self._db.get(TrackingSettings, user_id)

    def enabled(self) -> list[TrackingSettings]:
        return list(
            self._db.scalars(select(TrackingSettings).where(TrackingSettings.enabled.is_(True)))
        )

    def set(
        self, user_id: int, *, enabled: bool | None = None, times: dict[str, str] | None = None
    ) -> TrackingSettings:
        row = self.get(user_id)
        if row is None:
            row = TrackingSettings(
                user_id=user_id, enabled=False, breakfast_time="10:00", lunch_time="14:30",
                dinner_time="21:00", workout_time="19:30",
            )  # fmt: skip
            self._db.add(row)
            try:
                self._db.flush()
            except IntegrityError:
                self._db.rollback()
                row = self.get(user_id)
                if row is None:
                    raise
        if enabled is not None:
            row.enabled = enabled
        for slot, value in (times or {}).items():
            setattr(row, SLOT_COLUMNS[slot], value)
        self._db.commit()
        self._db.refresh(row)
        return row

    # ---------------------------------------------------------------- events

    def events(self, user_id: int, day: date) -> dict[str, str]:
        rows = self._db.scalars(
            select(TrackingEvent).where(TrackingEvent.user_id == user_id, TrackingEvent.day == day)
        )
        return {r.slot: r.kind for r in rows}

    def claim_nudge(self, user_id: int, day: date, slot: str) -> bool:
        """Insert-as-claim: True only for the caller that may send this nudge."""
        self._db.add(TrackingEvent(user_id=user_id, day=day, slot=slot, kind="nudged"))
        try:
            self._db.commit()
        except IntegrityError:
            self._db.rollback()
            return False
        return True

    def release_nudge(self, user_id: int, day: date, slot: str) -> None:
        """Sending failed: remove the claim so the next poll retries."""
        self._db.execute(
            delete(TrackingEvent).where(
                TrackingEvent.user_id == user_id,
                TrackingEvent.day == day,
                TrackingEvent.slot == slot,
                TrackingEvent.kind == "nudged",
            )
        )
        self._db.commit()

    def skip(self, user_id: int, day: date, slot: str) -> None:
        updated = self._db.execute(
            update(TrackingEvent)
            .where(
                TrackingEvent.user_id == user_id,
                TrackingEvent.day == day,
                TrackingEvent.slot == slot,
            )
            .values(kind="skipped")
        )
        if updated.rowcount == 0:  # type: ignore[attr-defined]
            self._db.add(TrackingEvent(user_id=user_id, day=day, slot=slot, kind="skipped"))
        try:
            self._db.commit()
        except IntegrityError:  # raced with a nudge insert: turn it into a skip
            self._db.rollback()
            self._db.execute(
                update(TrackingEvent)
                .where(
                    TrackingEvent.user_id == user_id,
                    TrackingEvent.day == day,
                    TrackingEvent.slot == slot,
                )
                .values(kind="skipped")
            )
            self._db.commit()