from datetime import date

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.models import UserProfile


class ProfileRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def get(self, user_id: int) -> UserProfile | None:
        return self._db.get(UserProfile, user_id)

    def create(
        self, user_id: int, timezone: str, currency: str, unit_system: str
    ) -> UserProfile:
        profile = UserProfile(
            user_id=user_id, timezone=timezone, currency=currency, unit_system=unit_system
        )
        self._db.add(profile)
        try:
            self._db.commit()
        except IntegrityError:
            # created concurrently (e.g. two quick messages from a new user)
            self._db.rollback()
            existing = self.get(user_id)
            if existing is None:
                raise
            return existing
        self._db.refresh(profile)
        return profile

    def update(
        self,
        profile: UserProfile,
        timezone: str | None = None,
        currency: str | None = None,
        unit_system: str | None = None,
    ) -> UserProfile:
        if timezone is not None:
            profile.timezone = timezone
        if currency is not None:
            profile.currency = currency
        if unit_system is not None:
            profile.unit_system = unit_system
        self._db.commit()
        self._db.refresh(profile)
        return profile

    def set_nudges(
        self,
        profile: UserProfile,
        *,
        agenda_enabled: bool | None = None,
        agenda_time: str | None = None,
        event_alert_minutes: int | None = None,
        clear_event_alerts: bool = False,
    ) -> UserProfile:
        if agenda_enabled is not None:
            profile.agenda_enabled = agenda_enabled
        if agenda_time is not None:
            profile.agenda_time = agenda_time
        if clear_event_alerts:
            profile.event_alert_minutes = None
        elif event_alert_minutes is not None:
            profile.event_alert_minutes = event_alert_minutes
        self._db.commit()
        self._db.refresh(profile)
        return profile

    def claim_agenda(self, user_id: int, day: date) -> bool:
        """Claim today's agenda: True only for the caller that gets to send it."""
        result = self._db.execute(
            update(UserProfile)
            .where(
                UserProfile.user_id == user_id,
                (UserProfile.last_agenda_on.is_(None)) | (UserProfile.last_agenda_on < day),
            )
            .values(last_agenda_on=day)
        )
        self._db.commit()
        return result.rowcount == 1  # type: ignore[attr-defined]

    def release_agenda(self, user_id: int, previous: date | None) -> None:
        """Sending failed: let the next poll try again."""
        self._db.execute(
            update(UserProfile).where(UserProfile.user_id == user_id).values(last_agenda_on=previous)
        )
        self._db.commit()

    def with_nudges(self) -> list[UserProfile]:
        return list(
            self._db.scalars(
                select(UserProfile).where(
                    (UserProfile.agenda_enabled.is_(True))
                    | (UserProfile.event_alert_minutes.is_not(None))
                )
            )
        )