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