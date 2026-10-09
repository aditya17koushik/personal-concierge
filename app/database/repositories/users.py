from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.models import User


class UserRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def get(self, external_id: str) -> User | None:
        return self._db.scalar(select(User).where(User.external_id == external_id))

    def get_or_create(self, external_id: str) -> User:
        user = self.get(external_id)
        if user:
            return user

        user = User(external_id=external_id)
        self._db.add(user)
        try:
            self._db.commit()
        except IntegrityError:
            # another request created the same user first
            self._db.rollback()
            existing = self.get(external_id)
            if existing is None:
                raise
            return existing

        self._db.refresh(user)
        return user