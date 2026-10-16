from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import ApiToken, User

TOUCH_INTERVAL = timedelta(seconds=60)


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


class ApiTokenRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def create(self, user_id: int, token_hash: str, label: str) -> ApiToken:
        token = ApiToken(user_id=user_id, token_hash=token_hash, label=label[:100])
        self._db.add(token)
        self._db.commit()
        self._db.refresh(token)
        return token

    def authenticate(self, token_hash: str, now: datetime) -> tuple[int, str] | None:
        """(user_id, external_id) for a valid, un-revoked token, else None."""
        row = self._db.execute(
            select(ApiToken, User.external_id)
            .join(User, User.id == ApiToken.user_id)
            .where(ApiToken.token_hash == token_hash, ApiToken.revoked_at.is_(None))
        ).first()
        if row is None:
            return None

        token, external_id = row
        user_id = token.user_id  # read before commit expires the instance
        last = token.last_used_at
        if last is None or now - _as_utc(last) > TOUCH_INTERVAL:
            token.last_used_at = now  # not on every request: avoids a write per call
            self._db.commit()
        return user_id, external_id

    def list_all(
        self,
    ) -> list[tuple[int, str, str, datetime, datetime | None, datetime | None]]:
        """(id, external_id, label, created_at, last_used_at, revoked_at)."""
        rows = self._db.execute(
            select(
                ApiToken.id,
                User.external_id,
                ApiToken.label,
                ApiToken.created_at,
                ApiToken.last_used_at,
                ApiToken.revoked_at,
            )
            .join(User, User.id == ApiToken.user_id)
            .order_by(ApiToken.id)
        )
        return [tuple(r) for r in rows]  # type: ignore[misc]

    def revoke(self, token_id: int, now: datetime) -> bool:
        token = self._db.get(ApiToken, token_id)
        if token is None or token.revoked_at is not None:
            return False
        token.revoked_at = now
        self._db.commit()
        return True