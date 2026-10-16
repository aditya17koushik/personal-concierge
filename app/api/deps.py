from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.database.repositories.api_tokens import ApiTokenRepository
from app.database.session import get_db
from app.security.api_tokens import hash_token

bearer_scheme = HTTPBearer(auto_error=False, description="Personal API token (pa_...)")


@dataclass(frozen=True)
class CurrentUser:
    """Who is calling, derived from the token and never from request data."""

    user_id: int
    external_id: str


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> CurrentUser:
    unauthorized = HTTPException(
        status_code=401,
        detail="Missing or invalid API token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None or not credentials.credentials.strip():
        raise unauthorized

    found = ApiTokenRepository(db).authenticate(
        hash_token(credentials.credentials), datetime.now(timezone.utc)
    )
    if found is None:
        raise unauthorized
    return CurrentUser(user_id=found[0], external_id=found[1])