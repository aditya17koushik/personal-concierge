from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database.session import get_db
from app.integrations.google.oauth import (
    GoogleAuthError,
    GoogleConfigError,
    GoogleOAuthService,
)

router = APIRouter(prefix="/auth/google", tags=["auth"])

UserIdQuery = Query(min_length=1, max_length=255)


def get_google_oauth_service(db: Session = Depends(get_db)) -> GoogleOAuthService:
    return GoogleOAuthService(db, get_settings())


@router.get("/login")
def login(
    user_id: str = UserIdQuery,
    service: GoogleOAuthService = Depends(get_google_oauth_service),
) -> RedirectResponse:
    """Open this in a browser: redirects to Google's consent screen."""
    try:
        return RedirectResponse(service.build_login_url(user_id))
    except GoogleConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/callback")
async def callback(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    service: GoogleOAuthService = Depends(get_google_oauth_service),
) -> dict[str, Any]:
    if error:
        raise HTTPException(
            status_code=400,
            detail="Google authorization was denied or cancelled.",
        )
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code or state.")

    try:
        cred = await service.handle_callback(code, state)
    except GoogleConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except GoogleAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "status": "connected",
        "email": cred.google_email,
        "message": "Google account connected. You can close this tab.",
    }


@router.get("/status")
def status(
    user_id: str = UserIdQuery,
    service: GoogleOAuthService = Depends(get_google_oauth_service),
) -> dict[str, Any]:
    return service.get_status(user_id)


@router.delete("")
async def disconnect(
    user_id: str = UserIdQuery,
    service: GoogleOAuthService = Depends(get_google_oauth_service),
) -> dict[str, bool]:
    return {"disconnected": await service.disconnect(user_id)}