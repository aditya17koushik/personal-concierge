from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, get_current_user
from app.config import Settings, get_settings
from app.database.session import get_db
from app.integrations.google.links import LoginLinkService
from app.integrations.google.oauth import (
    GoogleAuthError,
    GoogleConfigError,
    GoogleOAuthService,
)

router = APIRouter(prefix="/auth/google", tags=["auth"])


def get_google_oauth_service(db: Session = Depends(get_db)) -> GoogleOAuthService:
    return GoogleOAuthService(db, get_settings())


def get_login_link_service(db: Session = Depends(get_db)) -> LoginLinkService:
    return LoginLinkService(db, get_settings())


@router.post("/link")
def create_login_link(
    current: CurrentUser = Depends(get_current_user),
    links: LoginLinkService = Depends(get_login_link_service),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Create a single-use link for YOUR Google login. Open it in a browser."""
    code, _expires = links.create(current.user_id)
    return {
        "url": f"{settings.public_base_url.rstrip('/')}/auth/google/login?code={code}",
        "expires_in_minutes": settings.google_login_link_ttl_minutes,
        "note": "Single use. Only open links you requested yourself.",
    }


@router.get("/login")
def login(
    code: str = Query(min_length=8, max_length=128),
    links: LoginLinkService = Depends(get_login_link_service),
    service: GoogleOAuthService = Depends(get_google_oauth_service),
) -> RedirectResponse:
    """Opened in a browser from a single-use link; redirects to Google's consent screen."""
    try:
        external_id = links.consume(code)
        url = service.build_login_url(external_id)
    except GoogleConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except GoogleAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return RedirectResponse(url, headers={"Cache-Control": "no-store"})


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
    current: CurrentUser = Depends(get_current_user),
    service: GoogleOAuthService = Depends(get_google_oauth_service),
) -> dict[str, Any]:
    return service.get_status(current.external_id)


@router.delete("")
async def disconnect(
    current: CurrentUser = Depends(get_current_user),
    service: GoogleOAuthService = Depends(get_google_oauth_service),
) -> dict[str, bool]:
    return {"disconnected": await service.disconnect(current.external_id)}