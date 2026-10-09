from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import GoogleCredential


class GoogleCredentialRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def get(self, user_id: int) -> GoogleCredential | None:
        return self._db.scalar(
            select(GoogleCredential).where(GoogleCredential.user_id == user_id)
        )

    def upsert(
        self,
        user_id: int,
        google_email: str | None,
        access_token_enc: str,
        refresh_token_enc: str,
        expires_at: datetime,
        scopes: str,
    ) -> GoogleCredential:
        cred = self.get(user_id)
        if cred is None:
            cred = GoogleCredential(user_id=user_id)
            self._db.add(cred)

        cred.google_email = google_email
        cred.access_token_enc = access_token_enc
        cred.refresh_token_enc = refresh_token_enc
        cred.expires_at = expires_at
        cred.scopes = scopes

        self._db.commit()
        self._db.refresh(cred)
        return cred

    def update_access_token(
        self,
        cred: GoogleCredential,
        access_token_enc: str,
        expires_at: datetime,
        refresh_token_enc: str | None = None,
    ) -> GoogleCredential:
        cred.access_token_enc = access_token_enc
        cred.expires_at = expires_at
        if refresh_token_enc:
            cred.refresh_token_enc = refresh_token_enc
        self._db.commit()
        self._db.refresh(cred)
        return cred

    def delete(self, user_id: int) -> bool:
        cred = self.get(user_id)
        if cred is None:
            return False
        self._db.delete(cred)
        self._db.commit()
        return True