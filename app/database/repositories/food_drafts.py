from datetime import datetime
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import update
from sqlalchemy.orm import Session

from app.database.models import FoodDraft


class FoodDraftRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def add(
        self,
        user_id: int,
        description: str,
        calories: int,
        protein_g: float | None,
        carbs_g: float | None,
        fat_g: float | None,
        confidence: str,
    ) -> FoodDraft:
        def dec(v: float | None) -> Decimal | None:
            return None if v is None else Decimal(str(round(v, 1)))

        draft = FoodDraft(
            id=str(uuid4()),
            user_id=user_id,
            description=description,
            calories=calories,
            protein_g=dec(protein_g),
            carbs_g=dec(carbs_g),
            fat_g=dec(fat_g),
            confidence=confidence,
            status="pending",
        )
        self._db.add(draft)
        self._db.commit()
        self._db.refresh(draft)
        return draft

    def get(self, user_id: int, draft_id: str) -> FoodDraft | None:
        """Only the owner can see a draft."""
        draft = self._db.get(FoodDraft, draft_id)
        return draft if draft is not None and draft.user_id == user_id else None

    def claim(self, user_id: int, draft_id: str, status: str, not_before: datetime) -> bool:
        """pending -> logged/discarded, atomically, for the owner and not after expiry.
        Does NOT commit: the caller commits together with the meal it logs."""
        result = self._db.execute(
            update(FoodDraft)
            .where(
                FoodDraft.id == draft_id,
                FoodDraft.user_id == user_id,
                FoodDraft.status == "pending",
                FoodDraft.created_at >= not_before,
            )
            .values(status=status)
            .execution_options(synchronize_session=False)
        )
        return result.rowcount == 1  # type: ignore[attr-defined]