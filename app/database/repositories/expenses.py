from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database.models import Expense


@dataclass
class CategoryTotal:
    category: str
    currency: str
    total: Decimal
    count: int


class ExpenseRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def add(
        self,
        user_id: int,
        amount: Decimal,
        currency: str,
        category: str,
        description: str | None,
        spent_on: date,
    ) -> Expense:
        expense = Expense(
            user_id=user_id,
            amount=amount,
            currency=currency,
            category=category,
            description=description,
            spent_on=spent_on,
        )
        self._db.add(expense)
        self._db.commit()
        self._db.refresh(expense)
        return expense

    def find(
        self,
        user_id: int,
        start: date | None = None,
        end: date | None = None,
        category: str | None = None,
        limit: int = 20,
    ) -> list[Expense]:
        stmt = select(Expense).where(Expense.user_id == user_id)
        if start:
            stmt = stmt.where(Expense.spent_on >= start)
        if end:
            stmt = stmt.where(Expense.spent_on <= end)
        if category:
            stmt = stmt.where(Expense.category == category)
        stmt = stmt.order_by(Expense.spent_on.desc(), Expense.id.desc()).limit(limit)
        return list(self._db.scalars(stmt))

    def totals_by_category(
        self, user_id: int, start: date, end: date
    ) -> list[CategoryTotal]:
        stmt = (
            select(
                Expense.category,
                Expense.currency,
                func.sum(Expense.amount),
                func.count(Expense.id),
            )
            .where(
                Expense.user_id == user_id,
                Expense.spent_on >= start,
                Expense.spent_on <= end,
            )
            .group_by(Expense.category, Expense.currency)
            .order_by(func.sum(Expense.amount).desc())
        )
        return [
            CategoryTotal(category=c, currency=cur, total=Decimal(total), count=n)
            for c, cur, total, n in self._db.execute(stmt)
        ]