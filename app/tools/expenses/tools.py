from datetime import date as date_type
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.database.models import Expense
from app.database.repositories.expenses import ExpenseRepository
from app.tools.base import Tool, ToolContext


def _serialize(expense: Expense) -> dict[str, Any]:
    return {
        "id": expense.id,
        "amount": str(expense.amount),
        "currency": expense.currency,
        "category": expense.category,
        "description": expense.description,
        "date": expense.spent_on.isoformat(),
    }


# ---------------------------------------------------------------- add_expense


class AddExpenseArgs(BaseModel):
    amount: float = Field(gt=0, le=10_000_000, description="Amount spent (positive).")
    category: str = Field(
        min_length=1,
        max_length=50,
        description="Short lowercase category, e.g. food, groceries, transport, rent.",
    )
    description: str | None = Field(
        default=None, max_length=500, description="What the expense was for."
    )
    spent_on: date_type | None = Field(
        default=None,
        description="Date of the expense (YYYY-MM-DD). Omit to use today.",
    )
    currency: str | None = Field(
        default=None,
        min_length=3,
        max_length=3,
        description="3-letter currency code. Omit to use the user's default.",
    )


def add_expense(ctx: ToolContext, args: AddExpenseArgs) -> dict[str, Any]:
    expense = ExpenseRepository(ctx.db).add(
        user_id=ctx.user_id,
        amount=Decimal(str(args.amount)).quantize(Decimal("0.01")),
        currency=(args.currency or ctx.default_currency).upper(),
        category=args.category.strip().lower(),
        description=args.description,
        spent_on=args.spent_on or ctx.today,
    )
    return {"saved": _serialize(expense)}


# -------------------------------------------------------------- list_expenses


class ListExpensesArgs(BaseModel):
    start_date: date_type | None = Field(default=None, description="YYYY-MM-DD, inclusive.")
    end_date: date_type | None = Field(default=None, description="YYYY-MM-DD, inclusive.")
    category: str | None = Field(default=None, description="Filter by category.")
    limit: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def check_range(self) -> "ListExpensesArgs":
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date must not be before start_date")
        return self


def list_expenses(ctx: ToolContext, args: ListExpensesArgs) -> dict[str, Any]:
    expenses = ExpenseRepository(ctx.db).find(
        user_id=ctx.user_id,
        start=args.start_date,
        end=args.end_date,
        category=args.category.strip().lower() if args.category else None,
        limit=args.limit,
    )
    return {"count": len(expenses), "expenses": [_serialize(e) for e in expenses]}


# ------------------------------------------------------- get_expense_summary


class ExpenseSummaryArgs(BaseModel):
    start_date: date_type | None = Field(
        default=None, description="YYYY-MM-DD. Omit for the start of this month."
    )
    end_date: date_type | None = Field(
        default=None, description="YYYY-MM-DD. Omit for today."
    )

    @model_validator(mode="after")
    def check_range(self) -> "ExpenseSummaryArgs":
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date must not be before start_date")
        return self


def get_expense_summary(ctx: ToolContext, args: ExpenseSummaryArgs) -> dict[str, Any]:
    start = args.start_date or ctx.today.replace(day=1)
    end = args.end_date or ctx.today

    rows = ExpenseRepository(ctx.db).totals_by_category(ctx.user_id, start, end)

    grand_totals: dict[str, Decimal] = {}
    for row in rows:
        grand_totals[row.currency] = grand_totals.get(row.currency, Decimal("0")) + row.total

    return {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "by_category": [
            {
                "category": r.category,
                "currency": r.currency,
                "total": str(r.total),
                "count": r.count,
            }
            for r in rows
        ],
        "grand_total_by_currency": {k: str(v) for k, v in grand_totals.items()},
    }


EXPENSE_TOOLS: list[Tool] = [
    Tool(
        name="add_expense",
        description="Record a new expense for the user.",
        args_model=AddExpenseArgs,
        handler=add_expense,
        domain="expense",
        side_effects=True,
    ),
    Tool(
        name="list_expenses",
        description="List the user's recorded expenses, newest first, with optional filters.",
        args_model=ListExpensesArgs,
        handler=list_expenses,
        domain="expense",
    ),
    Tool(
        name="get_expense_summary",
        description="Total the user's spending by category over a date range.",
        args_model=ExpenseSummaryArgs,
        handler=get_expense_summary,
        domain="expense",
    ),
]