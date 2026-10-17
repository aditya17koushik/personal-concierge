from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database.session import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    external_id: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
    )

    name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class Expense(Base):
    __tablename__ = "expenses"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )

    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    currency: Mapped[str] = mapped_column(String(3))
    category: Mapped[str] = mapped_column(String(50), index=True)
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    spent_on: Mapped[date] = mapped_column(Date, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class GoogleCredential(Base):
    __tablename__ = "google_credentials"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        index=True,
    )

    google_email: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Both tokens are Fernet-encrypted before they are stored.
    access_token_enc: Mapped[str] = mapped_column(Text)
    refresh_token_enc: Mapped[str] = mapped_column(Text)

    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    scopes: Mapped[str] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class PendingAction(Base):
    """A change the assistant PROPOSED. Nothing runs until the user approves it."""

    __tablename__ = "pending_actions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # uuid4

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )

    tool_name: Mapped[str] = mapped_column(String(100))
    arguments: Mapped[dict[str, Any]] = mapped_column(JSON)

    # Built by code from the validated arguments (never written by the LLM),
    # so what the user approves is exactly what will run.
    summary: Mapped[str] = mapped_column(Text)
    request_text: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # pending | executing | executed | failed | rejected | expired
    status: Mapped[str] = mapped_column(String(20), index=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    decided_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class TelegramLink(Base):
    """Maps a verified Telegram account to an internal user."""

    __tablename__ = "telegram_links"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    telegram_user_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        index=True,
    )
    display_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class TelegramInvite(Base):
    """Single-use, expiring invite. Only a hash of the code is stored."""

    __tablename__ = "telegram_invites"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    code_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_by_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    used_by_telegram_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)


class ApiToken(Base):
    """Personal API token. Only a hash is stored; the plain token is shown once."""

    __tablename__ = "api_tokens"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    label: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class UserProfile(Base):
    __tablename__ = "user_profiles"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    timezone: Mapped[str] = mapped_column(String(64))
    currency: Mapped[str] = mapped_column(String(3))
    unit_system: Mapped[str] = mapped_column(String(10))  # metric | imperial
    # Telegram nudges (all opt-in): a daily agenda, and an alert before meetings.
    agenda_enabled: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false"
    )
    agenda_time: Mapped[str] = mapped_column(
        String(5), default="08:00", server_default="08:00"
    )  # HH:MM, in the user's timezone
    last_agenda_on: Mapped[date | None] = mapped_column(Date, nullable=True)
    event_alert_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class UsageCounter(Base):
    """Messages a user has sent on one local day (drives the daily limit)."""

    __tablename__ = "usage_counters"
    __table_args__ = (UniqueConstraint("user_id", "day", name="uq_usage_user_day"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    day: Mapped[date] = mapped_column(Date)
    messages: Mapped[int] = mapped_column(Integer, default=0)


class GoogleLoginLink(Base):
    """Single-use, short-lived link that starts the Google login for one user."""

    __tablename__ = "google_login_links"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    code_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class ConversationMessage(Base):
    """One turn half (user or assistant). Final texts only: tool results are never
    stored, so third-party email or calendar text is not kept beyond the reply."""

    __tablename__ = "conversation_messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    role: Mapped[str] = mapped_column(String(10))  # user | assistant
    content: Mapped[str] = mapped_column(Text)
    # An assistant reply built from email/calendar content may quote strangers.
    external_content: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class Reminder(Base):
    """Something the user asked to be messaged about later."""

    __tablename__ = "reminders"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    text: Mapped[str] = mapped_column(String(300))
    due_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    # The user's timezone when it was set: a daily 08:00 reminder stays at 08:00 local.
    timezone: Mapped[str] = mapped_column(String(64))
    repeat: Mapped[str | None] = mapped_column(String(10), nullable=True)  # daily|weekdays|weekly
    # pending | sending | sent | cancelled | failed | expired
    status: Mapped[str] = mapped_column(String(10), index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EventAlert(Base):
    """Remembers that a meeting alert was sent, so it is never sent twice."""

    __tablename__ = "event_alerts"
    __table_args__ = (
        UniqueConstraint("user_id", "event_id", "starts_at", name="uq_alert_user_event_start"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    event_id: Mapped[str] = mapped_column(String(128))
    starts_at: Mapped[str] = mapped_column(String(40))  # the event's start, as Google gave it
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )