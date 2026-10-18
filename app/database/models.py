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


class Meal(Base):
    """One thing the user ate. Numbers may be the user's or the model's estimate."""

    __tablename__ = "meals"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    eaten_on: Mapped[date] = mapped_column(Date, index=True)  # the user's local date
    meal_type: Mapped[str] = mapped_column(String(10))  # breakfast|lunch|dinner|snack
    description: Mapped[str] = mapped_column(String(300))
    calories: Mapped[int] = mapped_column(Integer)
    protein_g: Mapped[Decimal | None] = mapped_column(Numeric(6, 1), nullable=True)
    carbs_g: Mapped[Decimal | None] = mapped_column(Numeric(6, 1), nullable=True)
    fat_g: Mapped[Decimal | None] = mapped_column(Numeric(6, 1), nullable=True)
    estimated: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class Workout(Base):
    __tablename__ = "workouts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    done_on: Mapped[date] = mapped_column(Date, index=True)
    kind: Mapped[str] = mapped_column(String(12))  # strength|cardio|walk|run|cycle|yoga|sport|other
    description: Mapped[str] = mapped_column(String(300))
    duration_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    calories_burned: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class BodyWeight(Base):
    """Weigh-ins, always stored in kg. One per user per day (a new one replaces it)."""

    __tablename__ = "body_weights"
    __table_args__ = (UniqueConstraint("user_id", "measured_on", name="uq_weight_user_day"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    measured_on: Mapped[date] = mapped_column(Date, index=True)
    weight_kg: Mapped[Decimal] = mapped_column(Numeric(6, 2))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class FitnessGoal(Base):
    """A user's current goal (one per user): cut, bulk or maintain, with optional targets."""

    __tablename__ = "fitness_goals"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    mode: Mapped[str] = mapped_column(String(10))  # cut|bulk|maintain
    target_weight_kg: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)
    start_weight_kg: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)
    daily_calories: Mapped[int | None] = mapped_column(Integer, nullable=True)
    daily_protein_g: Mapped[int | None] = mapped_column(Integer, nullable=True)
    weekly_workouts: Mapped[int | None] = mapped_column(Integer, nullable=True)
    started_on: Mapped[date] = mapped_column(Date)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class CoachSettings(Base):
    """Opt-in coaching messages: an evening check-in, and a weekly review on one weekday."""

    __tablename__ = "coach_settings"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    enabled: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    checkin_time: Mapped[str] = mapped_column(
        String(5), default="20:00", server_default="20:00"
    )  # HH:MM in the user's timezone
    weekly_day: Mapped[int] = mapped_column(
        Integer, default=6, server_default="6"
    )  # 0 = Monday ... 6 = Sunday; that day's message is the weekly review
    last_sent_on: Mapped[date | None] = mapped_column(Date, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class FitnessProfile(Base):
    """What the user told us about themselves. Weight lives in body_weights, the goal in fitness_goals."""

    __tablename__ = "fitness_profiles"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    age_set_on: Mapped[date | None] = mapped_column(Date, nullable=True)  # age grows from here
    sex: Mapped[str | None] = mapped_column(String(10), nullable=True)
    height_cm: Mapped[int | None] = mapped_column(Integer, nullable=True)
    activity_level: Mapped[str | None] = mapped_column(String(12), nullable=True)
    experience: Mapped[str | None] = mapped_column(String(12), nullable=True)
    muscle_level: Mapped[str | None] = mapped_column(String(10), nullable=True)
    conditions: Mapped[str | None] = mapped_column(String(500), nullable=True)  # 'none' is an answer
    preferred_workouts: Mapped[str | None] = mapped_column(String(200), nullable=True)
    days_per_week: Mapped[int | None] = mapped_column(Integer, nullable=True)
    session_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    equipment: Mapped[str | None] = mapped_column(String(20), nullable=True)
    diet_pref: Mapped[str | None] = mapped_column(String(16), nullable=True)
    allergies: Mapped[str | None] = mapped_column(String(200), nullable=True)
    cuisine: Mapped[str | None] = mapped_column(String(100), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class FitnessPlan(Base):
    """A 7-day workout plan and 7-day meal plan. Versioned: one is active, older ones are kept."""

    __tablename__ = "fitness_plans"
    __table_args__ = (UniqueConstraint("user_id", "version", name="uq_plan_user_version"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(10), index=True)  # active | archived
    workout: Mapped[dict[str, Any]] = mapped_column(JSON)
    meals: Mapped[dict[str, Any]] = mapped_column(JSON)
    targets: Mapped[dict[str, Any]] = mapped_column(JSON)  # calories, macros, bmr, tdee, mode
    inputs: Mapped[dict[str, Any]] = mapped_column(JSON)  # what it was built from (stats, days)
    reason: Mapped[str] = mapped_column(String(300))  # 'first plan' or why it was changed
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class TrackingSettings(Base):
    """Opt-in nudges when a planned meal or workout has not been logged."""

    __tablename__ = "tracking_settings"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    enabled: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    breakfast_time: Mapped[str] = mapped_column(String(5), default="10:00", server_default="10:00")
    lunch_time: Mapped[str] = mapped_column(String(5), default="14:30", server_default="14:30")
    dinner_time: Mapped[str] = mapped_column(String(5), default="21:00", server_default="21:00")
    workout_time: Mapped[str] = mapped_column(String(5), default="19:30", server_default="19:30")


class TrackingEvent(Base):
    """One row per user, day and slot. 'nudged' = reminder sent (inserting it IS the claim, so a
    second worker cannot double-send); 'skipped' = the user said they skipped it or rest today."""

    __tablename__ = "tracking_events"
    __table_args__ = (UniqueConstraint("user_id", "day", "slot", name="uq_tracking_user_day_slot"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    day: Mapped[date] = mapped_column(Date)
    slot: Mapped[str] = mapped_column(String(10))  # breakfast|lunch|dinner|workout
    kind: Mapped[str] = mapped_column(String(8))  # nudged|skipped


class FoodDraft(Base):
    """A meal estimated from a photo, waiting for the user to tap [Log it]. The photo itself is
    never stored: only these numbers."""

    __tablename__ = "food_drafts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    description: Mapped[str] = mapped_column(String(300))
    calories: Mapped[int] = mapped_column(Integer)
    protein_g: Mapped[Decimal | None] = mapped_column(Numeric(6, 1), nullable=True)
    carbs_g: Mapped[Decimal | None] = mapped_column(Numeric(6, 1), nullable=True)
    fat_g: Mapped[Decimal | None] = mapped_column(Numeric(6, 1), nullable=True)
    confidence: Mapped[str] = mapped_column(String(6))  # low|medium|high
    status: Mapped[str] = mapped_column(String(10), default="pending")  # pending|logged|discarded
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )