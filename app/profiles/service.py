import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

from sqlalchemy.orm import Session

from app.database.models import UserProfile
from app.database.repositories.profiles import ProfileRepository

logger = logging.getLogger(__name__)

UNIT_SYSTEMS = ("metric", "imperial")
_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")


class ProfileError(Exception):
    """A safe-to-show reason a profile change was rejected."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class ProfileData:
    timezone: str
    currency: str
    unit_system: str
    agenda_enabled: bool = False
    agenda_time: str = "08:00"
    event_alert_minutes: int | None = None


_TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")
MAX_ALERT_MINUTES = 120


def normalize_time(value: str) -> str:
    match = _TIME_RE.match(value.strip())
    if not match:
        raise ProfileError("Use a 24-hour time like 08:00 or 7:30.")
    return f"{int(match.group(1)):02d}:{match.group(2)}"


def normalize_alert_minutes(value: str | int) -> int:
    try:
        minutes = int(str(value).strip())
    except ValueError:
        raise ProfileError("Use a number of minutes, for example /meetings 15, or /meetings off.")
    if not 1 <= minutes <= MAX_ALERT_MINUTES:
        raise ProfileError(f"Choose between 1 and {MAX_ALERT_MINUTES} minutes.")
    return minutes


def _data(profile: "UserProfile") -> ProfileData:
    return ProfileData(
        profile.timezone,
        profile.currency,
        profile.unit_system,
        bool(profile.agenda_enabled),
        profile.agenda_time or "08:00",
        profile.event_alert_minutes,
    )


@lru_cache(maxsize=1)
def _timezone_names() -> dict[str, str]:
    return {name.lower(): name for name in available_timezones()}


def normalize_timezone(name: str) -> str:
    """Canonical IANA name (accepts any letter case) or raises ProfileError."""
    candidate = name.strip()
    for attempt in (candidate, _timezone_names().get(candidate.lower(), "")):
        if not attempt:
            continue
        try:
            ZoneInfo(attempt)
            return attempt
        except (ZoneInfoNotFoundError, ValueError):
            continue
    raise ProfileError("Unknown timezone. Use a name like Asia/Kolkata or Europe/London.")


def normalize_currency(code: str) -> str:
    value = code.strip().upper()
    if not _CURRENCY_RE.match(value):
        raise ProfileError("Use a 3-letter currency code like INR or USD.")
    return value


def normalize_units(value: str) -> str:
    unit = value.strip().lower()
    if unit not in UNIT_SYSTEMS:
        raise ProfileError("Units must be 'metric' or 'imperial'.")
    return unit


def local_today(timezone_name: str, now: datetime) -> date:
    """The user's calendar date right now (`now` must be timezone-aware)."""
    return now.astimezone(ZoneInfo(timezone_name)).date()


def seconds_until_local_midnight(timezone_name: str, now: datetime) -> int:
    local = now.astimezone(ZoneInfo(timezone_name))
    next_midnight = (local + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return max(1, int((next_midnight - local).total_seconds()))


class ProfileService:
    def __init__(
        self,
        db: Session,
        default_currency: str = "INR",
        default_timezone: str = "UTC",
    ) -> None:
        self._repo = ProfileRepository(db)
        self._default_currency = default_currency
        self._default_timezone = default_timezone

    def _defaults(self) -> tuple[str, str]:
        try:
            timezone_name = normalize_timezone(self._default_timezone)
        except ProfileError:
            logger.warning("DEFAULT_TIMEZONE %r is invalid; using UTC", self._default_timezone)
            timezone_name = "UTC"
        try:
            currency = normalize_currency(self._default_currency)
        except ProfileError:
            currency = "USD"
        return timezone_name, currency

    def get_or_create(self, user_id: int) -> ProfileData:
        profile = self._repo.get(user_id)
        if profile is None:
            timezone_name, currency = self._defaults()
            profile = self._repo.create(user_id, timezone_name, currency, "metric")
        return _data(profile)

    def update(
        self,
        user_id: int,
        *,
        timezone: str | None = None,
        currency: str | None = None,
        unit_system: str | None = None,
    ) -> ProfileData:
        changes = {
            "timezone": normalize_timezone(timezone) if timezone is not None else None,
            "currency": normalize_currency(currency) if currency is not None else None,
            "unit_system": normalize_units(unit_system) if unit_system is not None else None,
        }
        self.get_or_create(user_id)  # make sure a row exists
        profile = self._repo.get(user_id)
        assert profile is not None
        profile = self._repo.update(profile, **changes)
        return _data(profile)

    def set_agenda(
        self, user_id: int, enabled: bool, time_text: str | None = None
    ) -> ProfileData:
        agenda_time = normalize_time(time_text) if time_text else None
        self.get_or_create(user_id)
        profile = self._repo.get(user_id)
        assert profile is not None
        return _data(
            self._repo.set_nudges(profile, agenda_enabled=enabled, agenda_time=agenda_time)
        )

    def set_event_alerts(self, user_id: int, minutes: str | int | None) -> ProfileData:
        """minutes=None switches meeting alerts off."""
        value = normalize_alert_minutes(minutes) if minutes is not None else None
        self.get_or_create(user_id)
        profile = self._repo.get(user_id)
        assert profile is not None
        return _data(
            self._repo.set_nudges(
                profile, event_alert_minutes=value, clear_event_alerts=value is None
            )
        )