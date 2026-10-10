from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import get_settings


settings = get_settings()


def _engine_options(url: str) -> dict[str, Any]:
    """Fail fast instead of hanging when Postgres is unreachable or locked.

    Postgres-only: SQLite (used in tests) does not accept these options.
    """
    if not url.startswith("postgresql"):
        return {}
    return {
        "pool_recycle": 1800,  # replace connections older than 30 min
        "pool_timeout": 10,  # wait at most 10s for a free pooled connection
        "connect_args": {
            "connect_timeout": 5,  # give up connecting after 5s
            "options": "-c statement_timeout=15000",  # no query runs > 15s
        },
    }


engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    **_engine_options(settings.database_url),
)


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()