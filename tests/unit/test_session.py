from app.database.session import _engine_options


def test_postgres_gets_timeouts():
    options = _engine_options("postgresql+psycopg://u:p@localhost:5433/db")

    assert options["connect_args"]["connect_timeout"] == 5
    assert "statement_timeout" in options["connect_args"]["options"]
    assert options["pool_timeout"] == 10


def test_sqlite_gets_no_postgres_only_options():
    assert _engine_options("sqlite://") == {}