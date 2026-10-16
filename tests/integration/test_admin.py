import pytest
from sqlalchemy.orm import Session, sessionmaker

import app.database.session as session_module
from app.admin.__main__ import create_token, list_tokens, main, revoke_token
from app.database.repositories.api_tokens import ApiTokenRepository
from app.security.api_tokens import hash_token
from datetime import datetime, timezone


def test_create_list_and_revoke(db: Session):
    token = create_token(db, "alice", "postman")

    found = ApiTokenRepository(db).authenticate(hash_token(token), datetime.now(timezone.utc))
    assert found is not None and found[1] == "alice"

    listing = "\n".join(list_tokens(db))
    assert "alice" in listing and "postman" in listing
    assert token not in listing  # the secret is never shown again

    assert revoke_token(db, 1) is True
    assert ApiTokenRepository(db).authenticate(hash_token(token), datetime.now(timezone.utc)) is None


def test_the_cli_prints_the_token_once(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(session_module, "SessionLocal", session_factory)

    assert main(["token-create", "--user", "me", "--label", "cli-test"]) == 0
    created = capsys.readouterr().out
    token = next(line for line in created.splitlines() if line.startswith("pa_"))

    assert main(["token-list"]) == 0
    listed = capsys.readouterr().out
    assert "cli-test" in listed and token not in listed

    assert main(["token-revoke", "1"]) == 0
    assert main(["token-revoke", "1"]) == 1  # already revoked
    assert main(["token-revoke", "99"]) == 1  # no such token