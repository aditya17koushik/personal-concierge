"""Admin commands.

    python -m app.admin token-create --user me --label postman
    python -m app.admin token-list
    python -m app.admin token-revoke 3
"""

import argparse
import sys
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.database.repositories.api_tokens import ApiTokenRepository
from app.database.repositories.users import UserRepository
from app.security.api_tokens import generate_token


def create_token(db: Session, external_id: str, label: str) -> str:
    user = UserRepository(db).get_or_create(external_id)
    plain, token_hash = generate_token()
    ApiTokenRepository(db).create(user.id, token_hash, label)
    return plain


def list_tokens(db: Session) -> list[str]:
    def fmt(value: datetime | None) -> str:
        return value.strftime("%Y-%m-%d %H:%M") if value else "-"

    lines = ["id  user  label  created  last_used  revoked"]
    for token_id, user, label, created, last_used, revoked in ApiTokenRepository(db).list_all():
        lines.append(f"{token_id}  {user}  {label}  {fmt(created)}  {fmt(last_used)}  {fmt(revoked)}")
    return lines


def revoke_token(db: Session, token_id: int) -> bool:
    return ApiTokenRepository(db).revoke(token_id, datetime.now(timezone.utc))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.admin")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("token-create", help="create an API token for a user")
    create.add_argument("--user", required=True, help="internal user id, e.g. me")
    create.add_argument("--label", default="cli")
    sub.add_parser("token-list", help="list tokens (never shows the secret)")
    revoke = sub.add_parser("token-revoke", help="revoke a token by id")
    revoke.add_argument("token_id", type=int)

    args = parser.parse_args(argv)

    from app.database.session import SessionLocal  # imported late: needs DATABASE_URL

    with SessionLocal() as db:
        if args.command == "token-create":
            token = create_token(db, args.user, args.label)
            print(f"Token for '{args.user}' ({args.label}). It is shown ONCE, so save it now:\n")
            print(token)
            print("\nUse it as:  Authorization: Bearer <token>")
        elif args.command == "token-list":
            print("\n".join(list_tokens(db)))
        else:
            ok = revoke_token(db, args.token_id)
            print("Revoked." if ok else "No such active token.")
            return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())