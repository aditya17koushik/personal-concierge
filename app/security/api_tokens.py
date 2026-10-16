import hashlib
import secrets

TOKEN_PREFIX = "pa_"


def hash_token(token: str) -> str:
    """Tokens are long random strings, so a plain SHA-256 is enough to store."""
    return hashlib.sha256(token.strip().encode()).hexdigest()


def generate_token() -> tuple[str, str]:
    """Returns (plain token to show once, hash to store)."""
    plain = TOKEN_PREFIX + secrets.token_urlsafe(32)
    return plain, hash_token(plain)