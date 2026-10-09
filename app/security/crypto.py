import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from app.config import get_settings


class CryptoError(Exception):
    pass


def _fernet(purpose: str) -> Fernet:
    secret = get_settings().app_secret_key
    if not secret:
        raise CryptoError("APP_SECRET_KEY is not set")
    # A separate derived key per purpose, so a value encrypted for one use
    # (e.g. OAuth state) can never be accepted for another (stored tokens).
    digest = hashlib.sha256(f"{purpose}:{secret}".encode()).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt(plaintext: str, purpose: str = "storage") -> str:
    return _fernet(purpose).encrypt(plaintext.encode()).decode()


def decrypt(token: str, purpose: str = "storage", ttl: int | None = None) -> str:
    try:
        return _fernet(purpose).decrypt(token.encode(), ttl=ttl).decode()
    except InvalidToken as exc:
        raise CryptoError("Invalid or expired token") from exc