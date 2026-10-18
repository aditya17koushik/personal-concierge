"""What is allowed to leave this process in a trace.

Traces go to Langfuse (cloud or self-hosted), so everything is cleaned first:
secrets are masked, long text is cut, images and binary data are replaced by a size, and
email or calendar content (written by third parties) is never sent. With
LANGFUSE_CAPTURE_CONTENT=false even the user's own words and the model's replies are left out:
only metadata (names, durations, token counts, decisions) is traced.
"""

import re
from typing import Any

MAX_STRING = 4000
MAX_ITEMS = 50
MAX_DEPTH = 6
NOT_CAPTURED = "[content not captured]"
EXTERNAL_REDACTED = "[redacted: email or calendar content]"

_SECRET_PATTERNS = [
    re.compile(r"\bpa_[A-Za-z0-9_\-]{8,}"),  # this app's API tokens
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),  # OpenAI style keys
    re.compile(r"\bpk-lf-[A-Za-z0-9\-]{8,}|\bsk-lf-[A-Za-z0-9\-]{8,}"),  # Langfuse keys
    re.compile(r"\b\d{6,}:[A-Za-z0-9_\-]{30,}"),  # Telegram bot tokens
    re.compile(r"\bya29\.[A-Za-z0-9_\-\.]{10,}"),  # Google access tokens
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_\-\.=]{12,}"),
]
_SENSITIVE_KEYS = ("token", "secret", "password", "authorization", "api_key", "apikey", "credential")


def mask_secrets(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[secret]", text)
    return text


def sanitize(value: Any, depth: int = 0) -> Any:
    """A JSON-safe, secret-free, size-limited copy of `value`."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        text = mask_secrets(value)
        return text if len(text) <= MAX_STRING else text[:MAX_STRING] + f"... [{len(text) - MAX_STRING} more chars]"
    if isinstance(value, (bytes, bytearray)):
        return f"[{len(value)} bytes]"
    if depth >= MAX_DEPTH:
        return "[nested]"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= MAX_ITEMS:
                out["..."] = f"{len(value) - MAX_ITEMS} more keys"
                break
            key = str(k)
            lowered = key.lower()
            # "tokens" (a count: prompt_tokens, max_tokens) is not a secret; "token" is.
            if not lowered.endswith("tokens") and any(s in lowered for s in _SENSITIVE_KEYS):
                out[key] = "[redacted]"
            else:
                out[key] = sanitize(v, depth + 1)
        return out
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        shown = [sanitize(v, depth + 1) for v in items[:MAX_ITEMS]]
        if len(items) > MAX_ITEMS:
            shown.append(f"... {len(items) - MAX_ITEMS} more items")
        return shown
    return mask_secrets(str(value))[:200]


class Redactor:
    """Applies the content policy. `capture_content=False` keeps only metadata."""

    def __init__(self, capture_content: bool = True) -> None:
        self.capture_content = capture_content

    def content(self, value: Any) -> Any:
        """For anything that contains words: the user's messages, replies, prompts, tool data."""
        if value is None:
            return None
        return sanitize(value) if self.capture_content else NOT_CAPTURED

    def metadata(self, value: Any) -> Any:
        """For names, counts, ids, decisions: always allowed (still cleaned)."""
        return sanitize(value)