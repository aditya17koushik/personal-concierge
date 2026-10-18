import logging
import re
from typing import Any, Protocol

import httpx

logger = logging.getLogger(__name__)

_TOKEN_IN_URL = re.compile(r"/bot\d+:[A-Za-z0-9_-]+")


class TokenRedactingFilter(logging.Filter):
    """Telegram puts the bot token in every request URL, and httpx logs request
    URLs at INFO. Redact the token from log records so it can never reach logs,
    wherever this client is used."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if _TOKEN_IN_URL.search(message):
            record.msg = _TOKEN_IN_URL.sub("/bot<redacted>", message)
            record.args = ()
        return True


def _install_log_redaction() -> None:
    for name in ("httpx", "httpcore"):
        target = logging.getLogger(name)
        if not any(isinstance(f, TokenRedactingFilter) for f in target.filters):
            target.addFilter(TokenRedactingFilter())


_install_log_redaction()


class TelegramError(Exception):
    """A Telegram API failure. Messages never contain the bot token."""

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.status = status


class TelegramAPI(Protocol):
    async def get_me(self) -> dict[str, Any]: ...
    async def delete_webhook(self) -> None: ...
    async def get_updates(self, offset: int | None, timeout: int) -> list[dict[str, Any]]: ...
    async def send_message(
        self, chat_id: int, text: str, reply_markup: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...
    async def edit_message_text(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        reply_markup: dict[str, Any] | None = None,
    ) -> None: ...
    async def answer_callback_query(
        self, callback_query_id: str, text: str | None = None
    ) -> None: ...
    async def send_chat_action(self, chat_id: int, action: str = "typing") -> None: ...
    async def get_file(self, file_id: str) -> dict[str, Any]: ...
    async def download_file(self, file_path: str, max_bytes: int) -> bytes: ...


class TelegramClient:
    """Minimal Telegram Bot API client (https://core.telegram.org/bots/api).

    The token is part of every request URL. URLs are never logged here, and the
    module installs a filter that redacts the token from httpx's own log lines.
    """

    def __init__(
        self,
        token: str,
        base_url: str = "https://api.telegram.org",
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 15.0,
    ) -> None:
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._transport = transport
        self._timeout = timeout

    async def _call(
        self,
        method: str,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        url = f"{self._base_url}/bot{self._token}/{method}"
        try:
            async with httpx.AsyncClient(transport=self._transport) as client:
                response = await client.post(
                    url, json=payload or {}, timeout=timeout or self._timeout
                )
        except httpx.HTTPError as exc:
            # `from None`: the original exception text can include the URL (token).
            raise TelegramError(f"Could not reach Telegram ({type(exc).__name__}).") from None

        try:
            data = response.json()
        except ValueError:
            raise TelegramError(
                "Telegram returned an invalid response.", status=response.status_code
            ) from None

        if not isinstance(data, dict) or not data.get("ok"):
            description = str((data or {}).get("description", "unknown error"))[:200]
            parameters = (data or {}).get("parameters") or {}
            raise TelegramError(
                description,
                retry_after=parameters.get("retry_after"),
                status=response.status_code,
            )
        return data.get("result")

    async def get_me(self) -> dict[str, Any]:
        return await self._call("getMe")

    async def delete_webhook(self) -> None:
        # Long polling and webhooks are mutually exclusive.
        await self._call("deleteWebhook", {"drop_pending_updates": False})

    async def get_updates(self, offset: int | None, timeout: int) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "timeout": timeout,
            "allowed_updates": ["message", "callback_query"],
        }
        if offset is not None:
            payload["offset"] = offset
        # The HTTP timeout must outlast the long-poll timeout.
        result = await self._call("getUpdates", payload, timeout=timeout + 10)
        return list(result or [])

    async def send_message(
        self, chat_id: int, text: str, reply_markup: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        # Plain text on purpose: model output can contain stray markup characters.
        payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
        if reply_markup:
            payload["reply_markup"] = reply_markup
        return await self._call("sendMessage", payload)

    async def edit_message_text(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        reply_markup: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {"chat_id": chat_id, "message_id": message_id, "text": text}
        # Omitting reply_markup removes the buttons.
        if reply_markup:
            payload["reply_markup"] = reply_markup
        await self._call("editMessageText", payload)

    async def answer_callback_query(
        self, callback_query_id: str, text: str | None = None
    ) -> None:
        payload: dict[str, Any] = {"callback_query_id": callback_query_id}
        if text:
            payload["text"] = text[:200]
        await self._call("answerCallbackQuery", payload)

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        await self._call("sendChatAction", {"chat_id": chat_id, "action": action})

    async def get_file(self, file_id: str) -> dict[str, Any]:
        """File metadata ({"file_path", "file_size", ...}) for a file the user sent."""
        return await self._call("getFile", {"file_id": file_id})

    async def download_file(self, file_path: str, max_bytes: int) -> bytes:
        """Download a file the user sent. Refuses anything larger than `max_bytes`."""
        url = f"{self._base_url}/file/bot{self._token}/{file_path}"
        try:
            async with httpx.AsyncClient(transport=self._transport) as client:
                async with client.stream("GET", url, timeout=self._timeout * 2) as response:
                    if response.status_code != 200:
                        raise TelegramError("Could not download the file.", status=response.status_code)
                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        data.extend(chunk)
                        if len(data) > max_bytes:
                            raise TelegramError("The file is too large.")
        except httpx.HTTPError as exc:
            raise TelegramError(f"Could not download the file ({type(exc).__name__}).") from None
        return bytes(data)