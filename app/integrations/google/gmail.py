import asyncio
import base64
import html
import logging
import re
from html.parser import HTMLParser
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://gmail.googleapis.com/gmail/v1/users/me"
MAX_BODY_CHARS = 6000


class GmailError(Exception):
    """Safe-to-show error about a Gmail request."""


class _TextExtractor(HTMLParser):
    _BLOCK_TAGS = {"br", "p", "div", "tr", "li", "h1", "h2", "h3", "h4"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def html_to_text(raw: str) -> str:
    parser = _TextExtractor()
    parser.feed(raw)
    return parser.text()


def _decode_body(data: str) -> str:
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _walk(
    part: dict[str, Any],
    plain: list[str],
    html_parts: list[str],
    attachments: list[dict[str, Any]],
) -> None:
    mime = part.get("mimeType", "")
    filename = part.get("filename") or ""
    body = part.get("body") or {}

    if filename:
        attachments.append(
            {"filename": filename, "mime_type": mime, "size": body.get("size", 0)}
        )
    elif mime == "text/plain" and body.get("data"):
        plain.append(_decode_body(body["data"]))
    elif mime == "text/html" and body.get("data"):
        html_parts.append(_decode_body(body["data"]))

    for child in part.get("parts") or []:
        _walk(child, plain, html_parts, attachments)


def _headers(message: dict[str, Any]) -> dict[str, str]:
    raw = (message.get("payload") or {}).get("headers") or []
    return {h["name"].lower(): h["value"] for h in raw if "name" in h and "value" in h}


def _summary(message: dict[str, Any]) -> dict[str, Any]:
    headers = _headers(message)
    return {
        "id": message.get("id"),
        "thread_id": message.get("threadId"),
        "from": headers.get("from", ""),
        "subject": headers.get("subject", "(no subject)"),
        "date": headers.get("date", ""),
        "snippet": html.unescape(message.get("snippet", "")),
        "unread": "UNREAD" in (message.get("labelIds") or []),
    }


class GmailClient:
    def __init__(
        self,
        access_token: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._token = access_token
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {self._token}"},
            transport=self._transport,
            timeout=20.0,
        )

    async def _get(
        self, client: httpx.AsyncClient, path: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            response = await client.get(path, params=params)
        except httpx.HTTPError as exc:
            logger.error("Could not reach Gmail: %s", exc)
            raise GmailError("Could not reach Gmail. Please try again.") from exc

        if response.status_code == 200:
            return response.json()

        # Log status and Google's error reason only, never message content.
        logger.warning("Gmail API returned %s for %s", response.status_code, path)
        if response.status_code == 401:
            raise GmailError("Gmail rejected the access token. Please reconnect Google.")
        if response.status_code == 403:
            raise GmailError(
                "Gmail access was denied. Check that the Gmail API is enabled "
                "and that all permissions were granted."
            )
        if response.status_code == 404:
            raise GmailError("Email not found.")
        if response.status_code == 429:
            raise GmailError("Gmail rate limit reached. Try again in a moment.")
        raise GmailError("Gmail returned an unexpected error.")

    async def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        async with self._client() as client:
            listing = await self._get(
                client, "/messages", {"q": query, "maxResults": max_results}
            )
            ids = [m["id"] for m in listing.get("messages") or []]
            if not ids:
                return []

            messages = await asyncio.gather(
                *(
                    self._get(
                        client,
                        f"/messages/{message_id}",
                        {
                            "format": "metadata",
                            "metadataHeaders": ["From", "Subject", "Date"],
                        },
                    )
                    for message_id in ids
                )
            )
        return [_summary(m) for m in messages]

    async def read(self, message_id: str) -> dict[str, Any]:
        async with self._client() as client:
            message = await self._get(
                client, f"/messages/{message_id}", {"format": "full"}
            )

        plain: list[str] = []
        html_parts: list[str] = []
        attachments: list[dict[str, Any]] = []
        _walk(message.get("payload") or {}, plain, html_parts, attachments)

        if plain:
            body = _clean("\n".join(plain))
        else:
            body = _clean(html_to_text("\n".join(html_parts)))

        truncated = len(body) > MAX_BODY_CHARS
        if truncated:
            body = body[:MAX_BODY_CHARS]

        headers = _headers(message)
        return {
            **_summary(message),
            "to": headers.get("to", ""),
            "cc": headers.get("cc", ""),
            "body": body,
            "body_truncated": truncated,
            "attachments": attachments,
        }