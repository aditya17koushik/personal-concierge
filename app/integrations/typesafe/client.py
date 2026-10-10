import asyncio
import logging
from typing import Any, Awaitable, Callable

import httpx

logger = logging.getLogger(__name__)

RETRY_STATUSES = {429, 529}  # rate limited / overloaded


class TypeSafeError(Exception):
    """Safe-to-log error about a TypeSafe request (never contains the API key)."""


class TypeSafeClient:
    """Minimal client for TypeSafe's System One API (POST /v1/systemone).

    Docs: https://docs.typesafe.ai/api
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.typesafe.ai",
        model: str = "jev-latest",
        timeout: float = 10.0,
        max_retries: int = 1,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._api_key = api_key
        self._url = f"{base_url.rstrip('/')}/v1/systemone"
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._transport = transport
        self._sleep = sleep

    async def evaluate(
        self, state: str | dict[str, Any] | list[Any], questions: dict[str, Any]
    ) -> dict[str, Any]:
        """Evaluate `state` against typed `questions`; returns the parsed body."""
        payload = {"state": state, "model": self._model, "questions": questions}
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(self._max_retries + 1):
            try:
                async with httpx.AsyncClient(
                    transport=self._transport, timeout=self._timeout
                ) as client:
                    response = await client.post(self._url, headers=headers, json=payload)
            except httpx.HTTPError as exc:
                # Deliberately not logging the payload: it contains user messages.
                logger.error("Could not reach TypeSafe: %s", type(exc).__name__)
                raise TypeSafeError("Could not reach TypeSafe.") from exc

            if response.status_code in RETRY_STATUSES:
                if attempt < self._max_retries:
                    await self._sleep(0.5 * (2**attempt))
                    continue
                raise TypeSafeError(f"TypeSafe is busy (HTTP {response.status_code}).")

            if response.status_code == 200:
                return self._parse(response)

            if response.status_code in (401, 403):
                raise TypeSafeError("TypeSafe rejected the API key.")
            if response.status_code == 422:
                logger.error("TypeSafe rejected the request: %s", response.text[:300])
                raise TypeSafeError("TypeSafe rejected the request as invalid.")
            raise TypeSafeError(f"TypeSafe returned HTTP {response.status_code}.")

        raise TypeSafeError("TypeSafe request failed.")  # pragma: no cover

    @staticmethod
    def _parse(response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise TypeSafeError("TypeSafe returned invalid JSON.") from exc
        if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
            raise TypeSafeError("TypeSafe response is missing 'answers'.")
        return data