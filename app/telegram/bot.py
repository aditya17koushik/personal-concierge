import asyncio
import logging
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Protocol

from app.integrations.telegram.client import TelegramAPI, TelegramError

logger = logging.getLogger(__name__)

MAX_BACKOFF_SECONDS = 30.0


class UpdateHandler(Protocol):
    async def handle(self, update: dict[str, Any]) -> None: ...


def _chat_key(update: dict[str, Any]) -> int:
    """Updates from the same chat are processed in order."""
    message = update.get("message") or {}
    if "chat" in message:
        return int(message["chat"].get("id", 0))
    callback = update.get("callback_query") or {}
    return int((callback.get("from") or {}).get("id", 0))


class TelegramBot:
    """Long-polling loop.

    - Offsets advance as soon as updates are dispatched (at-most-once): a crash
      mid-request can drop that message but never runs it twice.
    - Different chats are handled concurrently; one chat is handled in order.
    - A failing update never stops the loop; Telegram outages back off and retry.
    """

    def __init__(
        self,
        client: TelegramAPI,
        handler: UpdateHandler,
        poll_timeout: int = 30,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        max_concurrency: int = 5,
        dedupe_size: int = 1000,
    ) -> None:
        self._client = client
        self._handler = handler
        self._poll_timeout = poll_timeout
        self._sleep = sleep
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._locks: dict[int, asyncio.Lock] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._seen: OrderedDict[int, None] = OrderedDict()
        self._dedupe_size = dedupe_size

    async def run(self, stop: asyncio.Event | None = None) -> None:
        stop = stop or asyncio.Event()
        offset: int | None = None
        backoff = 1.0

        try:
            while not stop.is_set():
                try:
                    updates = await self._client.get_updates(offset, self._poll_timeout)
                except TelegramError as exc:
                    delay = exc.retry_after or backoff
                    logger.warning("Telegram polling failed: %s (retrying in %.0fs)", exc, delay)
                    backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
                    await self._sleep(delay)
                    continue

                backoff = 1.0
                for update in updates:
                    update_id = update.get("update_id")
                    if not isinstance(update_id, int):
                        continue
                    offset = max(offset or 0, update_id + 1)
                    if self._already_seen(update_id):
                        continue
                    self._dispatch(update)
        finally:
            await self._drain()

    def _already_seen(self, update_id: int) -> bool:
        if update_id in self._seen:
            return True
        self._seen[update_id] = None
        if len(self._seen) > self._dedupe_size:
            self._seen.popitem(last=False)
        return False

    def _dispatch(self, update: dict[str, Any]) -> None:
        lock = self._locks.setdefault(_chat_key(update), asyncio.Lock())
        task = asyncio.create_task(self._run_one(update, lock))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_one(self, update: dict[str, Any], lock: asyncio.Lock) -> None:
        async with lock:
            async with self._semaphore:
                try:
                    await self._handler.handle(update)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Unhandled error while handling a Telegram update")

    async def _drain(self) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)