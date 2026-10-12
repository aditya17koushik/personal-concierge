"""Run the Telegram bot:   python -m app.telegram
Check the token only:     python -m app.telegram --check
"""

import asyncio
import logging
import sys

from app.agent.factory import build_agent_service, build_approval_service
from app.config import get_settings
from app.database.session import SessionLocal
from app.integrations.telegram.client import TelegramClient, TelegramError
from app.telegram.bot import TelegramBot
from app.telegram.handler import TelegramHandler


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Telegram puts the bot token in every request URL. httpx logs request URLs
    # at INFO, which would write the token to the log, so keep it quiet.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


async def amain(check_only: bool) -> int:
    settings = get_settings()

    if not settings.telegram_bot_token:
        print("TELEGRAM_BOT_TOKEN is not set in .env")
        return 2
    if not settings.telegram_admin_id:
        print("TELEGRAM_ADMIN_ID is not set in .env (your numeric Telegram id)")
        return 2

    client = TelegramClient(settings.telegram_bot_token)
    try:
        me = await client.get_me()
    except TelegramError as exc:
        print(f"Telegram rejected the bot token or is unreachable: {exc}")
        return 1

    username = me.get("username")
    print(f"Connected as @{username} (admin id {settings.telegram_admin_id})")
    if check_only:
        return 0

    await client.delete_webhook()  # polling can't coexist with a webhook

    handler = TelegramHandler(
        client=client,
        session_factory=SessionLocal,
        settings=settings,
        agent_factory=lambda db: build_agent_service(db, settings),
        approvals_factory=lambda db: build_approval_service(db, settings),
        bot_username=username,
    )
    bot = TelegramBot(client, handler, poll_timeout=settings.telegram_poll_timeout)

    print("Listening for messages. Press Ctrl+C to stop.")
    await bot.run()
    return 0


def main() -> None:
    configure_logging()
    try:
        sys.exit(asyncio.run(amain("--check" in sys.argv)))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()