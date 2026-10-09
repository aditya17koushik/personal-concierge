#!/usr/bin/env python
"""End-to-end smoke test for a RUNNING Personal Agent server.

Usage (server must already be running):

    python scripts/smoke_test.py                       # core checks, user "smoke-test"
    python scripts/smoke_test.py --user-id me          # also tests Gmail if Google is connected
    python scripts/smoke_test.py --skip-llm            # only checks that don't call the LLM
    python scripts/smoke_test.py --base-url http://127.0.0.1:8000 --timeout 90

Notes:
  * Uses the real LLM, so it costs a few cents and the model's wording varies.
    Tool-usage checks can occasionally fail on a rewording; re-run before assuming a bug.
  * Writes a few test expenses (category "smoketest") for the chosen user.
    The default user "smoke-test" keeps your real data clean.
  * Exit code: 0 = all passed/skipped, 1 = at least one failure, 2 = server unreachable.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx

MARKER_AMOUNT = "777.77"


class Skip(Exception):
    """Raised by a check that cannot run in the current setup."""


@dataclass
class Result:
    name: str
    status: str  # PASS | FAIL | SKIP
    detail: str
    seconds: float


class Runner:
    def __init__(self, base_url: str, user_id: str, timeout: float) -> None:
        self.user_id = user_id
        self.client = httpx.Client(base_url=base_url, timeout=timeout)
        self.results: list[Result] = []
        self.google_connected = False

    # ------------------------------------------------------------ helpers

    def chat(self, message: str) -> dict[str, Any]:
        response = self.client.post(
            "/agent/chat", json={"user_id": self.user_id, "message": message}
        )
        if response.status_code != 200:
            raise AssertionError(f"HTTP {response.status_code}: {response.text[:200]}")
        body = response.json()
        for key in ("reply", "provider", "model", "tools_used"):
            assert key in body, f"response is missing '{key}'"
        return body

    def check(self, name: str, fn: Callable[[], str | None]) -> None:
        started = time.perf_counter()
        try:
            detail = fn() or ""
            status = "PASS"
        except Skip as exc:
            status, detail = "SKIP", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc)
        except httpx.TimeoutException:
            status, detail = "FAIL", "request timed out (try --timeout 120)"
        except httpx.HTTPError as exc:
            status, detail = "FAIL", f"network error: {exc}"
        elapsed = time.perf_counter() - started

        self.results.append(Result(name, status, detail, elapsed))
        print(f"[{status}] {name} ({elapsed:.1f}s)" + (f"\n       {detail}" if detail else ""))


def assert_tools(body: dict[str, Any], *expected: str) -> None:
    used = body["tools_used"]
    missing = [t for t in expected if t not in used]
    assert not missing, f"expected tools {list(expected)}, got {used}. Reply: {body['reply'][:160]!r}"


# ------------------------------------------------------------------ checks


def run_checks(r: Runner, skip_llm: bool) -> None:
    def health() -> str:
        res = r.client.get("/health")
        assert res.status_code == 200, f"HTTP {res.status_code}"
        assert res.json().get("status") == "ok", f"unexpected body: {res.text[:120]}"
        return f"service={res.json().get('service')}"

    def validation_blank() -> None:
        res = r.client.post("/agent/chat", json={"user_id": r.user_id, "message": "   "})
        assert res.status_code == 422, f"expected 422, got {res.status_code}"

    def validation_missing_user() -> None:
        res = r.client.post("/agent/chat", json={"message": "hi"})
        assert res.status_code == 422, f"expected 422, got {res.status_code}"

    def google_status() -> str:
        res = r.client.get("/auth/google/status", params={"user_id": r.user_id})
        assert res.status_code == 200, f"HTTP {res.status_code}: {res.text[:120]}"
        data = res.json()
        r.google_connected = bool(data.get("connected"))
        return f"connected as {data.get('email')}" if r.google_connected else "not connected"

    def login_redirect() -> None:
        res = r.client.get(
            "/auth/google/login",
            params={"user_id": r.user_id},
            follow_redirects=False,
        )
        assert res.status_code in (302, 307), f"expected redirect, got {res.status_code}: {res.text[:120]}"
        assert "accounts.google.com" in res.headers.get("location", ""), "does not redirect to Google"

    r.check("GET /health", health)
    r.check("validation: blank message -> 422", validation_blank)
    r.check("validation: missing user_id -> 422", validation_missing_user)
    r.check("GET /auth/google/status", google_status)
    r.check("GET /auth/google/login redirects to Google", login_redirect)

    if skip_llm:
        print("\n--skip-llm set: skipping all checks that call the LLM")
        return

    def general_chat() -> str:
        body = r.chat("Say hello in one short sentence")
        assert body["reply"].strip(), "empty reply"
        assert body["tools_used"] == [], f"unexpected tools: {body['tools_used']}"
        return f"{body['provider']} / {body['model']}"

    def add_expense() -> None:
        body = r.chat(f"I spent {MARKER_AMOUNT} on smoketest today, category smoketest")
        assert_tools(body, "add_expense")

    def add_expense_with_date() -> None:
        body = r.chat("Add 11.50 for smoketest on 2026-10-01, category smoketest")
        assert_tools(body, "add_expense")

    def list_expenses() -> None:
        body = r.chat("Show my last 5 expenses")
        assert_tools(body, "list_expenses")
        assert MARKER_AMOUNT in body["reply"], f"marker amount {MARKER_AMOUNT} not in reply: {body['reply'][:160]!r}"

    def summary() -> None:
        body = r.chat("How much did I spend this month by category?")
        assert_tools(body, "get_expense_summary")

    def missing_amount() -> None:
        body = r.chat("I bought something")
        assert body["tools_used"] == [], f"should ask for details, but used {body['tools_used']}"

    r.check("chat: general (no tools)", general_chat)
    r.check("chat: add expense (today)", add_expense)
    r.check("chat: add expense (explicit date)", add_expense_with_date)
    r.check("chat: list expenses (sees the one just added)", list_expenses)
    r.check("chat: monthly summary", summary)
    r.check("chat: missing amount -> asks, no tool", missing_amount)

    def need_google() -> None:
        if not r.google_connected:
            raise Skip(f"Google not connected for user '{r.user_id}' (use --user-id with a connected user)")

    def gmail_search() -> None:
        need_google()
        assert_tools(r.chat("Do I have any unread emails from the last 7 days?"), "search_emails")

    def gmail_read() -> None:
        need_google()
        body = r.chat("Summarize my most recent email")
        assert_tools(body, "search_emails", "read_email")

    def gmail_not_connected_message() -> None:
        if r.google_connected:
            raise Skip("user is connected; this check needs a user without Google")
        body = r.chat("Search my email for anything from last week")
        reply = body["reply"].lower()
        assert "connect" in reply or "login" in reply or "google" in reply, f"reply does not explain how to connect: {body['reply'][:160]!r}"

    def calendar_list() -> None:
        need_google()
        assert_tools(r.chat("What's on my calendar this week?"), "list_events")

    def calendar_details() -> None:
        need_google()
        body = r.chat("Look up my next calendar event over the next 30 days and give me its details")
        assert_tools(body, "list_events")

    r.check("calendar: list events", calendar_list)
    r.check("calendar: event details", calendar_details)
    r.check("gmail: search emails", gmail_search)
    r.check("gmail: read most recent email", gmail_read)
    r.check("gmail: unconnected user is told how to connect", gmail_not_connected_message)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test a running Personal Agent server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--user-id", default="smoke-test")
    parser.add_argument("--timeout", type=float, default=60.0, help="per-request timeout in seconds")
    parser.add_argument("--skip-llm", action="store_true", help="skip checks that call the LLM")
    args = parser.parse_args()

    runner = Runner(args.base_url, args.user_id, args.timeout)

    print(f"Target: {args.base_url}   user_id: {args.user_id}\n")
    try:
        runner.client.get("/health")
    except httpx.ConnectError:
        print(f"Cannot connect to {args.base_url}. Is uvicorn running?")
        return 2

    run_checks(runner, args.skip_llm)

    passed = sum(r.status == "PASS" for r in runner.results)
    failed = sum(r.status == "FAIL" for r in runner.results)
    skipped = sum(r.status == "SKIP" for r in runner.results)
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())