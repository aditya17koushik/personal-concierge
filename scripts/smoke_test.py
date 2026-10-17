#!/usr/bin/env python
"""End-to-end test of EVERY endpoint of a RUNNING Personal Agent server. No Postman needed.

Setup (once):
    python -m app.admin token-create --user smoke-test --label smoke   # prints a token ONCE

Run (server must already be running):
    python scripts/smoke_test.py --token pa_...              # or set SMOKE_API_TOKEN
    python scripts/smoke_test.py --skip-llm                  # only checks that don't call OpenAI/Jev
    python scripts/smoke_test.py --other-token pa_...         # also prove users can't see each other's approvals
    python scripts/smoke_test.py --include-disconnect         # also test DELETE /auth/google (disconnects Google!)
    python scripts/smoke_test.py --base-url http://127.0.0.1:8000 --timeout 90

What it covers: health, docs, authentication (every protected route without a token), input
validation, chat, Jev routing, approvals (list, filters, approve, reject, unknown ids, replay),
Google (status, single-use login link, callback error paths, optional disconnect), Gmail and
Calendar (skipped when Google isn't connected). At the end it reads the server's own route list
and reports any endpoint that no check exercised.

Notes:
  * Uses the real LLM, so it costs a few cents and the model's wording varies.
    Tool-usage checks can occasionally fail on a rewording; re-run before assuming a bug.
  * Writes a few test expenses (category "smoketest") for the token's user. Use a token for a
    throwaway user to keep your real data clean. Google checks are skipped for a user without
    a Google connection.
  * The Google callback's SUCCESS path needs your browser to finish the consent screen, so only
    its error paths are tested here.
  * Exit code: 0 = all passed/skipped, 1 = at least one failure, 2 = server unreachable.
"""
import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx
from urllib.parse import urlparse

MARKER_AMOUNT = "777.77"


QUICK_TIMEOUT = 10.0  # for checks that never call the LLM
MAX_CONSECUTIVE_TIMEOUTS = 2


class Skip(Exception):
    """Raised by a check that cannot run in the current setup."""


class Abort(Exception):
    """The server looks hung; stop instead of waiting on every remaining check."""


@dataclass
class Result:
    name: str
    status: str  # PASS | FAIL | SKIP
    detail: str
    seconds: float


class Runner:
    def __init__(
        self, base_url: str, token: str, timeout: float, other_token: str | None = None
    ) -> None:
        headers = {"Authorization": f"Bearer {token}"}
        quick_timeout = min(QUICK_TIMEOUT, timeout)
        hooks: dict[str, list[Callable[[httpx.Response], None]]] = {"response": [self._record]}
        self.calls: list[tuple[str, str, int]] = []  # (method, path, status) for the coverage report
        self.client = httpx.Client(
            base_url=base_url, timeout=timeout, headers=headers, event_hooks=hooks
        )
        self.quick = httpx.Client(
            base_url=base_url, timeout=quick_timeout, headers=headers, event_hooks=hooks
        )
        self.anon = httpx.Client(base_url=base_url, timeout=quick_timeout, event_hooks=hooks)
        self.other = (
            httpx.Client(
                base_url=base_url,
                timeout=quick_timeout,
                headers={"Authorization": f"Bearer {other_token}"},
                event_hooks=hooks,
            )
            if other_token
            else None
        )
        self.consecutive_timeouts = 0
        self.results: list[Result] = []
        self.google_connected = False

    def _record(self, response: httpx.Response) -> None:
        request = response.request
        self.calls.append((request.method, request.url.path, response.status_code))

    # ------------------------------------------------------------ helpers

    def chat(self, message: str) -> dict[str, Any]:
        response = self.client.post(
            "/agent/chat", json={"message": message}
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
            status, detail = "FAIL", "request timed out"
            self.consecutive_timeouts += 1
        except httpx.HTTPError as exc:
            status, detail = "FAIL", f"network error: {exc}"
        elapsed = time.perf_counter() - started
        if status != "FAIL" or "timed out" not in detail:
            self.consecutive_timeouts = 0

        self.results.append(Result(name, status, detail, elapsed))
        print(f"[{status}] {name} ({elapsed:.1f}s)" + (f"\n       {detail}" if detail else ""))

        if self.consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
            raise Abort()


def describe(body: dict[str, Any]) -> str:
    """Compact view of what Jev decided and what the assistant said."""
    d = body.get("decision") or {}
    return (
        f"jev action={d.get('action')} intents={d.get('intents')} "
        f"allowed={d.get('allowed_tools')} | reply={body['reply'][:200]!r}"
    )


def assert_tools(body: dict[str, Any], *expected: str) -> None:
    used = body["tools_used"]
    missing = [t for t in expected if t not in used]
    assert not missing, f"expected tools {list(expected)}, got {used}. {describe(body)}"


# ------------------------------------------------------------------ checks


def run_checks(r: Runner, skip_llm: bool, include_disconnect: bool = False) -> None:
    def health() -> str:
        res = r.quick.get("/health")
        assert res.status_code == 200, f"HTTP {res.status_code}"
        assert res.json().get("status") == "ok", f"unexpected body: {res.text[:120]}"
        return f"service={res.json().get('service')}"

    def validation_blank() -> None:
        res = r.quick.post("/agent/chat", json={"message": "   "})
        assert res.status_code == 422, f"expected 422, got {res.status_code}"

    def auth_no_token() -> str:
        protected = [
            ("POST", "/agent/chat"),
            ("GET", "/approvals"),
            ("POST", "/approvals/x/approve"),
            ("POST", "/approvals/x/reject"),
            ("GET", "/auth/google/status"),
            ("DELETE", "/auth/google"),
            ("POST", "/auth/google/link"),
            ("DELETE", "/agent/history"),
        ]
        for method, path in protected:
            res = r.anon.request(method, path, json={"message": "hi"} if "chat" in path else None)
            assert res.status_code == 401, f"{method} {path} -> {res.status_code}, expected 401"
        return f"{len(protected)} protected routes all return 401"

    def auth_bad_token() -> None:
        res = r.anon.post(
            "/agent/chat", json={"message": "hi"}, headers={"Authorization": "Bearer pa_wrong"}
        )
        assert res.status_code == 401, f"expected 401, got {res.status_code}"

    def history_clear() -> str:
        first = r.client.delete("/agent/history")
        assert first.status_code == 200, f"HTTP {first.status_code}: {first.text[:120]}"
        second = r.client.delete("/agent/history")
        assert second.status_code == 200 and second.json().get("deleted") == 0, f"second clear: {second.text[:120]}"
        return "history cleared; second clear deleted 0"

    def google_status() -> str:
        res = r.quick.get("/auth/google/status")
        assert res.status_code == 200, f"HTTP {res.status_code}: {res.text[:120]}"
        data = res.json()
        assert "connected" in data, f"unexpected body: {data}"
        r.google_connected = bool(data.get("connected"))
        return f"connected as {data.get('email')}" if r.google_connected else "not connected"

    def login_redirect() -> str:
        link = r.quick.post("/auth/google/link")
        assert link.status_code == 200, f"link -> HTTP {link.status_code}: {link.text[:120]}"
        parsed = urlparse(link.json()["url"])
        path = f"{parsed.path}?{parsed.query}"

        res = r.anon.get(path, follow_redirects=False)  # opened by a browser: no token
        assert res.status_code in (302, 307), f"expected redirect, got {res.status_code}: {res.text[:120]}"
        assert "accounts.google.com" in res.headers.get("location", ""), "does not redirect to Google"

        again = r.anon.get(path, follow_redirects=False)
        assert again.status_code == 400, f"link should be single-use, got {again.status_code}"
        return "single-use link works once"

    r.check("GET /health", health)
    r.check("validation: blank message -> 422", validation_blank)
    r.check("auth: no token -> 401", auth_no_token)
    r.check("auth: wrong token -> 401", auth_bad_token)
    r.check("GET /auth/google/status", google_status)
    r.check("history: DELETE /agent/history", history_clear)
    r.check("google: single-use login link -> redirect to Google", login_redirect)

    def docs_reachable() -> str:
        for path in ("/openapi.json", "/docs"):
            res = r.anon.get(path)
            assert res.status_code == 200, f"GET {path} -> {res.status_code}"
        paths = r.anon.get("/openapi.json").json().get("paths", {})
        assert "/agent/chat" in paths, "chat endpoint missing from the API spec"
        return f"{len(paths)} paths documented"

    def chat_validation() -> None:
        for payload in ({}, {"message": 5}, {"message": ""}, {"message": None}):
            res = r.quick.post("/agent/chat", json=payload)
            assert res.status_code == 422, f"{payload!r} -> {res.status_code}, expected 422"
        bad = r.quick.post(
            "/agent/chat", content=b"not json", headers={"Content-Type": "application/json"}
        )
        assert bad.status_code == 422, f"invalid JSON -> {bad.status_code}, expected 422"

    def approvals_list() -> str:
        for status in ("pending", "executed", "rejected", "expired", "failed", ""):
            res = r.quick.get("/approvals", params={"status": status})
            assert res.status_code == 200, f"status={status!r} -> {res.status_code}"
            items = res.json()
            assert isinstance(items, list), f"expected a list, got {type(items).__name__}"
            for item in items:
                for key in ("id", "tool", "summary", "status", "expires_at"):
                    assert key in item, f"approval is missing '{key}': {item}"
        return "all status filters return lists"

    def approvals_unknown_id() -> None:
        unknown = "00000000-0000-0000-0000-000000000000"
        for action in ("approve", "reject"):
            res = r.quick.post(f"/approvals/{unknown}/{action}")
            assert res.status_code == 404, f"{action} unknown id -> {res.status_code}, expected 404"

    def google_callback_errors() -> str:
        for params in ({}, {"error": "access_denied"}, {"code": "x"}):
            res = r.anon.get("/auth/google/callback", params=params)
            assert res.status_code == 400, f"{params!r} -> {res.status_code}, expected 400"
        forged = r.anon.get("/auth/google/callback", params={"code": "x", "state": "forged"})
        assert forged.status_code in (400, 503), f"forged state -> {forged.status_code}"
        assert "connected" not in forged.text.lower().replace("not connected", ""), forged.text
        return "no params / denied / missing state / forged state are all refused"

    def google_login_code_errors() -> None:
        assert r.anon.get("/auth/google/login").status_code == 422, "missing code should be 422"
        short = r.anon.get("/auth/google/login", params={"code": "short"})
        assert short.status_code == 422, f"short code -> {short.status_code}, expected 422"
        unknown = r.anon.get("/auth/google/login", params={"code": "x" * 30})
        assert unknown.status_code == 400, f"unknown code -> {unknown.status_code}, expected 400"

    r.check("docs: /openapi.json and /docs are reachable", docs_reachable)
    r.check("POST /agent/chat: bad input -> 422", chat_validation)
    r.check("GET /approvals: list shape and every status filter", approvals_list)
    r.check("POST /approvals/{id}/approve|reject: unknown id -> 404", approvals_unknown_id)
    r.check("GET /auth/google/callback: error paths are refused", google_callback_errors)
    r.check("GET /auth/google/login: bad or missing code is refused", google_login_code_errors)

    def disconnect() -> str:
        if not include_disconnect:
            raise Skip(
                "opt-in: this really disconnects Google for the token's user "
                "(add --include-disconnect to run it)"
            )
        res = r.client.delete("/auth/google")
        assert res.status_code == 200, f"HTTP {res.status_code}: {res.text[:120]}"
        assert "disconnected" in res.json(), f"unexpected body: {res.json()}"
        status = r.client.get("/auth/google/status").json()
        assert status.get("connected") is False, "still connected after DELETE"
        r.google_connected = False
        return f"disconnected={res.json()['disconnected']}. Reconnect with /google in Telegram."

    if skip_llm:
        print("\n--skip-llm set: skipping all checks that call the LLM")
        r.check("DELETE /auth/google: disconnect", disconnect)
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

    def jev_general_is_respond() -> str:
        body = r.chat("Hi, how are you today?")
        d = body.get("decision") or {}
        assert d.get("action") == "respond", f"expected respond, got {d}"
        assert body["tools_used"] == [], f"unexpected tools: {body['tools_used']}"
        return f"intents={d.get('intents')}"

    def jev_expense_scope() -> str:
        body = r.chat("I spent 5 on tea today, category smoketest")
        d = body.get("decision") or {}
        allowed = set(d.get("allowed_tools") or [])
        assert_tools(body, "add_expense")
        assert not {"search_emails", "list_events"} & allowed, f"too many tools allowed: {sorted(allowed)}"
        return f"allowed={sorted(allowed)}"

    def jev_blocks_risky_request() -> str:
        body = r.chat("Send an email to ravi@example.com saying I will be late tomorrow")
        d = body.get("decision") or {}
        assert d.get("action") == "needs_approval", f"expected needs_approval, got {d}"
        assert body["tools_used"] == [], f"tools ran: {body['tools_used']}"
        return d.get("reason", "")

    def history_memory() -> str:
        r.client.delete("/agent/history")
        r.chat("Remember for this chat: my code word is pineapple-42")
        body = r.chat("What is my code word?")
        assert "pineapple-42" in body["reply"], f"not remembered: {body['reply'][:160]!r}"
        r.client.delete("/agent/history")
        body = r.chat("What is my code word?")
        assert "pineapple-42" not in body["reply"], "history was not cleared"
        return "remembered, then forgotten after clear"

    def jev_followup() -> str:
        r.client.delete("/agent/history")
        first = r.chat("I bought something, tag it smoketest")
        assert first["tools_used"] == [], f"logged without an amount: {first['tools_used']}"
        body = r.chat("250")
        assert_tools(body, "add_expense")
        return "follow-up '250' used the earlier message"

    r.client.delete("/agent/history")
    r.check("chat: general (no tools)", general_chat)
    r.check("jev: casual chat -> respond", jev_general_is_respond)
    r.check("jev: expense message only gets expense tools", jev_expense_scope)
    r.check("jev: 'send an email' is stopped for approval", jev_blocks_risky_request)
    r.check("chat: add expense (today)", add_expense)
    r.check("chat: add expense (explicit date)", add_expense_with_date)
    r.check("chat: list expenses (sees the one just added)", list_expenses)
    r.check("chat: monthly summary", summary)
    r.check("chat: missing amount -> asks, no tool", missing_amount)
    r.check("history: remembers, then forgets after clear", history_memory)
    r.check("jev: short follow-up uses earlier message", jev_followup)

    def propose_delete(label: str) -> str:
        """Adds an expense, asks to delete it, returns the pending approval id."""
        assert_tools(r.chat(f"Add 9.99 for {label}, category smoketest"), "add_expense")
        body = r.chat(f"Delete the 9.99 {label} expense")
        d = body.get("decision") or {}
        assert d.get("action") == "needs_approval", describe(body)
        pending = body.get("pending_approvals") or []
        assert pending, f"no pending approval was created. {describe(body)}"
        assert "delete_expense" not in body["tools_used"], "deletion ran without approval!"
        return str(pending[0]["id"])

    def approval_approve() -> str:
        approval_id = propose_delete("smoketest-approve")

        listed = r.client.get("/approvals").json()
        assert approval_id in [a["id"] for a in listed], "proposal not in the pending list"

        isolation = "isolation not checked (pass --other-token)"
        if r.other is not None:
            seen = [a["id"] for a in r.other.get("/approvals").json()]
            assert approval_id not in seen, "ANOTHER USER CAN SEE THIS APPROVAL"
            for action in ("approve", "reject"):
                stolen = r.other.post(f"/approvals/{approval_id}/{action}")
                assert stolen.status_code == 404, f"another user could {action}: {stolen.status_code}"
            isolation = "another user can't see, approve or reject it"

        res = r.client.post(f"/approvals/{approval_id}/approve")
        assert res.status_code == 200, f"approve -> HTTP {res.status_code}: {res.text[:160]}"
        out = res.json()
        assert out["status"] == "executed", f"status={out['status']} result={out.get('result')}"
        assert "deleted" in (out.get("result") or {}), f"result={out.get('result')}"

        replay = r.client.post(f"/approvals/{approval_id}/approve")
        assert replay.status_code == 409, f"replay should be 409, got {replay.status_code}"
        return f"{out['summary']} | {isolation}"

    def approval_reject() -> str:
        approval_id = propose_delete("smoketest-reject")

        res = r.client.post(f"/approvals/{approval_id}/reject")
        assert res.status_code == 200, f"reject -> HTTP {res.status_code}: {res.text[:160]}"
        assert res.json()["status"] == "rejected"

        listed = r.client.get("/approvals").json()
        assert approval_id not in [a["id"] for a in listed], "rejected item still pending"
        return "rejected, and no longer pending"

    r.check("approval: propose -> approve -> executed (+ replay is 409)", approval_approve)
    r.check("approval: propose -> reject -> nothing deleted", approval_reject)

    def need_google() -> None:
        if not r.google_connected:
            raise Skip("Google is not connected for this token's user (run /google in Telegram first)")

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

    def calendar_free_time() -> None:
        need_google()
        assert_tools(r.chat("When am I free for a 30 minute call this week?"), "find_free_time")

    def calendar_create_is_gated() -> str:
        need_google()
        body = r.chat("Schedule an event called smoketest-event tomorrow at 10:00 for 30 minutes")
        assert "create_event" not in body["tools_used"], "event was created without approval!"
        pending = body.get("pending_approvals") or []
        assert pending, f"no approval was proposed. {describe(body)}"
        assert pending[0]["tool"] == "create_event", pending[0]
        approval_id = str(pending[0]["id"])
        res = r.client.post(f"/approvals/{approval_id}/reject")  # never actually create it
        assert res.status_code == 200, f"reject -> HTTP {res.status_code}"
        return pending[0].get("summary", "proposed, then rejected")

    def reminders_list() -> str:
        res = r.client.get("/reminders")
        assert res.status_code == 200, f"HTTP {res.status_code}: {res.text[:160]}"
        return f"{len(res.json())} pending"

    def reminders_cancel_unknown() -> None:
        res = r.client.delete("/reminders/999999999")
        assert res.status_code == 404, f"expected 404, got {res.status_code}"

    def reminder_create_and_cancel() -> str:
        body = r.chat("Remind me to smoketest-reminder in 3 hours")
        assert_tools(body, "create_reminder")
        listed = r.client.get("/reminders").json()
        mine = [x for x in listed if "smoketest-reminder" in x["text"]]
        assert mine, f"reminder not listed: {listed}"
        for item in mine:  # never leave a real reminder behind
            res = r.client.delete(f"/reminders/{item['id']}")
            assert res.status_code == 200, f"cancel -> HTTP {res.status_code}"
        assert not [x for x in r.client.get("/reminders").json() if "smoketest-reminder" in x["text"]]
        return "created by chat, listed, cancelled"

    def fitness_day() -> str:
        res = r.client.get("/fitness/day")
        assert res.status_code == 200, f"HTTP {res.status_code}: {res.text[:160]}"
        body = res.json()
        assert "calories" in body and "text" in body, body
        return f"{body['calories']} kcal logged today"

    def fitness_progress() -> str:
        res = r.client.get("/fitness/progress", params={"days": 30})
        assert res.status_code == 200, f"HTTP {res.status_code}: {res.text[:160]}"
        assert "notes" in res.json()
        return "ok"

    def fitness_goal() -> str:
        res = r.client.get("/fitness/goal")
        assert res.status_code in (200, 404), f"HTTP {res.status_code}: {res.text[:160]}"
        return "goal set" if res.status_code == 200 else "no goal yet (404 is normal)"

    def fitness_validation() -> None:
        assert r.client.get("/fitness/progress", params={"days": 0}).status_code == 422
        assert r.client.get("/fitness/day", params={"on": "tomorrow-ish"}).status_code == 422

    def fitness_meal_logged_as_estimate() -> str:
        before = r.client.get("/fitness/day").json()["calories"]
        body = r.chat("I had 2 boiled eggs and a slice of toast for breakfast")
        assert_tools(body, "log_meal")
        after = r.client.get("/fitness/day").json()
        assert after["calories"] > before, f"calories did not go up: {before} -> {after['calories']}"
        newest = max(after["meals"], key=lambda m: m["id"])
        assert newest["estimated"] is True, "an estimate must be marked as estimated"
        # clean up through the real approval flow (a human tap is the only way to delete)
        cleanup = r.chat(f"Delete meal {newest['id']}")
        proposals = [p for p in (cleanup.get("pending_approvals") or []) if p["tool"] == "delete_log_entry"]
        assert proposals, f"cleanup: no delete proposal. {describe(cleanup)}"
        res = r.client.post(f"/approvals/{proposals[0]['id']}/approve")
        assert res.status_code == 200, f"cleanup approve -> HTTP {res.status_code}"
        return f"+{after['calories'] - before} kcal logged as an estimate, then removed"

    r.check("GET /fitness/day", fitness_day)
    r.check("GET /fitness/progress", fitness_progress)
    r.check("GET /fitness/goal", fitness_goal)
    r.check("fitness: bad parameters are 422", fitness_validation)
    r.check("fitness: a meal in plain words is logged as an estimate", fitness_meal_logged_as_estimate)
    r.check("GET /reminders", reminders_list)
    r.check("DELETE /reminders/{id}: unknown id is 404", reminders_cancel_unknown)
    r.check("reminder: created by chat, listed, cancelled", reminder_create_and_cancel)
    r.check("calendar: free time", calendar_free_time)
    r.check("calendar: create event is proposed, never run (rejected)", calendar_create_is_gated)
    r.check("calendar: list events", calendar_list)
    r.check("calendar: event details", calendar_details)
    r.check("gmail: search emails", gmail_search)
    r.check("gmail: read most recent email", gmail_read)
    r.check("gmail: unconnected user is told how to connect", gmail_not_connected_message)
    r.check("DELETE /auth/google: disconnect", disconnect)  # last: it ends the Google checks above


ALWAYS_LLM = {
    ("POST", "/agent/chat"),
    ("POST", "/approvals/{approval_id}/approve"),
    ("POST", "/approvals/{approval_id}/reject"),
}


def report_coverage(r: Runner, skip_llm: bool, include_disconnect: bool) -> None:
    """Compare what we called against the server's own route list. An endpoint counts only
    if at least one call to it SUCCEEDED (2xx/3xx): a 401 or 404 proves nothing about it."""
    try:
        spec = r.anon.get("/openapi.json").json()
    except (httpx.HTTPError, ValueError):
        print("\nEndpoint coverage: could not read /openapi.json")
        return

    routes: list[tuple[str, str, re.Pattern[str]]] = []
    for path, operations in spec.get("paths", {}).items():
        regex = re.compile("^" + re.sub(r"\{[^}]+\}", "[^/]+", path) + "$")
        for method in operations:
            if method in ("get", "post", "put", "patch", "delete"):
                routes.append((method.upper(), path, regex))

    missing = [
        (method, path)
        for method, path, regex in routes
        if not any(m == method and regex.match(p) and 200 <= code < 400 for m, p, code in r.calls)
    ]
    done = len(routes) - len(missing)
    print(f"\nEndpoint coverage: {done} of {len(routes)} endpoints had a successful call")
    for method, path in missing:
        if path == "/auth/google/callback":
            why = "error paths are tested; the success path needs your browser to finish Google's consent screen"
        elif (method, path) == ("DELETE", "/auth/google") and not include_disconnect:
            why = "opt-in: --include-disconnect"
        elif skip_llm and (method, path) in ALWAYS_LLM:
            why = "skipped by --skip-llm"
        else:
            why = "no check covers it: add one"
        print(f"  not covered: {method} {path}\n               {why}")

    if r.other is None and not skip_llm:
        print("  tip: pass --other-token <another user's token> to also verify that users")
        print("       can't see or approve each other's approvals.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test a running Personal Agent server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--token", default=os.environ.get("SMOKE_API_TOKEN"), help="API token (or env SMOKE_API_TOKEN)"
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="per-request timeout in seconds")
    parser.add_argument("--skip-llm", action="store_true", help="skip checks that call the LLM")
    parser.add_argument(
        "--other-token",
        default=os.environ.get("SMOKE_OTHER_TOKEN"),
        help="a DIFFERENT user's token, to prove users can't see each other's approvals",
    )
    parser.add_argument(
        "--include-disconnect",
        action="store_true",
        help="also test DELETE /auth/google (this really disconnects Google for the token's user)",
    )
    args = parser.parse_args()

    if not args.token:
        print(
            "No API token. Create one with:\n"
            "  python -m app.admin token-create --user me --label smoke\n"
            "then run with --token pa_... or set SMOKE_API_TOKEN."
        )
        return 2

    runner = Runner(args.base_url, args.token, args.timeout, args.other_token)

    print(f"Target: {args.base_url}\n")
    try:
        runner.quick.get("/health")
    except httpx.ConnectError:
        print(f"Cannot connect to {args.base_url}. Is uvicorn running?")
        return 2
    except httpx.TimeoutException:
        print(
            "The server accepted the connection but did not answer /health.\n"
            "The process is probably hung (often a stuck database call).\n"
            "Stop uvicorn (Ctrl+C, or find it with `netstat -ano | findstr :8000` "
            "and `taskkill /PID <pid> /F`), check Postgres, then start it again."
        )
        return 2

    try:
        run_checks(runner, args.skip_llm, args.include_disconnect)
    except Abort:
        print(
            "\nStopping early: requests keep timing out. The server answers /health "
            "but hangs on requests that need the database or an external API.\n"
            "Check Postgres first:\n"
            "  docker ps\n"
            "  docker logs personal-agent-postgres --tail 20\n"
            "  docker exec -it personal-agent-postgres psql -U agent -d personal_agent -c \"SELECT 1;\""
        )

    report_coverage(runner, args.skip_llm, args.include_disconnect)

    passed = sum(r.status == "PASS" for r in runner.results)
    failed = sum(r.status == "FAIL" for r in runner.results)
    skipped = sum(r.status == "SKIP" for r in runner.results)
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())