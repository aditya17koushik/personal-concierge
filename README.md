# Personal Agent

A personal AI agent backend built with FastAPI. It talks to an LLM (OpenAI by default, Ollama optional), and is being extended step by step with expense tracking, a decision layer (Jev), Google integrations, approvals, messaging channels, and memory.

## Tech stack

- **API:** FastAPI + Uvicorn
- **Database:** PostgreSQL 17, SQLAlchemy, Alembic
- **Validation / config:** Pydantic, pydantic-settings
- **LLM:** OpenAI (`gpt-4o-mini`), Ollama (`llama3.2:latest`, disabled by default)
- **Decision layer:** TypeSafe AI Jev (System One API), which returns typed decisions with calibrated probabilities
- **Observability:** Langfuse (planned, Step 4)
- **Tests:** pytest, pytest-asyncio

## Project structure

```
.
├── app/
│   ├── main.py                 # FastAPI app, /health endpoint
│   ├── config.py               # Settings loaded from .env
│   ├── agent/                  # Agent service (Step 3)
│   ├── api/                    # HTTP routers
│   ├── database/
│   │   ├── session.py          # Engine, SessionLocal, Base, get_db
│   │   ├── models.py           # SQLAlchemy models
│   │   └── repositories/       # DB access layer
│   ├── decisions/              # Jev decision layer (Step 6)
│   ├── integrations/           # Google OAuth and other external services
│   ├── llm/
│   │   ├── base.py             # LLMProvider interface, LLMError
│   │   ├── openai_provider.py  # OpenAI implementation
│   │   ├── ollama_provider.py  # Ollama implementation (commented out)
│   │   └── factory.py          # get_llm_provider()
│   ├── messaging/              # Telegram / WhatsApp
│   ├── observability/          # Langfuse
│   ├── schemas/
│   │   └── llm.py              # Message, ToolCall, LLMResponse, Usage
│   └── tools/
│       ├── calendar/
│       ├── expenses/
│       └── gmail/
├── tests/
├── alembic/                    # Migrations
├── docker-compose.yml          # PostgreSQL
├── requirements.txt
└── .env.example
```

## Getting started

### 1. Prerequisites

- Python 3.11+
- Docker (for PostgreSQL)
- An OpenAI API key (or Ollama, see below)

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```env
DATABASE_URL=postgresql+psycopg://agent:agent_password@localhost:5433/personal_agent
LLM_PROVIDER=openai
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-4o-mini
```

Note: `docker-compose.yml` maps Postgres to host port **5433**, so `DATABASE_URL` must use 5433.

Never commit `.env`. Only `.env.example` belongs in version control.

### 4. Start PostgreSQL

```bash
docker compose up -d
```

### 5. Run migrations

```bash
alembic upgrade head
```

### 6. Start the API

```bash
uvicorn app.main:app --reload
```

- Health check: http://localhost:8000/health
- Interactive docs: http://localhost:8000/docs

## Authentication

Every endpoint except `/health`, Google's callback and the single-use login link needs a personal API
token. The server works out who you are **from the token**; no request carries a `user_id`.

```
python -m app.admin token-create --user me --label postman   # prints the token ONCE
python -m app.admin token-list                                # never shows secrets
python -m app.admin token-revoke 3
```

Send it as `Authorization: Bearer pa_...`. Only a hash is stored. In Postman, paste it into the
collection variable `token`. Telegram users never need a token: Telegram verifies who they are.

**Connecting Google** uses a single-use link that expires after `GOOGLE_LOGIN_LINK_TTL_MINUTES`. Get it
from `/google` in Telegram, or `POST /auth/google/link` with your token, and open it in a browser.
Links are created only for an already-verified user. Only open links you requested yourself: a link
someone else sends you would connect *your* Google account to *their* assistant.

## Profiles and limits

Each user has a profile (timezone, currency, units), set from Telegram with `/timezone Asia/Kolkata`,
`/currency INR`, `/units metric` and shown by `/profile`. "Today", relative dates and the currency used
for expenses follow the profile, so people in different timezones each get their own day.

`DAILY_MESSAGE_LIMIT` caps messages per user per local day. It is checked before Jev or any model
call, so a user at their limit costs nothing. The API returns `429` with `Retry-After`; Telegram
explains when it resets. Commands and approval buttons are free.

## Conversation history

The agent remembers the recent conversation per user. Only final texts are stored (not tool
traffic). A message window is replayed: last `HISTORY_MAX_MESSAGES`, within `HISTORY_TTL_MINUTES`,
capped at `HISTORY_MAX_CHARS`; older rows are pruned after `HISTORY_RETENTION_DAYS`. Replies built
from email or calendar content are flagged as external and treated as untrusted data. Short
follow-ups ("250") get a second Jev look that includes the user's earlier messages. Clear with
`DELETE /agent/history` or `/new` in Telegram. Run `alembic revision --autogenerate -m "add conversation history"`
then `alembic upgrade head`.

## Telegram

The bot uses long polling, so it needs no public URL and works on your own machine. It runs as its
own process and calls the same agent code as the API (Jev, tools and approvals included).

1. Create a bot with **@BotFather** (`/newbot`) and put the token in `.env` as `TELEGRAM_BOT_TOKEN`.
2. Get your numeric id from **@userinfobot** and set `TELEGRAM_ADMIN_ID`.
3. Run `alembic revision --autogenerate -m "add telegram"` then `alembic upgrade head`.
4. Check the token: `python -m app.telegram --check`
5. Start it (alongside uvicorn): `python -m app.telegram`

Access is invite-only. The admin is linked to the existing internal user (`TELEGRAM_ADMIN_EXTERNAL_ID`,
default `me`), so your expenses and Google connection carry over. Others join with `/invite`:

| Command | Who | What |
| --- | --- | --- |
| `/start <code>` | anyone | Redeem a single-use invite |
| `/help`, `/approvals` | linked users | Help, pending approvals |
| `/new` | linked users | Forget the current conversation |
| `/google` | linked users | Single-use Google login link |
| `/profile`, `/timezone`, `/currency`, `/units` | linked users | View or change your profile |
| `/today`, `/progress [days]`, `/goal` | linked users | Fitness day summary, trend and goal (no model involved) |
| `/reminders`, `/reminders cancel <id>` | linked users | List or cancel pending reminders |
| `/agenda on 08:00`, `/agenda off` | linked users | Daily morning summary |
| `/meetings 15`, `/meetings off` | linked users | Alert N minutes before meetings |
| `/invite`, `/users`, `/remove <id>` | admin | Manage who can use the bot |

Approve and Reject are buttons only. They are never tools the model can call, so text inside an
email or invite can't approve anything. Private chats only; groups are ignored.

## Fitness log

Write normally: "had 2 rotis and dal for lunch", "I did a 5 km run in 32 minutes", "I weigh 78.5",
"I want to cut, target 72 kg, 2000 calories", "how am I doing?", "calories left today?".

| Tool | What it does |
| --- | --- |
| `log_meal`, `update_meal`, `list_meals` | Meals with calories and optional protein/carbs/fat |
| `log_workout`, `list_workouts` | Strength, cardio, walk, run, cycle, yoga, sport, other |
| `log_weight`, `list_weights` | Weigh-ins (one per day; a new one replaces that day's) |
| `set_goal`, `get_goal` | Goal: `cut`, `bulk` or `maintain`, with optional target weight, calories, protein, workouts/week |
| `get_day_summary`, `get_progress` | Totals, what is left against the goal, trend and notes |
| `delete_log_entry` | Delete a meal, workout or weigh-in. **Needs your approval** |

How it behaves:
- Logging needs no approval, and the model never asks for numbers. If you state calories they are used exactly; otherwise it **estimates** and the entry is marked `estimated` (shown as "est."). Correct one with "that was 400 kcal" (`update_meal`); a corrected meal is no longer an estimate.
- All arithmetic (daily totals, calories left, weekly weight change, averages) is done in code, never by the model.
- Weights are stored in kg (two decimals) and shown in your unit (`/units metric|imperial`; say "lb" or "kg" to override once).
- Sanity checks: weights 25-400 kg, one meal up to 5000 kcal, no future dates, a cut needs a target below your current weight (bulk above), calorie targets 800-6000 (under 1200 saves with a "check with a professional" warning).
- Progress notes are plain facts, for example "steady pace for a cut", "weight is going UP while the goal is a cut", "not enough weigh-ins for a trend". Averages count only days with meals logged; the note says how many.
- This is not medical advice. Dieting tips for medical conditions, pregnancy or eating disorders are left to professionals.

Read-only API: `GET /fitness/day?on=YYYY-MM-DD`, `GET /fitness/progress?days=30`, `GET /fitness/goal`.
Telegram: `/today` (or `/today yesterday`), `/progress 7`, `/goal`.

After pulling this phase: `alembic revision --autogenerate -m "add fitness tables"`, check that the file
only creates `meals`, `workouts`, `body_weights` and `fitness_goals`, then `alembic upgrade head`.

## Reminders and scheduler

Ask in plain words: "remind me to call Ravi at 6pm", "remind me in 30 minutes", "remind me every
weekday at 9 to stretch". Reminders (`create_reminder`, `list_reminders`, `cancel_reminder`) need no
approval: they only message you. They are sent on Telegram.

- Times are your local wall-clock time, so set your zone first with `/timezone` (a daily 08:00 stays 08:00 across daylight saving).
- Repeats: daily, weekdays, weekly.
- Delivery is at-least-once with an atomic claim, so two workers never send the same reminder. Temporary Telegram errors are retried (after 1 and 5 minutes, 3 tries); a blocked bot fails at once.
- A reminder missed by more than `REMINDER_LATE_HOURS` (bot was off) is dropped, not sent late; a repeating one skips ahead. Slightly late ones say when they were due.
- **Morning agenda** (`/agenda on 08:00`): today's events plus your reminders, sent once a day within `AGENDA_WINDOW_HOURS` of the chosen time. Works without Google (reminders only).
- **Meeting alerts** (`/meetings 15`): one message per event, skipping declined, "free" and all-day events.
- Agenda and alert text is built by code; event titles are untrusted and are flattened and trimmed.
- The scheduler runs inside `python -m app.telegram`. **If that process is not running, nothing is sent.**

`GET /reminders` and `DELETE /reminders/{id}` expose the same list over the API.

After pulling this phase: `alembic revision --autogenerate -m "add reminders and nudges"`, check that the
file only adds `reminders`, `event_alerts` and the four new `user_profiles` columns, then `alembic upgrade head`.

## Calendar actions

Reading the calendar needs no approval. `find_free_time` is also read-only: it looks for gaps inside
working hours (default 09:00-18:00, weekdays) and ignores declined and "free" events.

Every change is a proposal the user approves with a button (or `POST /approvals/{id}/approve`):

| Tool | What it prepares |
| --- | --- |
| `create_event` | New event (timed or all-day), optional location and up to 10 guests |
| `update_event` | Move, rename, or change location or description of an existing event |
| `delete_event` | Delete one event (one occurrence of a recurring event) |
| `rsvp_event` | Accept, decline, or reply maybe to an invitation |

The approval text is built by code and states exactly what will happen, including when guests will
be emailed (invitations, changes, cancellations) or the organizer notified. Times are in the
calendar's own time zone. Gmail stays read-only.

Editing needs the extra Google permission `calendar.events`. Users who connected before Phase 14
must log in again (`/google` in Telegram); until then a change fails with a message saying so.

## Approvals

Risky actions are never executed straight from a chat message. When Jev rates a request high risk
and an approval-gated tool exists for it (today: `delete_expense`), the assistant only **proposes**
the change:

1. The model gets read-only lookup tools plus a proposal-only version of the risky tool.
2. A proposal is stored in `pending_actions`. Nothing runs.
3. The summary you approve is built by code from the validated arguments, not written by the model.
4. The chat response lists `pending_approvals` (id, tool, summary, expiry).
5. You decide:

```
GET  /approvals?user_id=me                      # pending by default; ?status=executed|rejected|expired|failed
POST /approvals/{id}/approve?user_id=me         # runs it
POST /approvals/{id}/reject?user_id=me
```

An approval runs at most once (atomic claim), expires after `APPROVAL_TTL_MINUTES`, and is scoped to
its owner. A high-risk request with no approval-gated tool (for example "send an email") is declined.

New approval-gated tools set `risk="high"` (or `medium` with `side_effects=True`) and must provide a
`summarize` function; the registry refuses to start without one.

## Database migrations

```bash
# Create a migration after changing models
alembic revision --autogenerate -m "describe the change"

# Apply migrations
alembic upgrade head

# Roll back one migration
alembic downgrade -1
```

Remember to import any new model modules in Alembic's `env.py` so autogenerate can see them.

## LLM providers

All LLM access goes through the `LLMProvider` interface in `app/llm/base.py`. The rest of the app uses the provider-agnostic types in `app/schemas/llm.py` and never imports the OpenAI SDK directly.

```python
from app.llm.factory import get_llm_provider
from app.schemas.llm import Message

provider = get_llm_provider()
response = await provider.chat([Message(role="user", content="Hello")])
print(response.content)
```

Tool calling uses the OpenAI function-calling schema and comes back as normalized `ToolCall` objects.

### Switching to Ollama

Ollama exposes an OpenAI-compatible API, so it reuses the same SDK with a different `base_url`.

1. Install Ollama and run `ollama pull llama3.2:latest`
2. Uncomment the code in `app/llm/ollama_provider.py`
3. Uncomment the import and the `ollama` branch in `app/llm/factory.py`
4. Set in `.env`:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3.2:latest
```

Local models are less reliable at tool calling than OpenAI models. Malformed tool arguments are handled gracefully (they become an empty dict), but expect occasional misses.

## Testing

```bash
python -m pytest tests -q
```

Run from the project root. The tests need `DATABASE_URL` to be set (your `.env` covers this) because settings are loaded on import. LLM tests use a fake client, so they do not call the real API.

## Configuration reference

| Variable | Default | Description |
| --- | --- | --- |
| `APP_NAME` | `personal-agent` | Service name |
| `APP_ENV` | `development` | Environment name |
| `DEBUG` | `true` | Debug mode |
| `DATABASE_URL` | required | SQLAlchemy URL (`postgresql+psycopg://...`) |
| `LLM_PROVIDER` | `openai` | `openai` or `ollama` |
| `OPENAI_API_KEY` | empty | Required when using OpenAI |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible endpoint |
| `OLLAMA_MODEL` | `llama3.2:latest` | Ollama model name |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | empty | Langfuse credentials (Step 4) |
| `LANGFUSE_BASE_URL` | `https://cloud.langfuse.com` | Langfuse host |
| `TELEGRAM_BOT_TOKEN` | empty | Telegram bot (Step 12) |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` / `GOOGLE_REDIRECT_URI` | empty | Google OAuth (Step 7) |
| `APP_SECRET_KEY` | empty | Application secret |
| `TELEGRAM_BOT_TOKEN` | empty | Bot token from @BotFather |
| `TELEGRAM_ADMIN_ID` | `0` | Your numeric Telegram id (from @userinfobot). `0` disables the bot |
| `TELEGRAM_ADMIN_EXTERNAL_ID` | `me` | Internal user the admin maps to, so existing data carries over |
| `TELEGRAM_MAX_USERS` | `10` | Cap on linked Telegram users |
| `TELEGRAM_INVITE_TTL_HOURS` | `24` | Invite code lifetime |
| `TELEGRAM_AGENT_TIMEOUT_SECONDS` | `90` | Give up on one request after this long |
| `PUBLIC_BASE_URL` | `http://127.0.0.1:8000` | Used to build the `/google` login link |
| `APPROVAL_TTL_MINUTES` | `30` | How long a proposed action can be approved before it expires |
| `DEFAULT_TIMEZONE` | `UTC` | Timezone for new profiles (users change theirs with `/timezone`) |
| `DAILY_MESSAGE_LIMIT` | `200` | Messages per user per local day. `0` disables the limit |
| `GOOGLE_LOGIN_LINK_TTL_MINUTES` | `10` | Lifetime of a single-use Google login link |
| `HISTORY_TTL_MINUTES` | `360` | Conversation is forgotten after this much silence |
| `HISTORY_MAX_MESSAGES` | `12` | Messages replayed to the model |
| `HISTORY_MAX_CHARS` | `6000` | Character cap on replayed history |
| `HISTORY_RETENTION_DAYS` | `7` | Stored messages older than this are deleted |
| `TYPESAFE_API_KEY` | empty | TypeSafe AI key for Jev. If empty, the app runs in read-only fallback mode |
| `TYPESAFE_BASE_URL` | `https://api.typesafe.ai` | TypeSafe API root |
| `JEV_MODEL` | `jev-latest` | Jev model. Aliases can move; pin e.g. `jev-1.13.0` in production |
| `JEV_TIMEOUT_SECONDS` | `10` | Per-request timeout |
| `JEV_DOMAIN_THRESHOLD` | `0.6` | Probability that a message needs a domain (email, ...) |
| `JEV_HIGH_RISK_THRESHOLD` | `0.3` | P(high risk) at or above this counts as high risk |
| `JEV_REFUSE_THRESHOLD` | `0.8` | Probability that a request is clearly harmful |
| `SCHEDULER_ENABLED` | `true` | Run the scheduler inside `python -m app.telegram` |
| `SCHEDULER_TICK_SECONDS` | `20` | How often due reminders are looked for |
| `REMINDER_MAX_PENDING` | `50` | Pending reminders per user |
| `REMINDER_LATE_HOURS` | `24` | A reminder missed by more than this is dropped |
| `REMINDER_MAX_DAYS_AHEAD` | `366` | Furthest a reminder can be set |
| `AGENDA_WINDOW_HOURS` | `2` | Agenda is sent only within this long after its time |
| `CALENDAR_POLL_SECONDS` | `120` | How often calendars are checked for agenda and meeting alerts |

## Roadmap

| Phase | Feature | Status |
| --- | --- | --- |
| 1-3 | Foundation, LLM layer (OpenAI, Ollama), `/agent/chat` | Done |
| 4 | Expenses | Done |
| 5-7 | Google login, Gmail (read), Calendar (read) | Done |
| 8 | Jev decision layer (TypeSafe) | Done |
| 9 | Approval system | Done |
| 10 | Safety hardening (DB timeouts, non-blocking event loop) | Done |
| 11 | Telegram bot | Done |
| 12 | Multi-user hardening (API tokens, single-use Google link, profiles, daily limits) | Done |
| 13 | Conversation history | Done |
| 14 | Calendar actions: create, move, delete, RSVP, free time (approval-gated) | Done |
| 15 | Reminders and scheduler (Telegram reminders, morning agenda, meeting alerts) | Done |
| 16 | Fitness: meals, workouts, weight, goals, progress | Done |
| 17 | Food photo analysis | Planned |
| 18 | Coach (goal-aware reminders and reviews) | Planned |
| 19 | Long-term memory | Planned |
