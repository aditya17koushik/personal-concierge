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
| `/google` | linked users | Single-use Google login link |
| `/profile`, `/timezone`, `/currency`, `/units` | linked users | View or change your profile |
| `/invite`, `/users`, `/remove <id>` | admin | Manage who can use the bot |

Approve and Reject are buttons only. They are never tools the model can call, so text inside an
email or invite can't approve anything. Private chats only; groups are ignored.

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
| `TYPESAFE_API_KEY` | empty | TypeSafe AI key for Jev. If empty, the app runs in read-only fallback mode |
| `TYPESAFE_BASE_URL` | `https://api.typesafe.ai` | TypeSafe API root |
| `JEV_MODEL` | `jev-latest` | Jev model. Aliases can move; pin e.g. `jev-1.13.0` in production |
| `JEV_TIMEOUT_SECONDS` | `10` | Per-request timeout |
| `JEV_DOMAIN_THRESHOLD` | `0.6` | Probability that a message needs a domain (email, ...) |
| `JEV_HIGH_RISK_THRESHOLD` | `0.3` | P(high risk) at or above this counts as high risk |
| `JEV_REFUSE_THRESHOLD` | `0.8` | Probability that a request is clearly harmful |

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
| 13 | Conversation history | Next |
| 14 | Calendar actions: create, move, delete, RSVP, free time (approval-gated) | Planned |
| 15 | Reminders and scheduler (Telegram reminders, morning agenda) | Planned |
| 16 | Fitness: meals, workouts, weight, goals, progress | Planned |
| 17 | Food photo analysis | Planned |
| 18 | Coach (goal-aware reminders and reviews) | Planned |
| 19 | Long-term memory | Planned |

Dropped: Hermes, WhatsApp, MCP. Later: Langfuse, 24/7 deployment, sending email as an
approval-gated tool.