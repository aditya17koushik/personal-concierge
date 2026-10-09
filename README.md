# Personal Agent

A personal AI agent backend built with FastAPI. It talks to an LLM (OpenAI by default, Ollama optional), and is being extended step by step with expense tracking, a decision layer (Jev), Google integrations, approvals, messaging channels, and memory.

## Tech stack

- **API:** FastAPI + Uvicorn
- **Database:** PostgreSQL 17, SQLAlchemy, Alembic
- **Validation / config:** Pydantic, pydantic-settings
- **LLM:** OpenAI (`gpt-4o-mini`), Ollama (`llama3.2:latest`, disabled by default)
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

## Roadmap

| Step | Feature | Status |
| --- | --- | --- |
| 1 | Repository foundation (FastAPI, PostgreSQL, SQLAlchemy, Alembic, Pydantic, config, logging) | Done |
| 2 | LLM abstraction (OpenAI, Ollama) | Done |
| 3 | `POST /agent/chat` | Next |
| 4 | Langfuse | Planned |
| 5 | Expense tools + database | Planned |
| 6 | Jev decision layer | Planned |
| 7 | Google OAuth | Planned |
| 8 | Gmail read/search | Planned |
| 9 | Google Calendar | Planned |
| 10 | Approval system | Planned |
| 11 | Hermes integration | Planned |
| 12 | Telegram | Planned |
| 13 | WhatsApp | Planned |
| 14 | Memory | Planned |
| 15 | Advanced personal planning | Planned |