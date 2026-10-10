from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    app_name: str = "personal-agent"
    app_env: str = "development"
    debug: bool = True

    # Database
    database_url: str

    # LLM
    # "ollama" is also accepted here, but its code is commented out in
    # app/llm/ollama_provider.py and app/llm/factory.py until you enable it.
    llm_provider: Literal["openai", "ollama"] = "openai"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Ollama
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.2:latest"

    # Expenses
    default_currency: str = "INR"

    # TypeSafe AI (Jev decision model)
    typesafe_api_key: str = ""
    typesafe_base_url: str = "https://api.typesafe.ai"
    jev_model: str = "jev-latest"  # pin e.g. "jev-1.13.0" in production
    jev_timeout_seconds: float = 10.0
    # Probability thresholds (Jev returns calibrated probabilities)
    jev_domain_threshold: float = 0.6  # message needs this domain (email, ...)
    jev_high_risk_threshold: float = 0.3  # P(high risk) at/above this => high risk
    jev_refuse_threshold: float = 0.8  # clearly harmful

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"

    # Telegram
    telegram_bot_token: str = ""

    # Google
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = ""

    # Security
    app_secret_key: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]