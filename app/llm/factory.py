from functools import lru_cache

from app.config import get_settings
from app.llm.base import LLMError, LLMProvider
from app.llm.openai_provider import OpenAIProvider

# from app.llm.ollama_provider import OllamaProvider  # <- uncomment for Ollama


def _traced(provider: LLMProvider) -> LLMProvider:
    """Every model call becomes a Langfuse generation (only when tracing is configured)."""
    from app.llm.traced import TracedProvider
    from app.observability import get_tracer

    return TracedProvider(provider) if get_tracer().enabled else provider


@lru_cache
def get_llm_provider() -> LLMProvider:
    settings = get_settings()

    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        return _traced(
            OpenAIProvider(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
            )
        )

    # --- Ollama (uncomment together with app/llm/ollama_provider.py) ---
    # if settings.llm_provider == "ollama":
    #     return OllamaProvider(
    #         base_url=settings.ollama_base_url,
    #         model=settings.ollama_model,
    #     )

    raise LLMError(f"Unsupported LLM_PROVIDER: {settings.llm_provider!r}")