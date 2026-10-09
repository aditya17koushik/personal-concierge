# ---------------------------------------------------------------------------
# OLLAMA PROVIDER (disabled by default)
#
# Ollama exposes an OpenAI-compatible API at http://localhost:11434/v1, so we
# reuse OpenAIProvider with a different base_url. No extra SDK needed.
#
# To enable:
#   1. Install Ollama and run:   ollama pull llama3.2:latest
#   2. Uncomment everything below
#   3. Uncomment the ollama branch in app/llm/factory.py
#   4. Set LLM_PROVIDER=ollama in .env
#
# Note: llama3.2 supports tool calling, but it is less reliable than OpenAI
# models -- expect occasional malformed tool arguments (handled gracefully by
# _parse_arguments in openai_provider.py).
# ---------------------------------------------------------------------------

# from app.llm.openai_provider import OpenAIProvider
#
#
# class OllamaProvider(OpenAIProvider):
#     name = "ollama"
#
#     def __init__(
#         self,
#         base_url: str = "http://localhost:11434/v1",
#         model: str = "llama3.2:latest",
#     ) -> None:
#         # Ollama ignores the API key, but the OpenAI SDK requires a non-empty one.
#         super().__init__(api_key="ollama", model=model, base_url=base_url)