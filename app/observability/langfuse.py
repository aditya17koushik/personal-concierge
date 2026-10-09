from langfuse import get_client

from app.config import get_settings


settings = get_settings()


langfuse = get_client()