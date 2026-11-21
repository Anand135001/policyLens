from google import genai
from google.genai import types

from app.core.config import get_settings


settings = get_settings()

if settings.gemini_api_key is None:
    raise RuntimeError("GEMINI_API_KEY is not configured")

client = genai.Client(
    api_key=settings.gemini_api_key.get_secret_value()
)


def generate_embedding(text: str, title: str | None = None) -> list[float]:
    formatted_text = f"title: {title or 'none'} | text: {text}"

    response = client.models.embed_content(
        model=settings.embedding_model,
        contents=formatted_text,
        config=types.EmbedContentConfig(
            output_dimensionality=settings.embedding_dimensions
        ),
    )

    embedding = response.embeddings[0].values

    if len(embedding) != settings.embedding_dimensions:
        raise ValueError("Unexpected embedding dimensions")

    return embedding


def generate_query_embedding(query: str) -> list[float]:
    formatted_query = f"task: search result | query: {query}"

    response = client.models.embed_content(
        model=settings.embedding_model,
        contents=formatted_query,
        config=types.EmbedContentConfig(
            output_dimensionality=settings.embedding_dimensions
        ),
    )

    embedding = response.embeddings[0].values

    if len(embedding) != settings.embedding_dimensions:
        raise ValueError("Unexpected embedding dimensions")

    return embedding
