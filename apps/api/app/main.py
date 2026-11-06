from fastapi import FastAPI
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="RAG system for evolving government documents",
    version=settings.app_version,
    debug=settings.debug,
)


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "service": "policylens-api",
    }
