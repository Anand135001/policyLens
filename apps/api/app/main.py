from fastapi import FastAPI


app = FastAPI(
    title="PolicyLens API",
    description="RAG system for evolving government documents",
    version="0.1.0",
)


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "service": "policylens-api",
    }
