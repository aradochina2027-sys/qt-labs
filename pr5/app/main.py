"""FastAPI-застосунок для пошуку."""

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import embeddings
from . import index
from . import keyword


app = FastAPI(
    title="Пошук у базі знань — ПР5"
)


# Шлях саме до HTML-сторінки
HTML_FILE = (
    Path(__file__).parent
    / "templates"
    / "index.html"
)


class SearchRequest(BaseModel):
    query: str
    mode: str = "both"
    top_k: int = index.DEFAULT_TOP_K
    filters: dict = Field(default_factory=dict)
    threshold: float | None = None


def hit_to_dict(hit):
    return {
        "score": hit.score,
        "text": hit.chunk.text,
        "source": hit.chunk.source,
        "metadata": hit.chunk.metadata
    }


@app.on_event("startup")
def startup():
    """Завантаження готового індексу."""

    app.state.index = None
    app.state.keyword_index = None

    try:
        app.state.index = index.load()

        app.state.keyword_index = keyword.build(
            app.state.index.chunks
        )

        print("Індекс успішно завантажено.")

    except Exception as error:
        print(
            "Індекс не завантажено:",
            error
        )


@app.get("/", response_class=HTMLResponse)
def home():
    """Головна HTML-сторінка."""

    if not HTML_FILE.exists():
        return HTMLResponse(
            content="<h1>Файл templates/index.html не знайдено</h1>",
            status_code=500
        )

    return HTMLResponse(
        content=HTML_FILE.read_text(
            encoding="utf-8"
        )
    )


@app.get("/api/status")
def status():
    """Інформація про стан індексу."""

    if app.state.index is None:
        return {
            "ready": False,
            "message": (
                "Індекс не збудовано. "
                "Виконайте python ingest.py"
            )
        }

    sources = {
        chunk.source
        for chunk in app.state.index.chunks
    }

    return {
        "ready": True,
        "documents": len(sources),
        "chunks": len(app.state.index.chunks),
        "model": app.state.index.model_name
    }


@app.post("/api/search")
def search(payload: SearchRequest):
    """Семантичний та ключовий пошук."""

    if app.state.index is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Індекс не завантажено. "
                "Виконайте python ingest.py"
            )
        )

    query = payload.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Введіть текст запиту."
        )

    if payload.mode not in (
        "semantic",
        "keyword",
        "both"
    ):
        raise HTTPException(
            status_code=400,
            detail="Невідомий режим пошуку."
        )

    if payload.top_k < 1 or payload.top_k > 20:
        raise HTTPException(
            status_code=400,
            detail="top_k має бути від 1 до 20."
        )

    result = {
        "query": query,
        "semantic": [],
        "keyword": [],
        "elapsed": {}
    }

    try:

        # СЕМАНТИЧНИЙ ПОШУК
        if payload.mode in (
            "semantic",
            "both"
        ):
            start = time.perf_counter()

            query_vector = embeddings.embed_query(
                query
            )

            semantic_hits = index.search(
                app.state.index,
                query_vector,
                top_k=payload.top_k,
                filters=payload.filters or None,
                threshold=payload.threshold
            )

            result["elapsed"]["semantic"] = (
                time.perf_counter() - start
            )

            result["semantic"] = [
                hit_to_dict(hit)
                for hit in semantic_hits
            ]

        # КЛЮЧОВИЙ ПОШУК
        if payload.mode in (
            "keyword",
            "both"
        ):
            start = time.perf_counter()

            keyword_hits = keyword.search(
                app.state.keyword_index,
                query,
                top_k=payload.top_k,
                filters=payload.filters or None
            )

            result["elapsed"]["keyword"] = (
                time.perf_counter() - start
            )

            result["keyword"] = [
                hit_to_dict(hit)
                for hit in keyword_hits
            ]

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    return result