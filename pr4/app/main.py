from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import llm


app = FastAPI(
    title="Помічник служби підтримки — ПР4"
)


INDEX_PAGE = (
    Path(__file__).parent
    / "templates"
    / "index.html"
)

CONTEXT_FILE = (
    Path(__file__).parent.parent
    / "context.md"
)


class Turn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Turn] = Field(default_factory=list)


def load_context() -> str:
    try:
        return CONTEXT_FILE.read_text(
            encoding="utf-8"
        )
    except OSError as exc:
        raise RuntimeError(
            "Не вдалося прочитати context.md"
        ) from exc


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_PAGE.read_text(
        encoding="utf-8"
    )


@app.post("/api/chat")
def api_chat(payload: ChatRequest):

    if not payload.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Введіть повідомлення."
        )

    history = [
        turn.model_dump()
        for turn in payload.history
    ]

    try:
        return llm.ask(
            payload.message,
            history,
            load_context()
        )

    except llm.LLMError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc)
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Внутрішня помилка сервера."
        ) from exc