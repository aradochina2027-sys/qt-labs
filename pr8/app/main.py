"""Веб-рівень застосунку: сторінка помічника і JSON-ендпоінти.

Цей файл не знає ані які інструменти є в моделі, ані як перевіряються
аргументи, ані якою моделлю й за якою інструкцією отримано відповідь —
усе це в `app/tools.py`, `app/llm.py` і поєднується в
`app/assistant.py`. Тут вирішується інше: що застосунок приймає від
сторінки, що віддає їй і з яким HTTP-статусом.

Запуск із папки pr8:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from shop import service

from . import assistant, tools

app = FastAPI(title="Помічник клієнта — ПР8")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class AskRequest(BaseModel):
    """Те, що надсилає сторінка.

    `customer_id` — клієнт, обраний на сторінці. Тут він заміняє вхід у
    кабінет: у справжньому застосунку веб-рівень узяв би його із сесії
    після автентифікації, а не з тіла запиту.
    """

    customer_id: str
    question: str


def answer_to_dict(result: assistant.Answer) -> dict:
    """Перетворити результат циклу на те, що піде на сторінку."""
    return {
        "answer": result.text,
        "calls": [asdict(call) for call in result.calls],
        "rounds": result.rounds,
        "stopped": result.stopped,
        "model": result.model,
        "elapsed": result.elapsed,
        "usage": result.usage,
    }


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Віддати сторінку помічника."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/customers")
def api_customers() -> list[dict]:
    """Клієнти, від імені яких можна «увійти» на сторінці."""
    return service.list_customers()


@app.get("/api/tools")
def api_tools() -> list[dict]:
    """Описи інструментів у тому вигляді, в якому їх бачить модель."""
    return tools.specs()


@app.post("/api/reset")
def api_reset() -> dict:
    """Повернути дані магазину до початкового стану: створені повернення
    й скасування зникають. Зручно між прогонами перевірки."""
    service.reset()
    return {"ok": True}


@app.post("/api/ask")
def api_ask(payload: AskRequest) -> dict:
    """Відповісти на питання клієнта й повернути журнал викликів."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Питання не може бути порожнім.")

    known = {c["customer_id"] for c in service.list_customers()}
    if payload.customer_id not in known:
        raise HTTPException(status_code=404, detail="Невідомий клієнт.")

    try:
        result = assistant.answer(question, payload.customer_id)
    except assistant.llm.LLMError as exc:
        # Деталі провайдера/traceback не показуємо клієнтові.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Внутрішня помилка застосунку.") from exc
    return answer_to_dict(result)
