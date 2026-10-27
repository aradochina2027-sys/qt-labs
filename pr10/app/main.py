
"""FastAPI: сторінка, чернетки, SSE та контрольовані помилки."""

import json
import logging
import uuid

from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import gateway, jobs, limits, samples


app = FastAPI(title="Помічник оператора — ПР10")

INDEX_PAGE = (
    Path(__file__).parent / "templates" / "index.html"
)


class DraftIn(BaseModel):
    request_id: str
    client: str = "anonymous"


def draft_to_dict(draft: jobs.Draft) -> dict:
    data = asdict(draft)
    data["events"] = len(draft.events)
    return data


def api_error(
    status_code: int,
    kind: str,
    message: str,
    retryable: bool = False,
    retry_after: int | None = None,
) -> JSONResponse:
    request_id = uuid.uuid4().hex[:12]

    headers = {}

    if retry_after is not None:
        headers["Retry-After"] = str(
            max(1, int(retry_after))
        )

    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content={
            "error": {
                "kind": kind,
                "message": message,
                "retryable": retryable,
                "request_id": request_id,
            }
        },
    )


@app.get("/", response_class=HTMLResponse)
def page():
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/health")
def api_health():
    return {
        "gateway": gateway.settings(),
        "limits": limits.settings(),
    }


@app.get("/api/requests")
def api_requests():
    return [
        samples.public(sample)
        for sample in samples.all_samples()
    ]


@app.post("/api/drafts", status_code=202)
def api_create(payload: DraftIn):
    sample = samples.get(payload.request_id)

    if sample is None:
        return api_error(
            400,
            "bad_request",
            "Невідоме звернення клієнта.",
        )

    try:
        limits.acquire(payload.client)

    except limits.RateLimited as exc:
        return api_error(
            429,
            "rate_limit",
            "Перевищено ліміт створення чернеток.",
            retryable=True,
            retry_after=exc.retry_after,
        )

    try:
        draft = jobs.start(sample, payload.client)
        return draft_to_dict(draft)

    except Exception:
        logging.exception(
            "Не вдалося запустити генерацію"
        )

        return api_error(
            503,
            "unavailable",
            "Не вдалося запустити генерацію. "
            "Спробуйте пізніше.",
            retryable=True,
        )


@app.get("/api/drafts")
def api_drafts():
    return [
        draft_to_dict(d)
        for d in jobs.list_drafts()
    ]


@app.get("/api/drafts/{draft_id}")
def api_draft(draft_id: str):
    draft = jobs.get(draft_id)

    if draft is None:
        return api_error(
            404,
            "not_found",
            "Чернетку не знайдено.",
        )

    return draft_to_dict(draft)


@app.get("/api/drafts/{draft_id}/events")
def api_events(draft_id: str, request: Request):
    draft = jobs.get(draft_id)

    if draft is None:
        return api_error(
            404,
            "not_found",
            "Чернетку не знайдено.",
        )

    try:
        since = max(
            0,
            int(
                request.headers.get("last-event-id")
                or 0
            ),
        )
    except ValueError:
        since = 0

    def stream() -> Iterator[str]:
        for event in jobs.subscribe(draft_id, since):
            if event is None:
                yield ": ping\n\n"
                continue

            data = json.dumps(
                event["data"],
                ensure_ascii=False,
            )

            yield (
                f"id: {event['id']}\n"
                f"event: {event['event']}\n"
                f"data: {data}\n\n"
            )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/drafts/{draft_id}/cancel")
def api_cancel(draft_id: str):
    try:
        draft = jobs.cancel(draft_id)
        return draft_to_dict(draft)

    except jobs.DraftNotFound:
        return api_error(
            404,
            "not_found",
            "Чернетку не знайдено.",
        )

    except jobs.DraftConflict:
        return api_error(
            409,
            "conflict",
            "Генерація вже завершена. "
            "Скасувати її неможливо.",
        )

    except Exception:
        logging.exception(
            "Помилка скасування чернетки %s",
            draft_id,
        )

        return api_error(
            503,
            "unavailable",
            "Не вдалося скасувати генерацію.",
            retryable=True,
        )
