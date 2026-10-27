
"""Фонові задачі генерації AI-чернеток."""

import datetime
import logging
import threading
import time
import uuid

from dataclasses import dataclass, field
from typing import Iterator

from .providers.base import ProviderError

TERMINAL = {"done", "failed", "cancelled"}


@dataclass
class Draft:
    draft_id: str
    request_id: str
    client: str
    status: str = "queued"
    text: str = ""
    attempts: int = 0
    provider: str | None = None
    model: str | None = None
    fallback: bool = False
    error: dict | None = None
    usage: dict | None = None
    elapsed: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    events: list[dict] = field(default_factory=list)


_drafts: dict[str, Draft] = {}
_order: list[str] = []

_lock = threading.RLock()
_changed = threading.Condition(_lock)

_cancellations: dict[str, threading.Event] = {}


class DraftNotFound(Exception):
    pass


class DraftConflict(Exception):
    pass


def _now() -> str:
    return datetime.datetime.now().isoformat(
        timespec="seconds"
    )


def new_draft(request_id: str, client: str) -> Draft:
    draft = Draft(
        draft_id=uuid.uuid4().hex[:8],
        request_id=request_id,
        client=client,
        created_at=_now(),
        updated_at=_now(),
    )

    with _lock:
        _drafts[draft.draft_id] = draft
        _order.insert(0, draft.draft_id)

    return draft


def get(draft_id: str) -> Draft | None:
    with _lock:
        return _drafts.get(draft_id)


def list_drafts(limit: int = 50) -> list[Draft]:
    with _lock:
        return [
            _drafts[draft_id]
            for draft_id in _order[:limit]
        ]


def emit(
    draft: Draft,
    event: str,
    data: dict | None = None,
) -> None:
    with _changed:
        draft.events.append({
            "id": len(draft.events) + 1,
            "event": event,
            "data": data or {},
        })

        draft.updated_at = _now()
        _changed.notify_all()


def subscribe(
    draft_id: str,
    since: int = 0,
    heartbeat: float = 15.0,
) -> Iterator[dict | None]:

    since = max(0, since)

    while True:
        with _changed:
            draft = _drafts.get(draft_id)

            if draft is None:
                return

            pending = draft.events[since:]

            if not pending:
                if draft.status in TERMINAL:
                    return

                _changed.wait(timeout=heartbeat)
                pending = draft.events[since:]

        if not pending:
            yield None
            continue

        for event in pending:
            yield event

        since += len(pending)


def start(sample: dict, client: str) -> Draft:
    """Створити чернетку й запустити генерацію у фоні."""

    from . import drafting, gateway

    draft = new_draft(sample["id"], client)
    cancel_flag = threading.Event()

    with _lock:
        _cancellations[draft.draft_id] = cancel_flag

    def is_cancelled() -> bool:
        return cancel_flag.is_set()

    def status(data: dict) -> None:
        with _lock:
            if draft.status in TERMINAL:
                return

            if "status" in data:
                draft.status = data["status"]

            if "attempt" in data:
                draft.attempts = data["attempt"]

            if "attempts" in data:
                draft.attempts = data["attempts"]

            if "provider" in data:
                draft.provider = data["provider"]

            if "fallback" in data:
                draft.fallback = data["fallback"]

            if data.get("first_token_elapsed") is not None:
                draft.elapsed["first_token"] = data[
                    "first_token_elapsed"
                ]

        emit(draft, "status", data)

    def delta(part: str) -> None:
        with _lock:
            if draft.status in TERMINAL or is_cancelled():
                return

            draft.text += part

        emit(draft, "delta", {"text": part})

    def worker() -> None:
        started = time.monotonic()

        try:
            if is_cancelled():
                raise ProviderError(
                    "cancelled",
                    "Генерацію скасовано",
                )

            status({
                "status": "running",
                "detail": "Початок генерації",
            })

            request = drafting.build_request(sample)

            result = gateway.generate(
                request,
                delta,
                status,
                is_cancelled,
            )

            with _lock:
                if is_cancelled():
                    raise ProviderError(
                        "cancelled",
                        "Генерацію скасовано",
                    )

                draft.status = "done"
                draft.text = result.text
                draft.provider = result.provider
                draft.model = result.model
                draft.fallback = (
                    "fallback" in result.provider
                )
                draft.usage = result.usage

                draft.elapsed["total"] = round(
                    time.monotonic() - started, 3
                )

                draft.warnings = drafting.check(
                    draft.text, sample
                )

                data = {
                    "text": draft.text,
                    "provider": draft.provider,
                    "model": draft.model,
                    "usage": draft.usage,
                    "elapsed": draft.elapsed.copy(),
                    "attempts": draft.attempts,
                    "fallback": draft.fallback,
                    "warnings": draft.warnings,
                }

                emit(draft, "done", data)

        except Exception as exc:
            if isinstance(exc, ProviderError):
                error = exc
            else:
                logging.exception(
                    "Unexpected draft failure: %s",
                    draft.draft_id,
                )

                error = ProviderError(
                    "unknown",
                    "Внутрішня помилка генерації",
                )

            with _lock:
                draft.elapsed["total"] = round(
                    time.monotonic() - started, 3
                )

                if is_cancelled() or error.kind == "cancelled":
                    draft.status = "cancelled"
                    event = "cancelled"
                    data = {}
                else:
                    draft.status = "failed"

                    draft.error = {
                        "kind": error.kind,
                        "message": error.message,
                        "retryable": error.retryable,
                        "request_id": draft.draft_id,
                    }

                    event = "error"
                    data = draft.error.copy()

                emit(draft, event, data)

        finally:
            with _lock:
                _cancellations.pop(draft.draft_id, None)

    thread = threading.Thread(
        target=worker,
        daemon=True,
        name=f"draft-{draft.draft_id}",
    )

    thread.start()
    return draft


def cancel(draft_id: str) -> Draft:
    """Попросити фонову задачу зупинитися."""

    with _lock:
        draft = _drafts.get(draft_id)

        if draft is None:
            raise DraftNotFound(draft_id)

        if draft.status in TERMINAL:
            raise DraftConflict(draft_id)

        flag = _cancellations.get(draft_id)

        if flag is not None:
            flag.set()

        return draft
