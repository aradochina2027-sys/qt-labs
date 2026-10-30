
"""Фонові генерації чернеток, SSE, скасування та журналювання."""

import contextvars
import datetime
import logging
import threading
import time
import uuid

from dataclasses import dataclass, field
from typing import Iterator

from . import drafting, gateway, telemetry
from .providers.base import ProviderError


TERMINAL = {"done", "failed", "cancelled"}

_log = logging.getLogger("app.jobs")


@dataclass
class Draft:
    draft_id: str
    request_id: str
    client: str

    http_request_id: str = ""
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

    cached: bool = False
    cost: float | None = None
    cost_estimated: bool = False
    fresh: bool = False

    created_at: str = ""
    updated_at: str = ""

    events: list[dict] = field(default_factory=list)
    cancelled_flag: bool = field(default=False, repr=False)


_drafts: dict[str, Draft] = {}
_order: list[str] = []

_lock = threading.Lock()
_changed = threading.Condition(_lock)


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(
        timespec="seconds"
    )


def new_draft(request_id: str, client: str) -> Draft:
    draft = Draft(
        draft_id=uuid.uuid4().hex[:8],
        request_id=request_id,
        client=client,
        http_request_id=telemetry.request_id_var.get() or "",
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
        draft.events.append(
            {
                "id": len(draft.events) + 1,
                "event": event,
                "data": data or {},
            }
        )

        draft.updated_at = _now()
        _changed.notify_all()


def subscribe(
    draft_id: str,
    since: int = 0,
    heartbeat: float = 15,
) -> Iterator[dict | None]:

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

        for item in pending:
            yield item

        since += len(pending)


def _run(draft: Draft, sample: dict) -> None:
    started = time.perf_counter()

    common = {
        "request_id": draft.http_request_id,
        "draft_id": draft.draft_id,
        "sample_id": draft.request_id,
        "client": draft.client,
    }

    with _lock:
        draft.status = "running"

    emit(draft, "status", {"status": "running"})

    try:
        request = drafting.build_request(sample)

        request.metadata.update(
            {
                **common,
                "http_request_id": draft.http_request_id,
                "fresh": draft.fresh,
            }
        )

        def delta(piece: str) -> None:
            with _lock:
                draft.text += piece

            emit(draft, "delta", {"text": piece})

        def status(changes: dict) -> None:
            with _lock:
                draft.attempts = int(
                    changes.get("attempt", draft.attempts)
                )

                draft.provider = changes.get(
                    "provider",
                    draft.provider,
                )

                draft.fallback = bool(
                    changes.get("fallback", draft.fallback)
                )

            emit(
                draft,
                "status",
                {"status": "running", **changes},
            )

        result = gateway.generate(
            request,
            delta,
            status,
            lambda: draft.cancelled_flag,
        )

        elapsed_total = round(
            time.perf_counter() - started,
            3,
        )

        with _lock:
            if draft.cancelled_flag:
                raise ProviderError(
                    "cancelled",
                    "Генерацію скасовано.",
                )

            draft.status = "done"
            draft.text = result.text

            draft.attempts = int(
                request.metadata.get(
                    "attempts",
                    draft.attempts,
                )
            )

            draft.provider = result.provider
            draft.model = result.model
            draft.usage = result.usage

            draft.cached = result.cached
            draft.cost = result.cost
            draft.cost_estimated = result.cost_estimated

            draft.elapsed = {
                "first_token": (
                    0.0
                    if result.cached
                    else request.metadata.get("first_token_s")
                ),
                "total": elapsed_total,
            }

            draft.warnings = drafting.check(
                result.text,
                sample,
            )

        event = {
            "text": draft.text,
            "provider": draft.provider,
            "model": draft.model,
            "attempts": draft.attempts,
            "fallback": draft.fallback,
            "usage": draft.usage,
            "elapsed": draft.elapsed,
            "warnings": draft.warnings,
            "cached": draft.cached,
            "cost": draft.cost,
            "cost_estimated": draft.cost_estimated,
        }

        emit(draft, "done", event)

        _log.info(
            "draft.done",
            extra={
                **common,
                "attempts": draft.attempts,
                "provider": draft.provider,
                "model": draft.model,
                "cached": draft.cached,
                "fallback": draft.fallback,
                "cost": draft.cost,
                "duration_ms": round(
                    elapsed_total * 1000,
                    2,
                ),
            },
        )

    except Exception as exc:
        if isinstance(exc, ProviderError):
            error = exc
        else:
            _log.exception(
                "draft.internal",
                extra=common,
            )

            error = ProviderError(
                "unknown",
                "Не вдалося створити чернетку.",
            )

        cancelled = (
            draft.cancelled_flag
            or error.kind == "cancelled"
        )

        with _lock:
            draft.status = (
                "cancelled"
                if cancelled
                else "failed"
            )

            draft.error = {
                "kind": (
                    "cancelled"
                    if cancelled
                    else error.kind
                ),
                "message": error.message,
                "retryable": error.retryable,
                "request_id": draft.http_request_id,
            }

        emit(
            draft,
            "cancelled" if cancelled else "error",
            {} if cancelled else draft.error,
        )

        _log.log(
            logging.INFO if cancelled else logging.ERROR,
            "draft.cancelled" if cancelled else "draft.failed",
            extra={
                **common,
                "kind": draft.error["kind"],
                "attempts": draft.attempts,
                "duration_ms": round(
                    (time.perf_counter() - started) * 1000,
                    2,
                ),
            },
        )


def start(
    sample: dict,
    client: str,
    fresh: bool = False,
) -> Draft:
    """Створити чернетку та запустити фонову генерацію."""

    draft = new_draft(
        sample["id"],
        client,
    )

    draft.fresh = fresh

    _log.info(
        "draft.created",
        extra={
            "request_id": draft.http_request_id,
            "draft_id": draft.draft_id,
            "sample_id": draft.request_id,
            "client": draft.client,
            "fresh": fresh,
        },
    )

    ctx = contextvars.copy_context()

    thread = threading.Thread(
        target=ctx.run,
        args=(_run, draft, sample),
        name=f"draft-{draft.draft_id}",
        daemon=True,
    )

    thread.start()

    return draft


def cancel(draft_id: str) -> Draft:
    """Позначити чернетку для скасування."""

    with _lock:
        draft = _drafts.get(draft_id)

        if draft is None:
            raise KeyError("Чернетки не знайдено.")

        if draft.status in TERMINAL:
            raise ValueError("Чернетка вже завершена.")

        draft.cancelled_flag = True

    return draft
