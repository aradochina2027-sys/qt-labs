
"""Шлюз AI: таймаути, повтори, обмеження та fallback."""

import os
import queue
import random
import threading
import time
import logging

from typing import Callable

from dotenv import load_dotenv
from .providers.base import Provider, ProviderError, Request, Result

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "mock")
FALLBACK = os.getenv("LLM_FALLBACK", "none")

TIMEOUT = float(os.getenv("LLM_TIMEOUT", "20"))
IDLE_TIMEOUT = float(os.getenv("LLM_IDLE_TIMEOUT", "8"))
TOTAL_TIMEOUT = float(os.getenv("LLM_TOTAL_TIMEOUT", "45"))

MAX_ATTEMPTS = max(1, int(os.getenv("LLM_MAX_ATTEMPTS", "3")))
BACKOFF_BASE = float(os.getenv("LLM_BACKOFF_BASE", "0.5"))
BACKOFF_MAX = float(os.getenv("LLM_BACKOFF_MAX", "8"))
MAX_CONCURRENT = max(1, int(os.getenv("LLM_MAX_CONCURRENT", "2")))

MODEL = os.getenv("LLM_MODEL") or None
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "600"))

_semaphore = threading.BoundedSemaphore(MAX_CONCURRENT)


def settings() -> dict:
    return {
        "provider": PROVIDER,
        "fallback": FALLBACK,
        "model": MODEL,
        "timeout": TIMEOUT,
        "idle_timeout": IDLE_TIMEOUT,
        "total_timeout": TOTAL_TIMEOUT,
        "max_attempts": MAX_ATTEMPTS,
        "max_concurrent": MAX_CONCURRENT,
    }


def make_provider(name: str) -> Provider:
    if name == "mock":
        from .providers.mock import MockProvider
        return MockProvider()

    from .providers.openai_compat import OpenAICompatProvider

    key = os.getenv("LLM_API_KEY", "").strip()

    if not key:
        raise ProviderError(
            "auth",
            "Ключ API не налаштовано",
            retryable=False,
        )

    model = MODEL if name == "openai" else name

    if not model:
        raise ProviderError(
            "bad_request",
            "Не вказано модель у налаштуваннях",
            retryable=False,
        )

    return OpenAICompatProvider(
        os.getenv("LLM_BASE_URL", ""),
        key,
        model,
        TIMEOUT,
    )


def _timed_stream(
    provider,
    request,
    on_delta,
    on_status,
    is_cancelled,
    attempt_deadline,
    total_deadline,
    began,
):
    """Читає потік у допоміжному потоці з контролем часу."""

    events = queue.Queue(maxsize=100)
    stop = threading.Event()

    def produce():
        stream = None

        try:
            stream = provider.stream(request)

            for chunk in stream:
                if stop.is_set():
                    break

                while not stop.is_set():
                    try:
                        events.put(("chunk", chunk), timeout=0.1)
                        break
                    except queue.Full:
                        continue

            if not stop.is_set():
                events.put(("end", None), timeout=1)

        except Exception as exc:
            if not stop.is_set():
                try:
                    events.put(("error", exc), timeout=1)
                except queue.Full:
                    pass

        finally:
            if stream is not None:
                close = getattr(stream, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        logging.exception("Помилка закриття потоку")

    worker = threading.Thread(
        target=produce,
        daemon=True,
        name="llm-stream",
    )
    worker.start()

    parts = []
    usage = None
    finish_reason = None
    first_token = None
    last_activity = time.monotonic()

    try:
        while True:
            if is_cancelled():
                raise ProviderError(
                    "cancelled", "Генерацію скасовано"
                )

            now = time.monotonic()

            if now >= total_deadline:
                raise ProviderError(
                    "timeout",
                    "Загальний час генерації вичерпано",
                    retryable=True,
                )

            if now >= attempt_deadline:
                raise ProviderError(
                    "timeout",
                    "Перевищено час однієї спроби",
                    retryable=True,
                )

            if now - last_activity >= IDLE_TIMEOUT:
                raise ProviderError(
                    "timeout",
                    "Потік не надсилає фрагментів",
                    retryable=True,
                )

            wait = min(
                0.1,
                attempt_deadline - now,
                total_deadline - now,
                IDLE_TIMEOUT - (now - last_activity),
            )

            try:
                event, payload = events.get(
                    timeout=max(0.001, wait)
                )
            except queue.Empty:
                continue

            if event == "error":
                if isinstance(payload, ProviderError):
                    raise payload

                logging.error(
                    "Невідомий збій провайдера: %s",
                    type(payload).__name__,
                )
                raise ProviderError(
                    "unknown",
                    "Непередбачена помилка генерації",
                    retryable=False,
                )

            if event == "end":
                break

            last_activity = time.monotonic()
            chunk = payload

            if chunk.text:
                if first_token is None:
                    first_token = last_activity

                    on_status({
                        "first_token_elapsed": round(
                            first_token - began, 3
                        ),
                    })

                parts.append(chunk.text)
                on_delta(chunk.text)

            if chunk.usage is not None:
                usage = chunk.usage

            if chunk.finish_reason is not None:
                finish_reason = chunk.finish_reason

        return (
            "".join(parts),
            usage,
            finish_reason,
            first_token,
        )

    finally:
        stop.set()


def generate(
    request: Request,
    on_delta: Callable[[str], None],
    on_status: Callable[[dict], None],
    is_cancelled: Callable[[], bool],
) -> Result:
    """Головна функція генерації."""

    began = time.monotonic()
    total_deadline = began + TOTAL_TIMEOUT

    last_error = ProviderError(
        "unavailable",
        "Сервіс моделі недоступний",
        retryable=True,
    )

    candidates = [PROVIDER]

    if FALLBACK not in ("none", "", PROVIDER):
        candidates.append(FALLBACK)

    if not _semaphore.acquire(
        timeout=max(0, min(2, TOTAL_TIMEOUT))
    ):
        raise ProviderError(
            "rate_limit",
            "Усі місця генерації зайняті",
            retryable=True,
            retry_after=2,
        )

    try:
        attempts = 0

        for index, provider_name in enumerate(candidates):
            fallback = index > 0
            max_tries = 1 if fallback else MAX_ATTEMPTS

            if fallback:
                on_status({
                    "fallback": True,
                    "detail": "Перехід до запасного провайдера",
                })

            for attempt in range(max_tries):
                if is_cancelled():
                    raise ProviderError(
                        "cancelled", "Генерацію скасовано"
                    )

                now = time.monotonic()

                if now >= total_deadline:
                    raise ProviderError(
                        "timeout",
                        "Вичерпано загальний бюджет часу",
                        retryable=True,
                    )

                attempts += 1

                on_status({
                    "status": "running",
                    "attempt": attempts,
                    "provider": provider_name,
                    "fallback": fallback,
                    "detail": (
                        f"Спроба {attempt + 1} із {max_tries}"
                    ),
                })

                delivered = []

                try:
                    provider = make_provider(provider_name)

                    attempt_deadline = min(
                        total_deadline,
                        time.monotonic() + TIMEOUT,
                    )

                    def send_delta(part):
                        delivered.append(part)
                        on_delta(part)

                    text, usage, finish, first = _timed_stream(
                        provider,
                        request,
                        send_delta,
                        on_status,
                        is_cancelled,
                        attempt_deadline,
                        total_deadline,
                        began,
                    )

                    on_status({
                        "attempts": attempts,
                        "first_token_elapsed": (
                            round(first - began, 3)
                            if first is not None else None
                        ),
                    })

                    return Result(
                        text=text,
                        provider=(
                            f"{provider.name} (fallback)"
                            if fallback else provider.name
                        ),
                        model=getattr(
                            provider, "model", provider_name
                        ),
                        usage=usage,
                        finish_reason=finish,
                        elapsed=round(
                            time.monotonic() - began, 3
                        ),
                    )

                except ProviderError as exc:
                    last_error = exc

                    logging.warning(
                        "Спроба %s, провайдер %s: %s",
                        attempts,
                        provider_name,
                        exc.kind,
                    )

                    on_status({
                        "attempts": attempts,
                        "detail": f"Збій: {exc.kind}",
                    })

                    if exc.kind == "cancelled":
                        raise

                    # Після показу частини тексту
                    # не починаємо відповідь повторно.
                    if delivered:
                        raise

                    if not exc.retryable:
                        break

                    if attempt + 1 >= max_tries:
                        break

                    pause = (
                        exc.retry_after
                        if exc.retry_after is not None
                        else min(
                            BACKOFF_MAX,
                            BACKOFF_BASE * (2 ** attempt),
                        ) * random.uniform(0.5, 1.5)
                    )

                    if (
                        time.monotonic() + pause
                        >= total_deadline
                    ):
                        break

                    on_status({
                        "detail": (
                            f"Очікування {pause:.2f} с перед повтором"
                        ),
                    })

                    end_pause = time.monotonic() + pause

                    while time.monotonic() < end_pause:
                        if is_cancelled():
                            raise ProviderError(
                                "cancelled",
                                "Генерацію скасовано",
                            )

                        time.sleep(
                            min(
                                0.05,
                                max(
                                    0,
                                    end_pause - time.monotonic(),
                                ),
                            )
                        )

            if time.monotonic() >= total_deadline:
                break

        raise last_error

    finally:
        _semaphore.release()
