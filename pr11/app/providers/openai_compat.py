
"""Провайдер для OpenAI-сумісного API, зокрема Gemini."""

import logging
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Iterator

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
)

from .base import Chunk, ProviderError, Request, Result


class OpenAICompatProvider:
    """Адаптер для OpenAI-сумісного API."""

    name = "openai"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
    ) -> None:
        self.model = model

        self.client = OpenAI(
            base_url=base_url or None,
            api_key=api_key,
            timeout=timeout,
            max_retries=0,
        )

    @staticmethod
    def _retry_after(exc: APIStatusError) -> float | None:
        value = exc.response.headers.get("retry-after")

        if not value:
            return None

        try:
            return max(0.0, float(value))
        except ValueError:
            pass

        try:
            date = parsedate_to_datetime(value)
            now = datetime.now(timezone.utc)

            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)

            return max(0.0, (date - now).total_seconds())

        except (TypeError, ValueError, OverflowError):
            return None

    @classmethod
    def _error(cls, exc: Exception) -> ProviderError:
        """Перетворює помилки SDK на єдиний формат."""

        if isinstance(exc, APITimeoutError):
            return ProviderError(
                "timeout",
                "Час очікування AI-провайдера вичерпано.",
                retryable=True,
            )

        if isinstance(exc, APIConnectionError):
            return ProviderError(
                "unavailable",
                "Немає з'єднання з AI-провайдером.",
                retryable=True,
            )

        if isinstance(exc, APIStatusError):
            code = exc.status_code

            if code == 429:
                kind = "rate_limit"
                message = (
                    "Провайдер тимчасово обмежив запити."
                )
                retryable = True

            elif code in (401, 403):
                kind = "auth"
                message = (
                    "Помилка доступу до AI-провайдера."
                )
                retryable = False

            elif code in (400, 404, 422):
                kind = "bad_request"
                message = (
                    "Провайдер відхилив запит."
                )
                retryable = False

            elif code in (408, 504):
                kind = "timeout"
                message = (
                    "Провайдер не відповів вчасно."
                )
                retryable = True

            elif code == 409 or code >= 500:
                kind = "unavailable"
                message = (
                    "AI-провайдер тимчасово недоступний."
                )
                retryable = True

            else:
                kind = "unknown"
                message = (
                    "Не вдалося отримати відповідь."
                )
                retryable = False

            return ProviderError(
                kind,
                message,
                retryable=retryable,
                retry_after=cls._retry_after(exc),
                detail=f"HTTP {code}",
            )

        logging.error(
            "Невідома помилка SDK: %s",
            type(exc).__name__,
        )

        return ProviderError(
            "unknown",
            "Непередбачена помилка AI-провайдера.",
            retryable=False,
            detail=type(exc).__name__,
        )

    def complete(self, request: Request) -> Result:
        """Отримує повну відповідь моделі."""

        started = time.monotonic()

        try:
            response = self.client.chat.completions.create(
                model=request.model or self.model,
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            choice = response.choices[0]

            return Result(
                text=choice.message.content or "",
                provider=self.name,
                model=response.model,
                usage=(
                    response.usage.model_dump()
                    if response.usage is not None
                    else None
                ),
                finish_reason=choice.finish_reason,
                elapsed=round(
                    time.monotonic() - started,
                    3,
                ),
            )

        except Exception as exc:
            raise self._error(exc) from None

    def stream(self, request: Request) -> Iterator[Chunk]:
        """
        Передає відповідь частинами.

        Важливо: include_usage=True запитує фактичну
        кількість токенів у кінці потоку.
        """

        response = None

        try:
            response = self.client.chat.completions.create(
                model=request.model or self.model,
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True,
                stream_options={
                    "include_usage": True,
                },
            )

            for item in response:
                usage = None

                if item.usage is not None:
                    usage = item.usage.model_dump()

                # Останній фрагмент може містити usage,
                # але не містити choices.
                # Його не можна пропускати.
                if not item.choices:
                    if usage is not None:
                        yield Chunk(
                            text="",
                            usage=usage,
                        )
                    continue

                choice = item.choices[0]

                yield Chunk(
                    text=choice.delta.content or "",
                    usage=usage,
                    finish_reason=choice.finish_reason,
                )

        except GeneratorExit:
            raise

        except Exception as exc:
            raise self._error(exc) from None

        finally:
            if response is not None:
                response.close()
