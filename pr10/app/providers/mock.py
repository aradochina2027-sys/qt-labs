
"""Підставний AI-провайдер для тестування без API."""

import os
import random
import time

from typing import Iterator

from dotenv import load_dotenv

from .base import Chunk, ProviderError, Request, Result
from .. import samples

load_dotenv()


class MockProvider:
    name = "mock"
    model = "mock-1"

    def stream(self, request: Request) -> Iterator[Chunk]:
        """Генерація тексту частинами з тестовими збоями."""

        kind = request.metadata.get(
            "failure_kind",
            os.getenv("MOCK_FAILURE_KIND", "unavailable"),
        )

        rate = float(request.metadata.get(
            "failure_rate",
            os.getenv("MOCK_FAILURE_RATE", "0"),
        ))

        rate = max(0.0, min(1.0, rate))
        should_fail = random.random() < rate

        first_delay = max(
            0.0,
            float(os.getenv("MOCK_FIRST_DELAY_MS", "600")) / 1000,
        )

        delay = max(
            0.0,
            float(os.getenv("MOCK_DELAY_MS", "40")) / 1000,
        )

        hang = max(
            0.0,
            float(os.getenv("MOCK_HANG_SECONDS", "30")),
        )

        if should_fail and kind in ("timeout", "idle_timeout"):
            time.sleep(hang)

            raise ProviderError(
                "timeout",
                "Час очікування провайдера вичерпано",
                retryable=True,
            )

        if should_fail and kind not in ("mid_stream", "idle_after_chunk"):
            failures = {
                "rate_limit": ("rate_limit", True),
                "auth": ("auth", False),
                "bad_request": ("bad_request", False),
                "unavailable": ("unavailable", True),
            }

            failure_kind, retryable = failures.get(
                kind, ("unavailable", True)
            )

            raise ProviderError(
                failure_kind,
                f"Тестовий збій: {failure_kind}",
                retryable=retryable,
                retry_after=(
                    1.0 if failure_kind == "rate_limit" else None
                ),
            )

        sample = samples.get(
            request.metadata.get("request_id", "")
        )

        if sample is None:
            answer = (
                "Добрий день! Будь ласка, уточніть "
                "інформацію щодо вашого звернення. "
                "З повагою, служба підтримки «Сузір'я»."
            )
        else:
            answer = sample["reply_example"]

        time.sleep(first_delay)

        pieces = [
            answer[i:i + 18]
            for i in range(0, len(answer), 18)
        ]

        for index, piece in enumerate(pieces):

            if index > 0:
                time.sleep(delay)

            if should_fail and kind == "mid_stream" and index >= 3:
                raise ProviderError(
                    "unavailable",
                    "Потік відповіді обірвався",
                    retryable=True,
                )

            if should_fail and kind == "idle_after_chunk" and index == 3:
                time.sleep(hang)

            yield Chunk(text=piece)

        yield Chunk(
            text="",
            usage={
                "completion_tokens": max(
                    1, len(answer) // 4
                ),
            },
            finish_reason="stop",
        )

    def complete(self, request: Request) -> Result:
        """Отримання повної відповіді."""

        started = time.monotonic()

        chunks = list(self.stream(request))

        text = "".join(
            chunk.text for chunk in chunks
        )

        usage = None
        finish_reason = None

        for chunk in chunks:
            if chunk.usage is not None:
                usage = chunk.usage

            if chunk.finish_reason is not None:
                finish_reason = chunk.finish_reason

        return Result(
            text=text,
            provider=self.name,
            model=self.model,
            usage=usage,
            finish_reason=finish_reason,
            elapsed=round(
                time.monotonic() - started, 3
            ),
        )
