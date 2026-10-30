
"""Обмеження частоти запитів від клієнтів."""

import math
import os
import threading
import time

from collections import defaultdict, deque

from dotenv import load_dotenv

load_dotenv()

PER_MINUTE = max(
    1,
    int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
)

WINDOW = 60

_records = defaultdict(deque)
_lock = threading.Lock()


class RateLimited(Exception):
    """Перевищено кількість дозволених запитів."""

    def __init__(self, retry_after: float):
        self.retry_after = retry_after

        super().__init__(
            "Перевищено ліміт запитів"
        )


def settings() -> dict:
    return {
        "per_minute": PER_MINUTE,
        "window_seconds": WINDOW,
    }


def acquire(client: str) -> None:
    """Перевірити дозвіл на створення чернетки."""

    client = client or "anonymous"
    now = time.monotonic()

    with _lock:
        requests = _records[client]

        # Прибираємо старі запити.
        while requests and now - requests[0] >= WINDOW:
            requests.popleft()

        # Перевіряємо кількість запитів.
        if len(requests) >= PER_MINUTE:
            retry_after = math.ceil(
                WINDOW - (now - requests[0])
            )

            raise RateLimited(
                max(1, retry_after)
            )

        # Запам'ятовуємо новий запит.
        requests.append(now)
