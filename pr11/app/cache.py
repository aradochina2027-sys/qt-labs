
"""Кеш AI-відповідей із підтримкою TTL та збереження у файл."""

import dataclasses
import hashlib
import json
import os
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

from .providers.base import Request, Result

load_dotenv()

ENABLED = os.getenv("CACHE_ENABLED", "1") == "1"
TTL_SECONDS = float(os.getenv("CACHE_TTL_SECONDS", "3600"))
MAX_ENTRIES = max(1, int(os.getenv("CACHE_MAX_ENTRIES", "500")))
CACHE_FILE = os.getenv("CACHE_FILE", "state/cache.json")

_lock = threading.RLock()
_entries = {}
_loaded = False

_counters = {
    "hits": 0,
    "misses": 0,
    "bypassed": 0,
}


def settings() -> dict:
    return {
        "enabled": ENABLED,
        "ttl_seconds": TTL_SECONDS,
        "max_entries": MAX_ENTRIES,
        "file": CACHE_FILE or None,
    }


def make_key(request: Request, prompt_version: str) -> str:
    """Створити унікальний ключ для AI-запиту."""

    payload = {
        "provider": request.metadata.get("provider"),
        "model": request.model,
        "prompt_version": prompt_version,
        "messages": request.messages,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }

    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def _save() -> None:
    """Атомарно записати кеш у JSON-файл."""

    if not CACHE_FILE:
        return

    path = Path(CACHE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(_entries, ensure_ascii=False),
        encoding="utf-8",
    )

    os.replace(temporary, path)


def _load() -> None:
    """Завантажити кеш із файлу."""

    global _loaded

    if _loaded:
        return

    _loaded = True

    if not CACHE_FILE:
        return

    try:
        data = json.loads(
            Path(CACHE_FILE).read_text(encoding="utf-8")
        )

        if isinstance(data, dict):
            _entries.update(data)

    except (OSError, ValueError):
        pass


def _cleanup() -> None:
    """Видалити прострочені та зайві записи."""

    now = time.time()

    for key in list(_entries):
        if _entries[key].get("expires", 0) <= now:
            del _entries[key]

    while len(_entries) > MAX_ENTRIES:
        oldest = min(
            _entries,
            key=lambda key: _entries[key].get("created", 0),
        )
        del _entries[oldest]


def bypass() -> None:
    """Зафіксувати обхід кешу (fresh=True)."""

    with _lock:
        _counters["bypassed"] += 1


def get(key: str) -> Result | None:
    """Повернути кешовану відповідь, якщо вона доступна."""

    if not ENABLED:
        return None

    with _lock:
        _load()
        _cleanup()

        item = _entries.get(key)

        if item is None:
            _counters["misses"] += 1
            return None

        try:
            result = Result(**item["result"])
        except (TypeError, KeyError):
            del _entries[key]
            _counters["misses"] += 1
            return None

        _counters["hits"] += 1

        result.cached = True
        result.cost = 0.0
        result.cost_estimated = False
        result.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        return result


def put(key: str, result: Result) -> None:
    """Зберегти успішну AI-відповідь."""

    if not ENABLED:
        return

    if result.finish_reason not in (None, "stop"):
        return

    with _lock:
        _load()

        now = time.time()

        _entries[key] = {
            "created": now,
            "expires": now + TTL_SECONDS,
            "result": dataclasses.asdict(result),
        }

        _cleanup()
        _save()


def stats() -> dict:
    """Повернути статистику використання кешу."""

    with _lock:
        _load()
        _cleanup()

        return {
            **_counters,
            "entries": len(_entries),
            "enabled": ENABLED,
        }
