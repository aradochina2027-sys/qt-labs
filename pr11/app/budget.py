
"""Денний ліміт витрат AI-застосунку."""

import json
import logging
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DAILY_USD = float(os.getenv("BUDGET_DAILY_USD", "1.00"))
WARN_AT = float(os.getenv("BUDGET_WARN_AT", "0.8"))
BUDGET_FILE = os.getenv("BUDGET_FILE", "state/budget.json")

_lock = threading.RLock()
_log = logging.getLogger("app.budget")


class BudgetExceeded(Exception):
    """Помилка перевищення денного бюджету."""

    def __init__(
        self,
        spent: float,
        limit: float,
        resets_at: str,
    ):
        self.spent = spent
        self.limit = limit
        self.resets_at = resets_at

        super().__init__(
            f"Денний бюджет {limit:.2f} USD вичерпано. "
            f"Витрачено: {spent:.6f} USD. "
            f"Відновлення: {resets_at}"
        )


def _today() -> str:
    """Поточна локальна дата."""
    return datetime.now().astimezone().date().isoformat()


def _reset_time() -> str:
    """Час відновлення бюджету."""
    now = datetime.now().astimezone()

    tomorrow = now.date() + timedelta(days=1)

    reset = datetime.combine(
        tomorrow,
        datetime.min.time(),
        tzinfo=now.tzinfo,
    )

    return reset.isoformat()


def _read() -> dict:
    """Завантажити стан бюджету з файлу."""

    try:
        data = json.loads(
            Path(BUDGET_FILE).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        data = {}

    if not isinstance(data, dict):
        data = {}

    if data.get("day") != _today():
        return {
            "day": _today(),
            "spent": 0.0,
        }

    return data


def _write(data: dict) -> None:
    """Атомарно зберегти стан бюджету."""

    path = Path(BUDGET_FILE)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_suffix(".tmp")

    temp.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )

    os.replace(temp, path)


def settings() -> dict:
    """Повернути налаштування бюджету."""

    return {
        "daily_usd": DAILY_USD,
        "warn_at": WARN_AT,
        "file": BUDGET_FILE,
    }


def check(expected_cost: float | None = None) -> None:
    """Перевірити, чи дозволено платний виклик."""

    with _lock:
        data = _read()
        spent = float(data.get("spent", 0.0))

        if (
            spent >= DAILY_USD
            or (
                expected_cost is not None
                and spent + expected_cost > DAILY_USD
            )
        ):
            raise BudgetExceeded(
                spent,
                DAILY_USD,
                _reset_time(),
            )


def record(cost: float | None) -> None:
    """Зарахувати вартість AI-виклику."""

    if cost is None:
        _log.warning("budget.unknown_cost")
        return

    if cost < 0:
        raise ValueError("Вартість не може бути від'ємною")

    with _lock:
        data = _read()

        old_spent = float(data.get("spent", 0.0))

        new_spent = round(old_spent + cost, 10)

        data["spent"] = new_spent

        _write(data)

        warning_limit = DAILY_USD * WARN_AT

        if old_spent < warning_limit <= new_spent:
            _log.warning(
                "budget.warning",
                extra={
                    "spent": new_spent,
                    "limit": DAILY_USD,
                },
            )

        if old_spent < DAILY_USD <= new_spent:
            _log.warning(
                "budget.exceeded",
                extra={
                    "spent": new_spent,
                    "limit": DAILY_USD,
                },
            )


def status() -> dict:
    """Отримати поточний стан бюджету."""

    with _lock:
        data = _read()

        spent = float(data.get("spent", 0.0))

        remaining = max(
            0.0,
            DAILY_USD - spent,
        )

        return {
            "day": data["day"],
            "spent": spent,
            "limit": DAILY_USD,
            "remaining": round(remaining, 10),
            "warning": spent >= DAILY_USD * WARN_AT,
            "exceeded": spent >= DAILY_USD,
            "resets_at": _reset_time(),
        }
