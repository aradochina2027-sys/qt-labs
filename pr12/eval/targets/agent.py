
"""Адаптер до помічника ПР8 для оцінювання ПР12.

ПР8 імпортується без зміни вихідного коду.
Mock для g14/g15 перевіряє роботу конвеєра без API.
"""

import dataclasses
import os
import sys
import time
import types
from pathlib import Path
from unittest.mock import patch

NAME = "agent"

_default_dir = Path(__file__).resolve().parents[3] / "pr8"

APP_DIR = Path(
    os.getenv("AGENT_APP_DIR") or _default_dir
).resolve()

_app: types.SimpleNamespace | None = None


def _mock_enabled() -> bool:
    return (
        os.getenv("LLM_PROVIDER", "")
        .strip()
        .lower() == "mock"
    )


def setup() -> dict:
    """Підключити незмінений застосунок ПР8."""
    global _app

    if not (APP_DIR / "app" / "assistant.py").exists():
        raise SystemExit(
            f"Не знайдено ПР8: {APP_DIR}. "
            "Перевірте AGENT_APP_DIR у .env."
        )

    sys.path.insert(0, str(APP_DIR))

    from app import assistant
    from shop import service

    _app = types.SimpleNamespace(
        assistant=assistant,
        service=service,
    )

    mock = _mock_enabled()

    return {
        "provider": "mock" if mock else "openai",
        "model": (
            "mock-agent-pr12"
            if mock
            else os.getenv("LLM_MODEL")
        ),
        "temperature": (
            0 if mock else os.getenv("LLM_TEMPERATURE")
        ),
        "max_rounds": getattr(
            assistant,
            "MAX_ROUNDS",
            None,
        ),
        "shop_today": os.getenv("SHOP_TODAY"),
        "app_dir": str(APP_DIR),
    }


def before_case(case: dict) -> None:
    """Відновити дані магазину перед кожним тестом."""
    _app.service.reset()


def _system_prompt() -> str | None:
    """Отримати системну інструкцію з ПР8."""
    try:
        from app import llm
    except ImportError:
        return None

    names = (
        "SYSTEM_PROMPT",
        "SYSTEM",
        "INSTRUCTION",
        "INSTRUCTIONS",
        "SYSTEM_MESSAGE",
    )

    for name in names:
        value = getattr(llm, name, None)

        if isinstance(value, str) and value.strip():
            return value

    builder = getattr(llm, "build_messages", None)

    if callable(builder):
        try:
            messages = builder("")

            for message in messages:
                if (
                    message.get("role") == "system"
                    and isinstance(
                        message.get("content"),
                        str,
                    )
                ):
                    return message["content"]

        except Exception:
            return None

    return None


def _as_dict(obj) -> dict:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)

    if isinstance(obj, dict):
        return dict(obj)

    return dict(vars(obj))


def _mock_response(case: dict) -> dict:
    """Підставна відповідь для перевірки конвеєра g14/g15."""

    case_id = case.get("id")

    replies = {
        "g14": (
            "Я можу допомогти перевірити інформацію "
            "про замовлення, але не маю повноважень "
            "змінювати його статус."
        ),
        "g15": (
            "Не можу оформити повернення для чужого "
            "замовлення або виконати фінансову операцію. "
            "Зверніться до власника замовлення."
        ),
    }

    if case_id not in replies:
        raise ValueError(
            "Mock для ПР8 підтримує лише g14/g15. "
            "Для повного набору використайте реальну модель."
        )

    return {
        "message": {
            "role": "assistant",
            "content": replies[case_id],
        },
        "finish_reason": "stop",
        "model": "mock-agent-pr12",
        "elapsed": 0.0,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def run(case: dict) -> dict:
    """Виконати приклад через assistant.answer() із ПР8."""

    question = case["input"]["question"]
    customer_id = case["input"]["customer_id"]

    started = time.perf_counter()

    try:
        if _mock_enabled():
            response = _mock_response(case)

            # Підміняємо лише зовнішній виклик моделі.
            # Код assistant.answer() у ПР8 виконується як є.
            with patch.object(
                _app.assistant.llm,
                "chat",
                return_value=response,
            ):
                answer = _app.assistant.answer(
                    question,
                    customer_id,
                )
        else:
            # Справжня AI-модель для фінальних тестів.
            answer = _app.assistant.answer(
                question,
                customer_id,
            )

    except NotImplementedError:
        raise

    except Exception as exc:
        kind = (
            getattr(exc, "kind", None)
            or type(exc).__name__
        )

        return {
            "text": None,
            "status": "error",
            "error": {
                "kind": kind,
                "message": str(exc),
            },
            "calls": [],
            "system_prompt": _system_prompt(),
            "elapsed": round(
                time.perf_counter() - started,
                3,
            ),
        }

    calls = [
        _as_dict(c)
        for c in (
            getattr(answer, "calls", None)
            or []
        )
    ]

    elapsed = getattr(answer, "elapsed", None)

    return {
        "text": getattr(answer, "text", None),
        "status": "ok",
        "error": None,
        "calls": calls,
        "system_prompt": _system_prompt(),
        "provider": (
            "mock" if _mock_enabled() else "openai"
        ),
        "model": getattr(answer, "model", None),
        "usage": getattr(answer, "usage", None),
        "cost": None,
        "rounds": getattr(answer, "rounds", None),
        "stopped": getattr(answer, "stopped", None),
        "elapsed_by_kind": (
            elapsed
            if isinstance(elapsed, dict)
            else None
        ),
        "elapsed": round(
            time.perf_counter() - started,
            3,
        ),
    }
