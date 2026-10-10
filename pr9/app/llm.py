
import os
import json
import time
from functools import lru_cache

from dotenv import load_dotenv
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    APIStatusError,
)

load_dotenv()

BASE_URL = os.getenv("LLM_BASE_URL")
API_KEY = os.getenv("LLM_API_KEY")
MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash")

TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2500"))
TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))


class LLMError(Exception):
    """Помилка під час роботи з AI-моделлю."""
    pass


@lru_cache(maxsize=1)
def get_client():
    """Створення клієнта OpenAI-сумісного API."""

    if not API_KEY:
        raise LLMError("Не задано LLM_API_KEY у файлі .env")

    if not BASE_URL:
        raise LLMError("Не задано LLM_BASE_URL у файлі .env")

    return OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        timeout=TIMEOUT,
        max_retries=0,
    )


SYSTEM_PROMPT = """
Ти — AI-помічник інтернет-магазину «Сузір'я».

Допомагай клієнтам:
- перевіряти стан замовлень;
- дізнаватися інформацію про товари;
- оформлювати повернення відповідно до правил;
- скасовувати замовлення відповідно до правил.

Використовуй лише перевірені дані магазину.

Не вигадуй замовлення, товари, статуси або результати операцій.

Не розкривай внутрішні дані магазину.

Відповідай українською мовою.

Якщо інформації недостатньо, попроси уточнення.
"""


def build_messages(question: str) -> list[dict]:
    """Початкові повідомлення для AI-агента."""

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": question,
        },
    ]


def get_usage(response):
    """Повертає статистику використання токенів."""

    usage = response.usage

    if usage is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    return {
        "prompt_tokens": usage.prompt_tokens or 0,
        "completion_tokens": usage.completion_tokens or 0,
        "total_tokens": usage.total_tokens or 0,
    }


def handle_error(error):
    """Перетворює помилки API на LLMError."""

    if isinstance(error, AuthenticationError):
        raise LLMError(
            "Неправильний API-ключ. Перевір .env."
        ) from error

    if isinstance(error, RateLimitError):
        raise LLMError(
            "Ліміт запитів до моделі вичерпано."
        ) from error

    if isinstance(error, APITimeoutError):
        raise LLMError(
            "Час очікування відповіді моделі вичерпано."
        ) from error

    if isinstance(error, APIConnectionError):
        raise LLMError(
            "Не вдалося підключитися до AI-сервісу."
        ) from error

    if isinstance(error, BadRequestError):
        raise LLMError(
            f"Модель відхилила запит: {error}"
        ) from error

    if isinstance(error, APIStatusError):
        raise LLMError(
            f"Помилка AI-сервісу: HTTP {error.status_code}"
        ) from error

    raise LLMError(
        f"Невідома помилка моделі: {error}"
    ) from error


def chat(
    messages: list[dict],
    tools: list[dict],
    tool_choice: str = "auto",
) -> dict:
    """
    Виклик моделі з інструментами.

    Використовується AI-агентом із ПР8.
    """

    client = get_client()
    started = time.perf_counter()

    try:
        kwargs = {
            "model": MODEL,
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        response = client.chat.completions.create(**kwargs)

        message = response.choices[0].message

        tool_calls = []

        for call in message.tool_calls or []:
            tool_calls.append({
                "id": call.id,
                "type": call.type,
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            })

        return {
            "text": message.content or "",
            "content": message.content or "",
            "message": message.model_dump(
                exclude_none=True
            ),
            "tool_calls": tool_calls,
            "model": response.model,
            "elapsed": time.perf_counter() - started,
            "usage": get_usage(response),
        }

    except Exception as error:
        if isinstance(error, LLMError):
            raise
        handle_error(error)


def complete(
    messages: list[dict],
    schema: dict | None = None,
) -> dict:
    """
    Виклик AI-моделі без інструментів.

    Використовується у workflow ПР9.

    Якщо передано schema, запитується JSON.
    Якщо schema=None, генерується звичайний текст.
    """

    client = get_client()
    started = time.perf_counter()

    try:
        kwargs = {
            "model": MODEL,
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        }

        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "workflow_result",
                    "schema": schema,
                },
            }

        response = client.chat.completions.create(**kwargs)

        message = response.choices[0].message
        content = (message.content or "").strip()

        if not content:
            raise LLMError(
                "Модель повернула порожню відповідь."
            )

        # Перевіряємо, чи отримали коректний JSON.
        # Перевірку полів за схемою виконує schema.py.
        if schema is not None:
            try:
                json.loads(content)
            except json.JSONDecodeError as error:
                raise LLMError(
                    "Модель повернула невалідний JSON: "
                    + content[:150]
                ) from error

        return {
            "text": content,
            "content": content,
            "model": response.model,
            "elapsed": time.perf_counter() - started,
            "usage": get_usage(response),
        }

    except Exception as error:
        if isinstance(error, LLMError):
            raise
        handle_error(error)
