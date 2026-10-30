
"""Розрахунок вартості AI-запитів з урахуванням thinking-токенів."""

import math
import os

from dotenv import load_dotenv

load_dotenv()

PRICE_INPUT = float(
    os.getenv("LLM_PRICE_INPUT_PER_1M", "0.30")
)

PRICE_OUTPUT = float(
    os.getenv("LLM_PRICE_OUTPUT_PER_1M", "2.50")
)

_cached = os.getenv("LLM_PRICE_CACHED_INPUT_PER_1M", "")

PRICE_CACHED_INPUT = (
    float(_cached)
    if _cached
    else PRICE_INPUT * 0.25
)

MOCK_BILLABLE = os.getenv("MOCK_BILLABLE", "1") == "1"


def settings() -> dict:
    """Повертає поточні тарифи."""

    return {
        "input_per_1m": PRICE_INPUT,
        "output_per_1m": PRICE_OUTPUT,
        "cached_input_per_1m": PRICE_CACHED_INPUT,
        "mock_billable": MOCK_BILLABLE,
        "currency": "USD",
    }


def _tokens(value):
    """Безпечно перетворює значення на кількість токенів."""

    if value is None:
        return None

    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _first_tokens(data: dict, *keys: str):
    """Шукає перше коректне поле з кількістю токенів."""

    for key in keys:
        if key in data:
            value = _tokens(data.get(key))
            if value is not None:
                return value

    return None


def estimate(
    usage: dict | None,
    *,
    provider: str | None = None,
    model: str | None = None,
    text: str | None = None,
) -> tuple[float | None, bool]:
    """
    Повертає (вартість USD, чи використано приблизні токени).

    Для Gemini:
    total_tokens може містити thinking-токени,
    яких немає у completion_tokens.

    Такі додаткові токени оплачуються за ціною
    вихідних токенів.

    Для інших OpenAI-сумісних моделей thinking-токени
    зазвичай вже входять у completion_tokens.
    """

    if provider == "mock" and not MOCK_BILLABLE:
        return 0.0, False

    data = usage if isinstance(usage, dict) else {}

    input_tokens = _first_tokens(
        data,
        "prompt_tokens",
        "input_tokens",
        "promptTokenCount",
    )

    output_tokens = _first_tokens(
        data,
        "completion_tokens",
        "output_tokens",
        "candidatesTokenCount",
    )

    total_tokens = _first_tokens(
        data,
        "total_tokens",
        "totalTokenCount",
    )

    details = (
        data.get("prompt_tokens_details")
        or data.get("input_tokens_details")
        or {}
    )

    cached_tokens = 0

    if isinstance(details, dict):
        cached_tokens = (
            _first_tokens(details, "cached_tokens") or 0
        )

    if cached_tokens == 0:
        cached_tokens = (
            _first_tokens(
                data,
                "cached_content_token_count",
                "cachedContentTokenCount",
            )
            or 0
        )

    estimated = False

    # Коли провайдер не повернув вихідні токени,
    # приблизно оцінюємо їх за довжиною тексту.
    if output_tokens is None and text:
        output_tokens = math.ceil(len(text) / 3)
        estimated = True

    # Немає жодної інформації для розрахунку.
    if input_tokens is None and output_tokens is None:
        return None, True

    if input_tokens is None:
        input_tokens = 0
        estimated = True

    if output_tokens is None:
        output_tokens = 0
        estimated = True

    billable_output_tokens = output_tokens

    model_name = (model or "").strip().lower()
    provider_name = (provider or "").strip().lower()

    is_gemini = (
        model_name.startswith("gemini")
        or provider_name == "gemini"
    )

    # У Gemini total_tokens може дорівнювати:
    # input + visible_output + thinking.
    #
    # Не додаємо reasoning_tokens окремо,
    # якщо вони вже враховані в completion_tokens.
    #
    # Додаткові токени беремо лише з різниці
    # між фактичним total_tokens та відомими полями.
    if is_gemini and total_tokens is not None:
        accounted_tokens = input_tokens + output_tokens

        if total_tokens > accounted_tokens:
            extra_output_tokens = (
                total_tokens - accounted_tokens
            )

            billable_output_tokens += extra_output_tokens

        elif total_tokens < accounted_tokens:
            # Непослідовні дані провайдера:
            # не вигадуємо кількість thinking-токенів.
            estimated = True

    cached_tokens = min(cached_tokens, input_tokens)

    regular_input_tokens = input_tokens - cached_tokens

    input_cost = (
        regular_input_tokens * PRICE_INPUT
    ) / 1_000_000

    cached_cost = (
        cached_tokens * PRICE_CACHED_INPUT
    ) / 1_000_000

    output_cost = (
        billable_output_tokens * PRICE_OUTPUT
    ) / 1_000_000

    total_cost = input_cost + cached_cost + output_cost

    return round(total_cost, 10), estimated
