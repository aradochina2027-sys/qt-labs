"""Конвеєр обробки документа: від байтів файлу до рішення."""

import time
from dataclasses import dataclass, field

from .images import ImageError, prepare
from .llm import LLMError, extract
from .rules import Issue, check


@dataclass
class Result:
    """Результат повного конвеєра обробки документа."""

    decision: str
    reasons: list[str] = field(default_factory=list)
    document: dict | None = None
    issues: list[Issue] = field(default_factory=list)
    image: dict = field(default_factory=dict)
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None

    # Сира JSON-відповідь моделі до перевірки schema.py.
    raw_response: str | None = None


def _has_useful_data(document: dict) -> bool:
    """Перевірити, чи вдалося взагалі щось вилучити з документа."""

    fields = [
        document.get("invoice_number"),
        document.get("invoice_date"),
        document.get("subtotal"),
        document.get("vat"),
        document.get("total"),
        document.get("supplier", {}).get("name"),
        document.get("supplier", {}).get("code"),
        document.get("supplier", {}).get("iban"),
        document.get("buyer", {}).get("name"),
        document.get("buyer", {}).get("code"),
    ]

    if any(
        value is not None and str(value).strip()
        for value in fields
    ):
        return True

    if document.get("items"):
        return True

    return False


def decide(
    document: dict,
    issues: list[Issue],
) -> tuple[str, list[str]]:
    """Вирішити долю документа.

    auto:
        рахунок і немає проблем;

    review:
        рахунок прочитаний, але є проблеми;

    reject:
        не рахунок або практично нічого не вдалося вилучити.
    """

    # Не рахунок на оплату.
    if document.get("document_type") != "invoice":
        return (
            "reject",
            [
                "Документ не визначено як рахунок на оплату."
            ],
        )

    # Рахунок, але корисних даних фактично немає.
    if not _has_useful_data(document):
        return (
            "reject",
            [
                "З документа не вдалося вилучити "
                "достатньо даних для обробки."
            ],
        )

    # Збираємо помилки правил.
    errors = [
        issue
        for issue in issues
        if issue.severity == "error"
    ]

    warnings = [
        issue
        for issue in issues
        if issue.severity == "warning"
    ]

    # Будь-яка критична помилка -> ручна перевірка.
    if errors:
        reasons = [
            issue.message
            for issue in errors
        ]

        return "review", reasons

    # Попередження також не повинні давати auto.
    if warnings:
        reasons = [
            issue.message
            for issue in warnings
        ]

        return "review", reasons

    # Тільки код може встановити auto.
    return (
        "auto",
        [
            "Документ пройшов усі програмні перевірки."
        ],
    )


def process(content: bytes) -> Result:
    """Обробити файл:

    підготовка -> модель -> схема -> правила -> рішення.
    """

    elapsed: dict[str, float] = {}

    # -----------------------------------------------------
    # 1. Підготовка зображення
    # -----------------------------------------------------

    started = time.perf_counter()

    try:
        prepared = prepare(content)

    except ImageError as exc:
        elapsed["prepare"] = (
            time.perf_counter() - started
        )

        return Result(
            decision="reject",
            reasons=[
                f"Не вдалося підготувати зображення: {exc}"
            ],
            elapsed=elapsed,
        )

    elapsed["prepare"] = (
        time.perf_counter() - started
    )

    image_info = {
        "original": prepared.original,
        "sent": prepared.sent,
        "mime": prepared.mime,
    }

    # -----------------------------------------------------
    # 2. Вилучення даних моделлю + перевірка схеми
    # -----------------------------------------------------

    started = time.perf_counter()

    try:
        model_result = extract(prepared)

    except LLMError as exc:
        elapsed["extraction"] = (
            time.perf_counter() - started
        )

        return Result(
            decision="review",
            reasons=[
                "Не вдалося отримати надійну "
                f"відповідь моделі: {exc}"
            ],
            image=image_info,
            elapsed=elapsed,
        )

    elapsed["extraction"] = model_result.get(
        "time",
        time.perf_counter() - started,
    )

    document = model_result["data"]

    raw_response = model_result.get(
        "raw_response"
    )

    # -----------------------------------------------------
    # 3. Програмні перевірки
    # -----------------------------------------------------

    started = time.perf_counter()

    try:
        issues = check(document)

    except Exception as exc:
        elapsed["checks"] = (
            time.perf_counter() - started
        )

        return Result(
            decision="review",
            reasons=[
                "Не вдалося завершити програмні "
                f"перевірки: {exc}"
            ],
            document=document,
            image=image_info,
            model=model_result.get("model"),
            elapsed=elapsed,
            usage=model_result.get("usage"),
            raw_response=raw_response,
        )

    elapsed["checks"] = (
        time.perf_counter() - started
    )

    # -----------------------------------------------------
    # 4. Остаточне рішення
    # -----------------------------------------------------

    decision, reasons = decide(
        document,
        issues,
    )

    return Result(
        decision=decision,
        reasons=reasons,
        document=document,
        issues=issues,
        image=image_info,
        model=model_result.get("model"),
        elapsed=elapsed,
        usage=model_result.get("usage"),
        raw_response=raw_response,
    )