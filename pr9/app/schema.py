
"""Схема розбору звернень клієнтів для ПР9."""

import json
from jsonschema import Draft202012Validator


class SchemaError(ValueError):
    """Помилка перевірки відповіді моделі."""


def request_schema() -> dict:
    """JSON Schema для класифікації звернення клієнта."""

    return {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": [
                    "return",
                    "cancel",
                    "status",
                    "other",
                    "unknown",
                ],
                "description": (
                    "return — повернення товару; "
                    "cancel — скасування замовлення; "
                    "status — перевірка стану замовлення; "
                    "other — інше звернення; "
                    "unknown — незрозуміле звернення "
                    "або декілька різних дій."
                ),
            },
            "order_id": {
                "type": ["string", "null"],
                "description": (
                    "Номер замовлення, який назвав клієнт. "
                    "Якщо номер відсутній, повернути null. "
                    "Не вигадувати номери."
                ),
            },
            "item": {
                "type": ["string", "null"],
                "description": (
                    "Назва товару зі слів клієнта. "
                    "Якщо товар не названо, повернути null."
                ),
            },
            "reason": {
                "type": ["string", "null"],
                "description": (
                    "Причина повернення зі слів клієнта. "
                    "Якщо причину не вказано, повернути null."
                ),
            },
        },
        "required": [
            "kind",
            "order_id",
            "item",
            "reason",
        ],
        "additionalProperties": False,
    }


def validate(raw: str) -> dict:
    """Перевіряє JSON-відповідь AI-моделі."""

    if not isinstance(raw, str):
        raise SchemaError(
            "Відповідь моделі повинна бути рядком."
        )

    if not raw.strip():
        raise SchemaError(
            "Модель повернула порожню відповідь."
        )

    try:
        data = json.loads(raw)

    except json.JSONDecodeError as error:
        raise SchemaError(
            f"Неправильний формат JSON: {error.msg}"
        ) from error

    validator = Draft202012Validator(request_schema())

    errors = sorted(
        validator.iter_errors(data),
        key=lambda error: list(map(str, error.path)),
    )

    if errors:
        error = errors[0]

        path = ".".join(
            str(part) for part in error.path
        ) or "root"

        raise SchemaError(
            f"Помилка у полі '{path}': {error.message}"
        )

    # Прибираємо зайві пробіли.
    for field in ("order_id", "item", "reason"):
        value = data[field]

        if isinstance(value, str):
            value = value.strip()
            data[field] = value if value else None

    # Номер замовлення не повинен містити
    # довільний текст.
    if data["order_id"] is not None:
        if not data["order_id"].isdigit():
            raise SchemaError(
                "Номер замовлення повинен складатися з цифр."
            )

    return data
