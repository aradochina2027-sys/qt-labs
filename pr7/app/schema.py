"""Контракт відповіді мультимодальної моделі."""

import json
from datetime import date

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


class SchemaError(Exception):
    """Відповідь моделі має неправильний формат."""


def output_schema() -> dict:
    """Повернути JSON Schema відповіді моделі."""

    nullable_string = {
        "type": ["string", "null"]
    }

    nullable_number = {
        "type": ["number", "null"]
    }

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "document_type": {
                "type": "string",
                "enum": ["invoice", "other"],
            },

            "invoice_number": nullable_string,

            "invoice_date": nullable_string,

            "valid_until": nullable_string,

            "supplier": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": nullable_string,
                    "code": nullable_string,
                    "iban": nullable_string,
                },
                "required": [
                    "name",
                    "code",
                    "iban",
                ],
            },

            "buyer": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": nullable_string,
                    "code": nullable_string,
                },
                "required": [
                    "name",
                    "code",
                ],
            },

            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": nullable_string,
                        "unit": nullable_string,
                        "quantity": nullable_number,
                        "price": nullable_string,
                        "amount": nullable_string,
                    },
                    "required": [
                        "name",
                        "unit",
                        "quantity",
                        "price",
                        "amount",
                    ],
                },
            },

            "subtotal": nullable_string,
            "vat": nullable_string,
            "total": nullable_string,
        },

        "required": [
            "document_type",
            "invoice_number",
            "invoice_date",
            "valid_until",
            "supplier",
            "buyer",
            "items",
            "subtotal",
            "vat",
            "total",
        ],
    }

    return schema


def _check_date(value, field_name: str) -> None:
    """Перевірити дату у форматі YYYY-MM-DD."""

    if value is None:
        return

    try:
        date.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        raise SchemaError(
            f"Поле '{field_name}' повинно містити дату "
            "у форматі YYYY-MM-DD або null."
        ) from exc


def validate(raw: str) -> dict:
    """Перевірити JSON-відповідь моделі."""

    if not raw or not raw.strip():
        raise SchemaError("Модель повернула порожню відповідь.")

    text = raw.strip()

    # На випадок, якщо модель обгорнула JSON у ```json ... ```
    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # Перетворюємо текст у JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SchemaError(
            f"Відповідь моделі не є коректним JSON: {exc.msg}."
        ) from exc

    # Перевіряємо JSON Schema
    validator = Draft202012Validator(output_schema())

    errors = sorted(
        validator.iter_errors(data),
        key=lambda error: list(error.absolute_path),
    )

    if errors:
        error = errors[0]

        path = ".".join(
            str(part) for part in error.absolute_path
        )

        if not path:
            path = "root"

        raise SchemaError(
            f"Помилка схеми в полі '{path}': {error.message}"
        )

    # Додаткова перевірка дат
    _check_date(data.get("invoice_date"), "invoice_date")
    _check_date(data.get("valid_until"), "valid_until")

    return data