import json

from jsonschema import validate as jsonschema_validate
from jsonschema.exceptions import ValidationError


def output_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "reply": {
                "type": "string"
            },
            "topic": {
                "type": "string",
                "enum": [
                    "order",
                    "delivery",
                    "payment",
                    "return",
                    "warranty",
                    "support",
                    "other"
                ]
            },
            "grounded": {
                "type": "boolean"
            },
            "needs_clarification": {
                "type": "boolean"
            },
            "handoff": {
                "type": "boolean"
            },
            "order_number": {
                "type": ["string", "null"],
                "pattern": "^[0-9]{6}$"
            }
        },
        "required": [
            "reply",
            "topic",
            "grounded",
            "needs_clarification",
            "handoff",
            "order_number"
        ],
        "additionalProperties": False
    }


def validate(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Відповідь моделі не є правильним JSON: {exc}"
        ) from exc

    try:
        jsonschema_validate(
            instance=data,
            schema=output_schema()
        )
    except ValidationError as exc:
        raise ValueError(
            f"Відповідь не пройшла JSON Schema: {exc.message}"
        ) from exc

    return data