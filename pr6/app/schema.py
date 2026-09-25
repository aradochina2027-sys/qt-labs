"""Схема та перевірка структурованої відповіді мовної моделі."""

import json


def output_schema() -> dict:
    """Повернути JSON Schema відповіді моделі."""

    return {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "Відповідь користувачеві тільки на основі наданого контексту.",
            },
            "sources": {
                "type": "array",
                "description": "Номери фрагментів контексту, на яких базується відповідь.",
                "items": {
                    "type": "integer",
                    "minimum": 1,
                },
                "uniqueItems": True,
            },
            "found": {
                "type": "boolean",
                "description": "Чи знайдено достатньо інформації у контексті.",
            },
        },
        "required": [
            "answer",
            "sources",
            "found",
        ],
        "additionalProperties": False,
    }


def validate(raw: str) -> dict:
    """Перевірити JSON-відповідь моделі."""

    if not isinstance(raw, str):
        raise ValueError(
            "Відповідь моделі повинна бути текстом."
        )

    raw = raw.strip()

    if not raw:
        raise ValueError(
            "Модель повернула порожню відповідь."
        )

    # На випадок, якщо модель обгорнула JSON у ```json ... ```
    if raw.startswith("```"):
        lines = raw.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        raw = "\n".join(lines).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Відповідь моделі не є коректним JSON: {exc.msg}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            "Відповідь моделі повинна бути JSON-об'єктом."
        )

    required = {
        "answer",
        "sources",
        "found",
    }

    missing = required - set(data)

    if missing:
        raise ValueError(
            "У відповіді моделі відсутні поля: "
            + ", ".join(sorted(missing))
        )

    extra = set(data) - required

    if extra:
        raise ValueError(
            "У відповіді моделі є зайві поля: "
            + ", ".join(sorted(extra))
        )

    answer = data["answer"]
    sources = data["sources"]
    found = data["found"]

    if not isinstance(answer, str):
        raise ValueError(
            "Поле 'answer' повинно бути рядком."
        )

    answer = answer.strip()

    if not answer:
        raise ValueError(
            "Поле 'answer' не може бути порожнім."
        )

    if not isinstance(sources, list):
        raise ValueError(
            "Поле 'sources' повинно бути списком."
        )

    checked_sources = []

    for ref in sources:
        # bool у Python є підкласом int,
        # тому перевіряємо його окремо.
        if isinstance(ref, bool) or not isinstance(ref, int):
            raise ValueError(
                "Кожне значення у 'sources' повинно бути цілим числом."
            )

        if ref < 1:
            raise ValueError(
                "Номери джерел у 'sources' повинні починатися з 1."
            )

        if ref not in checked_sources:
            checked_sources.append(ref)

    if not isinstance(found, bool):
        raise ValueError(
            "Поле 'found' повинно бути true або false."
        )

    # Якщо модель каже, що інформації немає,
    # вона не повинна посилатися на джерела.
    if found is False and checked_sources:
        raise ValueError(
            "При found=false список sources повинен бути порожнім."
        )

    return {
        "answer": answer,
        "sources": checked_sources,
        "found": found,
    }