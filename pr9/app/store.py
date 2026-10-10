
"""Сховище стану звернень для практичної роботи №9."""

import json
import os
import re
import tempfile
import threading

from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from .state import Run, Step


load_dotenv()

RUNS_DIR = Path(
    os.getenv("RUNS_DIR", "runs")
).resolve()

RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Захист від одночасного запису в одному процесі.
_LOCK = threading.RLock()

# Дозволені символи в ідентифікаторі звернення.
_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


def _get_path(run_id: str) -> Path:
    """Повертає безпечний шлях до JSON-файлу."""

    if not isinstance(run_id, str):
        raise ValueError("run_id повинен бути рядком")

    if not _ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "Недопустимий ідентифікатор звернення"
        )

    return RUNS_DIR / f"{run_id}.json"


def _run_to_dict(run: Run) -> dict:
    """Перетворює Run та вкладені Step на словник."""

    return asdict(run)


def _dict_to_run(data: dict) -> Run:
    """Відновлює Run зі словника JSON."""

    if not isinstance(data, dict):
        raise ValueError("Некоректні дані звернення")

    data = data.copy()

    steps_data = data.get("steps") or []

    if not isinstance(steps_data, list):
        raise ValueError("Некоректний журнал кроків")

    steps = []

    for step_data in steps_data:
        if not isinstance(step_data, dict):
            raise ValueError("Некоректний запис кроку")

        steps.append(Step(**step_data))

    data["steps"] = steps

    return Run(**data)


def save(run: Run) -> None:
    """
    Атомарно зберігає стан звернення.

    Спочатку створюється тимчасовий файл,
    потім він замінює попередній JSON.
    """

    if not isinstance(run, Run):
        raise TypeError("Очікується об'єкт Run")

    target = _get_path(run.run_id)

    data = _run_to_dict(run)

    # Перевіряємо, чи всі дані можна записати в JSON.
    content = json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )

    temp_path = None

    with _LOCK:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=RUNS_DIR,
                prefix=f".{run.run_id}_",
                suffix=".tmp",
                delete=False,
            ) as temp_file:

                temp_path = Path(temp_file.name)

                temp_file.write(content)
                temp_file.flush()

                # Скидаємо буфер на диск.
                os.fsync(temp_file.fileno())

            # Атомарно замінюємо попередній файл.
            os.replace(temp_path, target)

        finally:
            # Прибираємо тимчасовий файл у разі помилки.
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


def load(run_id: str) -> Run | None:
    """
    Завантажує звернення за ідентифікатором.

    Повертає None, якщо звернення не існує.
    """

    path = _get_path(run_id)

    with _LOCK:
        if not path.is_file():
            return None

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Не вдалося прочитати звернення {run_id}"
            ) from error

        run = _dict_to_run(data)

        if run.run_id != run_id:
            raise ValueError(
                "Ідентифікатор звернення не збігається"
            )

        return run


def list_runs(status: str | None = None) -> list[Run]:
    """
    Повертає всі звернення.

    Якщо задано status, фільтрує за статусом.
    Найновіші звернення розташовані першими.
    """

    result = []

    with _LOCK:
        if not RUNS_DIR.exists():
            return []

        for path in RUNS_DIR.glob("*.json"):

            try:
                run = load(path.stem)

                if run is None:
                    continue

                if status is not None:
                    if run.status != status:
                        continue

                result.append(run)

            except (ValueError, TypeError, OSError):
                # Один пошкоджений файл не повинен
                # зупиняти завантаження інших.
                continue

    result.sort(
        key=lambda run: (
            run.created_at or "",
            run.updated_at or "",
        ),
        reverse=True,
    )

    return result
