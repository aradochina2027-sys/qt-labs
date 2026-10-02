"""ПР8: eval із продовженням попереднього прогону.

- пропускає вже успішні питання з results.json;
- повторює тільки питання з помилками;
- чекає при rate limit і HTTP 503;
- після кожного питання одразу зберігає results.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path


# ---------------------------------------------------------
# Шляхи
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent

QUESTIONS_FILE = HERE / "questions.json"
RESULTS_FILE = HERE / "results.json"


# ---------------------------------------------------------
# Імпорти проєкту
# ---------------------------------------------------------

from app import assistant, llm
from shop import service


# ---------------------------------------------------------
# Налаштування eval
# ---------------------------------------------------------

# Пауза після rate limit / 503
RETRY_WAIT = int(
    os.getenv("EVAL_RETRY_WAIT", "120")
)

# Пауза між успішно обробленими питаннями
BETWEEN_QUESTIONS = int(
    os.getenv("EVAL_DELAY_SECONDS", "45")
)

# Максимальна кількість спроб одного питання
MAX_RETRIES = int(
    os.getenv("EVAL_MAX_RETRIES", "3")
)


# ---------------------------------------------------------
# Робота з results.json
# ---------------------------------------------------------

def load_old_results() -> dict[str, dict]:
    """Завантажити результати попереднього прогону."""
    if not RESULTS_FILE.exists():
        return {}

    try:
        rows = json.loads(
            RESULTS_FILE.read_text(encoding="utf-8")
        )

        return {
            row["id"]: row
            for row in rows
            if isinstance(row, dict) and row.get("id")
        }

    except Exception as exc:
        print(
            "Не вдалося прочитати старий results.json:"
        )
        print(exc)
        return {}


def save_results(
    questions: list[dict],
    results: dict[str, dict],
) -> None:
    """Зберегти результати у порядку questions.json."""

    rows = []

    for question in questions:
        question_id = question["id"]

        if question_id in results:
            rows.append(results[question_id])

    RESULTS_FILE.write_text(
        json.dumps(
            rows,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def is_success(row: dict | None) -> bool:
    """Чи питання вже успішно виконане."""
    if not row:
        return False

    if row.get("error"):
        return False

    return bool(row.get("answer"))


# ---------------------------------------------------------
# Перетворення журналу викликів
# ---------------------------------------------------------

def convert_calls(calls) -> list[dict]:
    rows = []

    for call in calls:
        try:
            rows.append(asdict(call))

        except TypeError:
            if isinstance(call, dict):
                rows.append(call)

            else:
                rows.append(
                    {
                        "name": getattr(
                            call,
                            "name",
                            None,
                        ),
                        "arguments": getattr(
                            call,
                            "arguments",
                            None,
                        ),
                        "status": getattr(
                            call,
                            "status",
                            None,
                        ),
                        "result": getattr(
                            call,
                            "result",
                            None,
                        ),
                    }
                )

    return rows


# ---------------------------------------------------------
# Чи треба повторювати помилку
# ---------------------------------------------------------

def should_retry(exc: llm.LLMError) -> bool:
    """Повторювати rate limit, 503, timeout та connection."""

    if exc.code == "rate_limit":
        return True

    if exc.code == "timeout":
        return True

    if exc.code == "connection":
        return True

    if (
        exc.code == "provider"
        and "503" in str(exc)
    ):
        return True

    return False


# ---------------------------------------------------------
# Основний прогін
# ---------------------------------------------------------

def main() -> None:

    print()
    print("=" * 65)
    print("ПР8 — ПРОДОВЖЕННЯ EVAL")
    print("=" * 65)

    if not QUESTIONS_FILE.exists():
        print(
            f"ПОМИЛКА: немає файла {QUESTIONS_FILE}"
        )
        return

    try:
        questions = json.loads(
            QUESTIONS_FILE.read_text(
                encoding="utf-8"
            )
        )

    except Exception as exc:
        print("Помилка questions.json:")
        print(exc)
        return

    results = load_old_results()

    successful = sum(
        1
        for q in questions
        if is_success(results.get(q["id"]))
    )

    remaining = len(questions) - successful

    print(
        f"Усього питань: {len(questions)}"
    )

    print(
        f"Уже успішно: {successful}"
    )

    print(
        f"Залишилось: {remaining}"
    )

    print(
        f"Пауза між питаннями: "
        f"{BETWEEN_QUESTIONS} с"
    )

    print(
        f"При rate limit / HTTP 503: "
        f"{RETRY_WAIT} с"
    )

    print()

    # -----------------------------------------------------
    # Питання
    # -----------------------------------------------------

    for position, q in enumerate(
        questions,
        start=1,
    ):

        question_id = q["id"]

        old_row = results.get(question_id)

        # Уже пройшло — не витрачаємо API
        if is_success(old_row):
            print(
                f"[{question_id}] "
                f"УЖЕ ГОТОВО — пропускаю."
            )
            continue

        print()
        print("-" * 65)

        print(
            f"[{question_id}] "
            f"{q['question']}"
        )

        print(
            f"Клієнт: {q['customer']}"
        )

        final_row = None

        # -------------------------------------------------
        # Спроби
        # -------------------------------------------------

        for attempt in range(
            1,
            MAX_RETRIES + 1,
        ):

            print(
                f"  Спроба {attempt}/{MAX_RETRIES}"
            )

            # Кожна спроба починається
            # з початкового стану магазину.
            service.reset()

            try:
                result = assistant.answer(
                    q["question"],
                    q["customer"],
                )

                calls = convert_calls(
                    result.calls
                )

                final_row = {
                    "id": q["id"],
                    "kind": q.get(
                        "kind",
                        "unknown",
                    ),
                    "customer": q["customer"],
                    "question": q["question"],
                    "expected_tools": q.get(
                        "expect_tools",
                        [],
                    ),
                    "answer": result.text,
                    "calls": calls,
                    "rounds": result.rounds,
                    "stopped": result.stopped,
                    "elapsed": result.elapsed,
                    "usage": result.usage,
                }

                print(
                    "  tools:",
                    [
                        call.get("name")
                        for call in calls
                    ],
                )

                print(
                    "  answer:",
                    result.text,
                )

                break

            # ---------------------------------------------
            # Контрольовані помилки LLM
            # ---------------------------------------------

            except llm.LLMError as exc:

                retry = should_retry(exc)

                if (
                    retry
                    and attempt < MAX_RETRIES
                ):
                    print(
                        f"  ТИМЧАСОВА ПОМИЛКА: "
                        f"{exc}"
                    )

                    print(
                        f"  Чекаю {RETRY_WAIT} с "
                        f"і повторюю {question_id}..."
                    )

                    time.sleep(RETRY_WAIT)
                    continue

                final_row = {
                    **q,
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }

                print(
                    "  ERROR:",
                    final_row["error"],
                )

                break

            # ---------------------------------------------
            # Інші помилки
            # ---------------------------------------------

            except Exception as exc:

                final_row = {
                    **q,
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }

                print(
                    "  ERROR:",
                    final_row["error"],
                )

                break

        # -------------------------------------------------
        # Записуємо результат
        # -------------------------------------------------

        if final_row is None:
            final_row = {
                **q,
                "error": (
                    "невідома помилка прогону"
                ),
            }

        results[question_id] = final_row

        save_results(
            questions,
            results,
        )

        print(
            "  Результат збережено:",
            RESULTS_FILE,
        )

        # -------------------------------------------------
        # Пауза
        # -------------------------------------------------

        if is_success(final_row):

            print(
                f"  Пауза "
                f"{BETWEEN_QUESTIONS} с..."
            )

            time.sleep(
                BETWEEN_QUESTIONS
            )

        else:
            print()
            print(
                "  Це питання поки не пройшло."
            )

            print(
                "  results.json збережено, "
                "його можна буде повторити пізніше."
            )

    # -----------------------------------------------------
    # Підсумок
    # -----------------------------------------------------

    save_results(
        questions,
        results,
    )

    successful = sum(
        1
        for q in questions
        if is_success(
            results.get(q["id"])
        )
    )

    failed = len(questions) - successful

    print()
    print("=" * 65)
    print("ПРОГІН ЗАВЕРШЕНО")
    print("=" * 65)

    print(
        f"Успішно: {successful}"
    )

    print(
        f"З помилками: {failed}"
    )

    print(
        f"Файл: {RESULTS_FILE}"
    )

    if failed == 0:
        print()
        print(
            "Усі питання успішно пройдені."
        )

    else:
        print()
        print(
            "Запусти цей файл пізніше ще раз — "
            "успішні питання будуть пропущені, "
            "а повторяться тільки невдалі."
        )


if __name__ == "__main__":
    main()