"""Автоматична перевірка PR7 на clean/degraded/own зразках."""

import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from pathlib import Path


# ---------------------------------------------------------
# Шляхи та імпорти
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.extraction import process  # noqa: E402


SAMPLES_DIR = ROOT / "samples"
EXPECTED_PATH = SAMPLES_DIR / "expected.json"

OUTPUT_PATH = ROOT / "eval" / "results.json"
SUMMARY_PATH = ROOT / "eval" / "summary.json"


# Пауза між УСПІШНИМИ API-запитами.
# При model_error/429 повторного запиту НЕ робимо.
REQUEST_DELAY = 10


# ---------------------------------------------------------
# Нормалізація
# ---------------------------------------------------------

def normalize_text(value):
    if value is None:
        return None

    value = str(value).strip()

    # Різні апострофи вважаємо однаковими.
    value = (
        value.replace("’", "'")
        .replace("ʼ", "'")
        .replace("`", "'")
    )

    value = re.sub(r"\s+", " ", value)

    return value.casefold()


def normalize_iban(value):
    if value is None:
        return None

    return re.sub(
        r"\s+",
        "",
        str(value),
    ).upper()


def normalize_money(value):
    if value is None:
        return None

    text = str(value).strip()

    text = (
        text.replace("\u00a0", "")
        .replace(" ", "")
        .replace(",", ".")
    )

    try:
        return Decimal(text).quantize(
            Decimal("0.01")
        )
    except (InvalidOperation, ValueError):
        return text


def normalize_number(value):
    if value is None:
        return None

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return str(value).strip()


# ---------------------------------------------------------
# expected.json -> схема застосунку
# ---------------------------------------------------------

def expected_to_document(fields: dict) -> dict:
    """Перетворити формат expected.json у формат app/schema.py."""

    document_type = fields.get("document_type")

    if document_type == "рахунок":
        document_type = "invoice"
    elif document_type is not None:
        document_type = "other"

    supplier = fields.get("supplier") or {}
    buyer = fields.get("buyer") or {}

    return {
        "document_type": document_type,
        "invoice_number": fields.get("number"),
        "invoice_date": fields.get("date"),
        "valid_until": fields.get("valid_until"),
        "supplier": {
            "name": supplier.get("name"),
            "code": supplier.get("code"),
            "iban": supplier.get("iban"),
        },
        "buyer": {
            "name": buyer.get("name"),
            "code": buyer.get("code"),
        },
        "items": fields.get("items") or [],
        "subtotal": fields.get("total_without_vat"),
        "vat": fields.get("vat"),
        "total": fields.get("total"),
    }


# ---------------------------------------------------------
# Розгортання вкладеного JSON
# ---------------------------------------------------------

def flatten(value, prefix=""):
    result = {}

    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = (
                f"{prefix}.{key}"
                if prefix
                else key
            )
            result.update(
                flatten(
                    child,
                    child_prefix,
                )
            )

    elif isinstance(value, list):
        if not value and prefix:
            result[prefix] = []

        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            result.update(
                flatten(
                    child,
                    child_prefix,
                )
            )

    else:
        result[prefix] = value

    return result


# ---------------------------------------------------------
# Порівняння значень
# ---------------------------------------------------------

MONEY_FIELDS = {
    "subtotal",
    "vat",
    "total",
}


def is_money_field(field: str) -> bool:
    if field in MONEY_FIELDS:
        return True

    return (
        field.endswith(".price")
        or field.endswith(".amount")
    )


def compare_value(
    field: str,
    expected_value,
    actual_value,
):
    # Обидва значення відсутні.
    if (
        expected_value is None
        and actual_value is None
    ):
        return "correct"

    # Очікувалося значення, але модель нічого не дала.
    if (
        expected_value is not None
        and actual_value is None
    ):
        return "omission"

    # У документі значення немає,
    # але модель його вигадала.
    if (
        expected_value is None
        and actual_value is not None
    ):
        return "hallucination"

    if field == "supplier.iban":
        expected_normalized = normalize_iban(
            expected_value
        )
        actual_normalized = normalize_iban(
            actual_value
        )

    elif is_money_field(field):
        expected_normalized = normalize_money(
            expected_value
        )
        actual_normalized = normalize_money(
            actual_value
        )

    elif field.endswith(".quantity"):
        expected_normalized = normalize_number(
            expected_value
        )
        actual_normalized = normalize_number(
            actual_value
        )

    else:
        expected_normalized = normalize_text(
            expected_value
        )
        actual_normalized = normalize_text(
            actual_value
        )

    if expected_normalized == actual_normalized:
        return "correct"

    return "error"


# ---------------------------------------------------------
# Порівняння документа
# ---------------------------------------------------------

def compare_documents(
    expected: dict,
    actual: dict,
) -> dict:
    expected_flat = flatten(expected)
    actual_flat = flatten(actual)

    keys = sorted(
        set(expected_flat)
        | set(actual_flat)
    )

    comparisons = {}

    counts = {
        "correct": 0,
        "error": 0,
        "omission": 0,
        "hallucination": 0,
    }

    for field in keys:
        expected_value = expected_flat.get(
            field
        )
        actual_value = actual_flat.get(
            field
        )

        status = compare_value(
            field,
            expected_value,
            actual_value,
        )

        comparisons[field] = {
            "expected": expected_value,
            "actual": actual_value,
            "status": status,
        }

        counts[status] += 1

    return {
        "fields": comparisons,
        "counts": counts,
    }


# ---------------------------------------------------------
# Критичні поля
# ---------------------------------------------------------

CRITICAL_FIELDS = {
    "supplier.iban",
    "supplier.code",
    "total",
}


def critical_stats(comparison: dict) -> dict:
    correct = 0
    total = 0
    details = {}

    for field in sorted(CRITICAL_FIELDS):
        info = comparison["fields"].get(
            field
        )

        if info is None:
            continue

        total += 1

        is_correct = (
            info["status"] == "correct"
        )

        if is_correct:
            correct += 1

        details[field] = {
            "status": info["status"],
            "expected": info["expected"],
            "actual": info["actual"],
        }

    return {
        "correct": correct,
        "total": total,
        "accuracy": (
            correct / total
            if total
            else None
        ),
        "fields": details,
    }


# ---------------------------------------------------------
# Правила
# ---------------------------------------------------------

def issue_codes(result) -> set[str]:
    codes = set()

    for issue in result.issues:
        code = getattr(
            issue,
            "code",
            None,
        )

        if code:
            codes.add(code)

    return codes


def normalize_issue_field(field):
    if not field:
        return None

    field = str(field)

    # items.1.amount -> items[1].amount
    field = re.sub(
        r"items\.(\d+)\.",
        r"items[\1].",
        field,
    )

    return field


def issue_fields(result) -> set[str]:
    fields = set()

    for issue in result.issues:
        field = getattr(
            issue,
            "field",
            None,
        )

        field = normalize_issue_field(
            field
        )

        if field:
            fields.add(field)

    return fields


def problem_fields(
    comparison: dict,
) -> list[str]:
    problems = []

    for field, info in comparison[
        "fields"
    ].items():
        if info["status"] in {
            "error",
            "hallucination",
        }:
            problems.append(field)

    return problems


def field_caught_by_rule(
    field: str,
    result,
) -> bool:
    fields = issue_fields(result)

    for issue_field in fields:
        if field == issue_field:
            return True

        if field.startswith(
            issue_field + "."
        ):
            return True

        if issue_field.startswith(
            field + "."
        ):
            return True

    return False


# ---------------------------------------------------------
# Перевірка успішності відповіді API
# ---------------------------------------------------------

def model_failed(result) -> bool:
    return result.document is None


# ---------------------------------------------------------
# Обробка одного файла
# ---------------------------------------------------------

def evaluate_file(
    relative_path: str,
    expected_entry: dict,
) -> dict:
    path = SAMPLES_DIR / relative_path

    print(
        f"\nОбробка: {relative_path}"
    )

    if not path.exists():
        print(
            "  Файл не знайдено."
        )

        return {
            "file": relative_path,
            "source": expected_entry.get(
                "source"
            ),
            "variant": expected_entry.get(
                "variant"
            ),
            "run_status": "file_not_found",
        }

    content = path.read_bytes()

    result = process(content)

    # Якщо API/модель не повернула документ,
    # НЕ рахуємо всі поля як omission.
    if model_failed(result):
        message = (
            result.reasons[0]
            if result.reasons
            else "Модель не повернула документ."
        )

        print(
            f"  ПОМИЛКА МОДЕЛІ/API: {message}"
        )

        return {
            "file": relative_path,
            "source": expected_entry.get(
                "source"
            ),
            "variant": expected_entry.get(
                "variant"
            ),
            "run_status": "model_error",
            "error": message,
            "actual_decision": result.decision,
            "reasons": result.reasons,
            "issues": [
                asdict(issue)
                for issue in result.issues
            ],
            "model": result.model,
            "image": result.image,
            "elapsed": result.elapsed,
            "usage": result.usage or {},
        }

    expected_document = (
        expected_to_document(
            expected_entry.get(
                "fields",
                {},
            )
        )
    )

    actual_document = result.document

    comparison = compare_documents(
        expected_document,
        actual_document,
    )

    actual_checks = sorted(
        issue_codes(result)
    )

    expected_checks = sorted(
        expected_entry.get(
            "expected_checks",
            [],
        )
    )

    expected_decision = (
        expected_entry.get(
            "expected_decision"
        )
    )

    decision_match = (
        result.decision
        == expected_decision
    )

    checks_match = (
        set(actual_checks)
        == set(expected_checks)
    )

    bad_fields = problem_fields(
        comparison
    )

    caught_fields = [
        field
        for field in bad_fields
        if field_caught_by_rule(
            field,
            result,
        )
    ]

    uncaught_fields = [
        field
        for field in bad_fields
        if field not in caught_fields
    ]

    # Тиха помилка =
    # неправильне/вигадане значення,
    # але документ отримав auto.
    silent_error_fields = (
        bad_fields
        if result.decision == "auto"
        else []
    )

    silent_error = bool(
        silent_error_fields
    )

    usage = result.usage or {}

    output = {
        "file": relative_path,
        "source": expected_entry.get(
            "source"
        ),
        "variant": expected_entry.get(
            "variant"
        ),
        "run_status": "ok",
        "expected_document": expected_document,
        "actual_document": actual_document,
        "comparison": comparison,
        "critical": critical_stats(
            comparison
        ),
        "expected_checks": expected_checks,
        "actual_checks": actual_checks,
        "checks_match": checks_match,
        "expected_decision": expected_decision,
        "actual_decision": result.decision,
        "decision_match": decision_match,
        "problem_fields": bad_fields,
        "caught_by_rules": caught_fields,
        "uncaught_by_rules": uncaught_fields,
        "silent_error": silent_error,
        "silent_error_fields": silent_error_fields,
        "reasons": result.reasons,
        "issues": [
            asdict(issue)
            for issue in result.issues
        ],
        "model": result.model,
        "image": result.image,
        "elapsed": result.elapsed,
        "usage": usage,
    }

    # Зберігаємо сиру відповідь моделі,
    # яку передає Result.raw_response.
    raw_response = getattr(
        result,
        "raw_response",
        None,
    )

    if raw_response is not None:
        output["raw_response"] = raw_response

    counts = comparison["counts"]

    print(
        "  "
        f"правильно={counts['correct']}, "
        f"помилки={counts['error']}, "
        f"пропуски={counts['omission']}, "
        f"вигадки={counts['hallucination']}"
    )

    print(
        "  "
        f"рішення: {result.decision} "
        f"(очікувалося {expected_decision})"
    )

    print(
        "  "
        f"правила: {actual_checks} "
        f"(очікувалося {expected_checks})"
    )

    if silent_error:
        print(
            "  !!! ТИХА ПОМИЛКА"
        )

        for field in silent_error_fields:
            print(
                f"      {field}"
            )

    return output


# ---------------------------------------------------------
# Робота зі старими results.json
# ---------------------------------------------------------

def load_old_results() -> dict:
    if not OUTPUT_PATH.exists():
        return {}

    try:
        data = json.loads(
            OUTPUT_PATH.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return {}

    if not isinstance(data, list):
        return {}

    result_map = {}

    for item in data:
        if not isinstance(item, dict):
            continue

        file_name = item.get("file")

        if file_name:
            result_map[
                file_name
            ] = item

    return result_map


def saved_result_is_successful(
    item: dict | None,
) -> bool:
    if not item:
        return False

    if (
        item.get("run_status")
        == "ok"
    ):
        return True

    # Сумісність зі старим results.json,
    # створеним попередньою версією eval.py.
    if (
        item.get("model")
        and item.get(
            "actual_document"
        )
        and "comparison" in item
    ):
        return True

    return False


def write_results(
    result_map: dict,
    entries: list,
):
    ordered_results = []

    for relative_path, _ in entries:
        if relative_path in result_map:
            ordered_results.append(
                result_map[
                    relative_path
                ]
            )

    OUTPUT_PATH.write_text(
        json.dumps(
            ordered_results,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------
# Зведення
# ---------------------------------------------------------

def build_summary(
    results: list[dict],
) -> dict:
    variants = defaultdict(
        lambda: {
            "files": 0,
            "correct": 0,
            "errors": 0,
            "omissions": 0,
            "hallucinations": 0,
            "caught_by_rules": 0,
            "uncaught_by_rules": 0,
            "silent_errors": 0,
            "silent_error_fields": 0,
            "decision_matches": 0,
            "checks_matches": 0,
            "prompt_tokens": 0,
            "critical_correct": 0,
            "critical_total": 0,
        }
    )

    decision_matrix = defaultdict(int)

    successful = 0
    failed = 0

    for item in results:
        if (
            item.get("run_status")
            != "ok"
        ):
            failed += 1
            continue

        if "comparison" not in item:
            continue

        successful += 1

        variant = (
            item.get("variant")
            or "unknown"
        )

        row = variants[
            variant
        ]

        row["files"] += 1

        counts = item[
            "comparison"
        ]["counts"]

        row["correct"] += (
            counts["correct"]
        )

        row["errors"] += (
            counts["error"]
        )

        row["omissions"] += (
            counts["omission"]
        )

        row[
            "hallucinations"
        ] += counts[
            "hallucination"
        ]

        row[
            "caught_by_rules"
        ] += len(
            item.get(
                "caught_by_rules",
                [],
            )
        )

        row[
            "uncaught_by_rules"
        ] += len(
            item.get(
                "uncaught_by_rules",
                [],
            )
        )

        if item.get(
            "silent_error"
        ):
            row[
                "silent_errors"
            ] += 1

        row[
            "silent_error_fields"
        ] += len(
            item.get(
                "silent_error_fields",
                [],
            )
        )

        if item.get(
            "decision_match"
        ):
            row[
                "decision_matches"
            ] += 1

        if item.get(
            "checks_match"
        ):
            row[
                "checks_matches"
            ] += 1

        usage = (
            item.get("usage")
            or {}
        )

        prompt_tokens = (
            usage.get(
                "prompt_tokens"
            )
        )

        if isinstance(
            prompt_tokens,
            int,
        ):
            row[
                "prompt_tokens"
            ] += prompt_tokens

        critical = item[
            "critical"
        ]

        row[
            "critical_correct"
        ] += critical[
            "correct"
        ]

        row[
            "critical_total"
        ] += critical[
            "total"
        ]

        matrix_key = (
            f"{item['expected_decision']}"
            f"->{item['actual_decision']}"
        )

        decision_matrix[
            matrix_key
        ] += 1

    for row in variants.values():
        if row[
            "critical_total"
        ]:
            row[
                "critical_accuracy"
            ] = (
                row[
                    "critical_correct"
                ]
                / row[
                    "critical_total"
                ]
            )
        else:
            row[
                "critical_accuracy"
            ] = None

        if row["files"]:
            row[
                "decision_accuracy"
            ] = (
                row[
                    "decision_matches"
                ]
                / row[
                    "files"
                ]
            )

            row[
                "checks_accuracy"
            ] = (
                row[
                    "checks_matches"
                ]
                / row[
                    "files"
                ]
            )
        else:
            row[
                "decision_accuracy"
            ] = None

            row[
                "checks_accuracy"
            ] = None

    return {
        "successful_files": successful,
        "failed_files": failed,
        "variants": dict(
            variants
        ),
        "decision_matrix": dict(
            decision_matrix
        ),
    }


def write_summary(
    result_map: dict,
    entries: list,
):
    ordered_results = []

    for relative_path, _ in entries:
        item = result_map.get(
            relative_path
        )

        if item:
            ordered_results.append(
                item
            )

    summary = build_summary(
        ordered_results
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    if not EXPECTED_PATH.exists():
        raise SystemExit(
            f"Не знайдено {EXPECTED_PATH}"
        )

    expected = json.loads(
        EXPECTED_PATH.read_text(
            encoding="utf-8"
        )
    )

    # Беремо clean + degraded + own.
    entries = [
        (
            relative_path,
            entry,
        )
        for (
            relative_path,
            entry,
        ) in expected.items()
        if (
            relative_path.startswith(
                "clean/"
            )
            or relative_path.startswith(
                "degraded/"
            )
            or relative_path.startswith(
                "own/"
            )
        )
    ]

    result_map = load_old_results()

    successful_before = sum(
        1
        for (
            relative_path,
            _,
        ) in entries
        if saved_result_is_successful(
            result_map.get(
                relative_path
            )
        )
    )

    print(
        f"Знайдено зразків: "
        f"{len(entries)}"
    )

    print(
        f"Вже успішно оброблено: "
        f"{successful_before}"
    )

    print(
        f"Залишилося: "
        f"{len(entries) - successful_before}"
    )

    stopped_by_api = False

    for number, (
        relative_path,
        entry,
    ) in enumerate(
        entries,
        start=1,
    ):
        old_result = result_map.get(
            relative_path
        )

        if saved_result_is_successful(
            old_result
        ):
            print(
                f"\n"
                f"[{number}/{len(entries)}] "
                f"ПРОПУСК — "
                f"вже є результат: "
                f"{relative_path}"
            )
            continue

        print(
            f"\n"
            f"[{number}/{len(entries)}] "
            f"НОВИЙ ЗАПИТ"
        )

        try:
            current_result = evaluate_file(
                relative_path,
                entry,
            )

        except KeyboardInterrupt:
            print(
                "\nЗупинено користувачем."
            )

            write_results(
                result_map,
                entries,
            )

            write_summary(
                result_map,
                entries,
            )

            return

        except Exception as exc:
            print(
                f"  ПОМИЛКА: {exc}"
            )

            current_result = {
                "file": relative_path,
                "source": entry.get(
                    "source"
                ),
                "variant": entry.get(
                    "variant"
                ),
                "run_status": "exception",
                "error": str(exc),
            }

        # Зберігаємо результат після КОЖНОГО файла.
        # model_error не вважається успішним,
        # тому наступний запуск повторить саме цей файл.
        result_map[
            relative_path
        ] = current_result

        write_results(
            result_map,
            entries,
        )

        write_summary(
            result_map,
            entries,
        )

        # При помилці API/моделі НЕ робимо повторний запит.
        # Це важливо для обмеженої добової квоти.
        if (
            current_result.get(
                "run_status"
            )
            == "model_error"
        ):
            print(
                "\nAPI/модель не відповіла."
            )

            print(
                "Повторний запит зараз НЕ виконується, "
                "щоб не витрачати квоту."
            )

            print(
                "Коли квота буде доступна, "
                "просто запусти:"
            )

            print(
                "python eval/eval.py"
            )

            stopped_by_api = True
            break

        # Пауза тільки після успішного запиту.
        if (
            current_result.get(
                "run_status"
            )
            == "ok"
        ):
            remaining_exists = any(
                not saved_result_is_successful(
                    result_map.get(
                        next_path
                    )
                )
                for (
                    next_path,
                    _,
                ) in entries[
                    number:
                ]
            )

            if remaining_exists:
                print(
                    f"  Пауза "
                    f"{REQUEST_DELAY} с..."
                )

                time.sleep(
                    REQUEST_DELAY
                )

    # Фінальне збереження.
    write_results(
        result_map,
        entries,
    )

    write_summary(
        result_map,
        entries,
    )

    successful_now = sum(
        1
        for (
            relative_path,
            _,
        ) in entries
        if saved_result_is_successful(
            result_map.get(
                relative_path
            )
        )
    )

    print(
        "\n=============================="
    )

    if (
        successful_now
        == len(entries)
    ):
        print(
            "ГОТОВО"
        )
    else:
        print(
            "ПОТОЧНИЙ РЕЗУЛЬТАТ"
        )

    print(
        "=============================="
    )

    print(
        f"Успішно: "
        f"{successful_now}/"
        f"{len(entries)}"
    )

    print(
        f"Результати: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Зведення:   "
        f"{SUMMARY_PATH}"
    )

    if stopped_by_api:
        print(
            "\nКоли ліміт відновиться, "
            "запусти цей самий скрипт ще раз."
        )


if __name__ == "__main__":
    main()
