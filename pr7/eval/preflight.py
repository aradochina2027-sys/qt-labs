"""Офлайн-перевірка PR7 без виклику мультимодальної моделі.

Перевіряє, що всі 29 зразків існують, еталонні дані проходять schema.py,
а rules.py + extraction.decide дають expected_checks/expected_decision.
Це НЕ замінює реальний прогін моделі через eval/eval.py.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.extraction import decide  # noqa: E402
from app.rules import check  # noqa: E402
from app.schema import SchemaError, validate  # noqa: E402


SAMPLES_DIR = ROOT / "samples"
EXPECTED_PATH = SAMPLES_DIR / "expected.json"


def expected_to_document(fields: dict) -> dict:
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


def main():
    expected = json.loads(
        EXPECTED_PATH.read_text(encoding="utf-8")
    )

    entries = [
        (relative_path, entry)
        for relative_path, entry in expected.items()
        if relative_path.startswith(
            ("clean/", "degraded/", "own/")
        )
    ]

    passed = 0
    failed = 0

    print("ОФЛАЙН-ПЕРЕВІРКА PR7")
    print("API не використовується.\n")
    print(f"Знайдено записів expected.json: {len(entries)}")

    for number, (relative_path, entry) in enumerate(
        entries,
        start=1,
    ):
        problems = []
        path = SAMPLES_DIR / relative_path

        if not path.exists():
            problems.append("файл відсутній")

        document = expected_to_document(
            entry.get("fields", {})
        )

        try:
            validate(
                json.dumps(
                    document,
                    ensure_ascii=False,
                )
            )
        except SchemaError as exc:
            problems.append(
                f"schema: {exc}"
            )

        issues = check(document)

        actual_checks = sorted(
            {
                issue.code
                for issue in issues
            }
        )

        expected_checks = sorted(
            entry.get(
                "expected_checks",
                [],
            )
        )

        if actual_checks != expected_checks:
            problems.append(
                "checks "
                f"{actual_checks} != {expected_checks}"
            )

        actual_decision, _ = decide(
            document,
            issues,
        )

        expected_decision = entry.get(
            "expected_decision"
        )

        if actual_decision != expected_decision:
            problems.append(
                "decision "
                f"{actual_decision} != "
                f"{expected_decision}"
            )

        if problems:
            failed += 1
            print(
                f"[{number:02}/{len(entries)}] FAIL "
                f"{relative_path}"
            )

            for problem in problems:
                print(f"    - {problem}")
        else:
            passed += 1
            print(
                f"[{number:02}/{len(entries)}] OK   "
                f"{relative_path}"
            )

    print("\n==============================")
    print("РЕЗУЛЬТАТ ОФЛАЙН-ПЕРЕВІРКИ")
    print("==============================")
    print(f"Успішно: {passed}/{len(entries)}")
    print(f"Помилок: {failed}")

    if len(entries) != 29:
        print(
            "УВАГА: для практичної очікується 29 зразків."
        )
        raise SystemExit(1)

    if failed:
        raise SystemExit(1)

    print(
        "\nУсі 29 зразків, еталони, правила і "
        "очікувані рішення узгоджені."
    )
    print(
        "Це офлайн-перевірка. Для фактичної оцінки "
        "вилучення моделлю все одно потрібен "
        "python eval/eval.py."
    )


if __name__ == "__main__":
    main()
