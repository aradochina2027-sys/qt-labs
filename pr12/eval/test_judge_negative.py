
"""ПР12: негативний контроль AI-судді."""

import json
from datetime import datetime, timezone
from pathlib import Path

import judge

HERE = Path(__file__).resolve().parent

CASES = HERE / "cases" / "drafts.json"
SOURCE = HERE / "runs" / "drafts-mock-checks-fixed.json"
RESULT = HERE / "runs" / "judge-negative-control.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    cases = read_json(CASES)
    source = read_json(SOURCE)

    case = next(
        item for item in cases
        if item["id"] == "d02"
    )

    saved = next(
        item for item in source["results"]
        if item["id"] == "d02"
    )

    output = dict(saved["output"])

    # Навмисно неправильна відповідь:
    # повернення заборонене правилами, але
    # текст підтверджує його та обіцяє компенсацію.
    output["text"] = (
        "Добрий день! Повернення фільтрів "
        "із замовлення 10412 схвалено. "
        "Ми приймемо обидві упаковки, хоча "
        "правила магазину це забороняють. "
        "Також гарантовано виплатимо "
        "компенсацію 500 гривень. "
        "Погодження керівника не потрібне."
    )

    criteria = [
        "refuses_properly",
        "no_extra_promises",
    ]

    print("Case: d02")
    print("Control: intentionally incorrect answer")
    print("Judge model:", judge.JUDGE_MODEL)
    print()

    verdict = judge.evaluate(
        case,
        output,
        criteria,
    )

    expected = {
        "refuses_properly": False,
        "no_extra_promises": False,
    }

    rows = []

    for item in verdict.criteria:
        correct = item.passed == expected[item.id]

        rows.append({
            "id": item.id,
            "expected": expected[item.id],
            "actual": item.passed,
            "correct": correct,
            "reason": item.reason,
        })

        print(
            f"{item.id}: "
            f"expected={expected[item.id]}, "
            f"actual={item.passed}, "
            f"correct={correct}"
        )

        print("Reason:", item.reason)
        print()

    success = all(row["correct"] for row in rows)

    result = {
        "label": "negative-judge-control",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case_id": "d02",
        "source_run": str(SOURCE),
        "generator_rerun": False,
        "model": verdict.model,
        "intentionally_corrupted": True,
        "answer": output["text"],
        "criteria": rows,
        "judge_usage": verdict.usage,
        "passed": success,
    }

    RESULT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("=" * 50)
    print("NEGATIVE CONTROL:", "PASS" if success else "FAIL")
    print("Saved:", RESULT)


if __name__ == "__main__":
    main()
