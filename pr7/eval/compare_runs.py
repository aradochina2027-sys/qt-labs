from pathlib import Path

code = r'''"""Порівняння двох прогонів PR7: baseline 1600 vs experiment 1200."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

BASELINE = ROOT / "eval" / "baseline" / "summary.json"
EXPERIMENT = ROOT / "eval" / "experiment-1200" / "summary.json"

OUTPUT_JSON = ROOT / "eval" / "comparison.json"
OUTPUT_MD = ROOT / "eval" / "comparison.md"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Не знайдено файл: {path}")

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def aggregate(summary: dict) -> dict:
    totals = {
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

    for row in (
        summary.get("variants", {})
        or {}
    ).values():
        for key in totals:
            value = row.get(key, 0)

            if isinstance(value, (int, float)):
                totals[key] += value

    field_total = (
        totals["correct"]
        + totals["errors"]
        + totals["omissions"]
        + totals["hallucinations"]
    )

    totals["field_total"] = field_total

    totals["field_accuracy"] = (
        totals["correct"] / field_total
        if field_total
        else None
    )

    totals["critical_accuracy"] = (
        totals["critical_correct"]
        / totals["critical_total"]
        if totals["critical_total"]
        else None
    )

    totals["decision_accuracy"] = (
        totals["decision_matches"]
        / totals["files"]
        if totals["files"]
        else None
    )

    totals["checks_accuracy"] = (
        totals["checks_matches"]
        / totals["files"]
        if totals["files"]
        else None
    )

    totals["successful_files"] = summary.get(
        "successful_files",
        0,
    )

    totals["failed_files"] = summary.get(
        "failed_files",
        0,
    )

    return totals


def pct(value):
    if value is None:
        return "—"

    return f"{value * 100:.2f}%"


def diff(new, old):
    if (
        isinstance(new, (int, float))
        and isinstance(old, (int, float))
    ):
        return new - old

    return None


def main():
    baseline_summary = load_json(
        BASELINE
    )

    experiment_summary = load_json(
        EXPERIMENT
    )

    baseline = aggregate(
        baseline_summary
    )

    experiment = aggregate(
        experiment_summary
    )

    keys = [
        "successful_files",
        "failed_files",
        "correct",
        "errors",
        "omissions",
        "hallucinations",
        "silent_errors",
        "silent_error_fields",
        "caught_by_rules",
        "uncaught_by_rules",
        "prompt_tokens",
        "critical_correct",
        "critical_total",
        "decision_matches",
        "checks_matches",
    ]

    result = {
        "baseline_1600": baseline,
        "experiment_1200": experiment,
        "difference_1200_minus_1600": {
            key: diff(
                experiment.get(key),
                baseline.get(key),
            )
            for key in keys
        },
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Порівняння експериментів PR7",
        "",
        "| Показник | 1600 | 1200 | Різниця |",
        "|---|---:|---:|---:|",
        (
            f"| Успішно оброблено файлів | "
            f"{baseline['successful_files']} | "
            f"{experiment['successful_files']} | "
            f"{experiment['successful_files'] - baseline['successful_files']} |"
        ),
        (
            f"| Правильні поля | "
            f"{baseline['correct']} | "
            f"{experiment['correct']} | "
            f"{experiment['correct'] - baseline['correct']} |"
        ),
        (
            f"| Помилки | "
            f"{baseline['errors']} | "
            f"{experiment['errors']} | "
            f"{experiment['errors'] - baseline['errors']} |"
        ),
        (
            f"| Пропуски | "
            f"{baseline['omissions']} | "
            f"{experiment['omissions']} | "
            f"{experiment['omissions'] - baseline['omissions']} |"
        ),
        (
            f"| Вигадки | "
            f"{baseline['hallucinations']} | "
            f"{experiment['hallucinations']} | "
            f"{experiment['hallucinations'] - baseline['hallucinations']} |"
        ),
        (
            f"| Тихі помилки (документи) | "
            f"{baseline['silent_errors']} | "
            f"{experiment['silent_errors']} | "
            f"{experiment['silent_errors'] - baseline['silent_errors']} |"
        ),
        (
            f"| Тихі помилки (поля) | "
            f"{baseline['silent_error_fields']} | "
            f"{experiment['silent_error_fields']} | "
            f"{experiment['silent_error_fields'] - baseline['silent_error_fields']} |"
        ),
        (
            f"| Точність полів | "
            f"{pct(baseline['field_accuracy'])} | "
            f"{pct(experiment['field_accuracy'])} | "
            f"— |"
        ),
        (
            f"| Точність критичних полів | "
            f"{pct(baseline['critical_accuracy'])} | "
            f"{pct(experiment['critical_accuracy'])} | "
            f"— |"
        ),
        (
            f"| Правильність рішень | "
            f"{pct(baseline['decision_accuracy'])} | "
            f"{pct(experiment['decision_accuracy'])} | "
            f"— |"
        ),
        (
            f"| Правильність правил | "
            f"{pct(baseline['checks_accuracy'])} | "
            f"{pct(experiment['checks_accuracy'])} | "
            f"— |"
        ),
        (
            f"| Prompt tokens | "
            f"{baseline['prompt_tokens']} | "
            f"{experiment['prompt_tokens']} | "
            f"{experiment['prompt_tokens'] - baseline['prompt_tokens']} |"
        ),
        "",
        "## Короткий висновок",
        "",
        "Після завершення обох прогонів використайте таблицю вище, "
        "щоб оцінити, як зменшення IMAGE_MAX_SIDE з 1600 до 1200 "
        "вплинуло на точність, критичні поля, рішення, тихі помилки "
        "та витрати токенів.",
        "",
    ]

    OUTPUT_MD.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("Готово.")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"Markdown: {OUTPUT_MD}")

    print()
    print(
        "Точність полів: "
        f"1600={pct(baseline['field_accuracy'])}, "
        f"1200={pct(experiment['field_accuracy'])}"
    )

    print(
        "Критичні поля: "
        f"1600={pct(baseline['critical_accuracy'])}, "
        f"1200={pct(experiment['critical_accuracy'])}"
    )

    print(
        "Тихі помилки: "
        f"1600={baseline['silent_errors']}, "
        f"1200={experiment['silent_errors']}"
    )


if __name__ == "__main__":
    main()
'''

path = Path("/mnt/data/compare_runs.py")
path.write_text(code, encoding="utf-8")
print(path)
