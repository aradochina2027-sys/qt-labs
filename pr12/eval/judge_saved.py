
"""ПР12: оцінювання збережених відповідей без повторної генерації."""

import argparse
import copy
import json
from pathlib import Path

import judge
import report

HERE = Path(__file__).resolve().parent

SOURCE = HERE / "runs" / "drafts-mock-checks-fixed.json"
CASES = HERE / "cases" / "drafts.json"
LABELS = HERE / "labels-human.json"
DEST = HERE / "runs" / "drafts-calibration-judged.json"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(data):
    temp = DEST.with_suffix(".tmp")
    temp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(DEST)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()

    source = load(SOURCE)
    labels = load(LABELS)

    cases = {
        case["id"]: case
        for case in load(CASES)
    }

    original = {
        f"{r['id']}#{r['attempt']}": r
        for r in source["results"]
    }

    if DEST.exists():
        previous = load(DEST)
        completed = {
            f"{r['id']}#{r['attempt']}": r
            for r in previous["results"]
            if any(
                check["name"] == "judge"
                and check.get("data", {}).get("criteria")
                for check in r["checks"]
            )
        }
    else:
        completed = {}

    def save_progress():
        rows = [
            completed[key]
            for key in labels
            if key in completed
        ]

        result = {
            "meta": {
                **source["meta"],
                "label": "calibration-saved-answers",
                "judge": True,
                "judge_model": judge.JUDGE_MODEL,
                "source_run": str(SOURCE),
                "partial": len(rows) < len(labels),
                "note": (
                    "Відповіді взяті зі збереженого mock-прогону; "
                    "повторної генерації не було."
                ),
            },
            "cases": [
                {
                    "id": key.split("#")[0],
                    "kind": cases[key.split("#")[0]]["kind"],
                    "about": cases[key.split("#")[0]].get("about", ""),
                }
                for key in labels
            ],
            "results": rows,
        }

        result["summary"] = report.summarize(result)
        save(result)

    calls = 0

    for key, human in labels.items():
        if key in completed:
            continue

        if calls >= args.limit:
            break

        if key not in original:
            raise SystemExit(f"No original result: {key}")

        case_id = key.split("#")[0]
        case = cases[case_id]
        original_row = original[key]

        output = original_row["output"]

        if output.get("text") != human["text"]:
            raise SystemExit(
                f"Text mismatch for {key}. "
                "The original answer was changed."
            )

        criteria_ids = list(human["criteria"])

        if not all(
            isinstance(value, bool)
            for value in human["criteria"].values()
        ):
            raise SystemExit(
                f"Human labels not completed: {key}"
            )

        print(f"Judging {key}...", flush=True)

        try:
            verdict = judge.evaluate(
                case,
                output,
                criteria_ids,
            )
        except Exception as exc:
            print(
                f"Judge error for {key}: "
                f"{type(exc).__name__}: {exc}"
            )
            print("Previous results remain saved.")
            return

        judge_check = {
            "name": "judge",
            "passed": verdict.overall,
            "detail": "; ".join(
                f"{c.id}: {c.reason}"
                for c in verdict.criteria
                if not c.passed
            ),
            "data": {
                "criteria": [
                    {
                        "id": c.id,
                        "passed": c.passed,
                        "reason": c.reason,
                    }
                    for c in verdict.criteria
                ],
                "overall": verdict.overall,
                "model": verdict.model,
                "usage": verdict.usage,
            },
        }

        row = copy.deepcopy(original_row)

        row["checks"] = [
            c for c in row["checks"]
            if c["name"] != "judge"
        ]

        row["checks"].append(judge_check)

        row["passed"] = all(
            c["passed"]
            for c in row["checks"]
        )

        completed[key] = row
        calls += 1

        save_progress()

        print(
            f"  Verdict: {verdict.overall}; "
            f"model: {verdict.model}"
        )

        for criterion in verdict.criteria:
            print(
                f"  {criterion.id}: "
                f"{criterion.passed}"
            )

    print()
    print(f"Completed: {len(completed)}/{len(labels)}")
    print(f"New judge calls: {calls}")
    print(f"Saved: {DEST}")


if __name__ == "__main__":
    main()
