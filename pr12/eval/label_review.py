
"""ПР12: ручне оцінювання відповідей для калібрування судді."""

import json
from pathlib import Path

PATH = Path(__file__).resolve().parent / "labels-human.json"


def save(data):
    PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    if not PATH.exists():
        print("ERROR: labels-human.json not found")
        return

    data = json.loads(PATH.read_text(encoding="utf-8"))

    total = sum(
        len(item["criteria"])
        for item in data.values()
    )

    completed = sum(
        isinstance(value, bool)
        for item in data.values()
        for value in item["criteria"].values()
    )

    print("=" * 65)
    print("PR12 - HUMAN JUDGE CALIBRATION")
    print(f"Cases: {len(data)}")
    print(f"Criteria: {total}")
    print(f"Already completed: {completed}")
    print("=" * 65)

    for case_id, item in data.items():
        pending = [
            criterion
            for criterion, value in item["criteria"].items()
            if not isinstance(value, bool)
        ]

        if not pending:
            continue

        print("\n" + "=" * 65)
        print("CASE:", case_id)
        print("TYPE:", item["kind"])
        print("PURPOSE:", item["about"])

        print("\nCUSTOMER MESSAGE:")
        print(item["question"])

        print("\nFACTS:")
        facts = item.get("facts", [])

        if isinstance(facts, list):
            for fact in facts:
                print("-", fact)
        else:
            print(facts)

        print("\nAI ANSWER:")
        print(item["text"])

        print("\n" + "-" * 65)

        for criterion in pending:
            description = item["rubric"][criterion]

            print("\nCRITERION:", criterion)
            print("RULE:", description)

            while True:
                choice = input(
                    "Pass? [y=yes / n=no / q=quit]: "
                ).strip().lower()

                if choice in ("y", "yes", "1"):
                    item["criteria"][criterion] = True
                    save(data)
                    break

                if choice in ("n", "no", "0"):
                    item["criteria"][criterion] = False
                    save(data)
                    break

                if choice in ("q", "quit"):
                    print("Progress saved.")
                    return

                print("Enter y, n or q.")

        print("\nCASE COMPLETED:", case_id)

    completed = sum(
        isinstance(value, bool)
        for item in data.values()
        for value in item["criteria"].values()
    )

    print("\n" + "=" * 65)
    print("CALIBRATION LABELS COMPLETED")
    print(f"Cases: {len(data)}")
    print(f"Criteria completed: {completed}/{total}")
    print("Saved to:", PATH)
    print("=" * 65)


if __name__ == "__main__":
    main()
