
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from app import workflow, store
from shop import service


def fake_complete(messages, schema=None):
    usage = {
        "prompt_tokens": 5,
        "completion_tokens": 5,
        "total_tokens": 10,
    }

    if schema is None:
        return {
            "text": "Ваше звернення оброблено.",
            "usage": usage,
        }

    return {
        "text": json.dumps({
            "kind": "status",
            "order_id": "10458",
            "item": None,
            "reason": None,
        }),
        "usage": usage,
    }


def main():
    print("=" * 50)
    print("ПР9 — ПЕРЕВІРКА ЗБОЇВ СЕРВІСУ")
    print("=" * 50)

    results = []

    with tempfile.TemporaryDirectory() as directory:
        with patch.object(store, "RUNS_DIR", Path(directory)):
            with patch.object(
                workflow.llm,
                "complete",
                side_effect=fake_complete,
            ):

                with patch.dict(
                    os.environ,
                    {"SHOP_FAILURE_RATE": "0.3"},
                ):
                    service.reset()

                    for number in range(1, 11):
                        print(f"\n=== REQUEST {number}/10 ===")

                        try:
                            run = workflow.start(
                                "Який стан замовлення 10458?",
                                "C-1001",
                            )

                            print("STATUS:", run.status)
                            print("ERROR:", run.stopped)

                            results.append({
                                "number": number,
                                "status": run.status,
                                "error": run.stopped,
                            })

                            assert run.status in (
                                "done",
                                "failed",
                                "stopped",
                                "handed_off",
                                "needs_info",
                            )

                        except Exception as error:
                            print("UNHANDLED ERROR:", error)
                            results.append({
                                "number": number,
                                "status": "exception",
                                "error": str(error),
                            })

    failures = sum(
        1 for item in results
        if item["status"] != "done"
    )

    print("\n" + "=" * 50)
    print("TOTAL:", len(results))
    print("DONE:", len(results) - failures)
    print("NOT DONE:", failures)
    print("=" * 50)

    assert len(results) == 10
    assert all(
        item["status"] != "exception"
        for item in results
    )

    print("\nFAILURE HANDLING TEST: OK")


if __name__ == "__main__":
    main()
