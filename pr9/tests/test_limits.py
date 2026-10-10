
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from app import workflow, store
from shop import service


def fake_complete(messages, schema=None):
    usage = {
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "total_tokens": 30,
    }

    if schema is None:
        return {
            "text": "Замовлення перевірено.",
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


def show_result(label, run):
    print(f"\n=== {label} ===")
    print("STATUS:", run.status)
    print("REASON:", run.stopped)
    print("STEPS:", len(run.steps))
    print("TOKENS:", run.usage)


def main():
    with tempfile.TemporaryDirectory() as directory:
        with patch.object(store, "RUNS_DIR", Path(directory)):
            with patch.object(
                workflow.llm,
                "complete",
                side_effect=fake_complete,
            ):
                service.reset()

                print("ПР9 — ПЕРЕВІРКА ОБМЕЖЕНЬ")

                # Тест 1: нормальна обробка
                normal = workflow.start(
                    "Де моє замовлення 10458?",
                    "C-1001",
                )
                show_result("NORMAL", normal)

                assert normal.status == "done"

                # Тест 2: ліміт кроків
                with patch.object(workflow, "MAX_STEPS", 1):
                    steps = workflow.start(
                        "Де моє замовлення 10458?",
                        "C-1001",
                    )

                show_result("MAX STEPS = 1", steps)
                assert steps.status in ("stopped", "failed")
                assert steps.stopped

                # Тест 3: бюджет токенів
                with patch.object(workflow, "TOKEN_BUDGET", 1):
                    tokens = workflow.start(
                        "Де моє замовлення 10458?",
                        "C-1001",
                    )

                show_result("TOKEN BUDGET = 1", tokens)
                assert tokens.status in ("stopped", "failed")
                assert tokens.stopped

                # Тест 4: нульовий час
                with patch.object(workflow, "TIMEOUT", 0):
                    timeout = workflow.start(
                        "Де моє замовлення 10458?",
                        "C-1001",
                    )

                show_result("TIMEOUT = 0", timeout)
                assert timeout.status in ("stopped", "failed")
                assert timeout.stopped

                print("\nALL LIMIT TESTS PASSED")


if __name__ == "__main__":
    main()
