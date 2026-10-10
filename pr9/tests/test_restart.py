
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from app import store, workflow
from shop import service


def fake_complete(messages, schema=None):
    usage = {
        "prompt_tokens": 5,
        "completion_tokens": 5,
        "total_tokens": 10,
    }

    if schema is None:
        return {
            "text": "Звернення оброблено.",
            "usage": usage,
        }

    return {
        "text": json.dumps({
            "kind": "cancel",
            "order_id": "10463",
            "item": None,
            "reason": None,
        }),
        "usage": usage,
    }


def main():
    print("=== RESTART TEST ===")

    service.reset()

    with tempfile.TemporaryDirectory() as directory:
        state_dir = Path(directory)

        with patch.object(store, "RUNS_DIR", state_dir):
            with patch.object(
                workflow.llm,
                "complete",
                side_effect=fake_complete,
            ):
                run = workflow.start(
                    "Хочу скасувати замовлення 10463",
                    "C-1001",
                )

                print("BEFORE RESTART:", run.status)
                assert run.status == "awaiting_approval"

                run_id = run.run_id

        # Інший процес Python читає збережений стан.
        check_code = """
import sys
from pathlib import Path
from app import store

store.RUNS_DIR = Path(sys.argv[1])
run = store.load(sys.argv[2])

assert run is not None
assert run.status == 'awaiting_approval'

print('AFTER RESTART:', run.status)
"""

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                check_code,
                str(state_dir),
                run_id,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )

        print(result.stdout.strip())

        with patch.object(store, "RUNS_DIR", state_dir):
            approved = workflow.resume(
                run_id,
                True,
                "Підтверджено після перезапуску",
            )

            print("AFTER APPROVAL:", approved.status)
            assert approved.status == "done"

            repeated = workflow.resume(run_id, True)
            assert repeated.status == "done"

        print("\nRESTART TEST: OK")


if __name__ == "__main__":
    main()
