
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from app import workflow, store
from shop import service


def fake_complete(messages, schema=None):
    """Підставна AI-модель без використання Gemini API."""

    text = str(messages).lower()

    if schema is None:
        return {
            "text": "Ваше звернення оброблено.",
            "usage": {"total_tokens": 10},
        }

    if "скасувати" in text:
        kind = "cancel"
        order_id = "10463"
        item = None
        reason = None

    elif "повернути" in text:
        kind = "return"
        order_id = "10421"
        item = "пилосос"
        reason = "не підійшов"

    else:
        kind = "status"
        order_id = "10458"
        item = None
        reason = None

    return {
        "text": json.dumps(
            {
                "kind": kind,
                "order_id": order_id,
                "item": item,
                "reason": reason,
            },
            ensure_ascii=False,
        ),
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 5,
            "total_tokens": 10,
        },
    }


def show(label, run):
    """Виведення результатів тестування."""

    print(
        label,
        "| route:", run.route,
        "| status:", run.status,
        "| error:", run.stopped,
        flush=True,
    )


def main():

    with tempfile.TemporaryDirectory() as temp_dir:

        with patch.object(
            store,
            "RUNS_DIR",
            Path(temp_dir),
        ):

            with patch.object(
                workflow.llm,
                "complete",
                side_effect=fake_complete,
            ):

                # ====================================
                # ТЕСТ 1. ПЕРЕВІРКА СТАНУ ЗАМОВЛЕННЯ
                # ====================================

                service.reset()

                print("\n=== TEST 1: STATUS ===")

                run = workflow.start(
                    "Де замовлення 10458?",
                    "C-1001",
                )

                show("STATUS", run)

                assert run.route == "status"
                assert run.status == "done"

                print("STATUS TEST: OK")

                # ====================================
                # ТЕСТ 2. СКАСУВАННЯ ЗАМОВЛЕННЯ
                # ====================================

                service.reset()

                print("\n=== TEST 2: CANCEL ===")

                run = workflow.start(
                    "Хочу скасувати замовлення 10463",
                    "C-1001",
                )

                show("BEFORE APPROVAL", run)

                assert run.route == "cancel"
                assert run.status == "awaiting_approval"

                approved = workflow.resume(
                    run.run_id,
                    True,
                    "Підтверджено оператором",
                )

                show("AFTER APPROVAL", approved)

                assert approved.status == "done"

                # ====================================
                # ТЕСТ 3. ПОВТОРНЕ ПІДТВЕРДЖЕННЯ
                # ====================================

                print("\n=== TEST 3: REPEATED APPROVAL ===")

                repeated = workflow.resume(
                    run.run_id,
                    True,
                )

                show("REPEATED", repeated)

                assert repeated.status == approved.status

                print("REPEATED APPROVAL TEST: OK")

                # ====================================
                # ТЕСТ 4. ВІДХИЛЕННЯ ОПЕРАТОРОМ
                # ====================================

                service.reset()

                print("\n=== TEST 4: REJECT ===")

                run = workflow.start(
                    "Хочу скасувати замовлення 10463",
                    "C-1001",
                )

                show("BEFORE REJECTION", run)

                assert run.status == "awaiting_approval"

                rejected = workflow.resume(
                    run.run_id,
                    False,
                    "Відхилено оператором",
                )

                show("REJECTED", rejected)

                assert rejected.status == "rejected"

                print("REJECTION TEST: OK")

                # ====================================
                # ТЕСТ 5. ЗБЕРЕЖЕННЯ СТАНУ
                # ====================================

                print("\n=== TEST 5: STORE ===")

                saved_runs = store.list_runs()

                print("SAVED RUNS:", len(saved_runs))

                assert len(saved_runs) == 3

                for saved_run in saved_runs:
                    restored = store.load(saved_run.run_id)

                    assert restored is not None
                    assert restored.run_id == saved_run.run_id

                print("STORE TEST: OK")

                print("\n" + "=" * 45)
                print("ALL LOCAL TESTS PASSED")
                print("=" * 45)


if __name__ == "__main__":
    main()
