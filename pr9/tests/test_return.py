
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from app import workflow, store
from shop import service


ORDER_ID = "10412"
CUSTOMER_ID = "C-1003"
PRODUCT_SKU = "SIR-R10"
PRODUCT_NAME = "Робот-пилосос «Сіріус R10»"


def fake_complete(messages, schema=None):
    """Тестова AI-модель без реальних API-запитів."""

    usage = {
        "prompt_tokens": 5,
        "completion_tokens": 5,
        "total_tokens": 10,
    }

    if schema is None:
        return {
            "text": "Ваше звернення щодо повернення оброблено.",
            "usage": usage,
        }

    return {
        "text": json.dumps(
            {
                "kind": "return",
                "order_id": ORDER_ID,
                "item": PRODUCT_SKU,
                "reason": "не підійшов",
            },
            ensure_ascii=False,
        ),
        "usage": usage,
    }


def main():
    print("=" * 50)
    print("ПР9 — ПЕРЕВІРКА ПОВЕРНЕННЯ ТОВАРУ")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as temp_dir:
        with patch.object(store, "RUNS_DIR", Path(temp_dir)):
            with patch.object(
                workflow.llm,
                "complete",
                side_effect=fake_complete,
            ):
                service.reset()

                print("\n=== TEST 1: ORDER ===")

                order = service.get_order(ORDER_ID)

                assert order["customer_id"] == CUSTOMER_ID
                assert order["status"] == "delivered"

                product = service.get_product(PRODUCT_SKU)

                assert product["returnable"] is True

                print("ORDER:", ORDER_ID)
                print("CUSTOMER:", CUSTOMER_ID)
                print("PRODUCT:", PRODUCT_NAME)
                print("RETURNABLE:", product["returnable"])

                print("\n=== TEST 2: START WORKFLOW ===")

                run = workflow.start(
                    "Хочу повернути робот-пилосос "
                    "Сіріус R10 із замовлення 10412, "
                    "тому що він мені не підійшов.",
                    CUSTOMER_ID,
                )

                print("ROUTE:", run.route)
                print("STATUS:", run.status)
                print("REPLY:", run.reply)
                print("ERROR:", run.stopped)
                print("PENDING:", run.pending)

                assert run.route == "return"
                assert run.status == "awaiting_approval"

                print("START TEST: OK")

                print("\n=== TEST 3: BEFORE APPROVAL ===")

                before = service.list_returns(ORDER_ID)

                print("RETURNS BEFORE:", len(before))

                assert len(before) == 0

                print("NO ACTION BEFORE APPROVAL: OK")

                print("\n=== TEST 4: APPROVE ===")

                approved = workflow.resume(
                    run.run_id,
                    True,
                    "Повернення підтверджено оператором",
                )

                print("STATUS:", approved.status)
                print("REPLY:", approved.reply)
                print("ERROR:", approved.stopped)

                assert approved.status == "done"

                after = service.list_returns(ORDER_ID)

                print("RETURNS AFTER:", len(after))

                assert len(after) == 1
                assert after[0]["sku"] == PRODUCT_SKU

                print("RETURN CREATED: OK")

                print("\n=== TEST 5: REPEATED APPROVAL ===")

                repeated = workflow.resume(
                    run.run_id,
                    True,
                )

                after_repeat = service.list_returns(ORDER_ID)

                print("STATUS:", repeated.status)
                print("RETURNS:", len(after_repeat))

                assert repeated.status == "done"
                assert len(after_repeat) == 1

                print("DUPLICATE PROTECTION: OK")

                print("\n=== TEST 6: STORAGE ===")

                restored = store.load(run.run_id)

                assert restored is not None
                assert restored.status == "done"
                assert restored.run_id == run.run_id

                print("RESTORED STATUS:", restored.status)
                print("STORAGE: OK")

                print("\n" + "=" * 50)
                print("ALL RETURN TESTS PASSED")
                print("=" * 50)


if __name__ == "__main__":
    main()
