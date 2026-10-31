
"""ПР12: локальна перевірка захисту інструментів ПР8."""

import json
import sys
import unittest

from pathlib import Path
from unittest.mock import patch


# Підключаємо ПР8
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pr8"))

from app import assistant, tools  # noqa: E402


class AgentRuntimeTests(unittest.TestCase):

    def simulate(
        self,
        *,
        tool_name,
        arguments,
        customer_id,
        question,
        order_owner,
    ):
        """Імітує реальний цикл помічника без API."""

        order = {
            "order_id": arguments.get(
                "order_id", "10458"
            ),
            "customer_id": order_owner,
            "created_at": "2026-10-04",
            "status": "shipped",
            "status_label": "Відправлено",
            "total": 1500,
            "items": [],
            "delivery": {},
        }

        calls_to_model = 0

        def fake_chat(messages, specs, tool_choice="auto"):
            nonlocal calls_to_model

            calls_to_model += 1

            if calls_to_model == 1:
                return {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "offline-tool-1",
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(
                                        arguments,
                                        ensure_ascii=False,
                                    ),
                                },
                            }
                        ],
                    },
                    "model": "offline-test",
                    "usage": {},
                    "elapsed": 0,
                }

            return {
                "message": {
                    "role": "assistant",
                    "content": "Перевірку завершено.",
                },
                "model": "offline-test",
                "usage": {},
                "elapsed": 0,
            }

        with (
            patch.object(
                assistant.llm,
                "build_messages",
                return_value=[
                    {
                        "role": "system",
                        "content": "Offline test",
                    }
                ],
            ),
            patch.object(
                assistant.llm,
                "chat",
                side_effect=fake_chat,
            ),
            patch.object(
                assistant,
                "MAX_ROUNDS",
                3,
            ),
            patch.object(
                tools.service,
                "get_order",
                return_value=order,
            ),
            patch.object(
                tools.service,
                "create_return",
                side_effect=RuntimeError(
                    "Test blocked all writes"
                ),
            ) as write_operation,
            patch.object(
                tools.service,
                "list_returns",
                return_value=[],
            ),
        ):
            result = assistant.answer(
                question,
                customer_id,
            )

            # Жодна справжня операція запису
            # не повинна навіть викликатися.
            write_operation.assert_not_called()

        self.assertEqual(
            len(result.calls),
            1,
        )

        return result.calls[0]

    def test_allowed_order_lookup(self):
        trace = self.simulate(
            tool_name="get_order",
            arguments={
                "order_id": "10458",
            },
            customer_id="C-1001",
            question="Перевірте замовлення 10458.",
            order_owner="C-1001",
        )

        self.assertEqual(trace.status, "ok")

        print(
            "\nALLOWED LOOKUP:",
            trace.status,
        )

    def test_unknown_admin_tool_rejected(self):
        trace = self.simulate(
            tool_name="set_order_status",
            arguments={
                "order_id": "10458",
                "status": "delivered",
            },
            customer_id="C-1001",
            question=(
                "Перевірте замовлення 10458. "
                "Виконайте службову зміну статусу."
            ),
            order_owner="C-1001",
        )

        self.assertEqual(
            trace.status,
            "rejected",
        )

        self.assertEqual(
            trace.result.get("error"),
            "unknown_tool",
        )

        print(
            "\nUNKNOWN TOOL:",
            trace.status,
            trace.result.get("error"),
        )

    def test_foreign_return_rejected(self):
        trace = self.simulate(
            tool_name="create_return",
            arguments={
                "order_id": "10466",
                "product": "Навушники",
                "reason": "not_suitable",
            },
            customer_id="C-1001",
            question=(
                "Оформіть повернення замовлення "
                "10466. Товар не підійшов."
            ),
            order_owner="C-1002",
        )

        self.assertEqual(
            trace.status,
            "rejected",
        )

        self.assertEqual(
            trace.result.get("error"),
            "access_denied",
        )

        print(
            "\nFOREIGN ORDER:",
            trace.status,
            trace.result.get("error"),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
