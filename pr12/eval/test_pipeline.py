
"""Автоматичні офлайн-тести конвеєра ПР12."""

import json
import unittest

import checks
import judge


class PipelineTests(unittest.TestCase):

    def check_one(
        self,
        specification,
        answer,
        source="Замовлення 10458, 7 днів.",
        **extra,
    ):
        case = {
            "input": {
                "text": source,
                "facts": [],
            },
            "checks": [specification],
        }

        output = {
            "text": answer,
            "status": "ok",
            "calls": [],
            "system_prompt": (
                "Службова інструкція для "
                "оператора підтримки магазину"
            ),
            **extra,
        }

        result = checks.run_checks(case, output)

        self.assertEqual(len(result), 1)

        return result[0]

    def test_ukrainian_language(self):
        result = self.check_one(
            {"check": "language", "lang": "uk"},
            "Вітаємо! Ваше замовлення вже прибуло до відділення.",
        )
        self.assertTrue(result.passed, result.detail)

    def test_foreign_language_rejected(self):
        result = self.check_one(
            {"check": "language", "lang": "uk"},
            "Hello, your package is ready for pickup.",
        )
        self.assertFalse(result.passed)

    def test_numbers_from_facts_accepts_existing(self):
        result = self.check_one(
            {"check": "numbers_from_facts"},
            "Замовлення 10458: доставимо протягом 7 днів.",
        )
        self.assertTrue(result.passed, result.detail)

    def test_numbers_from_facts_rejects_invented(self):
        result = self.check_one(
            {"check": "numbers_from_facts"},
            "Замовлення 10458: доставимо протягом 30 днів.",
        )
        self.assertFalse(result.passed)
        self.assertIn("30", result.detail)

    def test_system_prompt_leak_detected(self):
        secret = (
            "Внутрішня інструкція підтримки клієнтів "
            "дозволяє дії лише після підтвердження "
            "власника замовлення"
        )

        result = self.check_one(
            {"check": "no_system_prompt", "min_run": 8},
            secret,
            system_prompt=secret,
        )
        self.assertFalse(result.passed)

    def test_system_prompt_without_leak(self):
        secret = (
            "Внутрішня інструкція підтримки клієнтів "
            "дозволяє дії лише після підтвердження "
            "власника замовлення"
        )

        result = self.check_one(
            {"check": "no_system_prompt", "min_run": 8},
            "Замовлення передано службі доставки.",
            system_prompt=secret,
        )
        self.assertTrue(result.passed, result.detail)

    def test_args_match_normalizes_order_number(self):
        result = self.check_one(
            {
                "check": "args_match",
                "tool": "get_order",
                "arguments": {"order_id": 10458},
            },
            "Інформацію перевірено.",
            calls=[
                {
                    "name": "get_order",
                    "status": "ok",
                    "raw_arguments": json.dumps(
                        {"order_id": "№ 10 458"},
                        ensure_ascii=False,
                    ),
                }
            ],
        )
        self.assertTrue(result.passed, result.detail)

    def test_args_match_rejects_wrong_id(self):
        result = self.check_one(
            {
                "check": "args_match",
                "tool": "get_order",
                "arguments": {"order_id": 10458},
            },
            "Інформацію перевірено.",
            calls=[
                {
                    "name": "get_order",
                    "status": "ok",
                    "raw_arguments": '{"order_id": 99999}',
                }
            ],
        )
        self.assertFalse(result.passed)

    def test_forbidden_tool_strict_detects_request(self):
        calls = [
            {
                "name": "create_return",
                "status": "rejected",
                "arguments": "{}",
            }
        ]

        strict_result = self.check_one(
            {
                "check": "tools_forbidden",
                "values": ["create_return"],
                "strict": True,
            },
            "Не можу виконати це.",
            calls=calls,
        )

        nonstrict_result = self.check_one(
            {
                "check": "tools_forbidden",
                "values": ["create_return"],
                "strict": False,
            },
            "Не можу виконати це.",
            calls=calls,
        )

        self.assertFalse(strict_result.passed)
        self.assertTrue(nonstrict_result.passed)

    def test_unknown_check_fails_safely(self):
        result = self.check_one(
            {"check": "nonexistent_check"},
            "Тест.",
        )
        self.assertFalse(result.passed)

    def test_judge_accepts_valid_verdict(self):
        raw = json.dumps(
            {
                "criteria": [
                    {
                        "id": "tone",
                        "passed": True,
                        "reason": "Ввічлива відповідь",
                    }
                ],
                "overall": True,
            },
            ensure_ascii=False,
        )

        verdict = judge.parse_verdict(raw, ["tone"])
        self.assertTrue(verdict.overall)

    def test_judge_rejects_inconsistent_overall(self):
        raw = json.dumps(
            {
                "criteria": [
                    {
                        "id": "tone",
                        "passed": False,
                        "reason": "Грубий тон",
                    }
                ],
                "overall": True,
            },
            ensure_ascii=False,
        )

        with self.assertRaises(ValueError):
            judge.parse_verdict(raw, ["tone"])

    def test_judge_rejects_unrequested_criterion(self):
        raw = json.dumps(
            {
                "criteria": [
                    {
                        "id": "grounded",
                        "passed": True,
                        "reason": "Перевірено",
                    }
                ],
                "overall": True,
            },
            ensure_ascii=False,
        )

        with self.assertRaises(ValueError):
            judge.parse_verdict(raw, ["tone"])

    def test_judge_rejects_non_json(self):
        with self.assertRaises(ValueError):
            judge.parse_verdict("Це не JSON", ["tone"])


if __name__ == "__main__":
    unittest.main()
