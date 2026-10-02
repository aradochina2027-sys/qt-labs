"""Цикл «модель → інструменти → модель» для ПР8."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

from . import llm, tools

load_dotenv()
MAX_ROUNDS = int(os.getenv("TOOL_MAX_ROUNDS", "3"))


@dataclass
class ToolTrace:
    round: int
    name: str
    arguments: str
    status: str
    reason: str | None = None
    result: dict | list | str | None = None
    elapsed: float | None = None


@dataclass
class Answer:
    text: str
    calls: list[ToolTrace] = field(default_factory=list)
    rounds: int = 0
    stopped: str = "answer"
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


def _sum_usage(total: dict, current: dict | None) -> None:
    if not current:
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total[key] = total.get(key, 0) + int(current.get(key, 0) or 0)


def _tool_calls(message: dict) -> list[dict]:
    calls = message.get("tool_calls") or []
    return [c for c in calls if isinstance(c, dict)]


def _function_parts(call: dict) -> tuple[str, str, str]:
    fn = call.get("function") or {}
    return str(call.get("id") or ""), str(fn.get("name") or ""), str(fn.get("arguments") or "{}")


def answer(question: str, customer_id: str) -> Answer:
    """Відповісти клієнтові з контрольованим циклом викликів."""
    messages = llm.build_messages(question)
    tool_specs = tools.specs()
    ctx = tools.Context(customer_id=customer_id, question=question)

    traces: list[ToolTrace] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    model_elapsed = 0.0
    tools_elapsed = 0.0
    model_name: str | None = None

    max_rounds = max(1, MAX_ROUNDS)
    for round_no in range(1, max_rounds + 1):
        # Останнє дозволене звертання примушує модель сформулювати відповідь із уже наявних даних.
        choice = "none" if round_no == max_rounds else "auto"
        response = llm.chat(messages, tool_specs, tool_choice=choice)
        model_elapsed += float(response.get("elapsed") or 0)
        _sum_usage(usage, response.get("usage"))
        model_name = response.get("model") or model_name

        message = response["message"]
        messages.append(message)
        calls = _tool_calls(message)

        if not calls:
            text = str(message.get("content") or "").strip()
            if not text:
                text = "Не вдалося сформувати відповідь. Спробуйте уточнити питання."
            return Answer(
                text=text,
                calls=traces,
                rounds=round_no,
                stopped="answer",
                model=model_name,
                elapsed={"model": round(model_elapsed, 4), "tools": round(tools_elapsed, 4)},
                usage=usage,
            )

        # Захист для змінюючої операції: не виконуємо create_return у пакеті з іншими викликами.
        names = [_function_parts(c)[1] for c in calls]
        mixed_write = "create_return" in names and len(calls) > 1

        for call in calls:
            tool_call_id, name, raw_args = _function_parts(call)
            started = time.perf_counter()
            if mixed_write and name == "create_return":
                result = tools.ToolResult(
                    status="rejected",
                    content={
                        "ok": False,
                        "error": "write_in_parallel",
                        "message": "заявка на повернення не виконується в одному пакеті з іншими викликами; повторіть її окремим кроком",
                    },
                    reason="змінююча операція відхилена в пакетному виклику",
                )
            else:
                result = tools.call(name, raw_args, ctx)
            elapsed = time.perf_counter() - started
            tools_elapsed += elapsed

            traces.append(
                ToolTrace(
                    round=round_no,
                    name=name,
                    arguments=raw_args,
                    status=result.status,
                    reason=result.reason,
                    result=result.content,
                    elapsed=round(elapsed, 4),
                )
            )

            # На кожен tool_call повертаємо окремий результат із тим самим tool_call_id.
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(result.content, ensure_ascii=False),
                }
            )

    # Теоретично сюди не дійдемо, бо останній round має tool_choice='none'.
    return Answer(
        text="Досягнуто ліміту звертань до моделі. Спробуйте поставити коротше або конкретніше питання.",
        calls=traces,
        rounds=max_rounds,
        stopped="limit",
        model=model_name,
        elapsed={"model": round(model_elapsed, 4), "tools": round(tools_elapsed, 4)},
        usage=usage,
    )
