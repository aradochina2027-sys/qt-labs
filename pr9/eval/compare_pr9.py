"""ПР9 — відновлюване реальне порівняння workflow та агента.

Запуск із кореня pr9:
  python -m eval.compare_pr9 --check
  python -m eval.compare_pr9 --run
  python -m eval.compare_pr9 --summary

Жодної підставної моделі. При вичерпаній квоті прогін зупиняється;
успішні прогони зберігаються, а незавершені виконуються після наступного --run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

from app import assistant, workflow
from shop import service

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "requests.json"
RESULTS = ROOT / "comparison_results.json"
SUMMARY = ROOT / "comparison_summary.json"
BACKUP = ROOT / "comparison_results_before_resume.json"

EXTRA = [
    {"id": "own01", "kind": "звичайний стан", "customer": "C-1001",
     "text": "Перевірте стан замовлення 10458.", "expect_route": "status", "expect_action": None},
    {"id": "own02", "kind": "без номера", "customer": "C-1003",
     "text": "Хочу повернути товар, але не пам’ятаю номер замовлення.",
     "expect_route": "return", "expect_action": None},
    {"id": "own03", "kind": "дві дії", "customer": "C-1001",
     "text": "Скасуйте замовлення 10463 і поверніть навушники із 10421.",
     "expect_route": "декілька", "expect_action": None},
]

QUOTA_MARKERS = (
    "ratelimit", "rate limit", "429", "quota exceeded", "quota_exceeded",
    "resource_exhausted", "ліміт запитів", "квоту вичерпано",
    "exceeded your current quota", "retry in ",
)


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data) -> None:
    """Атомарне збереження одного файла результатів."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def dataset():
    if INPUT.exists():
        items = _read_json(INPUT, [])
    else:
        items = _read_json(ROOT / "requests.example.json", [])
        items.extend(EXTRA)
        _write_json(INPUT, items)
    if not isinstance(items, list) or len(items) < 20:
        raise ValueError("Потрібно щонайменше 20 звернень (17 готових + 3 власні)")
    ids = [x["id"] for x in items]
    if len(set(ids)) != len(ids):
        raise ValueError("Знайдено повторювані id")
    for item in items:
        if not isinstance(item.get("text"), str) or not item.get("customer"):
            raise ValueError(f"Неповне звернення: {item.get('id')}")
    return items


def safe_data(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        return value
    return str(value)


def shop_state():
    state = {"orders": {}, "returns": {}}
    for customer in service.list_customers():
        for order in service.list_orders(customer["customer_id"]):
            oid = order["order_id"]
            state["orders"][oid] = order["status"]
            state["returns"][oid] = [
                {"return_id": r.get("return_id"), "sku": r.get("sku"),
                 "reason": r.get("reason"), "quantity": r.get("quantity")}
                for r in service.list_returns(oid)
            ]
    return state


def is_quota_failure(row):
    parts = (row.get("error"), row.get("stopped"), row.get("answer"))
    msg = " ".join(str(v) for v in parts if v is not None).lower()
    return any(phrase in msg for phrase in QUOTA_MARKERS)


def is_retryable(row):
    """Квотні збої і системні помилки не вважаються завершеними вимірюваннями."""
    if is_quota_failure(row):
        return True
    status = str(row.get("status") or "").lower()
    # Не стираємо інші збої: вони є справжніми спостереженнями.
    return status == "error" and not row.get("trace") and not row.get("model_calls")


def execute(case, mode, approve=True):
    started = time.perf_counter()
    before = shop_state()
    problem = None
    try:
        if mode == "workflow":
            run = workflow.start(case["text"], case["customer"])
            initial_status = run.status
            if run.status == "awaiting_approval":
                run = workflow.resume(run.run_id, approve, "Порівняльний тест ПР9")
            data = {
                "route": run.route, "initial_status": initial_status,
                "status": run.status, "answer": run.reply,
                "trace": [safe_data(step) for step in run.steps],
                "model_calls": run.usage.get("model_calls"),
                "tokens": run.usage.get("total_tokens"),
                "usage": run.usage, "stopped": run.stopped,
            }
        else:
            answer = assistant.answer(case["text"], case["customer"])
            data = {
                "route": None, "initial_status": None,
                "status": str(answer.stopped or "done"),
                "answer": answer.text,
                "trace": [safe_data(call) for call in answer.calls],
                "model_calls": answer.rounds,
                "tokens": (answer.usage or {}).get("total_tokens"),
                "usage": answer.usage, "stopped": answer.stopped,
            }
    except Exception as exc:
        problem = f"{type(exc).__name__}: {exc}"
        data = {
            "route": None, "initial_status": None, "status": "error",
            "answer": None, "trace": [], "model_calls": None,
            "tokens": None, "usage": None, "stopped": problem,
        }
    after = shop_state()
    changed = {
        oid: [before["orders"].get(oid), status]
        for oid, status in after["orders"].items()
        if before["orders"].get(oid) != status
    }
    new_returns = {
        oid: [r for r in rows if r not in before["returns"].get(oid, [])]
        for oid, rows in after["returns"].items()
    }
    data.update({
        "id": case["id"], "kind": case.get("kind"),
        "customer": case["customer"], "mode": mode,
        "expected_route": case.get("expect_route"),
        "expected_action": case.get("expect_action"),
        "order_changes": changed,
        "new_returns": {k: v for k, v in new_returns.items() if v},
        "seconds": round(time.perf_counter() - started, 3),
        "error": problem,
    })
    return data


def _key(row):
    return (row.get("mode"), row.get("repetition"), row.get("id"))


def read_results():
    rows = _read_json(RESULTS, [])
    if not isinstance(rows, list):
        raise ValueError("comparison_results.json має бути списком")
    return rows


def prepare_resume():
    original = read_results()
    if original and not BACKUP.exists():
        shutil.copy2(RESULTS, BACKUP)
        print("Резервна копія попередніх результатів:", BACKUP)
    kept = []
    seen = set()
    for row in original:
        key = _key(row)
        if is_retryable(row):
            continue
        if key in seen:
            continue
        kept.append(row)
        seen.add(key)
    if len(kept) != len(original):
        print(f"Незавершені/квотні записи для повтору: {len(original) - len(kept)}")
        _write_json(RESULTS, kept)
    return kept


def make_summary(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row.get("kind"), row.get("mode")), []).append(row)
    summary = []
    for (kind, mode), group in grouped.items():
        def avg(field):
            vals = [float(v[field]) for v in group
                    if isinstance(v.get(field), (int, float)) and not isinstance(v[field], bool)]
            return round(statistics.mean(vals), 3) if vals else None
        summary.append({
            "kind": kind, "mode": mode, "runs": len(group),
            "errors": sum(x.get("status") == "error" for x in group),
            "avg_model_calls": avg("model_calls"),
            "avg_tokens": avg("tokens"), "avg_seconds": avg("seconds"),
            "manual_review_needed": True,
        })
    _write_json(SUMMARY, summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="лише перевірити набір, без API")
    group.add_argument("--run", action="store_true", help="запуск і продовження після зупинки")
    group.add_argument("--summary", action="store_true", help="підсумок без API")
    args = parser.parse_args()
    items = dataset()
    print("Звернень:", len(items), "Планових прогонів:", len(items) * 6)
    if args.summary:
        rows = read_results()
        make_summary(rows)
        print("Збережено підсумок:", SUMMARY, "Записів:", len(rows))
        return
    if not args.run:
        for case in items:
            print(case["id"], case.get("expect_route"), case["customer"])
        print("Перевірка набору завершена. API не викликався.")
        return

    rows = prepare_resume()
    completed = {_key(r) for r in rows}
    by_id = {c["id"]: c for c in items}
    for mode in ("workflow", "agent"):
        for repetition in (1, 2, 3):
            for case in items:
                key = (mode, repetition, case["id"])
                if key in completed:
                    continue
                service.reset()
                if case.get("after"):
                    prior = by_id.get(case["after"])
                    if prior is None:
                        raise ValueError(f"Немає звернення after={case['after']}")
                    prerequisite = execute(prior, mode)
                    if is_quota_failure(prerequisite):
                        print("Квота вичерпана під час підготовки", case["id"])
                        make_summary(rows)
                        return
                    if prerequisite["status"] == "error":
                        print("Не вдалося виконати попереднє звернення для", case["id"])
                        make_summary(rows)
                        return
                print(f"{mode} #{repetition} {case['id']}", flush=True)
                result = execute(case, mode)
                result["repetition"] = repetition
                if is_quota_failure(result):
                    print("Ліміт API вичерпано. Цей прогін не зараховано.")
                    print("Причина:", result.get("error") or result.get("stopped"))
                    make_summary(rows)
                    print("Після відновлення квоти повтори ту саму команду --run.")
                    return
                rows.append(result)
                completed.add(key)
                _write_json(RESULTS, rows)
                print("  ", result["status"], result["seconds"], "с", flush=True)
    make_summary(rows)
    print("Порівняння завершено:", RESULTS, SUMMARY)
    print("Потрібна ручна оцінка правильності та дій без підтвердження.")


if __name__ == "__main__":
    main()
