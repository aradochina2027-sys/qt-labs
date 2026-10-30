"""Аналіз журналу ПР11 у форматі JSON lines.

Запуск з папки pr11:
    python eval/analyze.py
    python eval/analyze.py --section slow
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def load(path: Path, since: str | None = None) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"Пропущено некоректний рядок {number}")
                continue
            if isinstance(item, dict) and (not since or str(item.get("ts", "")) >= since):
                records.append(item)
    return records


def attempts(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("event") == "llm.attempt"]


def amount(record: dict, field: str) -> float:
    try:
        return float(record.get(field) or 0)
    except (TypeError, ValueError):
        return 0.0


def percentile(values: list[float], percent: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    index = round((len(values) - 1) * percent)
    return values[index]


def summary(records: list[dict]) -> None:
    print(f"Усього записів: {len(records)}")
    if records:
        dates = [str(r.get("ts", "")) for r in records]
        print(f"Період: {min(dates)} — {max(dates)}")
    for event, count in Counter(r.get("event", "невідомо") for r in records).most_common():
        print(f"  {event}: {count}")


def slow(records: list[dict]) -> None:
    calls = [r for r in attempts(records) if r.get("duration_ms") is not None]
    if not calls:
        print("Поки немає записів llm.attempt із тривалістю.")
        return
    durations = [amount(r, "duration_ms") for r in calls]
    print(f"Викликів: {len(calls)}")
    print(f"Медіана: {median(durations):.2f} мс")
    print(f"P95: {percentile(durations, 0.95):.2f} мс")
    print(f"Максимум: {max(durations):.2f} мс")
    print("Найповільніші виклики:")
    for r in sorted(calls, key=lambda x: amount(x, "duration_ms"), reverse=True)[:10]:
        print(
            f"  draft={r.get('draft_id')} attempt={r.get('attempt')} "
            f"{r.get('provider')}/{r.get('model')} "
            f"total={amount(r, 'duration_ms'):.1f} мс "
            f"queue={amount(r, 'queue_ms'):.1f} мс "
            f"first_token={amount(r, 'first_token_ms'):.1f} мс"
        )
    by_provider = defaultdict(list)
    for r in calls:
        by_provider[str(r.get("provider", "невідомо"))].append(amount(r, "duration_ms"))
    print("P95 за провайдерами:")
    for name, durations in sorted(by_provider.items()):
        print(f"  {name}: {percentile(durations, 0.95):.2f} мс")
    print("Виклики за годинами:", dict(Counter(str(r.get("ts", ""))[:13] for r in calls)))


def failed(records: list[dict]) -> None:
    errors = [r for r in attempts(records) if r.get("status") not in ("ok", "success")]
    kinds = Counter(str(r.get("kind") or "unknown") for r in errors)
    print(f"Невдалих спроб: {len(errors)}; види помилок: {dict(kinds)}")
    final_errors = [r for r in records if r.get("event") in ("draft.failed", "draft.error")]
    print(f"Чернеток з остаточною помилкою: {len(final_errors)}")
    failed_first = {r.get("draft_id") for r in errors if r.get("attempt") == 1}
    completed = {r.get("draft_id") for r in records if r.get("event") == "draft.done"}
    print(f"Врятовано повторними спробами: {len(failed_first & completed)} із {len(failed_first)}")
    print(f"Вартість невдалих спроб: {sum(amount(r, 'cost') for r in errors):.8f} USD")


def repeated(records: list[dict]) -> None:
    cache_events = [r for r in records if r.get("event") in ("cache.hit", "cache.miss", "cache.bypass")]
    keys = Counter(r.get("cache_key") for r in cache_events if r.get("cache_key"))
    print("Повторні ключі кешу:")
    for key, count in keys.most_common():
        if count > 1:
            print(f"  {key[:12]}... — {count} разів")
    tries = Counter(r.get("draft_id") for r in attempts(records))
    print("Чернетки з кількома спробами:")
    for draft_id, count in tries.items():
        if count > 1:
            print(f"  {draft_id}: {count} спроб")
    series = Counter(
        (r.get("client"), str(r.get("ts", ""))[:16])
        for r in records if r.get("event") == "draft.created"
    )
    print("Клієнти з кількома замовленнями за хвилину:")
    for (client, minute), count in series.most_common():
        if count > 1:
            print(f"  {client} о {minute}: {count}")


def spend(records: list[dict]) -> None:
    calls = attempts(records)
    if not calls:
        print("Поки немає даних про вартість викликів.")
        return
    by_day, by_model, by_client = defaultdict(float), defaultdict(float), defaultdict(float)
    input_tokens = output_tokens = 0
    for r in calls:
        cost = amount(r, "cost")
        by_day[str(r.get("ts", ""))[:10]] += cost
        by_model[str(r.get("model") or "невідомо")] += cost
        by_client[str(r.get("client") or "невідомо")] += cost
        usage = r.get("usage") if isinstance(r.get("usage"), dict) else {}
        input_tokens += int(r.get("prompt_tokens") or usage.get("prompt_tokens") or 0)
        output_tokens += int(r.get("completion_tokens") or usage.get("completion_tokens") or 0)
    total = sum(by_day.values())
    done = {r.get("draft_id") for r in records if r.get("event") == "draft.done"}
    print(f"Разом: {total:.8f} USD; вхідні токени: {input_tokens}; вихідні: {output_tokens}")
    print(f"Середня вартість завершеної чернетки: {total / len(done):.8f} USD" if done else "Завершених чернеток немає")
    for label, grouping in (("За днями", by_day), ("За моделями", by_model), ("За клієнтами", by_client)):
        print(label + ":")
        for name, cost in sorted(grouping.items()):
            print(f"  {name}: {cost:.8f} USD")
    print(f"Невдалі спроби: {sum(amount(r, 'cost') for r in calls if r.get('status') not in ('ok', 'success')):.8f} USD")
    print(f"Fallback: {sum(amount(r, 'cost') for r in calls if r.get('fallback')):.8f} USD")
    print(f"Вартість оцінена: {sum(bool(r.get('cost_estimated')) for r in calls)} спроб")
    print(f"Вартість відсутня: {sum(r.get('cost') is None for r in calls)} спроб")


def cache_report(records: list[dict]) -> None:
    events = Counter(r.get("event") for r in records)
    hits = events["cache.hit"]
    misses = events["cache.miss"]
    bypasses = events["cache.bypass"]
    rate = 100 * hits / (hits + misses) if hits + misses else 0
    print(f"Влучань: {hits}; промахів: {misses}; fresh-обходів: {bypasses}")
    print(f"Частка влучань: {rate:.1f}%")
    print(f"Приблизно уникнуто звертань до провайдера: {hits}")
    print("Грошову економію визначаємо порівнянням replay з кешем і без кешу.")


SECTIONS = {
    "summary": summary,
    "slow": slow,
    "failed": failed,
    "repeated": repeated,
    "spend": spend,
    "cache": cache_report,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Аналіз журналу ПР11")
    parser.add_argument("log", nargs="?", default="logs/app.jsonl")
    parser.add_argument("--since", default=None)
    parser.add_argument("--section", choices=["all", *SECTIONS], default="all")
    args = parser.parse_args()
    path = Path(args.log)
    if not path.is_file():
        print(f"Журнал ще не створено: {path}")
        print("Спочатку потрібно запустити застосунок і створити чернетки.")
        return
    records = load(path, args.since)
    for name in (SECTIONS if args.section == "all" else [args.section]):
        print(f"\n=== {name.upper()} ===")
        SECTIONS[name](records)


if __name__ == "__main__":
    main()
