
"""Прогін замовлень через API застосунку.

Підтримує:
- послідовні та паралельні замовлення;
- повторні звернення для перевірки кешу;
- fresh=true для обходу кешу;
- збереження результатів у JSON;
- порівняння двох прогонів;
- облік вхідних, вихідних і додаткових токенів Gemini.
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


TERMINAL = {"done", "failed", "cancelled"}


def api(
    base: str,
    method: str,
    path: str,
    body: dict | None = None,
):
    data = (
        json.dumps(body).encode("utf-8")
        if body is not None
        else None
    )

    request = urllib.request.Request(
        base + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=30,
        ) as response:
            payload = response.read()

            return (
                response.status,
                json.loads(payload) if payload else {},
                dict(response.headers),
            )

    except urllib.error.HTTPError as exc:
        payload = exc.read()

        try:
            parsed = json.loads(payload) if payload else {}
        except ValueError:
            parsed = {
                "raw": payload.decode(
                    "utf-8",
                    "replace",
                )[:200]
            }

        return exc.code, parsed, dict(exc.headers)


def make_sequence(
    ids: list[str],
    orders: int,
    seed: int,
    repeat: float,
) -> list[str]:
    """Створює повторювану послідовність звернень."""

    rng = random.Random(seed)
    sequence = []

    for _ in range(orders):
        if sequence and rng.random() < repeat:
            sequence.append(
                rng.choice(sequence[-6:])
            )
        else:
            sequence.append(rng.choice(ids))

    return sequence


def order(
    base: str,
    request_id: str,
    client: str,
    fresh: bool,
    timeout: float,
) -> dict:
    """Створює чернетку та чекає на її завершення."""

    started = time.perf_counter()

    while True:
        status, body, headers = api(
            base,
            "POST",
            "/api/drafts",
            {
                "request_id": request_id,
                "client": client,
                "fresh": fresh,
            },
        )

        if status == 429:
            wait = float(
                headers.get("Retry-After") or 5
            )

            print(
                f"429 для {request_id}: "
                f"чекаю {wait:.0f} с",
                file=sys.stderr,
            )

            time.sleep(wait)
            continue

        if status >= 300:
            return {
                "request_id": request_id,
                "status": f"http {status}",
                "error": body,
                "wall": time.perf_counter() - started,
            }

        break

    draft_id = body.get("draft_id")
    snapshot = body

    deadline = time.perf_counter() + timeout

    while (
        snapshot.get("status") not in TERMINAL
        and time.perf_counter() < deadline
    ):
        time.sleep(0.25)

        status, snapshot, _ = api(
            base,
            "GET",
            f"/api/drafts/{draft_id}",
        )

        if status >= 300:
            snapshot = {
                "status": f"http {status}",
                "error": snapshot,
            }
            break

    snapshot["wall"] = (
        time.perf_counter() - started
    )

    snapshot.setdefault(
        "request_id",
        request_id,
    )

    return snapshot


def percentile(
    values: list[float],
    p: float,
) -> float | None:
    if not values:
        return None

    ordered = sorted(values)

    index = min(
        len(ordered) - 1,
        max(
            0,
            round((len(ordered) - 1) * p),
        ),
    )

    return ordered[index]


def token_number(value) -> int | None:
    """Перетворює дані API на кількість токенів."""

    if value is None or isinstance(value, bool):
        return None

    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None

    return max(0, number)


def collect_tokens(rows: list[dict]) -> dict:
    """Підсумовує статистику токенів усіх замовлень."""

    tokens_in = 0
    tokens_out = 0
    tokens_total = 0
    tokens_additional = 0
    tokens_billable_output = 0

    usage_known_for = 0
    total_known_for = 0

    for row in rows:
        usage = row.get("usage")

        if not isinstance(usage, dict):
            continue

        input_tokens = token_number(
            usage.get("prompt_tokens")
        )

        output_tokens = token_number(
            usage.get("completion_tokens")
        )

        total = token_number(
            usage.get("total_tokens")
        )

        if (
            input_tokens is None
            and output_tokens is None
            and total is None
        ):
            continue

        usage_known_for += 1

        if input_tokens is not None:
            tokens_in += input_tokens

        if output_tokens is not None:
            tokens_out += output_tokens
            tokens_billable_output += output_tokens

        if total is not None:
            tokens_total += total
            total_known_for += 1

        model = str(
            row.get("model") or ""
        ).lower()

        is_gemini = model.startswith("gemini")

        # У Gemini total_tokens може включати
        # thinking-токени, яких немає в
        # completion_tokens.
        #
        # Показуємо різницю окремо. Вона може
        # містити додаткові категорії токенів.
        if (
            is_gemini
            and input_tokens is not None
            and output_tokens is not None
            and total is not None
        ):
            additional = max(
                0,
                total - input_tokens - output_tokens,
            )

            tokens_additional += additional
            tokens_billable_output += additional

    return {
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_total": tokens_total,
        "tokens_additional": tokens_additional,
        "tokens_billable_output": tokens_billable_output,
        "usage_known_for": usage_known_for,
        "total_known_for": total_known_for,
    }


def summarize(rows: list[dict]) -> dict:
    """Формує підсумок прогону."""

    statuses = {}

    for row in rows:
        status = str(row.get("status"))

        statuses[status] = (
            statuses.get(status, 0) + 1
        )

    done = [
        row for row in rows
        if row.get("status") == "done"
    ]

    cached = [
        row for row in done
        if row.get("cached")
    ]

    walls = [
        row["wall"]
        for row in rows
        if isinstance(
            row.get("wall"),
            (int, float),
        )
    ]

    firsts = [
        row["elapsed"]["first_token"]
        for row in rows
        if isinstance(row.get("elapsed"), dict)
        and isinstance(
            row["elapsed"].get("first_token"),
            (int, float),
        )
    ]

    costs = [
        row["cost"]
        for row in rows
        if isinstance(
            row.get("cost"),
            (int, float),
        )
    ]

    attempts = sum(
        row.get("attempts") or 0
        for row in rows
        if not row.get("cached")
    )

    tokens = collect_tokens(rows)

    return {
        "orders": len(rows),
        "statuses": statuses,
        "cached_hits": len(cached),
        "provider_attempts": attempts,
        "fallbacks": sum(
            1 for row in rows
            if row.get("fallback")
        ),
        **tokens,
        "cost_total": (
            round(sum(costs), 8)
            if costs
            else None
        ),
        "cost_known_for": len(costs),
        "cost_estimated_count": sum(
            1 for row in rows
            if row.get("cost_estimated")
            and row.get("status") == "done"
        ),
        "wall_p50": percentile(walls, 0.5),
        "wall_p95": percentile(walls, 0.95),
        "first_token_p50": percentile(firsts, 0.5),
        "first_token_p95": percentile(firsts, 0.95),
    }


def fmt(value, digits: int = 2) -> str:
    if value is None:
        return "—"

    if isinstance(value, float):
        return f"{value:.{digits}f}"

    return str(value)


def print_rows(rows: list[dict]) -> None:
    print(
        f"{'#':>3} "
        f"{'зверн.':<7} "
        f"{'чернетка':<9} "
        f"{'статус':<10} "
        f"{'пров.':<7} "
        f"{'спроб':>5} "
        f"{'кеш':<4} "
        f"{'$':>11} "
        f"{'перший, с':>10} "
        f"{'разом, с':>9}"
    )

    for i, row in enumerate(rows, 1):
        elapsed = (
            row.get("elapsed")
            if isinstance(row.get("elapsed"), dict)
            else {}
        )

        print(
            f"{i:>3} "
            f"{row.get('request_id', ''):<7} "
            f"{str(row.get('draft_id', '')):<9} "
            f"{str(row.get('status')):<10} "
            f"{str(row.get('provider') or ''):<7} "
            f"{fmt(row.get('attempts')):>5} "
            f"{'так' if row.get('cached') else '':<4} "
            f"{fmt(row.get('cost'), 8):>11} "
            f"{fmt(elapsed.get('first_token')):>10} "
            f"{fmt(row.get('wall')):>9}"
        )


def print_summary(
    summary: dict,
    label: str = "",
) -> None:
    if label:
        print(f"\n== {label}")

    print(
        f"замовлень {summary['orders']}; "
        f"статуси {summary['statuses']}; "
        f"з кешу {summary['cached_hits']}; "
        f"спроб провайдера {summary['provider_attempts']}; "
        f"запасний варіант {summary['fallbacks']}"
    )

    print(
        f"токенів: "
        f"вхід {summary['tokens_in']}, "
        f"вихід {summary['tokens_out']}, "
        f"додаткові Gemini {summary['tokens_additional']}, "
        f"усього {summary['tokens_total']}"
    )

    print(
        f"вихідних для розрахунку Gemini: "
        f"{summary['tokens_billable_output']}; "
        f"usage є для {summary['usage_known_for']} "
        f"замовлень"
    )

    print(
        f"вартість разом "
        f"{fmt(summary['cost_total'], 8)} $ "
        f"(відома для {summary['cost_known_for']} "
        f"замовлень; приблизних: "
        f"{summary['cost_estimated_count']})"
    )

    print(
        f"тривалість, с: "
        f"медіана {fmt(summary['wall_p50'])}, "
        f"p95 {fmt(summary['wall_p95'])}; "
        f"до першого фрагмента: "
        f"медіана {fmt(summary['first_token_p50'])}, "
        f"p95 {fmt(summary['first_token_p95'])}"
    )


def compare(paths: list[str]) -> None:
    runs = []

    for path in paths:
        with open(
            path,
            encoding="utf-8",
        ) as file:
            runs.append((path, json.load(file)))

    keys = [
        "orders",
        "cached_hits",
        "provider_attempts",
        "fallbacks",
        "tokens_in",
        "tokens_out",
        "tokens_additional",
        "tokens_total",
        "tokens_billable_output",
        "usage_known_for",
        "cost_total",
        "wall_p50",
        "wall_p95",
        "first_token_p50",
        "first_token_p95",
    ]

    width = max(
        len(path)
        for path, _ in runs
    )

    print(
        f"{'':<24}"
        + "".join(
            f"{path:>{width + 2}}"
            for path, _ in runs
        )
    )

    for key in keys:
        print(
            f"{key:<24}"
            + "".join(
                f"{fmt(run['summary'].get(key), 8 if key == 'cost_total' else 2):>{width + 2}}"
                for _, run in runs
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--base",
        default="http://127.0.0.1:8000",
    )

    parser.add_argument(
        "--orders",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--repeat",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--client",
        default="replay",
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Обійти кеш для всіх замовлень.",
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=120,
    )

    parser.add_argument(
        "--out",
        help="Шлях для збереження результатів JSON.",
    )

    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="FILE",
    )

    args = parser.parse_args()

    if args.compare:
        compare(args.compare)
        return 0

    status, samples, _ = api(
        args.base,
        "GET",
        "/api/requests",
    )

    if status != 200 or not samples:
        print(
            f"Не вдалося отримати зразки: "
            f"{status} {samples}",
            file=sys.stderr,
        )
        return 1

    ids = [
        sample["id"]
        for sample in samples
    ]

    sequence = make_sequence(
        ids,
        args.orders,
        args.seed,
        args.repeat,
    )

    print(
        f"Послідовність ({args.orders}, "
        f"seed {args.seed}): "
        f"{' '.join(sequence)}\n"
    )

    status, health, _ = api(
        args.base,
        "GET",
        "/api/health",
    )

    started = time.perf_counter()

    if args.concurrency > 1:
        with ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as pool:
            rows = list(
                pool.map(
                    lambda rid: order(
                        args.base,
                        rid,
                        args.client,
                        args.fresh,
                        args.timeout,
                    ),
                    sequence,
                )
            )
    else:
        rows = [
            order(
                args.base,
                rid,
                args.client,
                args.fresh,
                args.timeout,
            )
            for rid in sequence
        ]

    total = time.perf_counter() - started

    print_rows(rows)

    summary = summarize(rows)
    summary["wall_total"] = total

    print_summary(summary)

    print(f"Увесь прогін: {total:.1f} с")

    if args.out:
        with open(
            args.out,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "args": vars(args),
                    "health": (
                        health
                        if status == 200
                        else None
                    ),
                    "sequence": sequence,
                    "rows": rows,
                    "summary": summary,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

        print(f"Записано {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
