
import json
import time
from pathlib import Path
from datetime import datetime

from shop import service
from app import workflow

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "requests.example.json"
OUTPUT_FILE = BASE_DIR / "smoke_results.json"


def main():
    requests = json.loads(
        INPUT_FILE.read_text(encoding="utf-8")
    )

    results = []

    print("=" * 55)
    print("ПР9 — АВТОМАТИЧНА ПЕРЕВІРКА WORKFLOW")
    print("Звернень:", len(requests))
    print("=" * 55, flush=True)

    for index, request in enumerate(requests, start=1):
        request_id = request["id"]
        customer_id = request["customer"]
        message = request["text"]

        print(
            f"\n[{index}/{len(requests)}] {request_id}",
            flush=True,
        )

        # Для незалежних тестів відновлюємо магазин.
        service.reset()

        started = time.perf_counter()

        try:
            run = workflow.start(message, customer_id)

            elapsed = round(
                time.perf_counter() - started, 2
            )

            expected_route = request.get("expect_route")
            route_matches = run.route == expected_route

            result = {
                "id": request_id,
                "customer": customer_id,
                "text": message,
                "expected_route": expected_route,
                "actual_route": run.route,
                "route_matches": route_matches,
                "status": run.status,
                "reply": run.reply,
                "run_id": run.run_id,
                "steps": [
                    {
                        "name": step.name,
                        "kind": step.kind,
                        "status": step.status,
                    }
                    for step in run.steps
                ],
                "usage": run.usage,
                "elapsed_seconds": elapsed,
                "error": run.stopped,
            }

            print("Маршрут:", run.route)
            print("Очікувався:", expected_route)
            print("Статус:", run.status)
            print("Час:", elapsed, "с")

        except Exception as error:
            result = {
                "id": request_id,
                "expected_route": request.get("expect_route"),
                "route_matches": False,
                "status": "exception",
                "error": str(error),
            }

            print("ПОМИЛКА:", error)

        results.append(result)

        # Зберігаємо результати після кожного тесту,
        # щоб вони не загубилися при перериванні.
        OUTPUT_FILE.write_text(
            json.dumps(
                results,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        if index < len(requests):
            time.sleep(2)

    matched = sum(
        1 for item in results
        if item.get("route_matches")
    )

    print("\n" + "=" * 55)
    print("ПЕРЕВІРКУ ЗАВЕРШЕНО")
    print("Усього:", len(results))
    print("Маршрут збігся:", matched)
    print("Маршрут не збігся:", len(results) - matched)
    print("Результати:", OUTPUT_FILE)
    print("=" * 55)


if __name__ == "__main__":
    main()
