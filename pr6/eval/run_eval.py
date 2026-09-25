import json
import sys
import time
from pathlib import Path

# Додаємо корінь pr6 до Python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import index, keyword, rag
from app.retrieval import build_context


QUESTIONS_FILE = ROOT / "eval" / "questions.json"
RESULTS_FILE = ROOT / "eval" / "results_run2.json"

# Пауза між запитами до LLM
REQUEST_DELAY = 15


def load_questions():
    """Завантажити тестові питання."""

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return data["питання"]


def source_to_dict(source):
    """Перетворити Source у словник для JSON."""

    return {
        "ref": source.ref,
        "score": round(float(source.score), 4),
        "source": source.chunk.source,
        "metadata": source.chunk.metadata,
        "text": source.chunk.text,
    }


def hit_to_dict(hit):
    """Перетворити Hit у словник для JSON."""

    return {
        "score": round(float(hit.score), 4),
        "source": hit.chunk.source,
        "metadata": hit.chunk.metadata,
        "text": hit.chunk.text,
    }


def is_infrastructure_error(result):
    """
    Визначити, чи відповідь є наслідком
    проблеми API, а не помилки RAG.
    """

    text = (result.text or "").lower()

    markers = [
        "модель тимчасово недоступна",
        "помилка генерації",
        "rate limit",
        "quota",
        "429",
        "503",
        "resource_exhausted",
        "недійсний api",
        "api-ключ",
    ]

    return any(
        marker in text
        for marker in markers
    )


def unique_documents(sources):
    """Отримати унікальні назви документів."""

    documents = []

    for source in sources:
        name = source.chunk.source

        if name not in documents:
            documents.append(name)

    return documents


def main():
    print("=" * 65)
    print("ОЦІНЮВАННЯ RAG-ПОМІЧНИКА")
    print("=" * 65)

    print("\nЗавантаження індексу...")

    search_index = index.load()

    keyword_index = keyword.build(
        search_index.chunks
    )

    questions = load_questions()

    print(f"Питань: {len(questions)}")
    print(
        f"Фрагментів в індексі: "
        f"{len(search_index.chunks)}"
    )

    results = []

    for number, item in enumerate(
        questions,
        start=1,
    ):
        question = item["питання"]

        if number > 1:
            print(
                f"\nОчікування {REQUEST_DELAY} секунд "
                "перед наступним запитом..."
            )
            time.sleep(REQUEST_DELAY)

        print("\n" + "=" * 65)
        print(
            f"[{number}/{len(questions)}] "
            f"{item['вид']}"
        )
        print(question)
        print("-" * 65)

        try:
            # Повний RAG pipeline
            result = rag.answer(
                question=question,
                index=search_index,
                keyword_index=keyword_index,
            )

            # Усі hits після retrieval
            retrieved = [
                hit_to_dict(hit)
                for hit in result.retrieved
            ]

            # Відтворюємо саме той контекст,
            # який формує build_context().
            context_text, context_sources = (
                build_context(
                    result.retrieved
                )
            )

            # Документи, які реально
            # потрапили в контекст моделі.
            context_documents = (
                unique_documents(
                    context_sources
                )
            )

            # Фрагменти фактичного контексту.
            context_chunks = [
                source_to_dict(source)
                for source in context_sources
            ]

            # Джерела, які модель використала
            # у фінальній відповіді.
            answer_sources = [
                source_to_dict(source)
                for source in result.sources
            ]

            expected_documents = item.get(
                "очікувані_документи",
                [],
            )

            expected_found = item.get(
                "очікується_відповідь",
                False,
            )

            # Для тестів, де відповідь очікується,
            # перевіряємо наявність усіх
            # очікуваних документів.
            #
            # Для no-answer тестів порожній список
            # expected_documents не означає,
            # що retrieval повинен бути порожнім.
            if expected_documents:
                expected_in_context = all(
                    document in context_documents
                    for document
                    in expected_documents
                )
            else:
                expected_in_context = None

            infrastructure_error = (
                is_infrastructure_error(result)
            )

            # Не оцінюємо found як помилку RAG,
            # якщо API фактично не відповіло.
            if infrastructure_error:
                found_correct = None
            else:
                found_correct = (
                    result.found
                    == expected_found
                )

            record = {
                "номер": number,
                "вид": item["вид"],
                "питання": question,

                "очікувані_документи":
                    expected_documents,

                "очікувана_відповідь":
                    item.get(
                        "очікувана_відповідь",
                        "",
                    ),

                "очікується_відповідь":
                    expected_found,

                "перевіряє":
                    item.get(
                        "перевіряє",
                        "",
                    ),

                "found":
                    result.found,

                "відповідь":
                    result.text,

                "інфраструктурна_помилка":
                    infrastructure_error,

                "found_правильний":
                    found_correct,

                "очікувані_документи_в_контексті":
                    expected_in_context,

                "документи_контексту":
                    context_documents,

                # Саме цей текст формується
                # для передачі моделі.
                "контекст_моделі":
                    context_text,

                # Окремо структуровані фрагменти
                # фактичного контексту.
                "фрагменти_контексту":
                    context_chunks,

                # Усі результати retrieval
                # до обмеження budget.
                "retrieved":
                    retrieved,

                # Джерела, на які послалася модель.
                "джерела_відповіді":
                    answer_sources,

                "модель":
                    result.model,

                "час":
                    result.elapsed,

                "токени":
                    result.usage,
            }

            results.append(record)

            print(
                "FOUND:",
                result.found,
                "| очікувалось:",
                expected_found,
            )

            if infrastructure_error:
                print(
                    "СТАТУС: "
                    "ІНФРАСТРУКТУРНА ПОМИЛКА API"
                )
            else:
                print(
                    "FOUND правильний:",
                    "ТАК"
                    if found_correct
                    else "НІ",
                )

            print(
                "Документи контексту:",
                ", ".join(context_documents)
                if context_documents
                else "немає",
            )

            if expected_in_context is not None:
                print(
                    "Очікувані документи "
                    "в контексті:",
                    "ТАК"
                    if expected_in_context
                    else "НІ",
                )
            else:
                print(
                    "Очікувані документи "
                    "в контексті: "
                    "не застосовується"
                )

            print("\nВідповідь:")
            print(result.text)

            if result.elapsed:
                retrieval_time = (
                    result.elapsed.get(
                        "retrieval",
                        0,
                    )
                )

                generation_time = (
                    result.elapsed.get(
                        "generation",
                        0,
                    )
                )

                print(
                    "\nЧас пошуку:",
                    round(
                        retrieval_time,
                        2,
                    ),
                    "с",
                )

                print(
                    "Час генерації:",
                    round(
                        generation_time,
                        2,
                    ),
                    "с",
                )

            if result.usage:
                print(
                    "Токени:",
                    result.usage.get(
                        "total_tokens",
                        0,
                    ),
                )

        except Exception as exc:
            print(
                "ПОМИЛКА:",
                type(exc).__name__,
                str(exc),
            )

            results.append({
                "номер": number,
                "вид": item.get(
                    "вид",
                    "",
                ),
                "питання": question,
                "інфраструктурна_помилка":
                    True,
                "помилка": (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            })

    # Збереження результатів
    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 65)
    print("ГОТОВО")
    print("=" * 65)

    print(
        "Результати збережено:",
        RESULTS_FILE,
    )

    total = len(results)

    infrastructure_errors = sum(
        1
        for item in results
        if item.get(
            "інфраструктурна_помилка"
        ) is True
    )

    evaluated = sum(
        1
        for item in results
        if item.get(
            "found_правильний"
        ) is not None
    )

    correct_found = sum(
        1
        for item in results
        if item.get(
            "found_правильний"
        ) is True
    )

    context_tests = sum(
        1
        for item in results
        if item.get(
            "очікувані_документи_в_контексті"
        ) is not None
    )

    context_ok = sum(
        1
        for item in results
        if item.get(
            "очікувані_документи_в_контексті"
        ) is True
    )

    print(f"\nУсього питань: {total}")

    print(
        "Інфраструктурних помилок:",
        infrastructure_errors,
    )

    print(
        f"Правильний found: "
        f"{correct_found}/{evaluated}"
    )

    print(
        "Очікувані документи "
        f"в контексті: "
        f"{context_ok}/{context_tests}"
    )


if __name__ == "__main__":
    main()