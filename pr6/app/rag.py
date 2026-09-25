"""Конвеєр RAG: пошук → контекст → модель → перевірка."""

import time
from dataclasses import dataclass, field

from .index import Hit, SearchIndex
from .keyword import KeywordIndex
from .retrieval import Source, retrieve, build_context
from .llm import ask, LLMError


@dataclass
class Answer:
    """Результат роботи RAG-конвеєра."""

    text: str
    found: bool
    sources: list[Source] = field(default_factory=list)
    retrieved: list[Hit] = field(default_factory=list)
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


NO_ANSWER = (
    "У базі знань немає достатньо інформації "
    "для відповіді на це питання."
)


def answer(
    question: str,
    index: SearchIndex,
    keyword_index: KeywordIndex | None,
    filters: dict | None = None,
) -> Answer:
    """Відповісти на питання тільки за базою знань."""

    question = question.strip()

    if not question:
        return Answer(
            text="Введіть питання.",
            found=False,
            sources=[],
            retrieved=[],
            elapsed={
                "retrieval": 0.0,
                "generation": 0.0,
            },
        )

    # -----------------------------
    # 1. Пошук
    # -----------------------------

    retrieval_started = time.perf_counter()

    try:
        hits = retrieve(
            query=question,
            index=index,
            keyword_index=keyword_index,
            filters=filters,
        )

        context, context_sources = build_context(hits)

    except Exception as exc:
        retrieval_time = (
            time.perf_counter() - retrieval_started
        )

        return Answer(
            text=f"Помилка пошуку: {exc}",
            found=False,
            sources=[],
            retrieved=[],
            elapsed={
                "retrieval": retrieval_time,
                "generation": 0.0,
            },
        )

    retrieval_time = (
        time.perf_counter() - retrieval_started
    )

    # -----------------------------
    # 2. Нічого не знайдено
    # -----------------------------

    if not hits or not context_sources or not context.strip():
        return Answer(
            text=NO_ANSWER,
            found=False,
            sources=[],
            retrieved=hits,
            elapsed={
                "retrieval": retrieval_time,
                "generation": 0.0,
            },
        )

    # -----------------------------
    # 3. Виклик мовної моделі
    # -----------------------------

    generation_started = time.perf_counter()

    try:
        llm_result = ask(
            question=question,
            context=context,
        )

    except LLMError as exc:
        generation_time = (
            time.perf_counter() - generation_started
        )

        return Answer(
            text=f"Модель тимчасово недоступна. {exc}",
            found=False,
            sources=[],
            retrieved=hits,
            elapsed={
                "retrieval": retrieval_time,
                "generation": generation_time,
            },
        )

    except Exception as exc:
        generation_time = (
            time.perf_counter() - generation_started
        )

        return Answer(
            text=f"Помилка генерації відповіді: {exc}",
            found=False,
            sources=[],
            retrieved=hits,
            elapsed={
                "retrieval": retrieval_time,
                "generation": generation_time,
            },
        )

    generation_time = (
        time.perf_counter() - generation_started
    )

    result = llm_result["result"]

    model_found = result["found"]
    refs = result["sources"]
    text = result["answer"]

    # -----------------------------
    # 4. Перевірка посилань
    # -----------------------------

    available = {
        source.ref: source
        for source in context_sources
    }

    # Модель сказала, що відповіді немає.
    if not model_found:
        return Answer(
            text=text or NO_ANSWER,
            found=False,
            sources=[],
            retrieved=hits,
            model=llm_result.get("model"),
            elapsed={
                "retrieval": retrieval_time,
                "generation": generation_time,
            },
            usage=llm_result.get("usage"),
        )

    # found=true, але немає жодного джерела.
    if not refs:
        return Answer(
            text=NO_ANSWER,
            found=False,
            sources=[],
            retrieved=hits,
            model=llm_result.get("model"),
            elapsed={
                "retrieval": retrieval_time,
                "generation": generation_time,
            },
            usage=llm_result.get("usage"),
        )

    # Перевіряємо, що модель не вигадала номер джерела.
    invalid_refs = [
        ref
        for ref in refs
        if ref not in available
    ]

    if invalid_refs:
        return Answer(
            text=NO_ANSWER,
            found=False,
            sources=[],
            retrieved=hits,
            model=llm_result.get("model"),
            elapsed={
                "retrieval": retrieval_time,
                "generation": generation_time,
            },
            usage=llm_result.get("usage"),
        )

    # Залишаємо лише джерела, на які реально
    # послалася модель.
    selected_sources = [
        available[ref]
        for ref in refs
    ]

    # -----------------------------
    # 5. Успішна відповідь
    # -----------------------------

    return Answer(
        text=text,
        found=True,
        sources=selected_sources,
        retrieved=hits,
        model=llm_result.get("model"),
        elapsed={
            "retrieval": retrieval_time,
            "generation": generation_time,
        },
        usage=llm_result.get("usage"),
    )