"""Відбір фрагментів для моделі та збирання контексту."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from .documents import Chunk
from .embeddings import embed_query
from .index import (
    Hit,
    SearchIndex,
    search as semantic_search,
    SIMILARITY_THRESHOLD,
)
from .keyword import KeywordIndex


load_dotenv()


CONTEXT_CHUNKS = int(
    os.getenv("RAG_CONTEXT_CHUNKS", "4")
)

CONTEXT_BUDGET = int(
    os.getenv("RAG_CONTEXT_BUDGET", "1500")
)

SEARCH_TOP_K = int(
    os.getenv("SEARCH_TOP_K", "8")
)


@dataclass
class Source:
    """Фрагмент, який був переданий моделі як джерело."""

    ref: int
    chunk: Chunk
    score: float


def _is_allowed(chunk: Chunk) -> bool:
    """
    Перевірити, чи можна передавати фрагмент
    клієнтській мовній моделі.

    Архівні та внутрішні документи блокуються
    ще до формування контексту.
    """

    metadata = chunk.metadata or {}

    audience = str(
        metadata.get("audience", "")
    ).strip().lower()

    status = str(
        metadata.get("status", "")
    ).strip().lower()

    category = str(
        metadata.get("category", "")
    ).strip().lower()

    source = str(
        chunk.source
    ).strip().lower()

    # Внутрішні документи
    blocked_audiences = {
        "персонал",
        "внутрішній",
        "внутрішнє",
        "internal",
        "staff",
        "operator",
        "employee",
    }

    if audience in blocked_audiences:
        return False

    # Додаткова перевірка категорії документа
    blocked_categories = {
        "внутрішнє",
        "внутрішній",
        "internal",
    }

    if category in blocked_categories:
        return False

    # Додатковий захист за назвою файла
    if "внутріш" in source:
        return False

    # Архівні та застарілі документи
    blocked_statuses = {
        "архів",
        "архівний",
        "архівна",
        "archive",
        "archived",
        "obsolete",
        "deprecated",
    }

    if status in blocked_statuses:
        return False

    return True


def retrieve(
    query: str,
    index: SearchIndex,
    keyword_index: KeywordIndex,
    filters: dict | None = None,
) -> list[Hit]:
    """
    Знайти релевантні фрагменти для питання.

    Спочатку виконується семантичний пошук,
    після чого відкидаються внутрішні,
    архівні та дубльовані фрагменти.
    """

    query = query.strip()

    if not query:
        return []

    # Шукаємо трохи більше кандидатів,
    # ніж реально буде передано моделі.
    candidate_k = max(
        SEARCH_TOP_K,
        CONTEXT_CHUNKS * 2,
    )

    # Створюємо embedding питання.
    # embed_query використовує модель,
    # задану в конфігурації.
    query_vector = embed_query(query)

    # Семантичний пошук
    hits = semantic_search(
        index=index,
        query_vector=query_vector,
        top_k=candidate_k,
        filters=filters,
        threshold=SIMILARITY_THRESHOLD,
    )

    result: list[Hit] = []
    seen = set()

    for hit in hits:
        chunk = hit.chunk

        # Не допускаємо внутрішні
        # та архівні документи.
        if not _is_allowed(chunk):
            continue

        # Захист від однакових фрагментів.
        key = (
            chunk.source,
            chunk.metadata.get("chunk"),
            chunk.text,
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(hit)

        # У контекст потрібно лише обмежене
        # число найкращих фрагментів.
        if len(result) >= CONTEXT_CHUNKS:
            break

    return result


def _estimate_tokens(text: str) -> int:
    """
    Приблизно оцінити кількість токенів.

    Для простого обмеження контексту
    використовується оцінка ~4 символи на токен.
    """

    if not text:
        return 0

    return max(
        1,
        (len(text) + 3) // 4,
    )


def build_context(
    hits: list[Hit],
    budget: int = CONTEXT_BUDGET,
) -> tuple[str, list[Source]]:
    """
    Побудувати контекст для мовної моделі.

    Кожен фрагмент отримує номер [1], [2]...
    Ці номери модель потім використовує
    у полі sources.
    """

    if not hits or budget <= 0:
        return "", []

    blocks: list[str] = []
    sources: list[Source] = []

    used_tokens = 0

    for hit in hits:
        if len(sources) >= CONTEXT_CHUNKS:
            break

        chunk = hit.chunk
        metadata = chunk.metadata or {}

        title = str(
            metadata.get(
                "title",
                chunk.source,
            )
        ).strip()

        section = str(
            metadata.get(
                "section",
                "",
            )
        ).strip()

        date = str(
            metadata.get(
                "date",
                metadata.get(
                    "updated",
                    "",
                ),
            )
        ).strip()

        ref = len(sources) + 1

        header_parts = [
            f"[{ref}]",
            f"Документ: {title}",
        ]

        if section:
            header_parts.append(
                f"Розділ: {section}"
            )

        if date:
            header_parts.append(
                f"Дата: {date}"
            )

        header = "\n".join(
            header_parts
        )

        block = (
            f"{header}\n"
            f"Текст:\n"
            f"{chunk.text.strip()}"
        )

        block_tokens = _estimate_tokens(
            block
        )

        # Не перевищуємо бюджет контексту.
        if (
            used_tokens + block_tokens
            > budget
        ):
            continue

        blocks.append(block)

        sources.append(
            Source(
                ref=ref,
                chunk=chunk,
                score=hit.score,
            )
        )

        used_tokens += block_tokens

    context = "\n\n---\n\n".join(
        blocks
    )

    return context, sources