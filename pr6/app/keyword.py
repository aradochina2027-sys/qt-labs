"""Пошук за ключовими словами BM25."""

import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from .documents import Chunk
from .index import (
    DEFAULT_TOP_K,
    Hit
)


@dataclass
class KeywordIndex:
    chunks: list[Chunk]
    extra: dict = field(
        default_factory=dict
    )


def tokenize(
    text: str
) -> list[str]:
    """Розбити текст на слова."""

    text = text.lower()

    tokens = re.findall(
        r"[0-9a-zа-яіїєґ]+"
        r"(?:[-'ʼ][0-9a-zа-яіїєґ]+)*",
        text,
        flags=re.IGNORECASE
    )

    return tokens


def build(
    chunks: list[Chunk]
) -> KeywordIndex:
    """Створити BM25-індекс."""

    tokenized_corpus = []

    for chunk in chunks:
        tokenized_corpus.append(
            tokenize(
                chunk.text
            )
        )

    bm25 = BM25Okapi(
        tokenized_corpus
    )

    return KeywordIndex(
        chunks=chunks,
        extra={
            "bm25": bm25,
            "tokenized_corpus":
                tokenized_corpus
        }
    )


def matches_filters(
    chunk: Chunk,
    filters: dict | None
) -> bool:

    if not filters:
        return True

    for key, expected in filters.items():

        if expected in (
            None,
            ""
        ):
            continue

        if key not in chunk.metadata:
            raise ValueError(
                f"Невідоме поле фільтра: {key}"
            )

        actual = str(
            chunk.metadata.get(
                key,
                ""
            )
        ).strip().lower()

        expected_value = str(
            expected
        ).strip().lower()

        if actual != expected_value:
            return False

    return True


def search(
    index: KeywordIndex,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filters: dict | None = None
) -> list[Hit]:
    """Виконати BM25-пошук."""

    if index is None:
        raise ValueError(
            "Keyword-індекс "
            "не завантажено."
        )

    if top_k < 1:
        raise ValueError(
            "top_k має бути не менше 1."
        )

    query_tokens = tokenize(
        query
    )

    if not query_tokens:
        return []

    bm25 = index.extra[
        "bm25"
    ]

    scores = bm25.get_scores(
        query_tokens
    )

    allowed_indices = []

    for i, chunk in enumerate(
        index.chunks
    ):
        if matches_filters(
            chunk,
            filters
        ):
            allowed_indices.append(i)

    ranked = sorted(
        allowed_indices,
        key=lambda i: scores[i],
        reverse=True
    )

    hits = []

    for i in ranked:

        score = float(
            scores[i]
        )

        if score <= 0:
            continue

        hits.append(
            Hit(
                chunk=index.chunks[i],
                score=score
            )
        )

        if len(hits) >= top_k:
            break

    return hits