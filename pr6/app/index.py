"""Векторний індекс і семантичний пошук."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from .documents import Chunk

load_dotenv()


INDEX_DIR = Path(__file__).parent.parent / "index"

DEFAULT_TOP_K = int(os.getenv("SEARCH_TOP_K", "5"))

_threshold = os.getenv(
    "SIMILARITY_THRESHOLD",
    ""
).strip()

SIMILARITY_THRESHOLD = (
    float(_threshold)
    if _threshold
    else None
)


@dataclass
class Hit:
    chunk: Chunk
    score: float


@dataclass
class SearchIndex:
    chunks: list[Chunk]
    vectors: np.ndarray
    model_name: str
    extra: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.chunks)


def build(
    chunks: list[Chunk],
    vectors: np.ndarray,
    model_name: str
) -> SearchIndex:
    """Створити векторний індекс."""

    vectors = np.asarray(
        vectors,
        dtype=np.float32
    )

    if len(chunks) == 0:
        raise ValueError(
            "Немає фрагментів для індексування."
        )

    if len(chunks) != len(vectors):
        raise ValueError(
            "Кількість фрагментів не збігається "
            "з кількістю векторів."
        )

    if vectors.ndim != 2:
        raise ValueError(
            "Вектори повинні бути двовимірною матрицею."
        )

    # Нормалізація векторів
    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1.0
    vectors = vectors / norms

    sources = {
        chunk.source
        for chunk in chunks
    }

    return SearchIndex(
        chunks=chunks,
        vectors=vectors,
        model_name=model_name,
        extra={
            "documents": len(sources),
            "chunks": len(chunks),
            "dimension": int(vectors.shape[1])
        }
    )


def save(
    index: SearchIndex,
    path: Path = INDEX_DIR
) -> None:
    """Зберегти індекс на диск."""

    path.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        path / "vectors.npy",
        index.vectors
    )

    data = {
        "model_name": index.model_name,
        "extra": index.extra,
        "chunks": []
    }

    for chunk in index.chunks:
        data["chunks"].append(
            {
                "text": chunk.text,
                "source": chunk.source,
                "metadata": chunk.metadata
            }
        )

    with open(
        path / "chunks.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )


def load(
    path: Path = INDEX_DIR
) -> SearchIndex:
    """Завантажити готовий індекс."""

    vectors_path = path / "vectors.npy"
    chunks_path = path / "chunks.json"

    if (
        not vectors_path.exists()
        or not chunks_path.exists()
    ):
        raise FileNotFoundError(
            "Індекс не збудовано. "
            "Виконайте python ingest.py"
        )

    vectors = np.load(vectors_path)

    with open(
        chunks_path,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    chunks = []

    for item in data["chunks"]:
        chunks.append(
            Chunk(
                text=item["text"],
                source=item["source"],
                metadata=item.get(
                    "metadata",
                    {}
                )
            )
        )

    if len(chunks) != len(vectors):
        raise ValueError(
            "Кількість фрагментів не відповідає "
            "кількості векторів."
        )

    return SearchIndex(
        chunks=chunks,
        vectors=np.asarray(
            vectors,
            dtype=np.float32
        ),
        model_name=data["model_name"],
        extra=data.get(
            "extra",
            {}
        )
    )


def matches_filters(
    chunk: Chunk,
    filters: dict | None
) -> bool:
    """Перевірити фільтри метаданих."""

    if not filters:
        return True

    for key, expected in filters.items():

        if expected in (None, ""):
            continue

        if key not in chunk.metadata:
            raise ValueError(
                f"Невідоме поле фільтра: {key}"
            )

        actual = str(
            chunk.metadata.get(key, "")
        ).strip().lower()

        expected_value = str(
            expected
        ).strip().lower()

        if actual != expected_value:
            return False

    return True


def search(
    index: SearchIndex,
    query_vector: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    filters: dict | None = None,
    threshold: float | None = SIMILARITY_THRESHOLD
) -> list[Hit]:
    """Виконати семантичний пошук."""

    if index is None:
        raise ValueError(
            "Індекс не завантажено."
        )

    if top_k < 1:
        raise ValueError(
            "top_k має бути не менше 1."
        )

    query_vector = np.asarray(
        query_vector,
        dtype=np.float32
    ).reshape(-1)

    if (
        query_vector.shape[0]
        != index.vectors.shape[1]
    ):
        raise ValueError(
            "Розмірність запиту не відповідає індексу."
        )

    norm = np.linalg.norm(query_vector)

    if norm == 0:
        raise ValueError(
            "Отримано нульовий вектор запиту."
        )

    query_vector = query_vector / norm

    allowed_indices = []

    for i, chunk in enumerate(index.chunks):
        if matches_filters(
            chunk,
            filters
        ):
            allowed_indices.append(i)

    if not allowed_indices:
        return []

    matrix = index.vectors[
        allowed_indices
    ]

    # Для нормалізованих векторів
    # скалярний добуток = cosine similarity.
    scores = matrix @ query_vector

    # Сортуємо від найбільшого score до найменшого.
    order = np.argsort(scores)[::-1]

    hits = []

    for position in order:

        original_index = allowed_indices[
            int(position)
        ]

        score = float(
            scores[position]
        )

        if (
            threshold is not None
            and score < threshold
        ):
            continue

        hits.append(
            Hit(
                chunk=index.chunks[
                    original_index
                ],
                score=score
            )
        )

        if len(hits) >= top_k:
            break

    return hits