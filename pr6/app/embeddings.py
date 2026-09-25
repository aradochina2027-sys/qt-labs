"""Створення embeddings для документів і запитів."""

import os

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "intfloat/multilingual-e5-small"
)

_model = None


def get_model():
    """Завантажити модель тільки один раз."""

    global _model

    if _model is None:
        print(
            f"Завантаження моделі: {MODEL_NAME}"
        )

        _model = SentenceTransformer(
            MODEL_NAME
        )

    return _model


def embed_passages(
    texts: list[str]
) -> np.ndarray:
    """Створити embeddings для фрагментів."""

    if not texts:
        return np.empty(
            (0, 0),
            dtype=np.float32
        )

    model = get_model()

    passages = [
        f"passage: {text}"
        for text in texts
    ]

    vectors = model.encode(
        passages,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return np.asarray(
        vectors,
        dtype=np.float32
    )


def embed_query(
    text: str
) -> np.ndarray:
    """Створити embedding запиту користувача."""

    model = get_model()

    query = f"query: {text.strip()}"

    vector = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return np.asarray(
        vector,
        dtype=np.float32
    )