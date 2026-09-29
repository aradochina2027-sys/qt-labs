"""Веб-рівень застосунку: сторінка розбору рахунків і JSON-ендпоінти."""

import os
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import extraction

load_dotenv()

app = FastAPI(title="Розбір рахунків — ПР7")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

UPLOAD_MAX_MB = float(
    os.getenv("UPLOAD_MAX_MB", "10")
)

UPLOAD_MAX_BYTES = int(
    UPLOAD_MAX_MB * 1024 * 1024
)

ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
}

app.mount(
    "/samples",
    StaticFiles(directory=SAMPLES_DIR),
    name="samples",
)


def result_to_dict(result: extraction.Result) -> dict:
    """Перетворити Result у JSON для веб-сторінки."""

    return {
        "decision": result.decision,
        "reasons": result.reasons,
        "document": result.document,
        "issues": [
            asdict(issue)
            for issue in result.issues
        ],
        "image": result.image,
        "model": result.model,
        "elapsed": result.elapsed,
        "usage": result.usage,
    }


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Віддати головну сторінку."""

    if not INDEX_PAGE.exists():
        raise HTTPException(
            status_code=500,
            detail="Не знайдено templates/index.html.",
        )

    return INDEX_PAGE.read_text(
        encoding="utf-8"
    )


@app.get("/api/samples")
def api_samples() -> dict:
    """Повернути перелік тестових зображень."""

    groups = {}

    if not SAMPLES_DIR.exists():
        return groups

    for folder in sorted(
        p
        for p in SAMPLES_DIR.iterdir()
        if p.is_dir()
    ):
        files = sorted(
            f.name
            for f in folder.iterdir()
            if (
                f.is_file()
                and f.suffix.lower() in IMAGE_SUFFIXES
            )
        )

        if files:
            groups[folder.name] = files

    return groups


@app.get("/api/health")
def health() -> dict:
    """Проста перевірка роботи FastAPI."""

    return {
        "status": "ok",
        "service": "pr7",
    }


@app.post("/api/extract")
async def api_extract(
    image: UploadFile = File(...)
) -> dict:
    """Прийняти зображення та запустити конвеєр."""

    filename = image.filename or ""

    suffix = Path(filename).suffix.lower()

    # ---------------------------------------------
    # 1. Перевірка розширення
    # ---------------------------------------------

    if suffix not in IMAGE_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=(
                "Непідтримуваний формат файлу. "
                "Дозволено PNG, JPG, JPEG або WEBP."
            ),
        )

    # ---------------------------------------------
    # 2. Перевірка MIME
    # ---------------------------------------------

    if (
        image.content_type
        and image.content_type
        not in ALLOWED_CONTENT_TYPES
    ):
        raise HTTPException(
            status_code=415,
            detail=(
                "Файл не має підтримуваного "
                "типу зображення."
            ),
        )

    # ---------------------------------------------
    # 3. Читаємо файл
    # ---------------------------------------------

    content = await image.read()

    # ---------------------------------------------
    # 4. Порожній файл
    # ---------------------------------------------

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Завантажено порожній файл.",
        )

    # ---------------------------------------------
    # 5. Завеликий файл
    # ---------------------------------------------

    if len(content) > UPLOAD_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Файл завеликий. Максимальний "
                f"розмір — {UPLOAD_MAX_MB:g} МБ."
            ),
        )

    # ---------------------------------------------
    # 6. Запуск повного конвеєра
    # ---------------------------------------------

    try:
        result = extraction.process(content)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Внутрішня помилка під час "
                f"обробки документа: {exc}"
            ),
        ) from exc

    return result_to_dict(result)