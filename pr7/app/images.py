"""Підготовка зображення до надсилання моделі."""

import io
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from PIL import Image, ImageOps, UnidentifiedImageError

load_dotenv()

UPLOAD_MAX_BYTES = int(float(os.getenv("UPLOAD_MAX_MB", "10")) * 1024 * 1024)
IMAGE_MAX_SIDE = int(os.getenv("IMAGE_MAX_SIDE", "1600"))

# Додатковий захист від дуже великих зображень.
MAX_PIXELS = 40_000_000

# Надто маленькі зображення одразу вважаємо непридатними.
MIN_SIDE = 300


class ImageError(Exception):
    """Файл не можна віддати моделі. Повідомлення — для користувача."""


@dataclass
class PreparedImage:
    """Зображення, готове до надсилання."""

    data: bytes
    mime: str
    original: dict
    sent: dict


def prepare(content: bytes) -> PreparedImage:
    """Перевірити файл і підготувати зображення для моделі."""

    # 1. Перевірка порожнього файла
    if not content:
        raise ImageError("Файл порожній.")

    # 2. Перевірка розміру файла
    if len(content) > UPLOAD_MAX_BYTES:
        max_mb = UPLOAD_MAX_BYTES / (1024 * 1024)
        raise ImageError(
            f"Файл завеликий. Максимальний розмір — {max_mb:g} МБ."
        )

    try:
        # 3. Відкриваємо байти як зображення
        with Image.open(io.BytesIO(content)) as source:
            source.verify()

        # verify() не дозволяє далі нормально працювати з тим самим
        # об'єктом, тому відкриваємо зображення повторно.
        with Image.open(io.BytesIO(content)) as source:
            # 4. Враховуємо EXIF-орієнтацію
            image = ImageOps.exif_transpose(source)

            width, height = image.size

            original = {
                "width": width,
                "height": height,
                "bytes": len(content),
            }

            # 5. Захист від надмірної кількості пікселів
            if width * height > MAX_PIXELS:
                raise ImageError(
                    "Зображення має надто велику роздільність."
                )

            # 6. Надто маленький знімок
            if width < MIN_SIDE or height < MIN_SIDE:
                raise ImageError(
                    "Зображення має надто малу роздільність. "
                    "Зробіть чіткіший або більший знімок документа."
                )

            # 7. Зменшуємо до IMAGE_MAX_SIDE
            if max(width, height) > IMAGE_MAX_SIDE:
                scale = IMAGE_MAX_SIDE / max(width, height)

                new_width = max(1, round(width * scale))
                new_height = max(1, round(height * scale))

                image = image.resize(
                    (new_width, new_height),
                    Image.Resampling.LANCZOS,
                )

            # 8. Переводимо у RGB
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")

            # 9. Зберігаємо як PNG без втрат
            output = io.BytesIO()
            image.save(
                output,
                format="PNG",
                optimize=True,
            )

            data = output.getvalue()

            sent = {
                "width": image.width,
                "height": image.height,
                "bytes": len(data),
            }

            # 10. Повертаємо підготовлене зображення
            return PreparedImage(
                data=data,
                mime="image/png",
                original=original,
                sent=sent,
            )

    except ImageError:
        raise

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        raise ImageError(
            "Файл не вдалося відкрити як коректне зображення."
        ) from exc