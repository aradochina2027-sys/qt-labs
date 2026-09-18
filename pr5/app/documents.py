"""Читання документів, метаданих і поділ на фрагменти."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = Path(__file__).parent.parent / "docs"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))


@dataclass
class Chunk:
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


def parse_front_matter(raw: str) -> tuple[dict, str]:
    """Відокремити метадані від основного тексту."""

    lines = raw.splitlines()

    if not lines or lines[0].strip() != "---":
        return {}, raw

    metadata = {}

    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body = "\n".join(lines[i + 1:]).lstrip("\n")
            return metadata, body

        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip()

    return {}, raw


def load_documents(docs_dir: Path = DOCS_DIR):
    """Прочитати всі документи з папки docs."""

    documents = []

    if not docs_dir.exists():
        raise FileNotFoundError(
            f"Папку docs не знайдено: {docs_dir}"
        )

    for path in sorted(docs_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue

        raw = path.read_text(encoding="utf-8")

        metadata, body = parse_front_matter(raw)

        documents.append(
            (
                path.name,
                metadata,
                body
            )
        )

    return documents


def split(
    text: str,
    source: str,
    metadata: dict
) -> list[Chunk]:
    """Поділити документ на осмислені фрагменти."""

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []

    current_parts = []
    current_length = 0
    chunk_number = 1

    title = metadata.get("title", source)

    def create_chunk(parts, number):
        if not parts:
            return

        body = "\n\n".join(parts).strip()

        chunk_text = f"{title}\n\n{body}"

        chunk_metadata = dict(metadata)
        chunk_metadata["chunk"] = number

        chunks.append(
            Chunk(
                text=chunk_text,
                source=source,
                metadata=chunk_metadata
            )
        )

    for paragraph in paragraphs:

        # Якщо один абзац дуже великий
        if len(paragraph) > CHUNK_SIZE:

            if current_parts:
                create_chunk(
                    current_parts,
                    chunk_number
                )

                chunk_number += 1
                current_parts = []
                current_length = 0

            step = max(
                1,
                CHUNK_SIZE - CHUNK_OVERLAP
            )

            for start in range(
                0,
                len(paragraph),
                step
            ):
                part = paragraph[
                    start:start + CHUNK_SIZE
                ]

                if part.strip():
                    create_chunk(
                        [part],
                        chunk_number
                    )

                    chunk_number += 1

                if start + CHUNK_SIZE >= len(paragraph):
                    break

            continue

        additional_length = (
            len(paragraph)
            + (2 if current_parts else 0)
        )

        if (
            current_parts
            and current_length + additional_length > CHUNK_SIZE
        ):
            create_chunk(
                current_parts,
                chunk_number
            )

            chunk_number += 1

            # Невелике перекриття між фрагментами
            last_paragraph = current_parts[-1]

            if len(last_paragraph) <= CHUNK_OVERLAP:
                current_parts = [last_paragraph]
                current_length = len(last_paragraph)
            else:
                current_parts = []
                current_length = 0

        current_parts.append(paragraph)

        current_length += (
            len(paragraph)
            + (2 if len(current_parts) > 1 else 0)
        )

    if current_parts:
        create_chunk(
            current_parts,
            chunk_number
        )

    return chunks


def load_chunks(
    docs_dir: Path = DOCS_DIR
) -> list[Chunk]:
    """Прочитати всі документи та створити фрагменти."""

    chunks = []

    documents = load_documents(docs_dir)

    for source, metadata, body in documents:

        document_chunks = split(
            body,
            source,
            metadata
        )

        chunks.extend(document_chunks)

    return chunks