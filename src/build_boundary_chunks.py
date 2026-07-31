"""Build conservative chunks using validated document boundaries.

This version deliberately avoids broad, regex-driven section inference.

Processing steps:

1. Apply validated body start and end boundaries.
2. Apply page-level exclusion policies.
3. Remove safe forms of residual line-number noise.
4. Merge obvious paragraph and cross-page sentence fragments.
5. Recognize only high-confidence headings.
6. Build chunks at complete sentence boundaries.
7. Preserve document and PDF-page provenance.
"""

from __future__ import annotations

import csv
import json
import random
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from build_chunks import (
    read_json,
    read_page_policy,
    read_page_records,
    read_registry,
    resolve_page_policy,
    text_sha256,
)


ROOT_DIR = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    ROOT_DIR / "configs" / "chunking_boundary_aware.json"
)
BOUNDARY_PATH = (
    ROOT_DIR / "docs" / "document_boundaries.csv"
)

OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "chunks"
REVIEW_DIR = ROOT_DIR / "data" / "processed" / "reviews"

JSONL_OUTPUT_PATH = OUTPUT_DIR / "boundary_chunks.jsonl"
CSV_OUTPUT_PATH = OUTPUT_DIR / "boundary_chunks.csv"

SUMMARY_OUTPUT_PATH = (
    ROOT_DIR / "docs" / "boundary_chunk_build_summary.csv"
)

REVIEW_OUTPUT_PATH = (
    REVIEW_DIR / "boundary_chunk_review.csv"
)


@dataclass(frozen=True)
class TextUnit:
    """A text unit with PDF-page provenance."""

    text: str
    page_start: int
    page_end: int
    unit_type: str = "body"


@dataclass
class ChunkDraft:
    """An intermediate chunk before final serialization."""

    units: list[TextUnit]
    section_heading: str


def read_boundary_config() -> dict[str, dict[str, str]]:
    """Read validated document boundaries by source_id."""

    with BOUNDARY_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    boundaries: dict[str, dict[str, str]] = {}

    for row in rows:
        source_id = row["source_id"].strip()

        if not source_id:
            raise ValueError(
                "Empty source_id in document boundaries."
            )

        if source_id in boundaries:
            raise ValueError(
                f"Duplicate document boundary: {source_id}"
            )

        boundaries[source_id] = row

    if not boundaries:
        raise ValueError(
            "No document boundaries were found."
        )

    return boundaries


def normalize_paragraph(text: str) -> str:
    """Normalize whitespace inside one extracted paragraph."""

    text = text.replace("\u00a0", " ")
    text = text.replace("\u3000", " ")
    return re.sub(r"\s+", " ", text).strip()


def trim_from_marker(
    text: str,
    marker: str,
    source_id: str,
    page_number: int,
) -> str:
    """Keep text from a validated marker onward."""

    position = text.find(marker)

    if position < 0:
        raise ValueError(
            f"Start marker not found for {source_id} "
            f"page {page_number}: {marker!r}"
        )

    return text[position:]


def trim_before_marker(
    text: str,
    marker: str,
    source_id: str,
    page_number: int,
) -> str:
    """Keep text before a validated tail marker."""

    position = text.find(marker)

    if position < 0:
        raise ValueError(
            f"Tail marker not found for {source_id} "
            f"page {page_number}: {marker!r}"
        )

    return text[:position]


def clean_residual_line_numbers(
    paragraph: str,
    source_id: str,
) -> str:
    """Remove only conservative forms of residual line-number noise."""

    text = paragraph.strip()

    # Remove standalone numeric text blocks.
    if re.fullmatch(r"\d{1,4}", text):
        return ""

    if source_id != "SRC005":
        return text

    # Remove a numeric prefix before English prose.
    text = re.sub(
        r"^\d{1,4}\s+(?=[A-Za-z])",
        "",
        text,
    )

    # Remove a trailing line number attached to a short uppercase heading.
    uppercase_heading = re.fullmatch(
        r"([A-Z][A-Z0-9 /,&()\-]{2,70})\s+\d{1,4}",
        text,
    )

    if uppercase_heading:
        text = uppercase_heading.group(1).strip()

    return text


def split_page_paragraphs(
    text: str,
    source_id: str,
) -> list[str]:
    """Split page text using extracted block boundaries."""

    paragraphs: list[str] = []

    for raw_paragraph in re.split(
        r"\n\s*\n",
        text,
    ):
        paragraph = normalize_paragraph(raw_paragraph)

        if not paragraph:
            continue

        paragraph = clean_residual_line_numbers(
            paragraph=paragraph,
            source_id=source_id,
        )

        if paragraph:
            paragraphs.append(paragraph)

    return paragraphs


def ends_with_complete_boundary(text: str) -> bool:
    """Return whether text appears to end at a sentence boundary."""

    stripped = text.rstrip()

    if not stripped:
        return True

    return stripped.endswith(
        (
            "。",
            "！",
            "？",
            "；",
            ".",
            "!",
            "?",
            ";",
        )
    )


def starts_with_list_item(text: str) -> bool:
    """Detect a likely numbered or bulleted item."""

    return bool(
        re.match(
            r"^(?:"
            r"\d+[.、）)]|"
            r"[（(][一二三四五六七八九十\d]+[）)]|"
            r"[①②③④⑤⑥⑦⑧⑨⑩]|"
            r"[-•]"
            r")",
            text.strip(),
        )
    )


def is_high_confidence_heading(
    text: str,
    language: str,
) -> bool:
    """Recognize only conservative section headings.

    Reference entries and DOI strings are already removed by validated
    document boundaries, but the rules remain intentionally restrictive.
    """

    stripped = text.strip()

    if not stripped or len(stripped) > 90:
        return False

    if stripped.endswith(
        (
            "。",
            "；",
            ".",
            ";",
            "，",
            ",",
        )
    ):
        return False

    chinese_patterns = [
        r"^[一二三四五六七八九十]+、\S.{0,60}$",
        r"^[（(][一二三四五六七八九十]+[）)]\S.{0,60}$",
        r"^(推荐意见|推荐建议|临床问题|问题)\s*\d+\S*.{0,60}$",
    ]

    if language.lower().startswith("zh"):
        return any(
            re.fullmatch(pattern, stripped)
            for pattern in chinese_patterns
        )

    if re.fullmatch(
        r"(RECOMMENDATION|QUESTION)\s+\d+\S*.{0,60}",
        stripped,
        re.IGNORECASE,
    ):
        return True

    letters = [
        character
        for character in stripped
        if character.isalpha()
    ]

    if not letters:
        return False

    uppercase_ratio = sum(
        character.isupper()
        for character in letters
    ) / len(letters)

    word_count = len(stripped.split())

    return (
        1 <= word_count <= 12
        and uppercase_ratio >= 0.85
    )


def should_merge_fragments(
    previous: TextUnit,
    current: TextUnit,
    language: str,
) -> bool:
    """Decide whether two extracted blocks form one interrupted sentence."""

    if previous.unit_type == "heading":
        return False

    if current.unit_type == "heading":
        return False

    if ends_with_complete_boundary(previous.text):
        return False

    if starts_with_list_item(current.text):
        return False

    # English continuations commonly start with a lowercase character.
    if language.lower().startswith("en"):
        first_character = current.text.lstrip()[:1]

        if first_character and first_character.islower():
            return True

    # For Chinese extracted blocks, lack of terminal punctuation is the
    # main conservative signal of a split sentence.
    return True


def join_fragments(
    previous: TextUnit,
    current: TextUnit,
    language: str,
) -> TextUnit:
    """Join two fragments while preserving their page span."""

    if language.lower().startswith("zh"):
        joined_text = previous.text + current.text
    else:
        if previous.text.endswith("-"):
            joined_text = (
                previous.text[:-1]
                + current.text
            )
        else:
            joined_text = (
                previous.text
                + " "
                + current.text
            )

    return TextUnit(
        text=normalize_paragraph(joined_text),
        page_start=min(
            previous.page_start,
            current.page_start,
        ),
        page_end=max(
            previous.page_end,
            current.page_end,
        ),
        unit_type="body",
    )


def build_clean_document_units(
    config: dict[str, Any],
) -> dict[str, list[TextUnit]]:
    """Apply validated boundaries and build cleaned document units."""

    registry = read_registry()
    boundaries = read_boundary_config()
    page_policy = read_page_policy()
    page_records = read_page_records()

    records_by_source: dict[
        str,
        dict[int, dict[str, Any]],
    ] = defaultdict(dict)

    for record in page_records:
        records_by_source[
            str(record["source_id"])
        ][
            int(record["pdf_page_number"])
        ] = record

    units_by_source: dict[
        str,
        list[TextUnit],
    ] = {}

    exclude_pending = bool(
        config["exclude_pending_review_pages"]
    )

    for source_id in sorted(boundaries):
        if source_id not in registry:
            raise KeyError(
                f"{source_id} is missing from source registry."
            )

        if source_id not in records_by_source:
            raise KeyError(
                f"No extracted pages found for {source_id}."
            )

        boundary = boundaries[source_id]

        body_start_page = int(
            boundary["body_start_page"]
        )
        body_end_page = int(
            boundary["body_end_page"]
        )
        tail_start_page = int(
            boundary["reference_start_page"]
        )

        start_marker = boundary[
            "body_start_marker"
        ].strip()

        tail_marker = boundary[
            "reference_start_marker"
        ].strip()

        language = registry[source_id][
            "language"
        ]

        source_units: list[TextUnit] = []
        previous_included_page: int | None = None

        for page_number in range(
            body_start_page,
            body_end_page + 1,
        ):
            record = records_by_source[
                source_id
            ].get(page_number)

            if record is None:
                raise KeyError(
                    f"Missing page record: "
                    f"{source_id} page {page_number}"
                )

            resolved_policy, _ = resolve_page_policy(
                source_id=source_id,
                page_number=page_number,
                policy_map=page_policy,
                exclude_pending_review_pages=exclude_pending,
            )

            if resolved_policy in {
                "exclude",
                "exclude_pending_review",
            }:
                previous_included_page = None
                continue

            text = str(
                record.get("text", "")
            )

            if page_number == body_start_page:
                text = trim_from_marker(
                    text=text,
                    marker=start_marker,
                    source_id=source_id,
                    page_number=page_number,
                )

            if (
                page_number == tail_start_page
                and tail_start_page <= body_end_page
            ):
                text = trim_before_marker(
                    text=text,
                    marker=tail_marker,
                    source_id=source_id,
                    page_number=page_number,
                )

            paragraphs = split_page_paragraphs(
                text=text,
                source_id=source_id,
            )

            page_units: list[TextUnit] = []

            for paragraph in paragraphs:
                unit_type = (
                    "heading"
                    if is_high_confidence_heading(
                        text=paragraph,
                        language=language,
                    )
                    else "body"
                )

                page_units.append(
                    TextUnit(
                        text=paragraph,
                        page_start=page_number,
                        page_end=page_number,
                        unit_type=unit_type,
                    )
                )

            pages_are_consecutive = (
                previous_included_page is not None
                and previous_included_page
                == page_number - 1
            )

            for page_unit in page_units:
                if (
                    source_units
                    and (
                        pages_are_consecutive
                        or source_units[-1].page_end
                        == page_number
                    )
                    and should_merge_fragments(
                        previous=source_units[-1],
                        current=page_unit,
                        language=language,
                    )
                ):
                    previous = source_units.pop()

                    source_units.append(
                        join_fragments(
                            previous=previous,
                            current=page_unit,
                            language=language,
                        )
                    )
                else:
                    source_units.append(page_unit)

            previous_included_page = page_number

        units_by_source[source_id] = source_units

    return units_by_source


def split_body_unit(
    unit: TextUnit,
) -> list[TextUnit]:
    """Split one body unit at complete sentence boundaries."""

    if unit.unit_type == "heading":
        return [unit]

    pieces = [
        piece.strip()
        for piece in re.split(
            r"(?<=[。！？；])|(?<=[.!?;])\s+",
            unit.text,
        )
        if piece.strip()
    ]

    if not pieces:
        return []

    return [
        TextUnit(
            text=piece,
            page_start=unit.page_start,
            page_end=unit.page_end,
            unit_type="body",
        )
        for piece in pieces
    ]


def render_units(
    units: list[TextUnit],
) -> str:
    """Render units as readable chunk text."""

    return "\n\n".join(
        unit.text
        for unit in units
    ).strip()


def trailing_overlap_units(
    units: list[TextUnit],
    overlap_characters: int,
) -> list[TextUnit]:
    """Select trailing complete body sentences as overlap."""

    if overlap_characters <= 0:
        return []

    selected: list[TextUnit] = []

    for unit in reversed(units):
        if unit.unit_type == "heading":
            break

        candidate = [unit, *selected]

        if (
            selected
            and len(render_units(candidate))
            > overlap_characters
        ):
            break

        selected = candidate

        if len(render_units(selected)) >= overlap_characters:
            break

    return selected


def finalize_draft(
    draft: ChunkDraft,
) -> dict[str, Any]:
    """Convert a draft to intermediate chunk metadata."""

    text = render_units(draft.units)

    return {
        "text": text,
        "page_start": min(
            unit.page_start
            for unit in draft.units
        ),
        "page_end": max(
            unit.page_end
            for unit in draft.units
        ),
        "section_heading":
            draft.section_heading,
    }


def merge_short_final_chunks(
    chunks: list[dict[str, Any]],
    minimum_characters: int,
    max_characters: int,
) -> list[dict[str, Any]]:
    """Merge a short trailing chunk when doing so is safe."""

    if len(chunks) < 2:
        return chunks

    merged: list[dict[str, Any]] = []

    for chunk in chunks:
        if (
            merged
            and len(str(chunk["text"]))
            < minimum_characters
            and chunk["section_heading"]
            == merged[-1]["section_heading"]
        ):
            combined_text = (
                str(merged[-1]["text"])
                + "\n\n"
                + str(chunk["text"])
            ).strip()

            if len(combined_text) <= max_characters:
                merged[-1]["text"] = combined_text
                merged[-1]["page_end"] = max(
                    int(merged[-1]["page_end"]),
                    int(chunk["page_end"]),
                )
                continue

        merged.append(chunk)

    return merged


def chunk_source_units(
    units: list[TextUnit],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build conservative sentence-boundary chunks."""

    max_characters = int(
        config["max_characters"]
    )
    minimum_characters = int(
        config["minimum_characters"]
    )
    overlap_characters = int(
        config["overlap_characters"]
    )

    atomic_units: list[TextUnit] = []

    for unit in units:
        atomic_units.extend(
            split_body_unit(unit)
        )

    drafts: list[dict[str, Any]] = []
    current_units: list[TextUnit] = []
    current_heading = ""

    for atom in atomic_units:
        if atom.unit_type == "heading":
            if current_units:
                drafts.append(
                    finalize_draft(
                        ChunkDraft(
                            units=current_units,
                            section_heading=current_heading,
                        )
                    )
                )

            current_units = [atom]
            current_heading = atom.text
            continue

        candidate = [
            *current_units,
            atom,
        ]

        if (
            current_units
            and len(render_units(candidate))
            > max_characters
        ):
            drafts.append(
                finalize_draft(
                    ChunkDraft(
                        units=current_units,
                        section_heading=current_heading,
                    )
                )
            )

            current_units = trailing_overlap_units(
                units=current_units,
                overlap_characters=
                    overlap_characters,
            )

            while (
                current_units
                and len(
                    render_units(
                        [*current_units, atom]
                    )
                )
                > max_characters
            ):
                current_units.pop(0)

        current_units.append(atom)

    if current_units:
        drafts.append(
            finalize_draft(
                ChunkDraft(
                    units=current_units,
                    section_heading=current_heading,
                )
            )
        )

    return merge_short_final_chunks(
        chunks=drafts,
        minimum_characters=minimum_characters,
        max_characters=max_characters,
    )


def build_all_chunks() -> list[dict[str, Any]]:
    """Build chunks for every configured source."""

    config = read_json(CONFIG_PATH)
    registry = read_registry()

    units_by_source = build_clean_document_units(
        config=config
    )

    all_chunks: list[dict[str, Any]] = []

    for source_id in sorted(units_by_source):
        source_chunks = chunk_source_units(
            units=units_by_source[source_id],
            config=config,
        )

        for index, chunk in enumerate(
            source_chunks,
            start=1,
        ):
            text = str(chunk["text"])

            all_chunks.append(
                {
                    "chunk_id": (
                        f"{source_id}-B{index:03d}"
                    ),
                    "source_id": source_id,
                    "title":
                        registry[source_id]["title"],
                    "short_name":
                        registry[source_id]["short_name"],
                    "language":
                        registry[source_id]["language"],
                    "local_filename":
                        registry[source_id][
                            "local_filename"
                        ],
                    "pdf_page_start":
                        int(chunk["page_start"]),
                    "pdf_page_end":
                        int(chunk["page_end"]),
                    "section_heading":
                        str(chunk["section_heading"]),
                    "content_type": "body_text",
                    "chunk_method":
                        config["chunk_method"],
                    "character_count": len(text),
                    "text_sha256":
                        text_sha256(text),
                    "text": text,
                }
            )

    return all_chunks


def choose_review_sample(
    chunks: list[dict[str, Any]],
    sample_size: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    """Choose a stable, source-balanced review sample."""

    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for chunk in chunks:
        grouped[
            str(chunk["source_id"])
        ].append(chunk)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    for source_id in sorted(grouped):
        source_chunks = grouped[source_id]

        indexes = {
            0,
            len(source_chunks) // 2,
            len(source_chunks) - 1,
        }

        for index in sorted(indexes):
            chunk = source_chunks[index]
            chunk_id = str(chunk["chunk_id"])

            if chunk_id not in selected_ids:
                selected.append(chunk)
                selected_ids.add(chunk_id)

        cross_page_chunk = next(
            (
                chunk
                for chunk in source_chunks
                if int(chunk["pdf_page_start"])
                != int(chunk["pdf_page_end"])
            ),
            None,
        )

        if cross_page_chunk is not None:
            chunk_id = str(
                cross_page_chunk["chunk_id"]
            )

            if chunk_id not in selected_ids:
                selected.append(cross_page_chunk)
                selected_ids.add(chunk_id)

    remaining = [
        chunk
        for chunk in chunks
        if str(chunk["chunk_id"])
        not in selected_ids
    ]

    random_generator = random.Random(
        random_seed
    )
    random_generator.shuffle(remaining)

    for chunk in remaining:
        if len(selected) >= sample_size:
            break

        selected.append(chunk)
        selected_ids.add(
            str(chunk["chunk_id"])
        )

    return selected[:sample_size]


def write_outputs(
    chunks: list[dict[str, Any]],
) -> None:
    """Write full local chunks, public statistics, and local review data."""

    config = read_json(CONFIG_PATH)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    REVIEW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with JSONL_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        for chunk in chunks:
            file.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False,
                )
                + "\n"
            )

    with CSV_OUTPUT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(chunks[0].keys()),
        )
        writer.writeheader()
        writer.writerows(chunks)

    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for chunk in chunks:
        grouped[
            str(chunk["source_id"])
        ].append(chunk)

    summary_rows: list[dict[str, Any]] = []

    for source_id in sorted(grouped):
        source_chunks = grouped[source_id]
        lengths = [
            int(chunk["character_count"])
            for chunk in source_chunks
        ]

        summary_rows.append(
            {
                "source_id": source_id,
                "chunk_count": len(source_chunks),
                "cross_page_chunk_count": sum(
                    int(chunk["pdf_page_start"])
                    != int(chunk["pdf_page_end"])
                    for chunk in source_chunks
                ),
                "chunks_below_minimum": sum(
                    length
                    < int(
                        config[
                            "minimum_characters"
                        ]
                    )
                    for length in lengths
                ),
                "minimum_chunk_characters":
                    min(lengths),
                "median_chunk_characters":
                    round(
                        statistics.median(lengths),
                        1,
                    ),
                "mean_chunk_characters":
                    round(
                        statistics.mean(lengths),
                        1,
                    ),
                "maximum_chunk_characters":
                    max(lengths),
            }
        )

    with SUMMARY_OUTPUT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    sample = choose_review_sample(
        chunks=chunks,
        sample_size=int(
            config["manual_review_sample_size"]
        ),
        random_seed=int(
            config["random_seed"]
        ),
    )

    review_rows: list[dict[str, Any]] = []

    for chunk in sample:
        review_rows.append(
            {
                "chunk_id": chunk["chunk_id"],
                "source_id": chunk["source_id"],
                "pdf_page_start":
                    chunk["pdf_page_start"],
                "pdf_page_end":
                    chunk["pdf_page_end"],
                "section_heading":
                    chunk["section_heading"],
                "character_count":
                    chunk["character_count"],
                "text_preview":
                    chunk["text"],
                "starts_mid_sentence": "",
                "ends_mid_sentence": "",
                "single_topic": "",
                "heading_correct": "",
                "author_metadata_noise": "",
                "reference_noise": "",
                "line_number_noise": "",
                "context_complete": "",
                "medical_conditions_preserved": "",
                "usable_for_retrieval": "",
                "review_notes": "",
            }
        )

    with REVIEW_OUTPUT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                review_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(review_rows)


def print_summary(
    chunks: list[dict[str, Any]],
) -> None:
    """Print source-level chunk statistics."""

    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for chunk in chunks:
        grouped[
            str(chunk["source_id"])
        ].append(chunk)

    print("\nBoundary-aware chunk build")
    print("=" * 100)

    for source_id in sorted(grouped):
        source_chunks = grouped[source_id]

        lengths = [
            int(chunk["character_count"])
            for chunk in source_chunks
        ]

        print(
            f"{source_id}: "
            f"chunks={len(source_chunks)} | "
            f"cross_page="
            f"{sum(int(chunk['pdf_page_start']) != int(chunk['pdf_page_end']) for chunk in source_chunks)} | "
            f"min={min(lengths)} | "
            f"median={statistics.median(lengths):.1f} | "
            f"mean={statistics.mean(lengths):.1f} | "
            f"max={max(lengths)}"
        )

    print("-" * 100)
    print(f"Total chunks: {len(chunks)}")
    print(f"Chunk JSONL: {JSONL_OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_OUTPUT_PATH}")
    print(f"Manual review: {REVIEW_OUTPUT_PATH}")
    print("=" * 100)


def main() -> None:
    """Build, save, and summarize boundary-aware chunks."""

    chunks = build_all_chunks()

    if not chunks:
        raise ValueError(
            "No boundary-aware chunks were created."
        )

    write_outputs(chunks)
    print_summary(chunks)


if __name__ == "__main__":
    main()