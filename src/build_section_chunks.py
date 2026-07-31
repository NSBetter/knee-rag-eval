"""Build page-spanning and section-aware guideline chunks.

This optimized chunker addresses errors found in the page-local
baseline:

1. Apply page exclusion and partial-page cleanup rules.
2. Remove remaining English preprint line numbers.
3. Join sentences split across consecutive PDF pages.
4. Start a new section at headings and recommendation statements.
5. Split only at paragraph or sentence boundaries.
6. Avoid mixing content from two adjacent sections.
7. Preserve source and PDF page spans for every chunk.
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
    is_reference_heading,
    read_json,
    read_page_policy,
    read_page_records,
    read_registry,
    resolve_page_policy,
    text_sha256,
)


ROOT_DIR = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    ROOT_DIR / "configs" / "chunking_section_aware.json"
)
CLEANUP_RULES_PATH = (
    ROOT_DIR / "docs" / "content_cleanup_rules.csv"
)

OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "chunks"
REVIEW_DIR = ROOT_DIR / "data" / "processed" / "reviews"

JSONL_OUTPUT_PATH = OUTPUT_DIR / "section_chunks.jsonl"
CSV_OUTPUT_PATH = OUTPUT_DIR / "section_chunks.csv"
REVIEW_OUTPUT_PATH = (
    REVIEW_DIR / "section_chunk_review.csv"
)
SUMMARY_OUTPUT_PATH = (
    ROOT_DIR / "docs" / "section_chunk_build_summary.csv"
)


@dataclass(frozen=True)
class TextUnit:
    """A paragraph or sentence with page provenance."""

    text: str
    page_start: int
    page_end: int


@dataclass
class Section:
    """A section delimited by a heading or recommendation."""

    heading: str
    units: list[TextUnit]


def read_cleanup_rules() -> dict[
    tuple[str, int],
    list[dict[str, str]],
]:
    """Read ordered partial-page cleanup rules."""

    if not CLEANUP_RULES_PATH.exists():
        return {}

    with CLEANUP_RULES_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    rules: dict[
        tuple[str, int],
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in rows:
        key = (
            row["source_id"].strip(),
            int(row["pdf_page_number"]),
        )
        rules[key].append(row)

    return dict(rules)


def normalize_paragraph(text: str) -> str:
    """Collapse internal whitespace in one paragraph."""

    text = text.replace("\u00a0", " ")
    text = text.replace("\u3000", " ")
    return re.sub(r"\s+", " ", text).strip()


def strip_preprint_line_numbers(
    text: str,
    source_id: str,
) -> str:
    """Remove residual line numbers from the English preprint.

    This rule is restricted to SRC005 because removing every numeric
    prefix globally could damage numbered recommendations.
    """

    if source_id != "SRC005":
        return text

    # Remove standalone numeric lines.
    text = re.sub(
        r"(?m)^\s*\d{1,4}\s*$",
        "",
        text,
    )

    # Remove numeric prefixes before English text.
    text = re.sub(
        r"(?m)^\s*\d{1,4}\s+(?=[A-Za-z])",
        "",
        text,
    )

    return text


def apply_cleanup_rules(
    text: str,
    source_id: str,
    page_number: int,
    cleanup_rules: dict[
        tuple[str, int],
        list[dict[str, str]],
    ],
) -> str:
    """Apply trim rules to one page in CSV order."""

    cleaned = text

    for rule in cleanup_rules.get(
        (source_id, page_number),
        [],
    ):
        action = rule["action"].strip().lower()
        marker = rule["marker"].strip()

        marker_index = cleaned.find(marker)

        if marker_index < 0:
            print(
                f"Warning: cleanup marker not found: "
                f"{source_id} page {page_number}, "
                f"marker={marker!r}"
            )
            continue

        if action == "trim_before":
            cleaned = cleaned[marker_index:]

        elif action == "trim_after":
            cleaned = cleaned[:marker_index]

        else:
            raise ValueError(
                f"Unsupported cleanup action {action!r} "
                f"for {source_id} page {page_number}"
            )

    return cleaned.strip()


def split_paragraphs(text: str) -> list[str]:
    """Split page text while preserving extracted block boundaries."""

    return [
        normalize_paragraph(paragraph)
        for paragraph in re.split(
            r"\n\s*\n",
            text,
        )
        if normalize_paragraph(paragraph)
    ]


def ends_with_sentence_boundary(text: str) -> bool:
    """Check whether a unit appears to end a complete sentence."""

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
            "：",
            ":",
        )
    )


def join_cross_page_text(
    left_text: str,
    right_text: str,
    language: str,
) -> str:
    """Join text that was divided only by a PDF page boundary."""

    if language.lower().startswith("zh"):
        return f"{left_text}{right_text}"

    return f"{left_text} {right_text}"


SECTION_PATTERNS = [
    re.compile(
        r"^[一二三四五六七八九十]+、"
    ),
    re.compile(
        r"^\d+(?:\.\d+)*[.、]\s*\S+"
    ),
    re.compile(
        r"^(推荐意见|推荐建议|临床问题|问题)\s*\d+"
    ),
    re.compile(
        r"^(recommendation|question)\s*\d+",
        re.IGNORECASE,
    ),
]


def starts_new_section(text: str) -> bool:
    """Detect major headings and recommendation boundaries."""

    stripped = text.strip()

    if any(
        pattern.match(stripped)
        for pattern in SECTION_PATTERNS
    ):
        return True

    if (
        len(stripped) <= 80
        and re.fullmatch(
            r"[A-Z][A-Z0-9 /,&():\-]+",
            stripped,
        )
    ):
        return True

    return False


def extract_section_heading(text: str) -> str:
    """Create a short label for section metadata and repetition."""

    stripped = text.strip()

    recommendation_match = re.match(
        r"^((?:推荐意见|推荐建议|临床问题|问题)\s*\d+)",
        stripped,
    )

    if recommendation_match:
        return recommendation_match.group(1)

    english_match = re.match(
        r"^((?:recommendation|question)\s*\d+)",
        stripped,
        re.IGNORECASE,
    )

    if english_match:
        return english_match.group(1)

    if len(stripped) <= 100:
        return stripped

    return ""


def split_sentences(unit: TextUnit) -> list[TextUnit]:
    """Split a paragraph into sentence-complete atomic units."""

    parts = [
        part.strip()
        for part in re.split(
            r"(?<=[。！？；])|(?<=[.!?;])\s+",
            unit.text,
        )
        if part.strip()
    ]

    if not parts:
        return []

    return [
        TextUnit(
            text=part,
            page_start=unit.page_start,
            page_end=unit.page_end,
        )
        for part in parts
    ]


def build_document_units(
    config: dict[str, Any],
) -> tuple[
    dict[str, list[TextUnit]],
    list[dict[str, Any]],
]:
    """Build cleaned units and merge cross-page sentence fragments."""

    registry = read_registry()
    page_policy = read_page_policy()
    cleanup_rules = read_cleanup_rules()
    page_records = read_page_records()

    exclude_pending = bool(
        config["exclude_pending_review_pages"]
    )
    reference_headings = [
        str(value)
        for value in config["reference_headings"]
    ]

    units_by_source: dict[
        str,
        list[TextUnit],
    ] = defaultdict(list)

    page_audit: list[dict[str, Any]] = []

    reached_references: dict[str, bool] = defaultdict(bool)
    previous_included_page: dict[str, int | None] = defaultdict(
        lambda: None
    )

    for record in page_records:
        source_id = str(record["source_id"])
        page_number = int(record["pdf_page_number"])

        if source_id not in registry:
            raise KeyError(
                f"Unknown source_id: {source_id}"
            )

        if reached_references[source_id]:
            page_audit.append(
                {
                    "source_id": source_id,
                    "pdf_page_number": page_number,
                    "status": "excluded_after_references",
                    "unit_count": 0,
                }
            )
            continue

        policy, reason = resolve_page_policy(
            source_id=source_id,
            page_number=page_number,
            policy_map=page_policy,
            exclude_pending_review_pages=exclude_pending,
        )

        if policy in {
            "exclude",
            "exclude_pending_review",
        }:
            page_audit.append(
                {
                    "source_id": source_id,
                    "pdf_page_number": page_number,
                    "status": policy,
                    "unit_count": 0,
                    "reason": reason,
                }
            )
            previous_included_page[source_id] = None
            continue

        page_text = strip_preprint_line_numbers(
            text=str(record.get("text", "")),
            source_id=source_id,
        )

        page_text = apply_cleanup_rules(
            text=page_text,
            source_id=source_id,
            page_number=page_number,
            cleanup_rules=cleanup_rules,
        )

        paragraphs = split_paragraphs(page_text)
        page_units: list[TextUnit] = []
        reference_found = False

        for paragraph in paragraphs:
            if is_reference_heading(
                paragraph=paragraph,
                reference_headings=reference_headings,
            ):
                reached_references[source_id] = True
                reference_found = True
                break

            page_units.append(
                TextUnit(
                    text=paragraph,
                    page_start=page_number,
                    page_end=page_number,
                )
            )

        source_units = units_by_source[source_id]

        pages_are_consecutive = (
            previous_included_page[source_id] is not None
            and previous_included_page[source_id]
            == page_number - 1
        )

        if (
            pages_are_consecutive
            and source_units
            and page_units
            and not ends_with_sentence_boundary(
                source_units[-1].text
            )
            and not starts_new_section(
                page_units[0].text
            )
        ):
            previous = source_units[-1]
            first_current = page_units.pop(0)

            source_units[-1] = TextUnit(
                text=join_cross_page_text(
                    left_text=previous.text,
                    right_text=first_current.text,
                    language=registry[source_id]["language"],
                ),
                page_start=previous.page_start,
                page_end=first_current.page_end,
            )

        source_units.extend(page_units)

        previous_included_page[source_id] = page_number

        page_audit.append(
            {
                "source_id": source_id,
                "pdf_page_number": page_number,
                "status": (
                    "included_before_references"
                    if reference_found
                    else "included"
                ),
                "unit_count": len(page_units),
                "reason": "",
            }
        )

    return dict(units_by_source), page_audit


def divide_into_sections(
    units: list[TextUnit],
) -> list[Section]:
    """Group text units without crossing detected section boundaries."""

    sections: list[Section] = []
    current_units: list[TextUnit] = []
    current_heading = ""

    for unit in units:
        if starts_new_section(unit.text):
            if current_units:
                sections.append(
                    Section(
                        heading=current_heading,
                        units=current_units,
                    )
                )

            current_units = [unit]
            current_heading = extract_section_heading(
                unit.text
            )
        else:
            current_units.append(unit)

    if current_units:
        sections.append(
            Section(
                heading=current_heading,
                units=current_units,
            )
        )

    return sections


def joined_text_length(
    units: list[TextUnit],
) -> int:
    """Calculate chunk length using paragraph separators."""

    if not units:
        return 0

    return sum(len(unit.text) for unit in units) + (
        2 * (len(units) - 1)
    )


def trailing_overlap(
    units: list[TextUnit],
    target_characters: int,
) -> list[TextUnit]:
    """Select complete trailing sentences as overlap."""

    selected: list[TextUnit] = []

    for unit in reversed(units):
        candidate = [unit, *selected]

        if (
            selected
            and joined_text_length(candidate)
            > target_characters
        ):
            break

        selected = candidate

        if joined_text_length(selected) >= target_characters:
            break

    return selected


def render_chunk_text(
    units: list[TextUnit],
    section_heading: str,
    repeat_heading: bool,
) -> str:
    """Render one chunk and optionally restore its section label."""

    text = "\n\n".join(
        unit.text
        for unit in units
    ).strip()

    if (
        repeat_heading
        and section_heading
        and not text.startswith(section_heading)
    ):
        text = f"{section_heading}\n\n{text}"

    return text


def chunk_section(
    section: Section,
    max_characters: int,
    overlap_characters: int,
    repeat_heading: bool,
) -> list[dict[str, Any]]:
    """Split one section only at complete sentence boundaries."""

    atoms: list[TextUnit] = []

    for unit in section.units:
        atoms.extend(split_sentences(unit))

    chunks: list[dict[str, Any]] = []
    current: list[TextUnit] = []

    for atom in atoms:
        candidate = [*current, atom]

        candidate_text = render_chunk_text(
            units=candidate,
            section_heading=section.heading,
            repeat_heading=repeat_heading,
        )

        if current and len(candidate_text) > max_characters:
            final_text = render_chunk_text(
                units=current,
                section_heading=section.heading,
                repeat_heading=repeat_heading,
            )

            chunks.append(
                {
                    "text": final_text,
                    "page_start": min(
                        unit.page_start
                        for unit in current
                    ),
                    "page_end": max(
                        unit.page_end
                        for unit in current
                    ),
                }
            )

            current = trailing_overlap(
                units=current,
                target_characters=overlap_characters,
            )

            while current:
                test_text = render_chunk_text(
                    units=[*current, atom],
                    section_heading=section.heading,
                    repeat_heading=repeat_heading,
                )

                if len(test_text) <= max_characters:
                    break

                current.pop(0)

        current.append(atom)

    if current:
        final_text = render_chunk_text(
            units=current,
            section_heading=section.heading,
            repeat_heading=repeat_heading,
        )

        chunks.append(
            {
                "text": final_text,
                "page_start": min(
                    unit.page_start
                    for unit in current
                ),
                "page_end": max(
                    unit.page_end
                    for unit in current
                ),
            }
        )

    return chunks


def build_chunks() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Build all section-aware chunks."""

    config = read_json(CONFIG_PATH)
    registry = read_registry()

    units_by_source, page_audit = build_document_units(
        config=config
    )

    all_chunks: list[dict[str, Any]] = []

    for source_id in sorted(units_by_source):
        sections = divide_into_sections(
            units_by_source[source_id]
        )

        source_chunk_number = 0

        for section_number, section in enumerate(
            sections,
            start=1,
        ):
            section_chunks = chunk_section(
                section=section,
                max_characters=int(
                    config["max_characters"]
                ),
                overlap_characters=int(
                    config["overlap_characters"]
                ),
                repeat_heading=bool(
                    config["repeat_section_heading"]
                ),
            )

            for chunk in section_chunks:
                source_chunk_number += 1
                chunk_text = str(chunk["text"])

                all_chunks.append(
                    {
                        "chunk_id": (
                            f"{source_id}-"
                            f"S{section_number:03d}-"
                            f"C{source_chunk_number:03d}"
                        ),
                        "source_id": source_id,
                        "title": registry[source_id]["title"],
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
                        "section_number": section_number,
                        "section_heading": section.heading,
                        "content_type": "body_text",
                        "chunk_method":
                            config["chunk_method"],
                        "character_count": len(chunk_text),
                        "text_sha256":
                            text_sha256(chunk_text),
                        "text": chunk_text,
                    }
                )

    return all_chunks, page_audit


def write_outputs(
    chunks: list[dict[str, Any]],
) -> None:
    """Write local full-text outputs and public statistics."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

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
        grouped[str(chunk["source_id"])].append(chunk)

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
                "minimum_chunk_characters": min(lengths),
                "median_chunk_characters":
                    round(statistics.median(lengths), 1),
                "mean_chunk_characters":
                    round(statistics.mean(lengths), 1),
                "maximum_chunk_characters": max(lengths),
            }
        )

    with SUMMARY_OUTPUT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    write_review_sample(chunks)


def choose_review_sample(
    chunks: list[dict[str, Any]],
    sample_size: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    """Choose a deterministic source-balanced sample."""

    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for chunk in chunks:
        grouped[str(chunk["source_id"])].append(chunk)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    for source_id in sorted(grouped):
        source_chunks = grouped[source_id]

        candidate_indexes = {
            0,
            len(source_chunks) // 2,
            len(source_chunks) - 1,
        }

        for index in sorted(candidate_indexes):
            chunk = source_chunks[index]
            chunk_id = str(chunk["chunk_id"])

            if chunk_id not in selected_ids:
                selected.append(chunk)
                selected_ids.add(chunk_id)

        cross_page = next(
            (
                chunk
                for chunk in source_chunks
                if chunk["pdf_page_start"]
                != chunk["pdf_page_end"]
            ),
            None,
        )

        if cross_page is not None:
            chunk_id = str(cross_page["chunk_id"])

            if chunk_id not in selected_ids:
                selected.append(cross_page)
                selected_ids.add(chunk_id)

    remaining = [
        chunk
        for chunk in chunks
        if str(chunk["chunk_id"]) not in selected_ids
    ]

    random_generator = random.Random(random_seed)
    random_generator.shuffle(remaining)

    for chunk in remaining:
        if len(selected) >= sample_size:
            break

        selected.append(chunk)
        selected_ids.add(str(chunk["chunk_id"]))

    return selected[:sample_size]


def write_review_sample(
    chunks: list[dict[str, Any]],
) -> None:
    """Write local optimized-chunk review sheet."""

    config = read_json(CONFIG_PATH)

    sample = choose_review_sample(
        chunks=chunks,
        sample_size=int(
            config["manual_review_sample_size"]
        ),
        random_seed=int(config["random_seed"]),
    )

    rows: list[dict[str, Any]] = []

    for chunk in sample:
        rows.append(
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
                "text_preview": chunk["text"],
                "starts_mid_sentence": "",
                "ends_mid_sentence": "",
                "single_section": "",
                "author_metadata_noise": "",
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
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    chunks: list[dict[str, Any]],
) -> None:
    """Print source-level statistics."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for chunk in chunks:
        grouped[str(chunk["source_id"])].append(chunk)

    print("\nSection-aware chunk build")
    print("=" * 100)

    for source_id in sorted(grouped):
        source_chunks = grouped[source_id]
        lengths = [
            int(chunk["character_count"])
            for chunk in source_chunks
        ]

        cross_page_count = sum(
            int(chunk["pdf_page_start"])
            != int(chunk["pdf_page_end"])
            for chunk in source_chunks
        )

        print(
            f"{source_id}: "
            f"chunks={len(source_chunks)} | "
            f"cross_page={cross_page_count} | "
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
    """Build and save optimized chunks."""

    chunks, _ = build_chunks()

    if not chunks:
        raise ValueError("No optimized chunks were created.")

    write_outputs(chunks)
    print_summary(chunks)


if __name__ == "__main__":
    main()