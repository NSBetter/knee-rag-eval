"""Build traceable baseline chunks from cleaned guideline pages.

Processing steps:

1. Read page-level JSONL produced by extract_corpus.py.
2. Apply the manually reviewed page content policy.
3. Stop indexing when a reference section begins.
4. Split text into paragraphs and sentences.
5. Build overlapping page-local chunks.
6. Save traceable chunk records as JSONL and CSV.
7. Generate aggregate statistics and a local manual-review sample.

Raw guideline text and chunk previews remain under data/processed/
and are excluded from the public Git repository.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT_DIR / "configs" / "chunking_baseline.json"
REGISTRY_PATH = ROOT_DIR / "docs" / "source_registry.csv"
POLICY_PATH = ROOT_DIR / "docs" / "page_content_policy.csv"

PAGE_DIR = ROOT_DIR / "data" / "processed" / "pages"
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "chunks"
REVIEW_DIR = ROOT_DIR / "data" / "processed" / "reviews"

CHUNK_JSONL_PATH = OUTPUT_DIR / "baseline_chunks.jsonl"
CHUNK_CSV_PATH = OUTPUT_DIR / "baseline_chunks.csv"
LOCAL_REVIEW_PATH = REVIEW_DIR / "baseline_chunk_review.csv"
SUMMARY_PATH = ROOT_DIR / "docs" / "chunk_build_summary.csv"


def read_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON configuration file."""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV file."""

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def read_registry() -> dict[str, dict[str, str]]:
    """Read the source registry and index records by source_id."""

    rows = read_csv_rows(REGISTRY_PATH)

    registry: dict[str, dict[str, str]] = {}

    for row in rows:
        source_id = row["source_id"].strip()

        if not source_id:
            raise ValueError("Empty source_id in source registry.")

        if source_id in registry:
            raise ValueError(
                f"Duplicate source_id in registry: {source_id}"
            )

        registry[source_id] = row

    return registry


def read_page_policy() -> dict[tuple[str, int], dict[str, str]]:
    """Read page-level inclusion and exclusion rules."""

    if not POLICY_PATH.exists():
        return {}

    rows = read_csv_rows(POLICY_PATH)
    policy: dict[tuple[str, int], dict[str, str]] = {}

    for row in rows:
        source_id = row["source_id"].strip()
        page_number = int(row["pdf_page_number"])

        key = (source_id, page_number)

        if key in policy:
            raise ValueError(
                f"Duplicate page policy for {source_id} page "
                f"{page_number}"
            )

        policy[key] = row

    return policy


def read_page_records() -> list[dict[str, Any]]:
    """Read all page-level extraction records."""

    records: list[dict[str, Any]] = []

    for jsonl_path in sorted(PAGE_DIR.glob("*_pages.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {jsonl_path}, "
                        f"line {line_number}: {exc}"
                    ) from exc

                records.append(record)

    if not records:
        raise FileNotFoundError(
            f"No page JSONL records found under {PAGE_DIR}"
        )

    records.sort(
        key=lambda row: (
            str(row["source_id"]),
            int(row["pdf_page_number"]),
        )
    )

    return records


def normalize_text(text: str) -> str:
    """Normalize spaces while preserving paragraph boundaries."""

    text = text.replace("\u00a0", " ")
    text = text.replace("\u3000", " ")

    paragraphs = [
        re.sub(r"[ \t]+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", text)
    ]

    return "\n\n".join(
        paragraph
        for paragraph in paragraphs
        if paragraph
    )


def normalized_heading(text: str) -> str:
    """Normalize a possible section heading for exact comparison."""

    normalized = re.sub(r"\s+", "", text).strip().lower()
    normalized = re.sub(r"^[一二三四五六七八九十\d.、（）()]+", "", normalized)
    return normalized.strip("：:")


def is_reference_heading(
    paragraph: str,
    reference_headings: list[str],
) -> bool:
    """Return whether a paragraph starts a reference section."""

    normalized = normalized_heading(paragraph)

    return any(
        normalized == normalized_heading(heading)
        for heading in reference_headings
    )


def split_long_paragraph(
    paragraph: str,
    max_characters: int,
) -> list[str]:
    """Split an overlong paragraph into sentence-level units."""

    if len(paragraph) <= max_characters:
        return [paragraph]

    sentences = [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[。！？；.!?;])\s*",
            paragraph,
        )
        if sentence.strip()
    ]

    units: list[str] = []

    for sentence in sentences:
        if len(sentence) <= max_characters:
            units.append(sentence)
            continue

        # Defensive hard split for unusually long text with no
        # usable sentence punctuation.
        for start in range(0, len(sentence), max_characters):
            piece = sentence[start : start + max_characters].strip()

            if piece:
                units.append(piece)

    return units


def build_text_units(
    text: str,
    max_characters: int,
    reference_headings: list[str],
) -> tuple[list[str], bool]:
    """Convert page text into paragraph or sentence units.

    Returns:
        units:
            Text units that may be included in chunks.
        reference_heading_found:
            Whether this page starts the reference section.
    """

    normalized = normalize_text(text)

    if not normalized:
        return [], False

    units: list[str] = []

    for paragraph in normalized.split("\n\n"):
        if is_reference_heading(
            paragraph=paragraph,
            reference_headings=reference_headings,
        ):
            return units, True

        units.extend(
            split_long_paragraph(
                paragraph=paragraph,
                max_characters=max_characters,
            )
        )

    return units, False


def joined_length(
    units: list[str],
    separator: str,
) -> int:
    """Calculate the final length of joined text units."""

    if not units:
        return 0

    return sum(len(unit) for unit in units) + (
        len(separator) * (len(units) - 1)
    )


def select_overlap_units(
    units: list[str],
    overlap_characters: int,
    separator: str,
) -> list[str]:
    """Select trailing units for the next chunk's overlap."""

    if overlap_characters <= 0:
        return []

    selected: list[str] = []

    for unit in reversed(units):
        candidate = [unit, *selected]

        if (
            selected
            and joined_length(candidate, separator)
            > overlap_characters
        ):
            break

        selected = candidate

        if joined_length(selected, separator) >= overlap_characters:
            break

    return selected


def chunk_text_units(
    units: list[str],
    max_characters: int,
    overlap_characters: int,
    separator: str,
) -> list[str]:
    """Build overlapping chunks without intentionally splitting units."""

    if overlap_characters >= max_characters:
        raise ValueError(
            "overlap_characters must be smaller than max_characters."
        )

    chunks: list[str] = []
    current_units: list[str] = []

    for unit in units:
        candidate = [*current_units, unit]

        if (
            current_units
            and joined_length(candidate, separator)
            > max_characters
        ):
            chunk_text = separator.join(current_units).strip()

            if chunk_text:
                chunks.append(chunk_text)

            current_units = select_overlap_units(
                units=current_units,
                overlap_characters=overlap_characters,
                separator=separator,
            )

            # If overlap plus the new unit exceeds the maximum,
            # drop the oldest overlap units until it fits.
            while (
                current_units
                and joined_length(
                    [*current_units, unit],
                    separator,
                )
                > max_characters
            ):
                current_units.pop(0)

        current_units.append(unit)

    final_text = separator.join(current_units).strip()

    if final_text:
        chunks.append(final_text)

    return chunks


def text_sha256(text: str) -> str:
    """Create a stable fingerprint for one text chunk."""

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def resolve_page_policy(
    source_id: str,
    page_number: int,
    policy_map: dict[tuple[str, int], dict[str, str]],
    exclude_pending_review_pages: bool,
) -> tuple[str, str]:
    """Resolve whether a page should be included."""

    policy_row = policy_map.get((source_id, page_number))

    if policy_row is None:
        return "include", ""

    chunk_policy = policy_row["chunk_policy"].strip().lower()
    reason = policy_row.get("reason", "").strip()

    if chunk_policy == "review" and exclude_pending_review_pages:
        return "exclude_pending_review", reason

    if chunk_policy not in {
        "include",
        "exclude",
        "review",
    }:
        raise ValueError(
            f"Unsupported chunk_policy={chunk_policy!r} for "
            f"{source_id} page {page_number}"
        )

    return chunk_policy, reason


def build_chunks() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Build chunks and collect page-level processing statistics."""

    config = read_json(CONFIG_PATH)
    registry = read_registry()
    policy_map = read_page_policy()
    page_records = read_page_records()

    max_characters = int(config["max_characters"])
    overlap_characters = int(config["overlap_characters"])
    separator = str(config["separator"])
    reference_headings = [
        str(value)
        for value in config["reference_headings"]
    ]

    exclude_pending = bool(
        config["exclude_pending_review_pages"]
    )
    stop_at_references = bool(
        config["stop_at_reference_section"]
    )

    chunks: list[dict[str, Any]] = []
    page_audit_rows: list[dict[str, Any]] = []

    reached_references: dict[str, bool] = defaultdict(bool)
    page_chunk_counters: dict[tuple[str, int], int] = defaultdict(int)

    for page_record in page_records:
        source_id = str(page_record["source_id"])
        page_number = int(page_record["pdf_page_number"])

        if source_id not in registry:
            raise KeyError(
                f"{source_id} is missing from source registry."
            )

        if reached_references[source_id]:
            page_audit_rows.append(
                {
                    "source_id": source_id,
                    "pdf_page_number": page_number,
                    "page_status": "excluded_after_references",
                    "reason": "Reference section already reached.",
                    "chunk_count": 0,
                }
            )
            continue

        resolved_policy, policy_reason = resolve_page_policy(
            source_id=source_id,
            page_number=page_number,
            policy_map=policy_map,
            exclude_pending_review_pages=exclude_pending,
        )

        if resolved_policy in {
            "exclude",
            "exclude_pending_review",
        }:
            page_audit_rows.append(
                {
                    "source_id": source_id,
                    "pdf_page_number": page_number,
                    "page_status": resolved_policy,
                    "reason": policy_reason,
                    "chunk_count": 0,
                }
            )
            continue

        units, reference_heading_found = build_text_units(
            text=str(page_record.get("text", "")),
            max_characters=max_characters,
            reference_headings=reference_headings,
        )

        page_chunks = chunk_text_units(
            units=units,
            max_characters=max_characters,
            overlap_characters=overlap_characters,
            separator=separator,
        )

        for chunk_text in page_chunks:
            page_chunk_counters[
                (source_id, page_number)
            ] += 1

            chunk_index = page_chunk_counters[
                (source_id, page_number)
            ]

            chunk_id = (
                f"{source_id}-"
                f"P{page_number:03d}-"
                f"C{chunk_index:02d}"
            )

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source_id": source_id,
                    "title": registry[source_id]["title"],
                    "short_name": registry[source_id]["short_name"],
                    "language": registry[source_id]["language"],
                    "local_filename":
                        registry[source_id]["local_filename"],
                    "pdf_page_start": page_number,
                    "pdf_page_end": page_number,
                    "content_type": "body_text",
                    "chunk_method": config["chunk_method"],
                    "max_characters": max_characters,
                    "overlap_characters": overlap_characters,
                    "character_count": len(chunk_text),
                    "text_sha256": text_sha256(chunk_text),
                    "text": chunk_text,
                }
            )

        page_status = "included"

        if reference_heading_found:
            page_status = "included_before_references"

            if stop_at_references:
                reached_references[source_id] = True

        page_audit_rows.append(
            {
                "source_id": source_id,
                "pdf_page_number": page_number,
                "page_status": page_status,
                "reason": (
                    "Reference heading found on this page."
                    if reference_heading_found
                    else ""
                ),
                "chunk_count": len(page_chunks),
            }
        )

    return chunks, page_audit_rows


def write_chunk_outputs(
    chunks: list[dict[str, Any]],
) -> None:
    """Save full local chunk outputs."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with CHUNK_JSONL_PATH.open(
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

    fieldnames = list(chunks[0].keys())

    with CHUNK_CSV_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(chunks)


def write_summary(
    chunks: list[dict[str, Any]],
    page_audit_rows: list[dict[str, Any]],
) -> None:
    """Save source-level aggregate statistics without guideline text."""

    chunks_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pages_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for chunk in chunks:
        chunks_by_source[str(chunk["source_id"])].append(chunk)

    for row in page_audit_rows:
        pages_by_source[str(row["source_id"])].append(row)

    source_ids = sorted(
        set(chunks_by_source) | set(pages_by_source)
    )

    summary_rows: list[dict[str, Any]] = []

    for source_id in source_ids:
        source_chunks = chunks_by_source.get(source_id, [])
        source_pages = pages_by_source.get(source_id, [])

        lengths = [
            int(chunk["character_count"])
            for chunk in source_chunks
        ]

        included_pages = sum(
            row["page_status"] in {
                "included",
                "included_before_references",
            }
            for row in source_pages
        )

        excluded_pages = len(source_pages) - included_pages

        summary_rows.append(
            {
                "source_id": source_id,
                "total_pages_processed": len(source_pages),
                "included_pages": included_pages,
                "excluded_pages": excluded_pages,
                "chunk_count": len(source_chunks),
                "minimum_chunk_characters":
                    min(lengths) if lengths else 0,
                "median_chunk_characters":
                    round(statistics.median(lengths), 1)
                    if lengths
                    else 0,
                "mean_chunk_characters":
                    round(statistics.mean(lengths), 1)
                    if lengths
                    else 0,
                "maximum_chunk_characters":
                    max(lengths) if lengths else 0,
            }
        )

    fieldnames = [
        "source_id",
        "total_pages_processed",
        "included_pages",
        "excluded_pages",
        "chunk_count",
        "minimum_chunk_characters",
        "median_chunk_characters",
        "mean_chunk_characters",
        "maximum_chunk_characters",
    ]

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def choose_review_sample(
    chunks: list[dict[str, Any]],
    sample_size: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    """Choose a deterministic, source-balanced review sample."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

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


def write_local_review_sample(
    chunks: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    """Save a local review sheet containing copyrighted text excerpts."""

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    sample = choose_review_sample(
        chunks=chunks,
        sample_size=int(
            config["manual_review_sample_size"]
        ),
        random_seed=int(config["random_seed"]),
    )

    review_rows: list[dict[str, Any]] = []

    for chunk in sample:
        review_rows.append(
            {
                "chunk_id": chunk["chunk_id"],
                "source_id": chunk["source_id"],
                "pdf_page_number": chunk["pdf_page_start"],
                "character_count": chunk["character_count"],
                "text_preview": chunk["text"],
                "semantic_coherence": "",
                "context_complete": "",
                "heading_caption_noise": "",
                "reference_noise": "",
                "medical_conditions_preserved": "",
                "usable_for_retrieval": "",
                "review_notes": "",
            }
        )

    fieldnames = list(review_rows[0].keys())

    with LOCAL_REVIEW_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(review_rows)


def print_summary(
    chunks: list[dict[str, Any]],
    page_audit_rows: list[dict[str, Any]],
) -> None:
    """Print a concise terminal summary."""

    chunks_by_source: dict[str, list[int]] = defaultdict(list)

    for chunk in chunks:
        chunks_by_source[str(chunk["source_id"])].append(
            int(chunk["character_count"])
        )

    print("\nBaseline chunk build")
    print("=" * 100)

    for source_id in sorted(chunks_by_source):
        lengths = chunks_by_source[source_id]

        print(
            f"{source_id}: "
            f"chunks={len(lengths)} | "
            f"min={min(lengths)} | "
            f"median={statistics.median(lengths):.1f} | "
            f"mean={statistics.mean(lengths):.1f} | "
            f"max={max(lengths)}"
        )

    status_counts: dict[str, int] = defaultdict(int)

    for row in page_audit_rows:
        status_counts[str(row["page_status"])] += 1

    print("-" * 100)
    print(f"Total chunks: {len(chunks)}")
    print(f"Page status counts: {dict(status_counts)}")
    print(f"Chunk JSONL: {CHUNK_JSONL_PATH}")
    print(f"Chunk CSV: {CHUNK_CSV_PATH}")
    print(f"Aggregate summary: {SUMMARY_PATH}")
    print(f"Local manual review: {LOCAL_REVIEW_PATH}")
    print("=" * 100)


def main() -> None:
    """Build, save, sample, and summarize baseline chunks."""

    config = read_json(CONFIG_PATH)
    chunks, page_audit_rows = build_chunks()

    if not chunks:
        raise ValueError(
            "No chunks were created. Check extraction and page policy."
        )

    write_chunk_outputs(chunks)
    write_summary(chunks, page_audit_rows)
    write_local_review_sample(
        chunks=chunks,
        config=config,
    )
    print_summary(chunks, page_audit_rows)


if __name__ == "__main__":
    main()