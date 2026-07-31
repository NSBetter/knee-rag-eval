"""Extract guideline text with layout-aware reading order.

Processing steps:

1. Read document-specific extraction settings.
2. Extract positioned text blocks from every PDF page.
3. Detect repeated headers and footers.
4. Remove page numbers and optional preprint line numbers.
5. Reorder two-column pages as left column followed by right column.
6. Save page-level JSONL and human-readable full text.
7. Generate cleaned previews for manual inspection.

Tables are retained as text in this stage but are not assumed to have
correct row-column structure. They will be processed separately later.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pymupdf


ROOT_DIR = Path(__file__).resolve().parents[1]

REGISTRY_PATH = ROOT_DIR / "docs" / "source_registry.csv"
CONFIG_PATH = ROOT_DIR / "docs" / "extraction_config.csv"
CORPUS_DIR = ROOT_DIR / "data" / "corpus"

PAGE_OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "pages"
TEXT_OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "full_text"
PREVIEW_OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "clean_previews"

HEADER_SCAN_RATIO = 0.15
FOOTER_SCAN_RATIO = 0.10
REPEATED_MARGIN_MIN_RATIO = 0.30


@dataclass(frozen=True)
class TextBlock:
    """A positioned text block extracted from a PDF page."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    block_number: int

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2


def parse_bool(value: str) -> bool:
    """Parse a CSV boolean field."""

    return value.strip().lower() in {"true", "1", "yes", "y"}


def normalize_space(text: str) -> str:
    """Collapse repeated whitespace for comparison."""

    return re.sub(r"\s+", " ", text).strip()


def normalize_margin_signature(text: str) -> str:
    """Normalize a header or footer so changing page numbers still match."""

    normalized = normalize_space(text).lower()
    normalized = re.sub(r"\d+", "#", normalized)
    return normalized


def is_cjk_character(character: str) -> bool:
    """Return whether a character belongs to a common CJK range."""

    if not character:
        return False

    return (
        "\u3400" <= character <= "\u4dbf"
        or "\u4e00" <= character <= "\u9fff"
    )


def remove_line_number_prefixes(text: str) -> str:
    """Remove preprint-style line numbers when they affect most lines.

    The function only activates when at least half of the non-empty lines
    begin with a one-to-three digit number followed by whitespace. This
    avoids removing ordinary numbered lists from most documents.
    """

    lines = [line.rstrip() for line in text.splitlines() if line.strip()]

    if not lines:
        return ""

    pattern = re.compile(r"^\s*\d{1,3}\s+(?=\S)")
    matched_count = sum(bool(pattern.match(line)) for line in lines)

    if matched_count / len(lines) < 0.50:
        return "\n".join(lines)

    cleaned_lines = [
        pattern.sub("", line).strip()
        for line in lines
    ]

    return "\n".join(line for line in cleaned_lines if line)


def join_wrapped_lines(text: str, language: str) -> str:
    """Join visual PDF lines into a readable paragraph.

    Chinese line wraps are usually joined without adding a space.
    English lines are usually joined with a space, while hyphenated words
    split across lines are reconnected.
    """

    lines = [
        normalize_space(line)
        for line in text.splitlines()
        if normalize_space(line)
    ]

    if not lines:
        return ""

    output = lines[0]

    for line in lines[1:]:
        previous_character = output[-1] if output else ""
        next_character = line[0] if line else ""

        if (
            output.endswith("-")
            and next_character.isalpha()
            and language.lower().startswith("en")
        ):
            output = output[:-1] + line
            continue

        if language.lower().startswith("zh"):
            if (
                is_cjk_character(previous_character)
                or is_cjk_character(next_character)
                or next_character in "，。；：！？、）》】"
                or previous_character in "（《【"
            ):
                output += line
            else:
                output += " " + line
        else:
            if next_character in ",.;:!?)]}":
                output += line
            else:
                output += " " + line

    return normalize_space(output)


def read_csv_by_key(
    path: Path,
    key_field: str,
) -> dict[str, dict[str, str]]:
    """Read a CSV file and index its records by one field."""

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    result: dict[str, dict[str, str]] = {}

    for row in rows:
        key = row[key_field].strip()

        if not key:
            raise ValueError(
                f"Empty key field {key_field!r} in {path}"
            )

        if key in result:
            raise ValueError(
                f"Duplicate {key_field}={key!r} in {path}"
            )

        result[key] = row

    return result


def raw_text_blocks(page: pymupdf.Page) -> list[TextBlock]:
    """Extract positioned text blocks from a page."""

    blocks: list[TextBlock] = []

    for raw_block in page.get_text("blocks", sort=False):
        if len(raw_block) < 7:
            continue

        x0, y0, x1, y1, text, block_number, block_type = raw_block[:7]

        # block_type == 0 represents text.
        if block_type != 0:
            continue

        if not str(text).strip():
            continue

        blocks.append(
            TextBlock(
                x0=float(x0),
                y0=float(y0),
                x1=float(x1),
                y1=float(y1),
                text=str(text),
                block_number=int(block_number),
            )
        )

    return blocks


def is_margin_block(
    block: TextBlock,
    page_height: float,
) -> bool:
    """Return whether a block is within the header or footer scan area."""

    return (
        block.y1 <= page_height * HEADER_SCAN_RATIO
        or block.y0 >= page_height * (1 - FOOTER_SCAN_RATIO)
    )


def detect_repeated_margin_signatures(
    document: pymupdf.Document,
) -> set[str]:
    """Find short header/footer blocks repeated across many pages."""

    signature_pages: dict[str, set[int]] = defaultdict(set)

    for page_index in range(document.page_count):
        page = document[page_index]
        page_height = page.rect.height

        for block in raw_text_blocks(page):
            if not is_margin_block(block, page_height):
                continue

            normalized = normalize_margin_signature(block.text)

            if not normalized:
                continue

            if len(normalized) > 180:
                continue

            signature_pages[normalized].add(page_index)

    minimum_pages = max(
        2,
        math.ceil(
            document.page_count * REPEATED_MARGIN_MIN_RATIO
        ),
    )

    return {
        signature
        for signature, pages in signature_pages.items()
        if len(pages) >= minimum_pages
    }


def clean_block_text(
    block_text: str,
    language: str,
    strip_line_numbers: bool,
) -> str:
    """Clean the content of one positioned text block."""

    text = block_text

    if strip_line_numbers:
        text = remove_line_number_prefixes(text)

    return join_wrapped_lines(
        text=text,
        language=language,
    )


def prepare_page_blocks(
    page: pymupdf.Page,
    language: str,
    strip_line_numbers: bool,
    repeated_margin_signatures: set[str],
) -> tuple[list[TextBlock], int]:
    """Clean blocks and remove repeated page-margin noise."""

    cleaned_blocks: list[TextBlock] = []
    removed_count = 0
    page_height = page.rect.height

    for block in raw_text_blocks(page):
        normalized_signature = normalize_margin_signature(block.text)

        if (
            is_margin_block(block, page_height)
            and normalized_signature in repeated_margin_signatures
        ):
            removed_count += 1
            continue

        # Remove standalone page-number blocks.
        if re.fullmatch(r"\s*\d+\s*", block.text):
            removed_count += 1
            continue

        cleaned_text = clean_block_text(
            block_text=block.text,
            language=language,
            strip_line_numbers=strip_line_numbers,
        )

        if not cleaned_text:
            continue

        cleaned_blocks.append(
            TextBlock(
                x0=block.x0,
                y0=block.y0,
                x1=block.x1,
                y1=block.y1,
                text=cleaned_text,
                block_number=block.block_number,
            )
        )

    return cleaned_blocks, removed_count


def sort_single_column(
    blocks: Iterable[TextBlock],
) -> list[TextBlock]:
    """Sort a single-column document in normal reading order."""

    return sorted(
        blocks,
        key=lambda block: (
            round(block.y0, 1),
            round(block.x0, 1),
            block.block_number,
        ),
    )


def sort_column_zone(
    blocks: list[TextBlock],
    page_midpoint: float,
) -> list[TextBlock]:
    """Read all left-column blocks, then all right-column blocks."""

    left_blocks = [
        block
        for block in blocks
        if block.center_x < page_midpoint
    ]

    right_blocks = [
        block
        for block in blocks
        if block.center_x >= page_midpoint
    ]

    sorting_key = lambda block: (
        round(block.y0, 1),
        round(block.x0, 1),
        block.block_number,
    )

    return [
        *sorted(left_blocks, key=sorting_key),
        *sorted(right_blocks, key=sorting_key),
    ]


def sort_two_columns(
    blocks: list[TextBlock],
    page_width: float,
) -> list[TextBlock]:
    """Sort a page containing two columns and possible full-width blocks.

    Full-width blocks act as vertical separators. Within each vertical
    region, the complete left column is read before the right column.
    """

    if not blocks:
        return []

    midpoint = page_width / 2

    full_width_blocks: list[TextBlock] = []
    column_blocks: list[TextBlock] = []

    for block in blocks:
        crosses_middle_area = (
            block.x0 < midpoint - page_width * 0.07
            and block.x1 > midpoint + page_width * 0.07
        )

        is_very_wide = block.width >= page_width * 0.72

        if crosses_middle_area or is_very_wide:
            full_width_blocks.append(block)
        else:
            column_blocks.append(block)

    full_width_blocks = sorted(
        full_width_blocks,
        key=lambda block: (
            round(block.y0, 1),
            round(block.x0, 1),
        ),
    )

    ordered_blocks: list[TextBlock] = []
    already_used: set[int] = set()
    current_top = float("-inf")

    def add_column_zone(
        zone_top: float,
        zone_bottom: float,
    ) -> None:
        zone_blocks = [
            block
            for block in column_blocks
            if id(block) not in already_used
            and zone_top <= block.center_y < zone_bottom
        ]

        for block in sort_column_zone(
            blocks=zone_blocks,
            page_midpoint=midpoint,
        ):
            ordered_blocks.append(block)
            already_used.add(id(block))

    for full_block in full_width_blocks:
        add_column_zone(
            zone_top=current_top,
            zone_bottom=full_block.y0,
        )

        ordered_blocks.append(full_block)
        current_top = max(current_top, full_block.y1)

    add_column_zone(
        zone_top=current_top,
        zone_bottom=float("inf"),
    )

    # Defensive fallback for blocks overlapping a separator.
    remaining_blocks = [
        block
        for block in column_blocks
        if id(block) not in already_used
    ]

    ordered_blocks.extend(
        sort_column_zone(
            blocks=remaining_blocks,
            page_midpoint=midpoint,
        )
    )

    return ordered_blocks


def select_preview_pages(page_count: int) -> list[int]:
    """Select representative zero-based page indexes."""

    candidates = [
        0,
        1,
        page_count // 2,
        page_count - 1,
    ]

    return sorted(
        {
            page_index
            for page_index in candidates
            if 0 <= page_index < page_count
        }
    )


def extract_document(
    registry_row: dict[str, str],
    config_row: dict[str, str],
) -> dict[str, object]:
    """Extract one registered PDF and save page-level outputs."""

    source_id = registry_row["source_id"].strip()
    title = registry_row["title"].strip()
    language = registry_row["language"].strip()
    local_filename = registry_row["local_filename"].strip()

    layout = config_row["layout"].strip().lower()
    strip_line_numbers = parse_bool(
        config_row["strip_line_numbers"]
    )
    remove_repeated_margins = parse_bool(
        config_row["remove_repeated_margins"]
    )

    if layout not in {"single_column", "two_column"}:
        raise ValueError(
            f"Unsupported layout {layout!r} for {source_id}"
        )

    pdf_path = CORPUS_DIR / local_filename

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found for {source_id}: {pdf_path}"
        )

    PAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    page_records: list[dict[str, object]] = []

    with pymupdf.open(pdf_path) as document:
        repeated_signatures = (
            detect_repeated_margin_signatures(document)
            if remove_repeated_margins
            else set()
        )

        for page_index in range(document.page_count):
            page = document[page_index]

            blocks, removed_count = prepare_page_blocks(
                page=page,
                language=language,
                strip_line_numbers=strip_line_numbers,
                repeated_margin_signatures=repeated_signatures,
            )

            if layout == "two_column":
                ordered_blocks = sort_two_columns(
                    blocks=blocks,
                    page_width=page.rect.width,
                )
            else:
                ordered_blocks = sort_single_column(blocks)

            page_text = "\n\n".join(
                block.text
                for block in ordered_blocks
                if block.text
            ).strip()

            page_records.append(
                {
                    "source_id": source_id,
                    "title": title,
                    "local_filename": local_filename,
                    "language": language,
                    "layout": layout,
                    "pdf_page_number": page_index + 1,
                    "text": page_text,
                    "text_character_count": len(page_text),
                    "text_block_count": len(ordered_blocks),
                    "removed_margin_block_count": removed_count,
                }
            )

    jsonl_path = PAGE_OUTPUT_DIR / f"{source_id.lower()}_pages.jsonl"

    with jsonl_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in page_records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    full_text_path = (
        TEXT_OUTPUT_DIR / f"{source_id.lower()}_full_text.txt"
    )

    full_text_sections: list[str] = [
        f"Source ID: {source_id}",
        f"Title: {title}",
        f"Local file: {local_filename}",
        "",
        "=" * 100,
        "",
    ]

    for record in page_records:
        full_text_sections.extend(
            [
                f"PDF PAGE {record['pdf_page_number']}",
                "-" * 100,
                str(record["text"]),
                "",
                "=" * 100,
                "",
            ]
        )

    full_text_path.write_text(
        "\n".join(full_text_sections),
        encoding="utf-8",
    )

    preview_path = (
        PREVIEW_OUTPUT_DIR
        / f"{source_id.lower()}_clean_preview.txt"
    )

    preview_sections: list[str] = [
        f"Source ID: {source_id}",
        f"Title: {title}",
        f"Layout: {layout}",
        "",
        "=" * 100,
        "",
    ]

    preview_indexes = select_preview_pages(len(page_records))

    for page_index in preview_indexes:
        record = page_records[page_index]

        preview_sections.extend(
            [
                f"PDF PAGE {record['pdf_page_number']}",
                "-" * 100,
                str(record["text"]),
                "",
                "=" * 100,
                "",
            ]
        )

    preview_path.write_text(
        "\n".join(preview_sections),
        encoding="utf-8",
    )

    return {
        "source_id": source_id,
        "page_count": len(page_records),
        "total_characters": sum(
            int(record["text_character_count"])
            for record in page_records
        ),
        "repeated_margin_signatures": len(repeated_signatures),
        "jsonl_path": jsonl_path,
        "full_text_path": full_text_path,
        "preview_path": preview_path,
    }


def main() -> None:
    """Extract every included source using its layout configuration."""

    registry = read_csv_by_key(
        path=REGISTRY_PATH,
        key_field="source_id",
    )

    extraction_config = read_csv_by_key(
        path=CONFIG_PATH,
        key_field="source_id",
    )

    included_rows = [
        row
        for row in registry.values()
        if row["include_status"].strip().lower() == "included"
    ]

    if not included_rows:
        raise ValueError("No included sources were found.")

    print("\nLayout-aware corpus extraction")
    print("=" * 100)

    results: list[dict[str, object]] = []

    for registry_row in included_rows:
        source_id = registry_row["source_id"].strip()

        if source_id not in extraction_config:
            raise KeyError(
                f"No extraction configuration for {source_id}"
            )

        result = extract_document(
            registry_row=registry_row,
            config_row=extraction_config[source_id],
        )

        results.append(result)

        print(
            f"{result['source_id']}: "
            f"pages={result['page_count']} | "
            f"characters={result['total_characters']} | "
            f"repeated_margin_patterns="
            f"{result['repeated_margin_signatures']}"
        )

    print("=" * 100)
    print(f"Extracted {len(results)} documents.")
    print(f"Page JSONL directory: {PAGE_OUTPUT_DIR}")
    print(f"Full text directory: {TEXT_OUTPUT_DIR}")
    print(f"Clean preview directory: {PREVIEW_OUTPUT_DIR}")


if __name__ == "__main__":
    main()