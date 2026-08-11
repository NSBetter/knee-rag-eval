"""Build Gold Corpus v1.3 for the knee OA medical RAG demo.

Strategies:
- SRC001: hierarchy-bounded sections with explicit layout repairs.
- SRC003: bind every clinical question to each included recommendation.
- Human-reviewed table evidence: import as manual_table_chunk.

Local outputs under data/processed/ are ignored by Git. The public summary
contains metadata only and can be committed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "gold_corpus_v1_3.json"
CLEANUP_CONFIG_PATH = ROOT / "configs" / "gold_corpus_cleanup_v1_3.json"
RECOMMENDATION_OVERRIDE_PATH = (
    ROOT / "data" / "processed" / "private"
    / "gold_corpus_recommendation_overrides_v1_3.json"
)
STRUCTURE_PATH = ROOT / "docs" / "document_structure_map.csv"
BOUNDARY_PATH = ROOT / "docs" / "document_boundaries.csv"
STRUCTURE_REPORT = ROOT / "docs" / "document_structure_validation.csv"
TABLE_REPORT = ROOT / "docs" / "manual_table_evidence_audit.csv"
PAGES_DIR = ROOT / "data" / "processed" / "pages"
MANUAL_TABLE_PATH = (
    ROOT / "data" / "processed" / "manual_tables"
    / "manual_table_evidence.csv"
)
GOLD_DIR = ROOT / "data" / "processed" / "gold_corpus"
JSONL_PATH = GOLD_DIR / "gold_corpus_v1_3.jsonl"
CSV_PATH = GOLD_DIR / "gold_corpus_v1_3.csv"
REVIEW_PATH = (
    ROOT / "data" / "processed" / "reviews"
    / "gold_corpus_v1_3_review.csv"
)
SUMMARY_PATH = ROOT / "docs" / "gold_corpus_v1_3_build_summary.csv"
DIAGNOSTICS_PATH = ROOT / "docs" / "gold_corpus_v1_3_diagnostics.csv"

AUTO_MODES = {"auto_text", "hybrid"}
MANUAL_MODES = {"manual_table", "hybrid"}


@dataclass(frozen=True)
class Config:
    corpus_version: str
    included_sources: tuple[str, ...]
    target_chars: int
    max_chars: int
    min_chars: int
    sentence_overflow_chars: int
    recommendation_context_max_chars: int
    review_sample_size: int
    random_seed: int


@dataclass(frozen=True)
class Boundary:
    source_id: str
    body_start_page: int
    body_start_marker: str
    reference_start_page: int
    reference_start_marker: str
    body_end_page: int


@dataclass
class Node:
    source_id: str
    node_id: str
    parent_id: str
    order: int
    node_type: str
    level: int
    pdf_page: int
    marker_text: str
    marker_occurrence: int
    display_title: str
    include_status: str
    extraction_mode: str
    notes: str
    marker_position: int = -1
    segment_text: str = ""
    page_start: int = 0
    page_end: int = 0


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = list(reader.fieldnames or [])
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    return rows, fields


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_config() -> Config:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config = Config(
        corpus_version=str(raw["corpus_version"]),
        included_sources=tuple(raw["included_sources"]),
        target_chars=int(raw["target_chars"]),
        max_chars=int(raw["max_chars"]),
        min_chars=int(raw["min_chars"]),
        sentence_overflow_chars=int(
            raw.get("sentence_overflow_chars", 300)
        ),
        recommendation_context_max_chars=int(
            raw.get("recommendation_context_max_chars", 420)
        ),
        review_sample_size=int(raw["review_sample_size"]),
        random_seed=int(raw["random_seed"]),
    )
    if not config.included_sources:
        raise ValueError("included_sources cannot be empty.")
    if not 0 < config.min_chars <= config.target_chars <= config.max_chars:
        raise ValueError(
            "Expected 0 < min_chars <= target_chars <= max_chars."
        )
    if config.sentence_overflow_chars < 0:
        raise ValueError("sentence_overflow_chars cannot be negative.")
    if config.recommendation_context_max_chars < 120:
        raise ValueError(
            "recommendation_context_max_chars must be at least 120."
        )
    return config



def load_cleanup_config() -> dict[str, Any]:
    if not CLEANUP_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Cleanup config not found: {CLEANUP_CONFIG_PATH}"
        )
    raw = json.loads(
        CLEANUP_CONFIG_PATH.read_text(encoding="utf-8")
    )

    raw.setdefault("content_anchor_node_ids", [])
    raw.setdefault("display_title_overrides", {})
    raw.setdefault("source_tail_stop_markers", {})
    raw.setdefault("node_cleanup_rules", {})
    raw.setdefault("high_risk_review_node_ids", [])
    raw.setdefault("allowed_missing_manual_evidence", [])
    raw.setdefault("recommendation_context_overrides", {})
    raw.setdefault("required_recommendation_context_override_ids", [])

    public_overrides = raw["recommendation_context_overrides"]
    if not isinstance(public_overrides, dict):
        raise ValueError(
            "recommendation_context_overrides must be a JSON object."
        )

    required_override_ids = [
        str(node_id)
        for node_id in raw["required_recommendation_context_override_ids"]
    ]

    if required_override_ids and not RECOMMENDATION_OVERRIDE_PATH.exists():
        raise FileNotFoundError(
            "Required local recommendation overrides not found: "
            f"{RECOMMENDATION_OVERRIDE_PATH}"
        )

    if RECOMMENDATION_OVERRIDE_PATH.exists():
        private_raw = json.loads(
            RECOMMENDATION_OVERRIDE_PATH.read_text(encoding="utf-8")
        )
        private_overrides = private_raw.get(
            "recommendation_context_overrides",
            {},
        )
        if not isinstance(private_overrides, dict):
            raise ValueError(
                "Private recommendation_context_overrides "
                "must be a JSON object."
            )
        public_overrides.update(private_overrides)

    missing_override_ids = [
        node_id
        for node_id in required_override_ids
        if not str(public_overrides.get(node_id, "")).strip()
    ]
    if missing_override_ids:
        raise ValueError(
            "Missing required recommendation overrides: "
            + ", ".join(missing_override_ids)
        )

    allowed_operations = {
        "remove_between",
        "trim_after",
        "replace",
        "regex_replace",
    }
    for node_id, rules in raw["node_cleanup_rules"].items():
        if not isinstance(rules, list):
            raise ValueError(
                f"Cleanup rules must be a list: {node_id}"
            )
        for rule in rules:
            operation = rule.get("operation", "")
            if operation not in allowed_operations:
                raise ValueError(
                    f"Unsupported cleanup operation "
                    f"{operation!r} for {node_id}"
                )
    return raw


def require_passing_reports() -> None:
    structure_rows, _ = read_csv(STRUCTURE_REPORT)
    table_rows, _ = read_csv(TABLE_REPORT)
    bad_structure = [
        row for row in structure_rows
        if row.get("overall_status") != "pass"
    ]
    bad_tables = [
        row for row in table_rows
        if row.get("validation_status") != "pass"
    ]
    if bad_structure:
        raise ValueError(
            f"Structure validation has {len(bad_structure)} non-pass rows."
        )
    if bad_tables:
        raise ValueError(
            f"Manual table audit has {len(bad_tables)} non-pass rows."
        )


def load_pages() -> dict[str, dict[int, str]]:
    pages: dict[str, dict[int, str]] = defaultdict(dict)
    for path in sorted(PAGES_DIR.glob("*_pages.jsonl")):
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON: {path}, line {line_number}"
                    ) from exc
                source_id = str(row["source_id"])
                page = int(row["pdf_page_number"])
                if page in pages[source_id]:
                    raise ValueError(
                        f"Duplicate page: {source_id} page {page}"
                    )
                pages[source_id][page] = str(row.get("text", ""))
    if not pages:
        raise FileNotFoundError(f"No page JSONL found under {PAGES_DIR}")
    return dict(pages)



def load_boundaries() -> dict[str, Boundary]:
    rows, fields = read_csv(BOUNDARY_PATH)
    required = {
        "source_id",
        "body_start_page",
        "body_start_marker",
        "reference_start_page",
        "reference_start_marker",
        "body_end_page",
    }
    missing = required - set(fields)
    if missing:
        raise ValueError(
            f"Missing boundary columns: {sorted(missing)}"
        )

    boundaries: dict[str, Boundary] = {}
    for row in rows:
        source_id = row["source_id"]
        if source_id in boundaries:
            raise ValueError(
                f"Duplicate document boundary: {source_id}"
            )
        boundaries[source_id] = Boundary(
            source_id=source_id,
            body_start_page=int(row["body_start_page"]),
            body_start_marker=row["body_start_marker"],
            reference_start_page=int(row["reference_start_page"]),
            reference_start_marker=row["reference_start_marker"],
            body_end_page=int(row["body_end_page"]),
        )
    return boundaries


def load_nodes(
    config: Config,
    cleanup: dict[str, Any],
) -> list[Node]:
    rows, fields = read_csv(STRUCTURE_PATH)
    required = {
        "source_id", "node_id", "parent_id", "order", "node_type",
        "level", "pdf_page", "marker_text", "marker_occurrence",
        "display_title", "include_status", "notes", "extraction_mode",
    }
    missing = required - set(fields)
    if missing:
        raise ValueError(f"Missing structure columns: {sorted(missing)}")

    title_overrides = cleanup["display_title_overrides"]

    nodes = [
        Node(
            source_id=row["source_id"],
            node_id=row["node_id"],
            parent_id=row["parent_id"],
            order=int(row["order"]),
            node_type=row["node_type"],
            level=int(row["level"]),
            pdf_page=int(row["pdf_page"]),
            marker_text=row["marker_text"],
            marker_occurrence=int(row["marker_occurrence"] or "1"),
            display_title=title_overrides.get(
                row["node_id"],
                row["display_title"],
            ),
            include_status=row["include_status"],
            extraction_mode=row["extraction_mode"],
            notes=row["notes"],
        )
        for row in rows
        if row["source_id"] in config.included_sources
    ]
    ids = [node.node_id for node in nodes]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate node_id found in selected sources.")
    for source_id in config.included_sources:
        selected = [node for node in nodes if node.source_id == source_id]
        if not selected:
            raise ValueError(f"No nodes found for {source_id}.")
        orders = [node.order for node in selected]
        if len(orders) != len(set(orders)):
            raise ValueError(f"Duplicate order values in {source_id}.")
    return nodes

def find_nth(text: str, marker: str, occurrence: int) -> int:
    if not marker or occurrence < 1:
        return -1
    start = 0
    position = -1
    for _ in range(occurrence):
        position = text.find(marker, start)
        if position < 0:
            return -1
        start = position + len(marker)
    return position



def locate_and_slice_nodes(
    nodes: list[Node],
    pages: dict[str, dict[int, str]],
    boundaries: dict[str, Boundary],
    cleanup: dict[str, Any],
) -> None:
    by_source: dict[str, list[Node]] = defaultdict(list)
    for node in nodes:
        page_text = pages.get(node.source_id, {}).get(node.pdf_page)
        if page_text is None:
            raise ValueError(
                f"Missing page: {node.source_id} page {node.pdf_page}"
            )
        node.marker_position = find_nth(
            page_text, node.marker_text, node.marker_occurrence
        )
        if node.marker_position < 0:
            raise ValueError(
                f"Marker not found after validation: {node.node_id}"
            )
        by_source[node.source_id].append(node)

    for source_id, source_nodes in by_source.items():
        source_nodes.sort(key=lambda item: item.order)
        boundary = boundaries.get(source_id)
        if boundary is None:
            raise ValueError(f"Boundary missing for {source_id}.")

        source_pages = pages[source_id]

        end_candidates: list[tuple[int, int, str]] = []

        reference_page_text = source_pages.get(
            boundary.reference_start_page
        )
        if reference_page_text is None:
            raise ValueError(
                f"Reference start page missing: {source_id} "
                f"page {boundary.reference_start_page}"
            )
        reference_position = reference_page_text.find(
            boundary.reference_start_marker
        )
        if reference_position < 0:
            raise ValueError(
                f"Reference start marker not found: {source_id} "
                f"{boundary.reference_start_marker!r}"
            )
        end_candidates.append((
            boundary.reference_start_page,
            reference_position,
            "document_boundary",
        ))

        for item in cleanup["source_tail_stop_markers"].get(
            source_id,
            [],
        ):
            page = int(item["page"])
            marker = str(item["marker"])
            page_text = source_pages.get(page)
            if page_text is None:
                raise ValueError(
                    f"Tail stop page missing: {source_id} page {page}"
                )
            position = page_text.find(marker)
            if position < 0:
                if item.get("required", True):
                    raise ValueError(
                        f"Tail stop marker not found: "
                        f"{source_id} {marker!r}"
                    )
                continue
            end_candidates.append((
                page,
                position,
                str(item.get("rule_id", "tail_stop")),
            ))

        end_page, end_position, _ = min(
            end_candidates,
            key=lambda item: (item[0], item[1]),
        )

        for index, node in enumerate(source_nodes):
            if index + 1 < len(source_nodes):
                next_node = source_nodes[index + 1]
                node_end_page = next_node.pdf_page
                node_end_position = next_node.marker_position
            else:
                node_end_page = end_page
                node_end_position = end_position

            if (node_end_page, node_end_position) < (
                node.pdf_page, node.marker_position
            ):
                raise ValueError(
                    f"Invalid node order around {node.node_id}."
                )

            parts: list[str] = []
            used_pages: list[int] = []
            for page in range(node.pdf_page, node_end_page + 1):
                if page not in source_pages:
                    raise ValueError(
                        f"Missing intermediate page: "
                        f"{source_id} page {page}"
                    )
                page_text = source_pages[page]
                if node.pdf_page == node_end_page:
                    part = page_text[
                        node.marker_position:node_end_position
                    ]
                elif page == node.pdf_page:
                    part = page_text[node.marker_position:]
                elif page == node_end_page:
                    part = page_text[:node_end_position]
                else:
                    part = page_text
                if part.strip():
                    parts.append(part)
                    used_pages.append(page)

            node.segment_text = "\n".join(parts)
            node.page_start = (
                min(used_pages) if used_pages else node.pdf_page
            )
            node.page_end = (
                max(used_pages) if used_pages else node.pdf_page
            )


def normalize_text(text: str) -> str:
    text = (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u00a0", " ")
        .replace("\u3000", " ")
    )
    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.splitlines()
    ]
    return re.sub(
        r"\s+",
        " ",
        " ".join(line for line in lines if line),
    ).strip()


def compact_cjk_spacing(text: str) -> str:
    """Remove obvious PDF line-wrap spaces without joining Latin words."""

    text = re.sub(
        r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])",
        "",
        text,
    )
    text = re.sub(
        r"\s+([，。；：！？、）】])",
        r"\1",
        text,
    )
    text = re.sub(
        r"([（【])\s+",
        r"\1",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def flexible_prefix_pattern(value: str) -> re.Pattern[str]:
    compact = "".join(value.split())
    body = r"\s*".join(
        re.escape(character)
        for character in compact
    )
    return re.compile(r"^" + body)


def marker_is_heading(
    node: Node,
    cleanup: dict[str, Any],
) -> bool:
    if node.node_id in set(
        cleanup["content_anchor_node_ids"]
    ):
        return False

    marker_key = "".join(node.marker_text.split())
    title_key = "".join(node.display_title.split())

    if not marker_key or not title_key:
        return False

    return (
        title_key.startswith(marker_key)
        or marker_key.startswith(title_key)
    )


def section_body_without_own_marker(
    node: Node,
    cleanup: dict[str, Any],
) -> str:
    """Remove a true section heading but preserve content anchors."""

    raw = normalize_text(node.segment_text)

    if not marker_is_heading(node, cleanup):
        return raw

    candidates = sorted(
        {
            node.display_title.strip(),
            node.marker_text.strip(),
        },
        key=len,
        reverse=True,
    )

    for candidate in candidates:
        if not candidate:
            continue
        match = flexible_prefix_pattern(candidate).match(raw)
        if match:
            raw = raw[match.end():]
            break

    raw = re.sub(
        r"^[\s:：、，,；;。]+",
        "",
        raw,
    )
    return normalize_text(raw)


def add_diagnostic(
    diagnostics: list[dict[str, Any]],
    *,
    source_id: str,
    node_id: str,
    category: str,
    status: str,
    detail: str,
    value: int | str = "",
) -> None:
    diagnostics.append({
        "source_id": source_id,
        "node_id": node_id,
        "category": category,
        "status": status,
        "detail": detail,
        "value": value,
    })


def apply_node_cleanup(
    node: Node,
    text: str,
    cleanup: dict[str, Any],
    diagnostics: list[dict[str, Any]],
) -> str:
    """Apply explicit, reviewable repairs to known layout failures."""

    result = normalize_text(text)
    rules = cleanup["node_cleanup_rules"].get(
        node.node_id,
        [],
    )

    for index, rule in enumerate(rules, start=1):
        operation = rule["operation"]
        rule_id = str(
            rule.get(
                "rule_id",
                f"{node.node_id}-R{index:02d}",
            )
        )
        before_length = len(result)

        if operation == "remove_between":
            start_marker = str(rule["start"])
            end_marker = str(rule["end"])
            start = result.find(start_marker)
            end = (
                result.find(
                    end_marker,
                    start + len(start_marker),
                )
                if start >= 0
                else -1
            )
            if start < 0 or end < 0:
                if rule.get("required", True):
                    raise ValueError(
                        f"Cleanup markers not found for {rule_id}: "
                        f"start={start}, end={end}"
                    )
                add_diagnostic(
                    diagnostics,
                    source_id=node.source_id,
                    node_id=node.node_id,
                    category="cleanup_rule",
                    status="not_applied",
                    detail=rule_id,
                )
                continue

            left = result[:start]
            if rule.get("keep_start", False):
                left += start_marker

            right = result[
                end + len(end_marker):
            ]
            if rule.get("keep_end", False):
                right = end_marker + right

            result = left + right

        elif operation == "trim_after":
            marker = str(rule["marker"])
            position = result.find(marker)
            if position < 0:
                if rule.get("required", True):
                    raise ValueError(
                        f"Cleanup marker not found for {rule_id}: "
                        f"{marker!r}"
                    )
                continue
            cut = position
            if rule.get("include_marker", False):
                cut += len(marker)
            result = result[:cut]

        elif operation == "replace":
            old = str(rule["old"])
            new = str(rule["new"])
            if old not in result:
                if rule.get("required", True):
                    raise ValueError(
                        f"Cleanup replacement not found for "
                        f"{rule_id}: {old!r}"
                    )
                continue
            result = result.replace(
                old,
                new,
                int(rule.get("count", -1)),
            )

        elif operation == "regex_replace":
            pattern = str(rule["pattern"])
            replacement = str(rule["replacement"])
            result, count = re.subn(
                pattern,
                replacement,
                result,
                count=int(rule.get("count", 0)),
            )
            if count == 0 and rule.get("required", True):
                raise ValueError(
                    f"Cleanup regex did not match for "
                    f"{rule_id}: {pattern!r}"
                )

        else:
            raise ValueError(
                f"Unsupported cleanup operation: {operation}"
            )

        result = normalize_text(result)
        add_diagnostic(
            diagnostics,
            source_id=node.source_id,
            node_id=node.node_id,
            category="cleanup_rule",
            status="applied",
            detail=rule_id,
            value=max(0, before_length - len(result)),
        )

    return compact_cjk_spacing(result)


def is_citation_only(text: str) -> bool:
    cleaned = re.sub(
        r"［[^］]{1,30}］|\[[^\]]{1,30}\]",
        "",
        text,
    )
    cleaned = re.sub(
        r"[^A-Za-z0-9\u3400-\u9fff]+",
        "",
        cleaned,
    )
    return len(cleaned) < 4



def recommendation_context(
    recommendation: Node,
    recommendation_text: str,
    cleanup: dict[str, Any],
    max_chars: int,
) -> str:
    """Return a complete recommendation core for continuation chunks.

    Explicit human-reviewed overrides are preferred for recommendations
    whose PDF text lacks a reliable punctuation boundary between the
    recommendation statement and its rationale.
    """

    override = str(
        cleanup["recommendation_context_overrides"].get(
            recommendation.node_id,
            "",
        )
    ).strip()
    if override:
        return compact_cjk_spacing(override)

    text = normalize_text(recommendation_text)

    if len(text) <= max_chars:
        return compact_cjk_spacing(text)

    # Prefer a complete sentence. Do not end reusable context at a comma
    # or list separator, because that creates a misleading partial rule.
    for delimiter in ("。", "！", "？", "；", "!", "?", ";"):
        position = text.find(delimiter)
        if 60 <= position < max_chars:
            return compact_cjk_spacing(text[: position + 1])

    # A short overrun is safer than truncating a medical recommendation.
    hard_limit = min(
        len(text),
        max_chars + 180,
    )
    for delimiter in ("。", "！", "？", "；", "!", "?", ";"):
        position = text.find(delimiter, max_chars, hard_limit)
        if position >= 0:
            return compact_cjk_spacing(text[: position + 1])

    # This fallback is used only for recommendations not covered by an
    # explicit override. It retains a substantial prefix without adding
    # invented punctuation or wording.
    return compact_cjk_spacing(text[:hard_limit])


STRONG_BOUNDARIES = ("。", "！", "？", "；", "!", "?", ";")


def split_long(
    text: str,
    max_chars: int,
    overflow_chars: int,
) -> list[str]:
    """Split long text without preferring commas as medical boundaries.

    max_chars is a soft limit. A complete sentence may exceed it by up
    to overflow_chars. Comma-level splitting is a final fallback only.
    """

    parts: list[str] = []
    remaining = text.strip()

    while len(remaining) > max_chars:
        window = remaining[: max_chars + 1]

        strong_cut = max(
            window.rfind(delimiter)
            for delimiter in STRONG_BOUNDARIES
        )

        if strong_cut >= int(max_chars * 0.45):
            cut = strong_cut + 1

        else:
            hard_limit = min(
                len(remaining),
                max_chars + overflow_chars,
            )
            next_positions = [
                remaining.find(delimiter, max_chars, hard_limit + 1)
                for delimiter in STRONG_BOUNDARIES
            ]
            next_positions = [
                position
                for position in next_positions
                if position >= 0
            ]

            if next_positions:
                cut = min(next_positions) + 1
            else:
                # Prefer a numbered-list boundary over a comma cut.
                list_positions = [
                    match.start()
                    for match in re.finditer(
                        r"(?<![\d.])(?:[1-9]|1\d)\.\s*",
                        window,
                    )
                    if match.start() >= int(max_chars * 0.45)
                ]
                if list_positions:
                    cut = list_positions[-1]
                else:
                    # Last-resort clause split near the soft limit.
                    clause_cut = max(
                        window.rfind(delimiter)
                        for delimiter in ("，", ",", "、", "：", ":")
                    )
                    cut = (
                        clause_cut + 1
                        if clause_cut >= int(max_chars * 0.85)
                        else max_chars
                    )

        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()

    if remaining:
        parts.append(remaining)

    return parts


def sentence_units(
    text: str,
    max_chars: int,
    overflow_chars: int,
) -> list[str]:
    units: list[str] = []
    for raw in re.findall(r".+?(?:[。！？；!?;]+|$)", text):
        raw = raw.strip()
        if not raw:
            continue
        units.extend(
            [raw]
            if len(raw) <= max_chars
            else split_long(
                raw,
                max_chars,
                overflow_chars,
            )
        )
    return units


def pack_text(
    text: str,
    target_chars: int,
    max_chars: int,
    min_chars: int,
    overflow_chars: int = 0,
) -> list[str]:
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""

    for unit in sentence_units(
        text,
        max_chars,
        overflow_chars,
    ):
        effective_max = max_chars + overflow_chars

        if current and len(current) + len(unit) > effective_max:
            chunks.append(current.strip())
            current = unit
        else:
            current += unit

        if (
            len(current) >= target_chars
            and current.endswith(STRONG_BOUNDARIES)
        ):
            chunks.append(current.strip())
            current = ""

    if current:
        chunks.append(current.strip())

    if (
        len(chunks) >= 2
        and len(chunks[-1]) < min_chars
        and len(chunks[-2]) + len(chunks[-1])
        <= max_chars + overflow_chars
    ):
        chunks[-2] += chunks[-1]
        chunks.pop()

    return [chunk for chunk in chunks if chunk]


def sha256_text(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def title_path(node: Node, node_map: dict[str, Node]) -> list[str]:
    path: list[str] = []
    seen: set[str] = set()
    current: Node | None = node
    while current is not None:
        if current.node_id in seen:
            raise ValueError(f"Parent cycle at {current.node_id}.")
        seen.add(current.node_id)
        path.append(
            current.display_title or current.marker_text or current.node_id
        )
        if not current.parent_id:
            break
        current = node_map.get(current.parent_id)
        if current is None:
            raise ValueError(
                f"Parent missing for title path: {node.node_id}"
            )
    return list(reversed(path))


def make_chunk(
    *,
    chunk_id: str,
    source_id: str,
    source_order: int,
    node_id: str,
    content_type: str,
    display_title: str,
    path: list[str],
    page_start: int,
    page_end: int,
    extraction_mode: str,
    review_status: str,
    build_method: str,
    text: str,
    corpus_version: str,
    question_node_id: str = "",
    recommendation_node_ids: list[str] | None = None,
    table_id: str = "",
    context_repeated: bool = False,
) -> dict[str, Any]:
    text = normalize_text(text)
    if not text:
        raise ValueError(f"Empty chunk: {chunk_id}")
    retrieval = "\n".join(
        part for part in [
            f"主题：{display_title}" if display_title else "",
            f"章节：{' > '.join(path)}" if path else "",
            text,
        ]
        if part
    )
    return {
        "chunk_id": chunk_id,
        "source_id": source_id,
        "source_order": source_order,
        "node_id": node_id,
        "content_type": content_type,
        "display_title": display_title,
        "title_path": path,
        "pdf_page_start": page_start,
        "pdf_page_end": page_end,
        "cross_page": page_start != page_end,
        "extraction_mode": extraction_mode,
        "review_status": review_status,
        "build_method": build_method,
        "question_node_id": question_node_id,
        "recommendation_node_ids": recommendation_node_ids or [],
        "table_id": table_id,
        "context_repeated": context_repeated,
        "text": text,
        "retrieval_text": retrieval,
        "char_count": len(text),
        "text_sha256": sha256_text(text),
        "corpus_version": corpus_version,
    }



def build_section_chunks(
    nodes: list[Node],
    node_map: dict[str, Node],
    config: Config,
    cleanup: dict[str, Any],
    diagnostics: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    chunks: list[dict[str, Any]] = []
    skipped = 0

    for node in sorted(nodes, key=lambda item: (item.source_id, item.order)):
        if node.include_status != "included":
            continue
        if node.extraction_mode not in AUTO_MODES:
            continue
        if (
            node.source_id == "SRC003"
            and node.node_type in {"question", "recommendation"}
        ):
            continue

        text = section_body_without_own_marker(
            node,
            cleanup,
        )
        text = apply_node_cleanup(
            node,
            text,
            cleanup,
            diagnostics,
        )

        if not text or is_citation_only(text):
            skipped += 1
            add_diagnostic(
                diagnostics,
                source_id=node.source_id,
                node_id=node.node_id,
                category="structural_only",
                status="skipped",
                detail="no independent retrievable body",
                value=len(text),
            )
            continue

        parts = pack_text(
            text,
            config.target_chars,
            config.max_chars,
            config.min_chars,
            config.sentence_overflow_chars,
        )
        for index, part in enumerate(parts, start=1):
            chunks.append(
                make_chunk(
                    chunk_id=(
                        f"{node.source_id}-SEC-"
                        f"{node.order:03d}-C{index:02d}"
                    ),
                    source_id=node.source_id,
                    source_order=node.order,
                    node_id=node.node_id,
                    content_type="section_text",
                    display_title=node.display_title,
                    path=title_path(node, node_map),
                    page_start=node.page_start,
                    page_end=node.page_end,
                    extraction_mode=node.extraction_mode,
                    review_status=(
                        "auto_extracted_structure_validated"
                    ),
                    build_method=(
                        "hierarchy_bounded_section_v3"
                    ),
                    text=part,
                    corpus_version=config.corpus_version,
                    context_repeated=index > 1,
                )
            )
    return chunks, skipped


def build_question_chunks(
    nodes: list[Node],
    node_map: dict[str, Node],
    config: Config,
    cleanup: dict[str, Any],
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    questions = [
        node for node in nodes
        if (
            node.source_id == "SRC003"
            and node.node_type == "question"
            and node.include_status == "included"
            and node.extraction_mode in AUTO_MODES
        )
    ]

    for question in sorted(questions, key=lambda item: item.order):
        recommendations = sorted(
            [
                node for node in nodes
                if (
                    node.source_id == question.source_id
                    and node.parent_id == question.node_id
                    and node.node_type == "recommendation"
                    and node.include_status == "included"
                    and node.extraction_mode in AUTO_MODES
                )
            ],
            key=lambda item: item.order,
        )
        if not recommendations:
            raise ValueError(
                f"Included question has no recommendation: "
                f"{question.node_id}"
            )

        question_text = apply_node_cleanup(
            question,
            normalize_text(question.segment_text),
            cleanup,
            diagnostics,
        )
        if len(question_text) < 4:
            raise ValueError(
                f"Question text too short: {question.node_id}"
            )

        question_group: list[dict[str, Any]] = []

        for recommendation in recommendations:
            rec_text = apply_node_cleanup(
                recommendation,
                normalize_text(recommendation.segment_text),
                cleanup,
                diagnostics,
            )
            if len(rec_text) < 4:
                raise ValueError(
                    f"Recommendation text too short: "
                    f"{recommendation.node_id}"
                )

            rec_context = recommendation_context(
                recommendation,
                rec_text,
                cleanup,
                max_chars=(
                    config.recommendation_context_max_chars
                ),
            )

            room = (
                config.max_chars
                - len(question_text)
                - len(rec_context)
                - 2
            )
            if room < 220:
                raise ValueError(
                    f"Question and recommendation context are "
                    f"too long: {recommendation.node_id}"
                )

            rec_parts = pack_text(
                rec_text,
                min(
                    room,
                    max(
                        300,
                        config.target_chars
                        - len(question_text)
                        - len(rec_context),
                    ),
                ),
                room,
                min(
                    config.min_chars,
                    max(100, room // 3),
                ),
                config.sentence_overflow_chars,
            )

            for index, part in enumerate(rec_parts, start=1):
                if index == 1:
                    combined = (
                        f"{question_text}\n{part}"
                    )
                else:
                    combined = (
                        f"{question_text}\n"
                        f"{rec_context}\n"
                        f"{part}"
                    )

                question_group.append(
                    make_chunk(
                        chunk_id=(
                            f"SRC003-QR-{question.order:03d}-"
                            f"{recommendation.order:03d}-"
                            f"C{index:02d}"
                        ),
                        source_id="SRC003",
                        source_order=recommendation.order,
                        node_id=question.node_id,
                        content_type=(
                            "question_recommendation"
                        ),
                        display_title=question.display_title,
                        path=title_path(question, node_map),
                        page_start=min(
                            question.page_start,
                            recommendation.page_start,
                        ),
                        page_end=max(
                            question.page_end,
                            recommendation.page_end,
                        ),
                        extraction_mode="auto_text",
                        review_status=(
                            "auto_extracted_structure_validated"
                        ),
                        build_method=(
                            "question_recommendation_v3"
                        ),
                        text=combined,
                        corpus_version=config.corpus_version,
                        question_node_id=question.node_id,
                        recommendation_node_ids=[
                            recommendation.node_id
                        ],
                        context_repeated=index > 1,
                    )
                )

        chunks.extend(question_group)

    return chunks


def build_manual_chunks(
    node_map: dict[str, Node],
    config: Config,
) -> tuple[list[dict[str, Any]], set[str]]:
    rows, fields = read_csv(MANUAL_TABLE_PATH)
    required = {
        "record_id", "source_id", "node_id", "pdf_page", "table_id",
        "display_title", "text", "include_status", "extraction_mode",
        "review_status",
    }
    missing = required - set(fields)
    if missing:
        raise ValueError(
            f"Missing manual table columns: {sorted(missing)}"
        )

    chunks: list[dict[str, Any]] = []
    covered_node_ids: set[str] = set()

    for row in rows:
        if row["source_id"] not in config.included_sources:
            continue
        if row["include_status"] != "included":
            continue
        if row["review_status"] != "human_verified":
            raise ValueError(
                f"Manual row is not human_verified: "
                f"{row['record_id']}"
            )

        node = node_map.get(row["node_id"])
        if node is None:
            raise ValueError(
                f"Manual evidence node missing: "
                f"{row['node_id']}"
            )
        if node.extraction_mode not in MANUAL_MODES:
            raise ValueError(
                f"Node is not manual_table/hybrid: "
                f"{node.node_id}"
            )

        covered_node_ids.add(node.node_id)
        page = int(row["pdf_page"])

        chunks.append(
            make_chunk(
                chunk_id=row["record_id"],
                source_id=row["source_id"],
                source_order=node.order,
                node_id=node.node_id,
                content_type="manual_table_chunk",
                display_title=row["display_title"],
                path=title_path(node, node_map),
                page_start=page,
                page_end=page,
                extraction_mode="manual_table",
                review_status="human_verified",
                build_method="manual_table_import_v1",
                text=row["text"],
                corpus_version=config.corpus_version,
                table_id=row["table_id"],
            )
        )

    return chunks, covered_node_ids


def audit_manual_coverage(
    nodes: list[Node],
    covered_node_ids: set[str],
    cleanup: dict[str, Any],
    diagnostics: list[dict[str, Any]],
) -> None:
    allowed = set(
        cleanup["allowed_missing_manual_evidence"]
    )

    for node in nodes:
        if node.include_status != "included":
            continue
        if node.extraction_mode not in MANUAL_MODES:
            continue
        if node.node_id in covered_node_ids:
            continue

        if node.extraction_mode == "manual_table":
            status = (
                "allowed_for_mvp"
                if node.node_id in allowed
                else "error"
            )
            add_diagnostic(
                diagnostics,
                source_id=node.source_id,
                node_id=node.node_id,
                category="missing_manual_evidence",
                status=status,
                detail=node.display_title,
            )
            if status == "error":
                raise ValueError(
                    f"Missing manual evidence for "
                    f"{node.node_id}: {node.display_title}"
                )
        else:
            add_diagnostic(
                diagnostics,
                source_id=node.source_id,
                node_id=node.node_id,
                category="hybrid_without_manual_evidence",
                status="warning",
                detail=node.display_title,
            )



def validate_chunks(
    chunks: list[dict[str, Any]],
    config: Config,
) -> list[str]:
    if not chunks:
        raise ValueError("Gold corpus is empty.")

    ids = [str(chunk["chunk_id"]) for chunk in chunks]
    hashes = [str(chunk["text_sha256"]) for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk_id found.")
    if len(hashes) != len(set(hashes)):
        raise ValueError("Duplicate chunk text found.")

    actual_sources = {
        str(chunk["source_id"])
        for chunk in chunks
    }
    missing_sources = (
        set(config.included_sources)
        - actual_sources
    )
    if missing_sources:
        raise ValueError(
            f"No chunks for sources: {sorted(missing_sources)}"
        )

    forbidden_patterns = {
        "reference_section": r"参考文献",
        "workgroup_roster": (
            r"共识工作组成员名单|"
            r"首席临床专家：|秘书组："
        ),
        "known_layout_address": (
            r"Kunming City,\s*Yunnan Province"
        ),
        "table_header_pollution": (
            r"表[1-7]骨关节炎"
            r"(?:临床分期|口服药物治疗推荐|"
            r"外用药物治疗推荐|物理治疗推荐|"
            r"常见骨关节炎的临床表现)"
        ),
    }

    warnings: list[str] = []

    for chunk in chunks:
        if chunk["source_id"] not in config.included_sources:
            raise ValueError(
                f"Unexpected source: {chunk['source_id']}"
            )
        if not chunk["text"]:
            raise ValueError(
                f"Empty text: {chunk['chunk_id']}"
            )
        if is_citation_only(str(chunk["text"])):
            raise ValueError(
                f"Citation-only chunk: {chunk['chunk_id']}"
            )

        for label, pattern in forbidden_patterns.items():
            if re.search(pattern, str(chunk["text"])):
                raise ValueError(
                    f"Forbidden {label} in "
                    f"{chunk['chunk_id']}"
                )

        if chunk["content_type"] == "question_recommendation":
            if not chunk["question_node_id"]:
                raise ValueError(
                    f"Missing question_node_id: "
                    f"{chunk['chunk_id']}"
                )
            if not chunk["recommendation_node_ids"]:
                raise ValueError(
                    f"Missing recommendation ID: "
                    f"{chunk['chunk_id']}"
                )

        if (
            chunk["content_type"] == "manual_table_chunk"
            and chunk["review_status"] != "human_verified"
        ):
            raise ValueError(
                f"Unverified manual chunk: "
                f"{chunk['chunk_id']}"
            )

        length = int(chunk["char_count"])
        if length < config.min_chars:
            warnings.append(
                f"short_chunk:{chunk['chunk_id']}:{length}"
            )
        if length > config.max_chars:
            warnings.append(
                f"long_chunk:{chunk['chunk_id']}:{length}"
            )

    return warnings

def csv_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    row = dict(chunk)
    row["title_path"] = " > ".join(chunk["title_path"])
    row["recommendation_node_ids"] = "|".join(
        chunk["recommendation_node_ids"]
    )
    row["cross_page"] = str(bool(chunk["cross_page"])).lower()
    row["context_repeated"] = str(
        bool(chunk["context_repeated"])
    ).lower()
    return row


def write_gold(chunks: list[dict[str, Any]]) -> None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    with JSONL_PATH.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )
    write_csv(CSV_PATH, [csv_chunk(chunk) for chunk in chunks])


def build_summary(
    chunks: list[dict[str, Any]],
    config: Config,
    warnings: list[str],
    skipped: int,
) -> list[dict[str, Any]]:
    groups: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for chunk in chunks:
        groups[(chunk["source_id"], chunk["content_type"])].append(chunk)

    rows: list[dict[str, Any]] = []
    for (source_id, content_type), group in sorted(groups.items()):
        lengths = [int(chunk["char_count"]) for chunk in group]
        rows.append({
            "corpus_version": config.corpus_version,
            "source_id": source_id,
            "content_type": content_type,
            "chunk_count": len(group),
            "cross_page_count": sum(
                bool(chunk["cross_page"]) for chunk in group
            ),
            "context_repeated_count": sum(
                bool(chunk["context_repeated"]) for chunk in group
            ),
            "human_verified_count": sum(
                chunk["review_status"] == "human_verified"
                for chunk in group
            ),
            "min_chars": min(lengths),
            "median_chars": round(statistics.median(lengths), 1),
            "mean_chars": round(statistics.mean(lengths), 1),
            "max_chars": max(lengths),
            "below_min_count": sum(
                length < config.min_chars for length in lengths
            ),
            "above_max_count": sum(
                length > config.max_chars for length in lengths
            ),
            "build_warning_count": len(warnings),
            "skipped_structural_only_count": skipped,
        })

    lengths = [int(chunk["char_count"]) for chunk in chunks]
    rows.append({
        "corpus_version": config.corpus_version,
        "source_id": "ALL",
        "content_type": "ALL",
        "chunk_count": len(chunks),
        "cross_page_count": sum(
            bool(chunk["cross_page"]) for chunk in chunks
        ),
        "context_repeated_count": sum(
            bool(chunk["context_repeated"]) for chunk in chunks
        ),
        "human_verified_count": sum(
            chunk["review_status"] == "human_verified"
            for chunk in chunks
        ),
        "min_chars": min(lengths),
        "median_chars": round(statistics.median(lengths), 1),
        "mean_chars": round(statistics.mean(lengths), 1),
        "max_chars": max(lengths),
        "below_min_count": sum(
            length < config.min_chars for length in lengths
        ),
        "above_max_count": sum(
            length > config.max_chars for length in lengths
        ),
        "build_warning_count": len(warnings),
        "skipped_structural_only_count": skipped,
    })
    return rows



def review_sample(
    chunks: list[dict[str, Any]],
    size: int,
    seed: int,
    cleanup: dict[str, Any],
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    high_risk_nodes = set(
        cleanup["high_risk_review_node_ids"]
    )

    selected_by_id: dict[
        str,
        dict[str, Any],
    ] = {}

    def add(chunk: dict[str, Any]) -> None:
        selected_by_id[chunk["chunk_id"]] = chunk

    chunk_by_id = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    for chunk in chunks:
        if chunk["content_type"] == "manual_table_chunk":
            add(chunk)
        if chunk["node_id"] in high_risk_nodes:
            add(chunk)
        if bool(chunk["context_repeated"]):
            add(chunk)

            # Review the first chunk of the same recommendation group,
            # otherwise continuation quality is judged without seeing
            # the opening recommendation and rationale.
            group_prefix = chunk["chunk_id"].rsplit("-C", 1)[0]
            first_chunk = chunk_by_id.get(
                f"{group_prefix}-C01"
            )
            if first_chunk is not None:
                add(first_chunk)

    # Always inspect the last retrievable chunk from each source.
    for source_id in sorted({
        chunk["source_id"] for chunk in chunks
    }):
        source_chunks = [
            chunk for chunk in chunks
            if chunk["source_id"] == source_id
        ]
        add(max(
            source_chunks,
            key=lambda item: (
                int(item["source_order"]),
                item["chunk_id"],
            ),
        ))

    remaining = [
        chunk for chunk in chunks
        if chunk["chunk_id"] not in selected_by_id
    ]
    rng.shuffle(remaining)

    for chunk in remaining:
        if len(selected_by_id) >= size:
            break
        add(chunk)

    selected = list(selected_by_id.values())
    selected.sort(
        key=lambda item: (
            item["source_id"],
            int(item["source_order"]),
            item["chunk_id"],
        )
    )

    rows = []
    for chunk in selected:
        rows.append({
            "chunk_id": chunk["chunk_id"],
            "source_id": chunk["source_id"],
            "content_type": chunk["content_type"],
            "node_id": chunk["node_id"],
            "display_title": chunk["display_title"],
            "title_path": " > ".join(chunk["title_path"]),
            "pdf_page_start": chunk["pdf_page_start"],
            "pdf_page_end": chunk["pdf_page_end"],
            "char_count": chunk["char_count"],
            "context_repeated": str(
                bool(chunk["context_repeated"])
            ).lower(),
            "text": chunk["text"],
            "starts_mid_sentence": "",
            "ends_mid_sentence": "",
            "single_topic": "",
            "question_recommendation_complete": "",
            "title_path_correct": "",
            "medical_conditions_preserved": "",
            "author_or_reference_noise": "",
            "usable_for_retrieval": "",
            "review_notes": "",
        })
    return rows


def main() -> None:
    config = load_config()
    cleanup = load_cleanup_config()
    diagnostics: list[dict[str, Any]] = []

    print("\nGold Corpus v1.3 build")
    print("=" * 90)
    print("Sources:", ", ".join(config.included_sources))
    print(
        "Chunk limits:",
        f"target={config.target_chars}",
        f"max={config.max_chars}",
        f"min={config.min_chars}",
        f"sentence_overflow={config.sentence_overflow_chars}",
    )

    require_passing_reports()
    pages = load_pages()
    boundaries = load_boundaries()
    nodes = load_nodes(config, cleanup)

    locate_and_slice_nodes(
        nodes,
        pages,
        boundaries,
        cleanup,
    )
    node_map = {
        node.node_id: node
        for node in nodes
    }

    section_chunks, skipped = build_section_chunks(
        nodes,
        node_map,
        config,
        cleanup,
        diagnostics,
    )
    question_chunks = build_question_chunks(
        nodes,
        node_map,
        config,
        cleanup,
        diagnostics,
    )
    manual_chunks, covered_manual_nodes = (
        build_manual_chunks(
            node_map,
            config,
        )
    )

    audit_manual_coverage(
        nodes,
        covered_manual_nodes,
        cleanup,
        diagnostics,
    )

    chunks = (
        section_chunks
        + question_chunks
        + manual_chunks
    )
    chunks.sort(
        key=lambda item: (
            item["source_id"],
            int(item["source_order"]),
            item["content_type"],
            item["chunk_id"],
        )
    )

    warnings = validate_chunks(
        chunks,
        config,
    )
    write_gold(chunks)
    write_csv(
        SUMMARY_PATH,
        build_summary(
            chunks,
            config,
            warnings,
            skipped,
        ),
    )

    if diagnostics:
        write_csv(
            DIAGNOSTICS_PATH,
            diagnostics,
        )
    else:
        write_csv(
            DIAGNOSTICS_PATH,
            [{
                "source_id": "ALL",
                "node_id": "",
                "category": "build",
                "status": "pass",
                "detail": "no diagnostics",
                "value": "",
            }],
        )

    write_csv(
        REVIEW_PATH,
        review_sample(
            chunks,
            config.review_sample_size,
            config.random_seed,
            cleanup,
        ),
    )

    print("-" * 90)
    print("Total chunks:", len(chunks))
    print(
        "By source:",
        dict(Counter(
            chunk["source_id"]
            for chunk in chunks
        )),
    )
    print(
        "By content type:",
        dict(Counter(
            chunk["content_type"]
            for chunk in chunks
        )),
    )
    print(
        "Skipped structural-only nodes:",
        skipped,
    )
    print("Warnings:", len(warnings))
    for warning in warnings[:20]:
        print(" -", warning)
    if len(warnings) > 20:
        print(
            f" ... {len(warnings) - 20} more"
        )
    print("JSONL:", JSONL_PATH)
    print("CSV:", CSV_PATH)
    print("Review sample:", REVIEW_PATH)
    print("Public summary:", SUMMARY_PATH)
    print("Diagnostics:", DIAGNOSTICS_PATH)
    print("=" * 90)


if __name__ == "__main__":
    main()

