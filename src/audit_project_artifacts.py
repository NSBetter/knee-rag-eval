"""Audit required project artifacts and their Git status.

Run from anywhere inside the knee-rag-eval repository:

    uv run python src/audit_project_artifacts.py

The script does not contact GitHub. Run `git fetch origin` first when
you need an up-to-date comparison with the remote branch.
"""

from __future__ import annotations

import csv
import glob
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Artifact:
    pattern: str
    category: str
    requirement: str
    visibility: str
    purpose: str


ARTIFACTS: tuple[Artifact, ...] = (
    Artifact(".gitignore", "governance", "required", "public", "Protect local and restricted data"),
    Artifact("README.md", "governance", "required", "public", "Project entry point"),
    Artifact("PROJECT_CHARTER.md", "governance", "required", "public", "Scope and objectives"),
    Artifact("DECISIONS.md", "governance", "required", "public", "Engineering decision log"),
    Artifact("docs/engineering_artifact_inventory.md", "governance", "required", "public", "Artifact inventory"),
    Artifact("docs/page_content_policy.csv", "document_governance", "required", "public", "Page-level content policy"),
    Artifact("docs/document_boundaries.csv", "document_governance", "required", "public", "Validated body boundaries"),
    Artifact("docs/document_boundary_validation.csv", "document_governance", "required", "public", "Boundary audit"),
    Artifact("docs/index_source_policy.csv", "document_governance", "required", "public", "MVP source selection"),
    Artifact("docs/document_structure_map.csv", "document_governance", "required", "public", "Human-reviewed structure"),
    Artifact("docs/document_structure_validation.csv", "document_governance", "required", "public", "Structure audit"),
    Artifact("docs/manual_table_evidence_audit.csv", "manual_tables", "required", "public", "Public table evidence audit"),
    Artifact("configs/gold_corpus_v1_3.json", "gold_corpus", "required", "public", "Final corpus build config"),
    Artifact("configs/gold_corpus_cleanup_v1_3.json", "gold_corpus", "required", "public", "Explicit cleanup rules"),
    Artifact("src/build_gold_corpus.py", "gold_corpus", "required", "public", "Final corpus builder"),
    Artifact("docs/gold_corpus_v1_2_review_summary.csv", "gold_corpus", "recommended", "public", "Pre-optimization review summary"),
    Artifact("docs/gold_corpus_v1_3_build_summary.csv", "gold_corpus", "required", "public", "Final build statistics"),
    Artifact("docs/gold_corpus_v1_3_diagnostics.csv", "gold_corpus", "required", "public", "Final build diagnostics"),
    Artifact("src/validate_document_boundaries.py", "validation", "required", "public", "Boundary validator"),
    Artifact("src/validate_structure_map.py", "validation", "required", "public", "Structure validator"),
    Artifact("src/validate_manual_table_evidence.py", "validation", "required", "public", "Manual table validator"),
    Artifact("src/audit_project_artifacts.py", "validation", "required", "public", "Project artifact auditor"),
    Artifact("configs/chunking_baseline.json", "experiments", "recommended", "public", "Baseline config"),
    Artifact("configs/chunking_section_aware.json", "experiments", "recommended", "public", "Section-regex experiment"),
    Artifact("configs/chunking_boundary_aware.json", "experiments", "recommended", "public", "Boundary-aware experiment"),
    Artifact("src/build_chunks.py", "experiments", "recommended", "public", "Baseline builder"),
    Artifact("src/build_section_chunks.py", "experiments", "recommended", "public", "Failed section-aware builder"),
    Artifact("src/build_boundary_chunks.py", "experiments", "recommended", "public", "Boundary-aware builder"),
    Artifact("src/audit_tables.py", "experiments", "recommended", "public", "Table audit experiment"),
    Artifact("src/compare_table_extractors.py", "experiments", "recommended", "public", "Table extractor comparison"),
    Artifact("docs/chunk_build_summary.csv", "experiments", "recommended", "public", "Baseline summary"),
    Artifact("docs/section_chunk_build_summary.csv", "experiments", "recommended", "public", "Section experiment summary"),
    Artifact("docs/boundary_chunk_build_summary.csv", "experiments", "recommended", "public", "Boundary experiment summary"),
    Artifact("data/corpus/*.pdf", "local_data", "required", "local", "Original guideline PDFs"),
    Artifact("data/processed/pages/*_pages.jsonl", "local_data", "required", "local", "Page-level extracted text"),
    Artifact("data/processed/full_text/*.txt", "local_data", "recommended", "local", "Full-text previews"),
    Artifact("data/processed/manual_tables/manual_table_evidence.csv", "local_data", "required", "local", "Verified table evidence"),
    Artifact("data/processed/manual_tables/review/*.xlsx", "local_data", "recommended", "local", "Human review workbook"),
    Artifact("data/processed/gold_corpus/gold_corpus_v1_3.jsonl", "local_data", "required", "local", "Final corpus JSONL"),
    Artifact("data/processed/gold_corpus/gold_corpus_v1_3.csv", "local_data", "required", "local", "Final corpus CSV"),
    Artifact("data/processed/reviews/gold_corpus_v1_3_review.csv", "local_data", "required", "local", "Final review with text"),
)


def run_git(
    root: Path,
    *args: str,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=check,
    )


def repository_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            "Not inside a Git repository. Open the knee-rag-eval project first."
        )
    return Path(result.stdout.strip())


def matched_paths(root: Path, pattern: str) -> list[Path]:
    return sorted(
        Path(path)
        for path in glob.glob(str(root / pattern), recursive=True)
        if Path(path).is_file()
    )


def is_tracked(root: Path, path: Path) -> bool:
    relative = path.relative_to(root).as_posix()
    result = run_git(
        root,
        "ls-files",
        "--error-unmatch",
        "--",
        relative,
    )
    return result.returncode == 0


def is_ignored(root: Path, path: Path) -> bool:
    relative = path.relative_to(root).as_posix()
    result = run_git(
        root,
        "check-ignore",
        "-q",
        "--",
        relative,
    )
    return result.returncode == 0


def last_commit(root: Path, path: Path) -> str:
    relative = path.relative_to(root).as_posix()
    result = run_git(
        root,
        "log",
        "-1",
        "--format=%h|%ad|%s",
        "--date=short",
        "--",
        relative,
    )
    return result.stdout.strip()


def working_tree_state(root: Path, path: Path) -> str:
    relative = path.relative_to(root).as_posix()
    result = run_git(
        root,
        "status",
        "--porcelain",
        "--",
        relative,
    )
    return result.stdout.strip().replace("\n", " | ")


def remote_summary(root: Path) -> dict[str, str]:
    branch = run_git(
        root,
        "branch",
        "--show-current",
    ).stdout.strip()

    upstream_result = run_git(
        root,
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
    )

    if upstream_result.returncode != 0:
        return {
            "branch": branch,
            "upstream": "",
            "behind": "",
            "ahead": "",
            "remote_status": "no_upstream_configured",
        }

    upstream = upstream_result.stdout.strip()
    counts = run_git(
        root,
        "rev-list",
        "--left-right",
        "--count",
        f"{upstream}...HEAD",
    )

    if counts.returncode != 0:
        return {
            "branch": branch,
            "upstream": upstream,
            "behind": "",
            "ahead": "",
            "remote_status": "comparison_failed",
        }

    behind, ahead = counts.stdout.strip().split()
    return {
        "branch": branch,
        "upstream": upstream,
        "behind": behind,
        "ahead": ahead,
        "remote_status": (
            "synced"
            if behind == "0" and ahead == "0"
            else "not_synced"
        ),
    }


def main() -> None:
    root = repository_root()
    output = root / "docs" / "project_artifact_audit.csv"
    rows: list[dict[str, str]] = []

    for artifact in ARTIFACTS:
        matches = matched_paths(root, artifact.pattern)

        if not matches:
            rows.append(
                {
                    "pattern": artifact.pattern,
                    "matched_path": "",
                    "category": artifact.category,
                    "requirement": artifact.requirement,
                    "visibility": artifact.visibility,
                    "exists": "no",
                    "tracked": "",
                    "ignored": "",
                    "working_tree_state": "",
                    "last_commit": "",
                    "status": (
                        "missing_required"
                        if artifact.requirement == "required"
                        else "missing_recommended"
                    ),
                    "purpose": artifact.purpose,
                }
            )
            continue

        for path in matches:
            tracked = is_tracked(root, path)
            ignored = is_ignored(root, path)

            if artifact.visibility == "public":
                if not tracked:
                    status = (
                        "public_required_untracked"
                        if artifact.requirement == "required"
                        else "public_recommended_untracked"
                    )
                else:
                    status = "ok"
            else:
                if tracked:
                    status = "local_file_should_not_be_tracked"
                elif not ignored:
                    status = "local_file_not_ignored"
                else:
                    status = "ok"

            rows.append(
                {
                    "pattern": artifact.pattern,
                    "matched_path": path.relative_to(root).as_posix(),
                    "category": artifact.category,
                    "requirement": artifact.requirement,
                    "visibility": artifact.visibility,
                    "exists": "yes",
                    "tracked": "yes" if tracked else "no",
                    "ignored": "yes" if ignored else "no",
                    "working_tree_state": working_tree_state(root, path),
                    "last_commit": last_commit(root, path) if tracked else "",
                    "status": status,
                    "purpose": artifact.purpose,
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1

    remote = remote_summary(root)
    porcelain = run_git(root, "status", "--porcelain").stdout.strip()

    print("\nProject artifact audit")
    print("=" * 88)
    print(f"Repository: {root}")
    print(f"Rows: {len(rows)}")
    print(f"Artifact status: {status_counts}")
    print(f"Working tree clean: {'yes' if not porcelain else 'no'}")
    print(
        "Remote: "
        f"branch={remote['branch']} "
        f"upstream={remote['upstream'] or 'none'} "
        f"behind={remote['behind'] or 'unknown'} "
        f"ahead={remote['ahead'] or 'unknown'} "
        f"status={remote['remote_status']}"
    )
    print(f"Audit CSV: {output}")
    print("-" * 88)

    action_rows = [
        row
        for row in rows
        if row["status"] not in {"ok", "missing_recommended"}
    ]

    if action_rows:
        print("Items requiring action:")
        for row in action_rows:
            print(
                f" - {row['status']}: "
                f"{row['matched_path'] or row['pattern']}"
            )
    else:
        print("No required artifact problems detected.")

    if remote["ahead"] not in {"", "0"}:
        print(
            f"Local branch has {remote['ahead']} commit(s) not present "
            "in the configured upstream. Run `git push` after review."
        )

    if remote["behind"] not in {"", "0"}:
        print(
            f"Local branch is behind the upstream by "
            f"{remote['behind']} commit(s). Review before pushing."
        )

    print("=" * 88)


if __name__ == "__main__":
    main()
