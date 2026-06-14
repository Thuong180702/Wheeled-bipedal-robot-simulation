#!/usr/bin/env python3
"""Phase 3 of deep outputs cleanup: extract small summary artifacts.

Reads the deep-outputs audit JSON and, for every directory classified
SUMMARIZE_THEN_DELETE, copies small summary artifacts into
``archive/cleanup_2026-06-13/output_summaries/<DIR_NAME>/`` and writes a
per-directory ``manifest.json``. A global manifest is written to
``docs/repo_cleanup/execution/deep_outputs_summary_extraction_manifest.json``.

No directory is deleted here. This step only reads ``outputs/`` and writes
into the archive tree. Run :mod:`scripts.clean_outputs_bulk` afterwards to
perform deletion.

Copy rules (all must hold for a file to be copied):
- name matches one of the summary glob patterns (case-insensitive), and
- file size is <= MAX_COPY_BYTES (10 MB).

If a directory yields no copyable summaries, its manifest records
``copied_files: []`` with a note "no small summaries found"; deletion is
still allowed (the audit already confirmed it is regenerable + protected
paths excluded).
"""
from __future__ import annotations

import fnmatch
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AUDIT_JSON = REPO / "docs/repo_cleanup/execution/deep_outputs_audit.json"
ARCHIVE_ROOT = REPO / "archive/cleanup_2026-06-13/output_summaries"
GLOBAL_MANIFEST = (
    REPO / "docs/repo_cleanup/execution/deep_outputs_summary_extraction_manifest.json"
)

MAX_COPY_BYTES = 10 * 1024 * 1024  # 10 MB

# Case-insensitive name patterns for "small useful summary" artifacts.
SUMMARY_PATTERNS = [
    "*.json",
    "*summary*.json",
    "*decision*.json",
    "*final*.json",
    "*metrics*.csv",
    "*comparison*.csv",
    "*window*.csv",
    "*.md",
    "*.txt",
]

# Only extract from these classes.
EXTRACT_CLASSES = {"SUMMARIZE_THEN_DELETE"}


def matches_summary(name: str) -> bool:
    low = name.lower()
    return any(fnmatch.fnmatch(low, pat) for pat in SUMMARY_PATTERNS)


def human(n: int) -> str:
    f = float(n)
    for unit in ("B", "K", "M", "G", "T"):
        if f < 1024.0 or unit == "T":
            return f"{f:.1f}{unit}"
        f /= 1024.0
    return f"{f:.1f}T"


def extract_dir(entry: dict) -> dict:
    """Extract summaries for one directory entry; return its manifest."""
    src_rel = entry["directory"]
    src = REPO / src_rel
    name = entry["name"]
    dest = ARCHIVE_ROOT / name
    dest.mkdir(parents=True, exist_ok=True)

    copied: list[dict] = []
    skipped_large: list[dict] = []

    if src.is_dir():
        for path in sorted(src.rglob("*")):
            if not path.is_file():
                continue
            if not matches_summary(path.name):
                continue
            size = path.stat().st_size
            rel_inside = path.relative_to(src).as_posix()
            if size > MAX_COPY_BYTES:
                skipped_large.append({"file": rel_inside, "size_bytes": size})
                continue
            # Flatten nested paths into the dest using a path-safe name to
            # avoid collisions while keeping provenance.
            flat = rel_inside.replace("/", "__")
            target = dest / flat
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied.append({"file": rel_inside, "size_bytes": size})

    manifest = {
        "original_path": src_rel,
        "original_size": entry.get("size_human", human(entry.get("size_bytes", 0))),
        "original_size_bytes": entry.get("size_bytes", 0),
        "classification": entry.get("classification"),
        "references_count": len(entry.get("references", [])),
        "copied_files": copied,
        "skipped_large_files": skipped_large,
        "deletion_allowed": True,
        "reason": (
            "Summaries extracted; raw regenerable output safe to delete."
            if copied
            else "no small summaries found; raw regenerable output safe to delete."
        ),
    }
    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> int:
    data = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    dirs = data["directories"]
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)

    targets = [d for d in dirs if d.get("classification") in EXTRACT_CLASSES]
    global_manifest: list[dict] = []
    total_copied = 0
    total_skipped = 0
    failures: list[dict] = []

    for entry in targets:
        try:
            m = extract_dir(entry)
            global_manifest.append(m)
            total_copied += len(m["copied_files"])
            total_skipped += len(m["skipped_large_files"])
        except Exception as exc:  # continue, record failure
            failures.append({"directory": entry.get("directory"), "error": repr(exc)})

    summary = {
        "extract_classes": sorted(EXTRACT_CLASSES),
        "max_copy_bytes": MAX_COPY_BYTES,
        "patterns": SUMMARY_PATTERNS,
        "dirs_processed": len(targets),
        "total_files_copied": total_copied,
        "total_large_files_skipped": total_skipped,
        "failures": failures,
        "per_directory": global_manifest,
    }
    GLOBAL_MANIFEST.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"dirs_processed={len(targets)}")
    print(f"total_files_copied={total_copied}")
    print(f"total_large_files_skipped={total_skipped}")
    print(f"failures={len(failures)}")
    if failures:
        for f in failures:
            print("FAIL", f)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
