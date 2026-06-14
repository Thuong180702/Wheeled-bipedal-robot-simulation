#!/usr/bin/env python3
"""Guarded deletion of approved raw output directories after summary extraction.

Reads docs/repo_cleanup/execution/deep_outputs_audit.json and deletes only
directories classified SUMMARIZE_THEN_DELETE or DELETE_DIRECT, with hard
refusals for protected paths and a requirement that SUMMARIZE_THEN_DELETE dirs
have an extracted-summary manifest on disk.

Never deletes:
  - outputs/balance
  - outputs/physical_target_height_setups
  - backup_checkpoints

Logs every action to deep_outputs_cleanup_delete_log.csv and continues on
individual failures, reporting them at the end.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXEC_DIR = REPO / "docs" / "repo_cleanup" / "execution"
AUDIT_JSON = EXEC_DIR / "deep_outputs_audit.json"
SUMMARY_ROOT = REPO / "archive" / "cleanup_2026-06-13" / "output_summaries"
DELETE_LOG = EXEC_DIR / "deep_outputs_cleanup_delete_log.csv"

DELETABLE = {"SUMMARIZE_THEN_DELETE", "DELETE_DIRECT"}

# Hard refusal: a candidate is protected only when it is EXACTLY one of these
# paths or lives INSIDE one of them. Substring matching is intentionally NOT
# used so that sibling dirs like outputs/balance_core_validation (which merely
# share the "balance" prefix) are not falsely protected.
PROTECTED_PATHS = frozenset(
    {
        (REPO / "outputs" / "balance").resolve(),
        (REPO / "outputs" / "physical_target_height_setups").resolve(),
        (REPO / "backup_checkpoints").resolve(),
    }
)


def is_protected(candidate: Path) -> bool:
    """True only if candidate is exactly a protected path or nested within one.

    Uses resolved paths and parent containment — NOT substring matching — so
    that ``outputs/balance_core_validation`` is NOT treated as protected while
    ``outputs/balance`` and ``outputs/balance/rl/seed42`` are.
    """
    resolved = (candidate if candidate.is_absolute() else (REPO / candidate)).resolve()
    if resolved in PROTECTED_PATHS:
        return True
    return any(p in resolved.parents for p in PROTECTED_PATHS)


def _self_check() -> None:
    assert is_protected(Path("outputs/balance")) is True, "outputs/balance must be protected"
    assert (
        is_protected(Path("outputs/balance/rl/seed42")) is True
    ), "outputs/balance/rl/seed42 must be protected"
    assert (
        is_protected(Path("outputs/balance_core_validation")) is False
    ), "outputs/balance_core_validation must NOT be protected"
    assert (
        is_protected(Path("outputs/physical_target_height_setups")) is True
    ), "setups dir must be protected"
    assert is_protected(Path("backup_checkpoints")) is True, "backup_checkpoints must be protected"


def main() -> int:
    _self_check()

    if not AUDIT_JSON.exists():
        print(f"FATAL: audit json missing: {AUDIT_JSON}")
        return 2

    data = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    directories = data["directories"]

    rows = []
    failures = []
    deleted = 0
    freed_bytes = 0
    skipped_protected = 0
    skipped_missing_manifest = 0

    for entry in directories:
        rel = entry["directory"].replace("\\", "/")
        name = entry["name"]
        classification = entry["classification"]
        size_bytes = entry.get("size_bytes", 0)
        abs_path = REPO / rel

        # 1) Only consider deletable classifications.
        if classification not in DELETABLE:
            rows.append((rel, classification, "SKIP_PROTECT_CLASS", "not a deletable class", size_bytes))
            continue

        # 2) Hard refusal on protected substrings.
        if is_protected(rel):
            skipped_protected += 1
            rows.append((rel, classification, "REFUSED_PROTECTED", "path matches protected substring", size_bytes))
            print(f"REFUSED (protected): {rel}")
            continue

        # 3) SUMMARIZE_THEN_DELETE requires an extracted manifest on disk.
        if classification == "SUMMARIZE_THEN_DELETE":
            manifest = SUMMARY_ROOT / name / "manifest.json"
            if not manifest.exists():
                skipped_missing_manifest += 1
                rows.append((rel, classification, "REFUSED_NO_MANIFEST", str(manifest), size_bytes))
                print(f"REFUSED (no manifest): {rel}")
                continue

        # 4) Path must exist and live under outputs/.
        if not abs_path.exists():
            rows.append((rel, classification, "ALREADY_GONE", "path not present", size_bytes))
            continue
        if "outputs" not in rel.split("/"):
            rows.append((rel, classification, "REFUSED_OUTSIDE_OUTPUTS", "not under outputs/", size_bytes))
            print(f"REFUSED (outside outputs/): {rel}")
            continue

        # 5) Delete.
        try:
            shutil.rmtree(abs_path)
            deleted += 1
            freed_bytes += size_bytes
            rows.append((rel, classification, "DELETED", "rmtree ok", size_bytes))
        except Exception as exc:  # noqa: BLE001 - continue on individual failure
            failures.append((rel, str(exc)))
            rows.append((rel, classification, "FAILED", str(exc), size_bytes))
            print(f"FAILED: {rel}: {exc}")

    # Write log.
    with DELETE_LOG.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["directory", "classification", "action", "reason", "size_bytes"])
        writer.writerows(rows)

    print("=== deep outputs cleanup summary ===")
    print(f"deleted_dirs={deleted}")
    print(f"freed_bytes={freed_bytes} ({freed_bytes / (1024**3):.2f} G)")
    print(f"refused_protected={skipped_protected}")
    print(f"refused_no_manifest={skipped_missing_manifest}")
    print(f"failures={len(failures)}")
    for rel, exc in failures:
        print(f"  FAIL {rel}: {exc}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
