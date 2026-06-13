#!/usr/bin/env python3
"""Delete the 12 remaining outputs/balance_core* diagnostic directories.

Uses the exact-path PROTECTED_PATHS from clean_outputs_bulk.py so the same
guard protects seed checkpoints / setups / backup_checkpoints. Writes a CSV
delete log to balance_core_delete_log.csv. Does NOT read the audit JSON — it
operates on an explicit allow-list of balance_core dirs only.
"""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from clean_outputs_bulk import REPO, is_protected  # noqa: E402

ARCHIVE = REPO / "archive" / "cleanup_2026-06-13"
SUMMARY_ROOT = REPO / "archive" / "cleanup_2026-06-13" / "output_summaries"
DELETE_LOG = REPO / "docs" / "repo_cleanup" / "execution" / "balance_core_delete_log.csv"

# All 12 remaining balance_core diagnostic dirs — nothing else touched.
BALANCE_CORE_DIRS = [
    "balance_core_validation",
    "balance_core_position_containment",
    "balance_core_position_containment_e0b",
    "balance_core_position_aware_precheck_5000",
    "balance_core_e0_cleanup_validation_5000",
    "balance_core_e0_cleanup_validation",
    "balance_core_position_aware_precheck_1000",
    "balance_core_true_height_variants",
    "balance_core_extended_height_range",
    "balance_core_longevity_height_sweep",
    "balance_core_extended_longevity",
    "balance_core_height_recovery",
]


def dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            try:
                total += (Path(root) / fn).stat().st_size
            except OSError:
                pass
    return total


def main() -> int:
    # Self-check that the fixed guard is correct.
    def _assert(cond: bool, msg: str) -> None:
        if not cond:
            raise AssertionError(msg)

    _assert(is_protected(Path("outputs/balance")) is True, "balance must be protected")
    _assert(
        is_protected(Path("outputs/balance/rl/seed42")) is True,
        "seed42 checkpoint must be protected",
    )
    _assert(
        is_protected(Path("outputs/balance_core_validation")) is False,
        "balance_core_validation must NOT be protected",
    )
    _assert(
        is_protected(Path("outputs/physical_target_height_setups")) is True,
        "setups must be protected",
    )
    print("self-check: PASS")

    rows: list[list[str | int]] = []
    failures: list[tuple[str, str]] = []
    deleted = 0
    freed_bytes = 0

    for name in BALANCE_CORE_DIRS:
        candidate = REPO / "outputs" / name

        # Safety: must not be a protected path
        if is_protected(candidate):
            rows.append([f"outputs/{name}", "REFUSED_PROTECTED", "guard returned True", 0])
            print(f"REFUSED (protected): outputs/{name}")
            continue

        # Safety: manifest must exist
        manifest = SUMMARY_ROOT / name / "manifest.json"
        if not manifest.exists():
            rows.append([f"outputs/{name}", "REFUSED_NO_MANIFEST", str(manifest), 0])
            print(f"REFUSED (no manifest): outputs/{name}")
            continue

        # Safety: must be untracked
        res = subprocess.run(
            ["git", "ls-files", f"outputs/{name}"],
            cwd=str(REPO), capture_output=True, text=True,
        )
        if res.stdout.strip():
            rows.append([f"outputs/{name}", "REFUSED_TRACKED", "has tracked files", 0])
            print(f"REFUSED (tracked): outputs/{name}")
            continue

        # Must exist on disk
        if not candidate.exists():
            rows.append([f"outputs/{name}", "SKIP", "not present", 0])
            print(f"SKIP (already gone): outputs/{name}")
            continue

        sz = dir_size(candidate)
        try:
            shutil.rmtree(candidate)
            deleted += 1
            freed_bytes += sz
            rows.append([f"outputs/{name}", "DELETED", "ok", sz])
            print(f"DELETED outputs/{name} ({sz / (1024**2):.1f}M)")
        except Exception as exc:
            failures.append((f"outputs/{name}", str(exc)))
            rows.append([f"outputs/{name}", "FAILED", str(exc), sz])
            print(f"FAILED outputs/{name}: {exc}")

    with DELETE_LOG.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["directory", "action", "reason", "size_bytes"])
        writer.writerows(rows)

    print("=== summary ===")
    print(f"deleted={deleted}")
    print(f"freed_bytes={freed_bytes} ({freed_bytes / (1024**3):.2f}G)")
    print(f"failures={len(failures)}")
    for d, e in failures:
        print(f"  FAIL {d}: {e}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())