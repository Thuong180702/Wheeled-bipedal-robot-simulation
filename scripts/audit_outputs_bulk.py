#!/usr/bin/env python3
"""Audit first-level outputs/ directories for the deep-cleanup pass.

Read-only. Lists every ``outputs/*`` directory, computes its size, checks
whether any kept doc/script/README/paper references it, and classifies each
directory into one of: PROTECT / SUMMARIZE_THEN_DELETE / DELETE_DIRECT / REVIEW.

Writes:
    docs/repo_cleanup/execution/deep_outputs_audit.csv
    docs/repo_cleanup/execution/deep_outputs_audit.json
    docs/repo_cleanup/execution/deep_outputs_audit.md

Nothing is deleted or moved by this script.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / "outputs"
EXEC_DIR = REPO / "docs" / "repo_cleanup" / "execution"

# Directories that must never be touched.
PROTECTED_NAMES = {"balance", "physical_target_height_setups"}

# Where we look for references to a directory name.
REFERENCE_ROOTS = [
    REPO / "README.md",
    REPO / "CLAUDE.md",
    REPO / "docs",
    REPO / "scripts",
    REPO / "paper",
]

# Reference hits inside these paths do NOT count as "kept" references:
# the archived experiment docs and the cleanup-audit docs themselves.
REFERENCE_EXCLUDE_SUBSTRINGS = [
    str((REPO / "archive").resolve()).replace("\\", "/"),
    str((EXEC_DIR).resolve()).replace("\\", "/"),
    str((REPO / "docs" / "repo_cleanup").resolve()).replace("\\", "/"),
]

SIZE_SUMMARIZE_THRESHOLD = 50 * 1024 * 1024  # 50 MB


def dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def human(nbytes: int) -> str:
    val = float(nbytes)
    for unit in ("B", "K", "M", "G", "T"):
        if val < 1024 or unit == "T":
            return f"{val:.1f}{unit}"
        val /= 1024
    return f"{val:.1f}T"


def find_references(dirname: str) -> list[str]:
    """Return kept files (outside archive/exec) that mention outputs/<dirname>."""
    needle = f"outputs/{dirname}"
    hits: list[str] = []
    try:
        # git grep is fast and respects the working tree; fall back to rg-less scan.
        proc = subprocess.run(
            ["git", "grep", "-l", "-F", needle, "--",
             "README.md", "CLAUDE.md", "docs/", "scripts/", "paper/"],
            cwd=str(REPO), capture_output=True, text=True,
        )
        candidates = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    except Exception:
        candidates = []

    for c in candidates:
        cp = str((REPO / c).resolve()).replace("\\", "/")
        if any(excl in cp for excl in REFERENCE_EXCLUDE_SUBSTRINGS):
            continue
        hits.append(c)
    return sorted(set(hits))


def classify(name: str, size: int, refs: list[str], has_summary: bool) -> tuple[str, str]:
    if name in PROTECTED_NAMES:
        return "PROTECT", "Trained checkpoints / setup JSONs — never delete."
    if size >= SIZE_SUMMARIZE_THRESHOLD:
        ref_note = (f"referenced by {len(refs)} kept file(s)" if refs
                    else "no kept reference found")
        return "SUMMARIZE_THEN_DELETE", (
            f"Large ({human(size)}) generated diagnostic/raw output; {ref_note}; "
            "regenerable. Extract small summaries first."
        )
    # Small dir
    if size == 0:
        return "DELETE_DIRECT", "Empty directory."
    if not has_summary and not refs:
        return "DELETE_DIRECT", (
            f"Small ({human(size)}) generated output, no summary files, "
            "no kept reference."
        )
    return "SUMMARIZE_THEN_DELETE", (
        f"Small ({human(size)}) generated output; "
        + ("has summary artifacts; " if has_summary else "")
        + (f"referenced by {len(refs)} kept file(s); " if refs else "")
        + "extract summaries then delete."
    )


SUMMARY_GLOBS = ("*.json", "*.md", "*.txt")


def has_small_summary(path: Path) -> bool:
    for pattern in SUMMARY_GLOBS:
        for f in path.rglob(pattern):
            try:
                if f.is_file() and f.stat().st_size <= 10 * 1024 * 1024:
                    return True
            except OSError:
                pass
    return False


def main() -> None:
    if not OUTPUTS.is_dir():
        raise SystemExit(f"outputs/ not found at {OUTPUTS}")

    EXEC_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for child in sorted(p for p in OUTPUTS.iterdir() if p.is_dir()):
        name = child.name
        size = dir_size_bytes(child)
        refs = find_references(name)
        summary = has_small_summary(child)
        cls, reason = classify(name, size, refs, summary)
        rows.append({
            "directory": f"outputs/{name}",
            "name": name,
            "size_bytes": size,
            "size_human": human(size),
            "references": refs,
            "n_references": len(refs),
            "has_small_summary": summary,
            "classification": cls,
            "reason": reason,
        })

    # also account for top-level files directly under outputs/
    top_files = [p for p in OUTPUTS.iterdir() if p.is_file()]

    rows.sort(key=lambda r: r["size_bytes"], reverse=True)

    # JSON
    (EXEC_DIR / "deep_outputs_audit.json").write_text(
        json.dumps({
            "outputs_root": "outputs",
            "total_dirs": len(rows),
            "top_level_files": [p.name for p in top_files],
            "directories": rows,
        }, indent=2),
        encoding="utf-8",
    )

    # CSV
    with (EXEC_DIR / "deep_outputs_audit.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["directory", "size_bytes", "size_human", "classification",
                    "n_references", "has_small_summary", "reason"])
        for r in rows:
            w.writerow([r["directory"], r["size_bytes"], r["size_human"],
                        r["classification"], r["n_references"],
                        r["has_small_summary"], r["reason"]])

    # MD
    by_cls: dict[str, list[dict]] = {}
    for r in rows:
        by_cls.setdefault(r["classification"], []).append(r)

    lines = ["# Deep Outputs Audit", "",
             f"Total first-level `outputs/*` directories: **{len(rows)}**", ""]
    for cls in ("PROTECT", "SUMMARIZE_THEN_DELETE", "DELETE_DIRECT", "REVIEW"):
        group = by_cls.get(cls, [])
        tot = human(sum(r["size_bytes"] for r in group))
        lines.append(f"## {cls} ({len(group)} dirs, {tot})")
        lines.append("")
        if group:
            lines.append("| Directory | Size | Refs | Summary | Reason |")
            lines.append("|---|---|---|---|---|")
            for r in group:
                lines.append(
                    f"| `{r['directory']}` | {r['size_human']} | {r['n_references']} | "
                    f"{'yes' if r['has_small_summary'] else 'no'} | {r['reason']} |"
                )
        lines.append("")
    (EXEC_DIR / "deep_outputs_audit.md").write_text("\n".join(lines), encoding="utf-8")

    # Console summary
    for cls in ("PROTECT", "SUMMARIZE_THEN_DELETE", "DELETE_DIRECT", "REVIEW"):
        group = by_cls.get(cls, [])
        print(f"{cls}: {len(group)} dirs, {human(sum(r['size_bytes'] for r in group))}")
    print(f"top-level files under outputs/: {[p.name for p in top_files]}")


if __name__ == "__main__":
    main()
