#!/usr/bin/env python3
"""Merge the sharded post-fix WBC campaign back into the canonical layout.

scripts/run_wbc_postfix_campaign.sh runs the 225-scenario three-arm batch as
70 independent shards, each writing its own JSONL. This concatenates them
(de-duplicating on the batch runner's own (scenario, arm, suite) key, keeping
the freshest entry) into outputs/phase3d_full_batch_execution/, which is where
phase3d_full_batch_execution.py's report generator reads from.

After merging, regenerate the campaign report with:
  .venv/bin/python scripts/phase3d_full_batch_execution.py \
      --full --resume --skip-truth-check

Usage:
  .venv/bin/python scripts/merge_wbc_shards.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHARD_ROOT = ROOT / "outputs" / "wbc_postfix_shards"
DEST = ROOT / "outputs" / "phase3d_full_batch_execution"


def main() -> None:
    shards = sorted(p for p in SHARD_ROOT.glob("*/full_batch_results.jsonl"))
    if not shards:
        sys.exit(f"No shard results under {SHARD_ROOT}")

    merged: dict[tuple, dict] = {}
    for path in shards:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            key = (entry.get("scenario"), entry.get("arm"), entry.get("suite"))
            merged[key] = entry  # later shard wins; shards are disjoint anyway

    DEST.mkdir(parents=True, exist_ok=True)
    out = DEST / "full_batch_results.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for entry in merged.values():
            f.write(json.dumps(entry, default=str) + "\n")

    # Headline three-arm tally, so the merge itself reports something checkable.
    tally = {"v3_falls": 0, "wbc_only_falls": 0, "assist_falls": 0}
    blocked = 0
    per_suite: dict[str, dict] = {}
    for entry in merged.values():
        if entry.get("blocked"):
            blocked += 1
            continue
        fc = (entry.get("comparison") or {}).get("fall_comparison", {})
        s = per_suite.setdefault(entry.get("suite", "?"),
                                 {"n": 0, **{k: 0 for k in tally}})
        s["n"] += 1
        for k in tally:
            tally[k] += int(fc.get(k, 0))
            s[k] += int(fc.get(k, 0))

    print(f"Shards merged: {len(shards)}")
    print(f"Scenarios:     {len(merged)} ({blocked} blocked)")
    print(f"Wrote:         {out}")
    print("\nFalls by suite (V3 / WBC-only / V3+assist):")
    for suite, s in sorted(per_suite.items()):
        print(f"  {suite:<13} n={s['n']:>3}  "
              f"{s['v3_falls']:>6} / {s['wbc_only_falls']:>6} / {s['assist_falls']:>6}")
    print(f"  {'TOTAL':<13} n={len(merged) - blocked:>3}  "
          f"{tally['v3_falls']:>6} / {tally['wbc_only_falls']:>6} / "
          f"{tally['assist_falls']:>6}")


if __name__ == "__main__":
    main()
