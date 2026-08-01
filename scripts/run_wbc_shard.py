#!/usr/bin/env python3
"""Run one shard of the post-fix WBC campaign into its own output directory.

``phase3d_full_batch_execution.py`` hard-codes ``OUTPUT_DIR`` and has no seed
filter, so the full 225-scenario three-arm campaign is one ~69 h serial job
(~0.22 s per QP-arm control step, two QP arms per scenario). This wrapper
rebinds those module globals before calling ``main()``, which makes the
campaign shardable across cores. ``scripts/merge_wbc_shards.py`` stitches the
per-shard JSONL files back into the canonical output directory for reporting.

Nothing about the physics, controllers, scenarios, or metrics changes: each
shard runs the unmodified suite runners over a subset of
(suite x height variant x seed).

Usage:
  .venv/bin/python scripts/run_wbc_shard.py \
      --tag step_e__nominal --suite step_e --height nominal
  .venv/bin/python scripts/run_wbc_shard.py \
      --tag random_push__nominal__s201 --suite random_push --height nominal \
      --seeds 201,202,203,204,205
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHARD_ROOT = ROOT / "outputs" / "wbc_postfix_shards"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="Shard directory name.")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--height", required=True)
    ap.add_argument("--seeds", default="",
                    help="Comma-separated seed subset owned by this shard.")
    args, passthrough = ap.parse_known_args()

    spec = importlib.util.spec_from_file_location(
        "phase3d_batch", ROOT / "scripts" / "phase3d_full_batch_execution.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase3d_batch"] = mod
    spec.loader.exec_module(mod)

    shard_dir = SHARD_ROOT / args.tag
    shard_dir.mkdir(parents=True, exist_ok=True)
    mod.OUTPUT_DIR = shard_dir
    mod.JSONL_PATH = shard_dir / "full_batch_results.jsonl"
    # Otherwise every shard would overwrite the same campaign report in
    # docs/validation/ concurrently. The real report is written once, after
    # merging, by rerunning the batch script itself.
    mod.REPORT_PATH = shard_dir / "shard_report.md"

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        # main() reads these module globals when it builds the suite kwargs,
        # so narrowing them here narrows exactly this shard's scenario set.
        mod.STEP_D_SEEDS = seeds
        mod.SINGLE_PUSH_SEEDS = seeds
        mod.RANDOM_PUSH_SEEDS = seeds

    # --full keeps the paper's step counts; --resume makes a shard restartable
    # without wiping its own JSONL. The V3 truth check is run once by the
    # launcher instead of 70 times (same commit, same controller, same result).
    sys.argv = [
        "phase3d_full_batch_execution.py",
        "--full", "--resume", "--skip-truth-check",
        "--suite", args.suite,
        "--height", args.height,
        *passthrough,
    ]
    mod.main()


if __name__ == "__main__":
    main()
