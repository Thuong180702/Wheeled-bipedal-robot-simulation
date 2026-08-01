#!/usr/bin/env python3
"""Compare WBC task-weight modes side by side and name the best one.

Reads the shards written by scripts/run_wbc_taskweight_ablation.sh plus the
main campaign's balanced_default shard (reused as the ablation baseline), and
ranks the modes on the metric that decides the paper's claim: how much of the
episode the WBC-only arm spends in a fallen state.

The point of the table is to let the paper say "even the best of four weight
modes" rather than "the one weighting we happened to run".

Usage:
  .venv/bin/python scripts/summarize_wbc_ablation.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHARD_ROOT = ROOT / "outputs" / "wbc_postfix_shards"

# mode -> shard directory. balanced_default reuses the main campaign's shard,
# which is the identical scenario set at that mode.
MODES = {
    "balanced_default": SHARD_ROOT / "random_push__nominal__s201",
    "torso_priority": SHARD_ROOT / "ablation__torso_priority",
    "posture_priority": SHARD_ROOT / "ablation__posture_priority",
    "com_priority": SHARD_ROOT / "ablation__com_priority",
}


def load(path: Path) -> list[dict]:
    f = path / "full_batch_results.jsonl"
    if not f.exists():
        return []
    return [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]


def stats(entries: list[dict]) -> dict | None:
    live = [e for e in entries if not e.get("blocked")]
    if not live:
        return None
    steps = fallen = v3_fallen = 0
    roll: list[float] = []
    pitch: list[float] = []
    height: list[float] = []
    for e in live:
        c = e.get("comparison") or {}
        n = int(c.get("n_steps", 0))
        steps += n
        fc = c.get("fall_comparison", {})
        fallen += int(fc.get("wbc_only_falls", 0))
        v3_fallen += int(fc.get("v3_falls", 0))
        pm = c.get("physical_metrics", {}).get("wbc_only", {}) or {}
        for src, dst in ((("roll_rms_rad"), roll), (("pitch_rms_rad"), pitch)):
            v = pm.get(src)
            if isinstance(v, (int, float)) and not math.isnan(v):
                dst.append(math.degrees(float(v)))
        h = pm.get("height_rms")
        if isinstance(h, (int, float)) and not math.isnan(h):
            height.append(float(h))
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    return {
        "n": len(live),
        "fallen_pct": 100.0 * fallen / steps if steps else float("nan"),
        "v3_fallen_pct": 100.0 * v3_fallen / steps if steps else float("nan"),
        "roll_deg": mean(roll),
        "pitch_deg": mean(pitch),
        "height_m": mean(height),
    }


def main() -> None:
    rows = {m: stats(load(p)) for m, p in MODES.items()}
    have = {m: s for m, s in rows.items() if s}

    print(f"{'WBC task-weight mode':<20}{'N':>4}{'fallen %':>11}{'roll deg':>11}"
          f"{'pitch deg':>11}{'base h (m)':>12}")
    print("-" * 69)
    for m in MODES:
        s = rows[m]
        if not s:
            print(f"{m:<20}{'--- not run ---':>49}")
            continue
        print(f"{m:<20}{s['n']:>4}{s['fallen_pct']:>10.1f}%{s['roll_deg']:>11.2f}"
              f"{s['pitch_deg']:>11.2f}{s['height_m']:>12.3f}")

    if not have:
        print("\nNo ablation shards found yet.")
        return

    best = min(have.items(), key=lambda kv: kv[1]["fallen_pct"])
    acc = next(iter(have.values()))["v3_fallen_pct"]
    print("-" * 69)
    print(f"ACC (V3) on the same scenarios: {acc:.1f}% of episode fallen")
    print(f"Best WBC mode: {best[0]} at {best[1]['fallen_pct']:.1f}% fallen")
    if best[1]["fallen_pct"] > 50.0:
        print("=> No weighting rescues WBC; the paper may state 'even the best of "
              f"{len(have)} weight modes'.")
    else:
        print("=> A weighting materially changes the result. Do NOT reuse the "
              "balanced_default numbers in the paper; re-run the full campaign "
              f"at task_mode={best[0]}.")


if __name__ == "__main__":
    main()
