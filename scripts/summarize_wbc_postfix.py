#!/usr/bin/env python3
"""Aggregate the merged post-fix WBC campaign into paper Table XI-B rows.

Reads the merged three-arm JSONL written by scripts/merge_wbc_shards.py and
emits the WBC-only vs ACC(V3) comparison in the same metric set as
`tab:wbc_w1_full` in the paper: falls are summed over scenarios, physical
metrics are averaged over non-blocked scenarios, and angles are converted from
radians to degrees.

Aggregation matches the table's own semantics: one unweighted mean per
scenario, over the mixed batch (fixed-height, transitions, pushes), which is
why ACC's pitch RMS here is much larger than its clean-standing value.

Usage:
  .venv/bin/python scripts/summarize_wbc_postfix.py
  .venv/bin/python scripts/summarize_wbc_postfix.py --jsonl <path>
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSONL = ROOT / "outputs" / "phase3d_full_batch_execution" / "full_batch_results.jsonl"

# (paper row label, comparison sub-dict, key, unit conversion)
ROWS = [
    ("Pitch RMS (deg)", "physical_metrics", "pitch_rms_rad", math.degrees),
    ("Roll RMS (deg)", "physical_metrics", "roll_rms_rad", math.degrees),
    ("Yaw drift (deg)", "physical_metrics", "yaw_drift_rms_rad", math.degrees),
    # NB: height_rms is sqrt(mean(h^2)) of the ABSOLUTE base height
    # (offline_three_arm_counterfactual.py:1898), not a tracking error. A LOW
    # value means the robot is on the floor, so the ratio must not be read as
    # "comparable height performance". Same trap on torque: a collapsed arm
    # draws less torque because it has stopped holding itself up.
    ("Base height (RMS, m) [hi=upright]", "physical_metrics", "height_rms", None),
    ("Planar drift (m)", "physical_metrics", "planar_drift_max_m", None),
    ("Torque RMS (Nm) [lo iff upright]", "torque_comparison", "rms_tau", None),
]
ARMS = {"v3": "v3", "wbc": "wbc_only", "assist": "assist"}


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    args = ap.parse_args()

    entries = [json.loads(ln) for ln in args.jsonl.read_text().splitlines() if ln.strip()]
    live = [e for e in entries if not e.get("blocked")]
    blocked = len(entries) - len(live)

    falls = {a: 0 for a in ARMS}
    push_falls = {a: 0 for a in ARMS}
    series: dict[tuple[str, str], list[float]] = {}
    # `*_falls` counts control STEPS in a fallen state, not fall events
    # (offline_three_arm_counterfactual.py:1880), so the only comparable
    # headline is the fraction of the episode spent fallen.
    total_steps = 0
    push_steps = 0
    scenarios_ever_fell = {a: 0 for a in ARMS}

    for e in live:
        comp = e.get("comparison") or {}
        fc = comp.get("fall_comparison", {})
        is_push = str(e.get("suite", "")).endswith("push")
        n_steps = int(comp.get("n_steps", 0))
        total_steps += n_steps
        if is_push:
            push_steps += n_steps
        for short, key in ARMS.items():
            n = int(fc.get(f"{key}_falls", 0))
            falls[short] += n
            if n > 0:
                scenarios_ever_fell[short] += 1
            if is_push:
                push_falls[short] += n
        for _, sub, key, _conv in ROWS:
            block = comp.get(sub, {})
            for short, key_arm in ARMS.items():
                v = (block.get(key_arm) or {}).get(key)
                if isinstance(v, (int, float)) and not math.isnan(v):
                    series.setdefault((short, key), []).append(float(v))

    def ratio(w: float | None, v: float | None) -> str:
        if w is None or v is None or v == 0:
            return "---"
        return f"{w / v:.2f}x"

    print(f"Source:     {args.jsonl}")
    print(f"Scenarios:  {len(live)} evaluated ({blocked} blocked)")
    print(f"Suites:     {dict(Counter(e.get('suite') for e in live))}\n")
    print(f"{'Metric':<34}{'WBC-only':>12}{'ACC (V3)':>12}{'Assist':>12}{'WBC/ACC':>14}")
    print("-" * 80)
    print(f"{'Fallen-state steps (of ' + str(total_steps) + ')':<34}"
          f"{falls['wbc']:>12}{falls['v3']:>12}{falls['assist']:>12}"
          f"{ratio(falls['wbc'], falls['v3']):>14}")
    pct = lambda n, d: "---" if not d else f"{100.0 * n / d:.1f}%"
    print(f"{'  as % of episode':<34}{pct(falls['wbc'], total_steps):>12}"
          f"{pct(falls['v3'], total_steps):>12}{pct(falls['assist'], total_steps):>12}"
          f"{'':>14}")
    print(f"{'Push fallen-state steps':<34}{push_falls['wbc']:>12}{push_falls['v3']:>12}"
          f"{push_falls['assist']:>12}{ratio(push_falls['wbc'], push_falls['v3']):>14}")
    print(f"{'  as % of push episodes':<34}{pct(push_falls['wbc'], push_steps):>12}"
          f"{pct(push_falls['v3'], push_steps):>12}"
          f"{pct(push_falls['assist'], push_steps):>12}{'':>14}")
    print(f"{'Scenarios with any fall':<34}{scenarios_ever_fell['wbc']:>12}"
          f"{scenarios_ever_fell['v3']:>12}{scenarios_ever_fell['assist']:>12}"
          f"{'of ' + str(len(live)):>14}")
    for label, _sub, key, conv in ROWS:
        vals = {}
        for short in ARMS:
            m = _mean(series.get((short, key), []))
            vals[short] = conv(m) if (conv and m is not None) else m
        fmt = lambda x: "---" if x is None else f"{x:.3f}"
        print(f"{label:<34}{fmt(vals['wbc']):>12}{fmt(vals['v3']):>12}"
              f"{fmt(vals['assist']):>12}{ratio(vals['wbc'], vals['v3']):>14}")


if __name__ == "__main__":
    main()
