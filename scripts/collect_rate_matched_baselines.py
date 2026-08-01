#!/usr/bin/env python3
"""Re-run every classical baseline at 50 Hz *and* 100 Hz (rate-matched to ACC).

Answers the reviewer question behind Table VII: ACC runs at 100 Hz and the
classical baselines at their 50 Hz design rate, so ACC enjoys a 2x control
bandwidth advantage. This script removes the confound by running each baseline
at both rates under an otherwise identical protocol (same seeds, same episode
count, same 20 s wall-clock horizon, clean sensors, 500 Hz physics).

Episode length is held constant in *seconds*, not steps: 1000 steps at 50 Hz
and 2000 steps at 100 Hz both cover 20 s.

Usage:
  .venv/bin/python scripts/collect_rate_matched_baselines.py
  .venv/bin/python scripts/collect_rate_matched_baselines.py --n-episodes 20
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (paper row label, --controller, --baseline-config, paper N)
VARIANTS = [
    ("LQR_TWIP", "baseline_lqr", "configs/baseline_lqr.yaml", 60),
    ("LQR_AW", "baseline_lqr_aw", "configs/baseline_lqr_anti_windup.yaml", 60),
    ("LQR_DT", "baseline_lqr_torque", "configs/baseline_lqr_torque.yaml", 20),
    ("LQR_6State", "baseline_coupled_lqr", "configs/baseline_coupled_lqr.yaml", 20),
    ("PI_AW_DT", "baseline_pi_aw", "configs/baseline_pi_aw.yaml", 20),
    ("LQR_6State_DT", "baseline_coupled_lqr_torque",
     "configs/baseline_coupled_lqr_torque.yaml", 20),
]

RATES = (50.0, 100.0)
EPISODE_SECONDS = 20.0
SEEDS = [0, 42, 123]  # paper-documented seeds


def run_one(label: str, ctrl: str, cfg: str, hz: float, n_ep: int,
            out_root: Path) -> dict:
    num_steps = int(round(EPISODE_SECONDS * hz))
    out_dir = out_root / f"{ctrl}_{int(hz)}hz"
    cmd = [
        sys.executable, str(ROOT / "scripts/eval_balance.py"),
        "--controller", ctrl,
        "--baseline-config", cfg,
        "--scenarios", "nominal",
        "--num-episodes", str(n_ep),
        "--num-steps", str(num_steps),
        "--control-hz", str(hz),
        *[a for s in SEEDS for a in ("--seeds", str(s))],
        "--output-dir", str(out_dir),
    ]
    print(f"  [{label} @ {hz:g} Hz] {num_steps} steps x {n_ep} ep ...", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                          cwd=str(ROOT))
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"    FAILED rc={proc.returncode}\n{proc.stderr[-800:]}")
        return {"error": proc.stderr[-1500:], "rc": proc.returncode}

    res_path = out_dir / "eval_results.json"
    if not res_path.exists():
        return {"error": "no eval_results.json", "path": str(res_path)}
    data = json.loads(res_path.read_text())
    rows = data["results"] if isinstance(data.get("results"), list) else [data]
    row = rows[0]
    keep = (
        "survival_time_mean_s", "survival_time_std_s", "fall_rate",
        "survival_rate", "pitch_rms_deg", "roll_rms_deg", "height_rmse_mm",
        "torque_rms_nm", "wheel_speed_rms_rads",
    )
    out = {k: row[k] for k in keep if k in row}
    out["_all"] = row
    out["_elapsed_s"] = round(dt, 1)
    out["_num_steps"] = num_steps
    print(f"    surv={out.get('survival_time_mean_s')}s "
          f"pitch={out.get('pitch_rms_deg'):.3f} "
          f"roll={out.get('roll_rms_deg'):.2f} ({dt:.0f}s)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=20)
    ap.add_argument("--output-dir", type=str,
                    default=str(ROOT / "outputs/rate_matched_baselines"))
    args = ap.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    results: dict = {"_metadata": {
        "n_episodes": args.n_episodes,
        "seeds": SEEDS,
        "rates_hz": list(RATES),
        "episode_seconds": EPISODE_SECONDS,
        "physics_hz": 500,
        "scenario": "nominal",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }}

    print("=" * 72)
    print(f"RATE-MATCHED BASELINES — {len(VARIANTS)} variants x {len(RATES)} rates, "
          f"N={args.n_episodes}")
    print("=" * 72)

    for label, ctrl, cfg, paper_n in VARIANTS:
        print(f"\n{'-' * 72}\n{label}  ({ctrl})")
        results[label] = {"controller": ctrl, "config": cfg, "paper_n": paper_n}
        for hz in RATES:
            results[label][f"{int(hz)}hz"] = run_one(
                label, ctrl, cfg, hz, args.n_episodes, out_root)
        # incremental save so a crash late in the sweep does not lose earlier rows
        (out_root / "results.json").write_text(json.dumps(results, indent=2,
                                                          default=str))

    print(f"\nSaved: {out_root / 'results.json'}")


if __name__ == "__main__":
    main()
