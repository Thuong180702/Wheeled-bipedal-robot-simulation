#!/usr/bin/env python3
"""Sensitivity of the coupled 6-state LQR baselines to the hip-roll coupling.

The 6-state coupled LQR model closes its roll channel with an empirical
constant b_{r,h} = -5.0 (hip roll angle -> roll acceleration), which is not
available in closed form from the planar pendulum model. This script measures
how much the baseline's closed-loop behaviour actually depends on that number,
by re-deriving the Riccati gain across a range of b_{r,h} and re-running the
standard nominal evaluation at each value.

Both coupled variants are swept: the PID-servo path (`baseline_coupled_lqr`)
and the direct-torque path (`baseline_coupled_lqr_torque`), which scales the
same constant by 1/(m*l_com).

Usage:
  .venv/bin/python scripts/sweep_b_roll_hip.py --episodes 20
  .venv/bin/python scripts/sweep_b_roll_hip.py --episodes 20 --variant servo
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

VARIANTS = {
    "servo": ("baseline_coupled_lqr", "configs/baseline_coupled_lqr.yaml",
              "coupled_lqr"),
    "torque": ("baseline_coupled_lqr_torque",
               "configs/baseline_coupled_lqr_torque.yaml",
               "baseline_coupled_lqr_torque"),
}
NOMINAL = -5.0

# Command ceiling on the roll channel: the hip-roll joint range for the servo
# path, the hip-roll actuator torque limit for the direct-torque path.
ROLL_CMD_LIMIT = {"servo": 0.7, "torque": 30.0}


def roll_gain(variant: str, b: float) -> float:
    """|K| from roll angle to the roll-channel command, for this b_{r,h}."""
    if variant == "servo":
        from wheeled_biped.controllers.coupled_lqr_3d import (
            _compute_coupled_lqr_gains as gains)
    else:
        from wheeled_biped.controllers.coupled_lqr_3d_torque import (
            _compute_coupled_lqr_gains_dt as gains)
    return float(abs(gains(b_roll_hip=b)[1, 2]))


def run_one(controller: str, cfg_path: Path, episodes: int,
            out_dir: Path, control_hz: float) -> dict:
    cmd = [
        sys.executable, str(ROOT / "scripts/eval_balance.py"),
        "--controller", controller,
        "--baseline-config", str(cfg_path),
        "--scenarios", "nominal",
        "--num-episodes", str(episodes),
        "--num-steps", "1000",
        "--control-hz", str(control_hz),
        "--seeds", "0", "--seeds", "42", "--seeds", "123",
        "--output-dir", str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if proc.returncode != 0:
        return {"error": proc.stderr[-800:]}
    results = out_dir / "eval_results.json"
    if not results.exists():
        return {"error": f"no results at {results}"}
    return json.loads(results.read_text())


KEEP = ("fall_rate", "survival_rate", "survival_time_mean_s",
        "survival_time_std_s", "roll_rms_deg", "pitch_rms_deg",
        "xy_drift_max_m", "torque_rms_nm", "wheel_speed_rms_rads")


def summarize(raw: dict) -> dict:
    """Pull the comparable scalars out of the nominal row of eval_results.json."""
    if "error" in raw:
        return raw
    rows = [r for r in raw.get("results", []) if r.get("scenario") == "nominal"]
    if not rows:
        return {"error": "no nominal row"}
    return {k: rows[0][k] for k in KEEP if k in rows[0]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--variant", choices=[*VARIANTS, "both"], default="both")
    ap.add_argument("--values", type=float, nargs="+",
                    default=[-1.0, -2.0, -3.0, -5.0, -8.0, -12.0, -20.0])
    ap.add_argument("--control-hz", type=float, default=100.0,
                    help="Control rate. 100 matches the reported baseline "
                         "table; 50 is the coupled model's design rate.")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "outputs" / "b_roll_hip_sweep" / "results.json")
    args = ap.parse_args()

    variants = list(VARIANTS) if args.variant == "both" else [args.variant]
    work = args.output.parent / "work"
    work.mkdir(parents=True, exist_ok=True)

    results: dict = {"nominal_value": NOMINAL, "episodes": args.episodes,
                     "control_hz": args.control_hz, "variants": {}}

    for var in variants:
        controller, cfg_rel, cfg_key = VARIANTS[var]
        base_cfg = yaml.safe_load((ROOT / cfg_rel).read_text())
        rows = []
        print(f"\n=== {var} ({controller}) ===", flush=True)
        for b in args.values:
            cfg = json.loads(json.dumps(base_cfg))       # deep copy
            cfg.setdefault(cfg_key, {})["b_roll_hip"] = b
            tag = f"{var}_b{b:+.1f}".replace(".", "p")
            cfg_path = work / f"{tag}.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg))

            t0 = time.time()
            metrics = summarize(run_one(controller, cfg_path, args.episodes,
                                        work / tag, args.control_hz))
            k_roll = roll_gain(var, b)
            # Roll excursion at which the roll command hits its ceiling. Compare
            # against the roll RMS actually observed: if the excursion exceeds
            # this, the channel is saturated and the gain cannot matter.
            sat_deg = np.degrees(ROLL_CMD_LIMIT[var] / k_roll)
            row = {"b_roll_hip": b, "k_roll": k_roll,
                   "roll_saturation_deg": sat_deg, **metrics}
            rows.append(row)
            mark = "  <- nominal" if b == NOMINAL else ""
            print(f"  b={b:+6.1f}  |K_roll|={k_roll:8.2f}  "
                  f"sat@{sat_deg:6.2f}deg  "
                  f"fall={metrics.get('fall_rate', 0) * 100:5.1f}%  "
                  f"surv={metrics.get('survival_time_mean_s', 0):6.3f}s  "
                  f"roll_rms={metrics.get('roll_rms_deg', 0):7.3f}deg  "
                  f"[{time.time() - t0:.0f}s]{mark}", flush=True)
        results["variants"][var] = rows

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
