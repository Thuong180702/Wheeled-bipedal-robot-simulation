#!/usr/bin/env python3
"""Collect structured JSON for all 4 LQR baseline rows (Table III in paper).

Runs each LQR variant with N episodes, saves per-episode metrics + aggregates.
Matches paper methodology: 10s episodes (1000 steps), clean sensors, 100Hz ctrl.

Usage:
  .venv/bin/mjpython scripts/collect_lqr_baselines.py --n-episodes 20
  .venv/bin/mjpython scripts/collect_lqr_baselines.py --n-episodes 60 --output-dir outputs/paper_statistics
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run_lqr_eval(controller_name: str, config_path: str, n_episodes: int,
                 seeds: list[int], out_dir: Path) -> dict:
    """Run eval_balance.py for one LQR variant, parse results."""
    import subprocess

    seed_args = []
    for s in seeds:
        seed_args.extend(["--seeds", str(s)])

    cmd = [
        sys.executable, str(ROOT / "scripts/eval_balance.py"),
        "--controller", controller_name,
        "--config", config_path,
        "--scenarios", "nominal",
        "--num-episodes", str(n_episodes),
        "--num-steps", "1000",
        *seed_args,
        "--output-dir", str(out_dir / controller_name),
    ]

    print(f"  Running: {' '.join(cmd[-8:])}...", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                            cwd=str(ROOT))
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s (rc={result.returncode})")

    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")
        return {"error": result.stderr[-1000:], "rc": result.returncode}

    # Parse eval_results.json
    results_path = out_dir / controller_name / "eval_results.json"
    if results_path.exists():
        return json.load(results_path.open())
    return {"error": "no results file", "path": str(results_path)}


def main():
    p = argparse.ArgumentParser(description="Collect LQR baseline data for paper Table III")
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--output-dir", type=str,
                   default=str(ROOT / "outputs/paper_statistics"))
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # LQR variants matching paper Table III
    variants = [
        ("baseline_lqr",         "configs/baseline_lqr.yaml",
         "LQR_TWIP",            60),
        ("baseline_lqr_aw",      "configs/baseline_lqr_anti_windup.yaml",
         "LQR_AW",              60),
        ("baseline_lqr_torque",  "configs/baseline_lqr_torque.yaml",
         "LQR_DirectTorque",    20),
        ("baseline_coupled_lqr", "configs/baseline_coupled_lqr.yaml",
         "LQR_Coupled6State",   20),
    ]

    print("=" * 72)
    print(f"LQR BASELINE COLLECTION — {len(variants)} variants, N={args.n_episodes} each")
    print(f"Output: {out_dir}")
    print("=" * 72)

    all_results = {}
    seeds = [0, 42, 123]  # paper-documented seeds

    for ctrl_name, config_path, label, paper_n in variants:
        print(f"\n{'─'*72}")
        print(f"[{label}] controller={ctrl_name}, config={config_path}")
        result = run_lqr_eval(ctrl_name, config_path, args.n_episodes, seeds, out_dir)
        all_results[label] = {
            "controller": ctrl_name,
            "config": config_path,
            "paper_n": paper_n,
            "actual_n": args.n_episodes,
            "result": result,
        }

    # Save aggregate
    out_path = out_dir / "lqr_baselines.json"
    all_results["_metadata"] = {
        "n_episodes": args.n_episodes,
        "seeds": seeds,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    json.dump(all_results, out_path.open("w"), indent=2, default=str)
    print(f"\nSaved: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
