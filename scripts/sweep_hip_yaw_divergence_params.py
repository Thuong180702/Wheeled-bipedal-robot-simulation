"""Real-simulation parameter sweep for the mode-based hip-yaw divergence
controller.

For each parameter combination this script invokes the simulator with
``--enable-mode-hip-yaw-divergence`` set to the candidate's gains, runs
D4_medium_push_low (low_0p330, 60N, 1000 steps), and writes telemetry to
``outputs/mode_based_hip_yaw_divergence_sweep/sweep_<kp>_<kd>_<max>_<soft>/``.

After the sweep completes, the existing
``wheeled_biped.validation.sweep_hip_yaw_divergence_params.run_sweep``
function will read each candidate's telemetry and report the parsed
``hip_yaw_abs_max`` for downstream gate evaluation.

Run:

    python scripts/sweep_hip_yaw_divergence_params.py
"""
from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups"
SWEEP_DIR = ROOT / "outputs" / "mode_based_hip_yaw_divergence_sweep"

# Fixed sagittal profile (low-band v2)
PROFILE = "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"
HEIGHT = "low_0p330"
STEPS = 1000
PUSH_MAG = 60
PUSH_DUR = 5
PUSH_INT = 150

# Parameter grid: kp, kd, max_torque, soft_gain
GRID = [
    {"kp": 1.0, "kd": 0.10, "max_torque": 1.0, "soft_gain": 0.20},
    {"kp": 2.0, "kd": 0.10, "max_torque": 1.5, "soft_gain": 0.25},
    {"kp": 5.0, "kd": 0.20, "max_torque": 2.0, "soft_gain": 0.25},
]


def _dir_for(p: dict) -> Path:
    return SWEEP_DIR / (
        f"sweep_{p['kp']:.2f}_{p['kd']:.2f}_"
        f"{p['max_torque']:.2f}_{p['soft_gain']:.2f}"
    )


def _run_one(p: dict) -> dict:
    out_dir = _dir_for(p)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup = SETUP_DIR / f"{HEIGHT}_setup.json"
    if not setup.exists():
        return {"status": "missing_setup", "params": p, "out_dir": str(out_dir)}

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", PROFILE,
        "--height-variant-setup", str(setup),
        "--steps", str(STEPS),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(STEPS),
        "--write-run-summary-sidecar",
        "--output-dir", str(out_dir),
        "--enable-mode-hip-yaw-divergence",
        "--mode-hip-yaw-div-kp", str(p["kp"]),
        "--mode-hip-yaw-div-kd", str(p["kd"]),
        "--mode-hip-yaw-div-max-torque", str(p["max_torque"]),
        "--mode-hip-yaw-div-soft-limit-rad", "0.30",
        "--mode-hip-yaw-div-soft-gain", str(p["soft_gain"]),
        "--push-enabled",
        "--push-magnitude-n", str(float(PUSH_MAG)),
        "--push-duration-steps", str(PUSH_DUR),
        "--push-interval-steps", str(PUSH_INT),
    ]
    print(f"\n[SWEEP] {p}", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "params": p, "out_dir": str(out_dir)}
    elapsed = time.time() - t0
    return {
        "status": "ok" if result.returncode == 0 else f"rc={result.returncode}",
        "params": p,
        "out_dir": str(out_dir),
        "elapsed_s": round(elapsed, 1),
    }


def main() -> int:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    log = SWEEP_DIR / "sweep_log.csv"
    with log.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["status", "kp", "kd", "max_torque", "soft_gain", "out_dir", "elapsed_s"])
        for p in GRID:
            r = _run_one(p)
            writer.writerow([
                r["status"], p["kp"], p["kd"], p["max_torque"], p["soft_gain"],
                r["out_dir"], r.get("elapsed_s", ""),
            ])
            print(f"  -> {r['status']} ({r.get('elapsed_s','?')}s)", flush=True)
    print(f"\nSweep log: {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())