"""D4/D5 mode-divergence authority limit sweep.

Runs the D baseline and F candidates across D4/D5 push cases.
Each F candidate varies only mode-div authority parameters
(kp, kd, max_torque) on top of the D_MODE_HIP_YAW_DIV_V1 base.

Output:
    outputs/mode_divergence_authority_limit_sweep/d4_d5_focused_sweep/
        sweep_metrics.csv   — aggregate across candidates
        <case>/<candidate>/telemetry_*.csv

Run:
    python scripts/run_d4_d5_mode_div_authority_sweep.py
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Canonical D profile
PROFILE_D = "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1"

OUT_BASE = ROOT / "outputs" / "mode_divergence_authority_limit_sweep" / "d4_d5_focused_sweep"

# D4/D5 push cases
PUSH_CASES = [
    ("D4_medium_push_low", "low_0p330", 1000, 60, 5, 150),
    ("D5_large_push_high", "high_0p480", 1000, 90, 5, 200),
]

# Candidate grid: (name_suffix, kp, kd, max_torque)
# Each F candidate keeps the same sagittal/base profile and only changes
# the three mode-div authority parameters.
CANDIDATES = [
    # D current-best baseline (reference)
    ("D_baseline", 5.0, 0.20, 2.0),
    # Conservative
    ("F1_kp5_mt3", 5.0, 0.20, 3.0),
    ("F2_kp5_mt5", 5.0, 0.20, 5.0),
    ("F3_kp5_kd3_mt5", 5.0, 0.30, 5.0),
    # Balanced
    ("F4_kp75_kd3_mt5", 7.5, 0.30, 5.0),
    ("F5_kp75_kd5_mt75", 7.5, 0.50, 7.5),
    # Aggressive diagnostic
    ("F6_kp10_kd5_mt75", 10.0, 0.50, 7.5),
    ("F7_kp10_kd75_mt10", 10.0, 0.75, 10.0),
    ("F8_kp15_kd1_mt10", 15.0, 1.00, 10.0),
]

# Mode-div parameters that stay the same across all runs
SOFT_LIMIT_RAD = 0.30
SOFT_GAIN = 0.25
REF_SOURCE = "target"


def _build_cmd(
    profile: str,
    height_label: str,
    steps: int,
    push_mag: int,
    push_dur: int,
    push_int: int,
    kp: float,
    kd: float,
    max_torque: float,
    out_dir: Path,
) -> list[str]:
    """Build the simulation CLI command for a candidate."""
    setup_path = (
        ROOT / "outputs" / "physical_target_height_setups" / f"{height_label}_setup.json"
    )
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", profile,
        "--height-variant-setup", str(setup_path),
        "--steps", str(steps),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(steps),
        "--write-run-summary-sidecar",
        "--output-dir", str(out_dir),
        # Mode-div flags
        "--enable-mode-hip-yaw-divergence",
        "--mode-hip-yaw-div-kp", str(float(kp)),
        "--mode-hip-yaw-div-kd", str(float(kd)),
        "--mode-hip-yaw-div-max-torque", str(float(max_torque)),
        "--mode-hip-yaw-div-soft-limit-rad", str(SOFT_LIMIT_RAD),
        "--mode-hip-yaw-div-soft-gain", str(SOFT_GAIN),
        "--mode-hip-yaw-div-ref-source", REF_SOURCE,
        # Push flags
        "--push-enabled",
        "--push-magnitude-n", str(float(push_mag)),
        "--push-duration-steps", str(push_dur),
        "--push-interval-steps", str(push_int),
    ]
    return cmd


def _find_telemetry_csv(out_dir: Path) -> Path | None:
    """Find the latest telemetry CSV in out_dir."""
    csvs = sorted(out_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0] if csvs else None


def _run_candidate(
    case_id: str,
    height_label: str,
    steps: int,
    push_mag: int,
    push_dur: int,
    push_int: int,
    candidate_name: str,
    kp: float,
    kd: float,
    max_torque: float,
) -> tuple[Path | None, int, float, float, float, float, float, float, float, int, int]:
    """Run one simulation and extract key metrics.

    Returns tuple:
        (telemetry_path, actual_rows, hy_abs_max, pitch_max_deg, sup_max,
         roll_rms_deg, yaw_err_max, mode_div_tau_max, tau_final_max,
         sat_rows, sign_ok_rows)
    """
    out_dir = OUT_BASE / case_id / candidate_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = _build_cmd(
        profile=PROFILE_D,
        height_label=height_label,
        steps=steps,
        push_mag=push_mag,
        push_dur=push_dur,
        push_int=push_int,
        kp=kp,
        kd=kd,
        max_torque=max_torque,
        out_dir=out_dir,
    )

    print(f"  [{candidate_name}] {case_id} ... ", end="", flush=True)
    t0 = time.time()
    log_path = out_dir / "sim.log"
    with open(log_path, "w") as log_f:
        result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, timeout=600)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"FAILED (rc={result.returncode}) in {elapsed:.0f}s")
        return (None, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    tele_path = _find_telemetry_csv(out_dir)
    if tele_path is None:
        print(f"NO TELEMETRY in {elapsed:.0f}s")
        return (None, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    # Parse telemetry
    with open(tele_path, newline="") as f:
        rows = list(csv.DictReader(f))

    n = len(rows)
    hy = max(float(r["hip_yaw_abs_max"]) for r in rows)
    pitch = max(abs(float(r["pitch_error"])) for r in rows) * 180 / 3.14159265
    sup = max(abs(float(r.get("support_position_error_scaled_m", 0)))) if "support_position_error_scaled_m" in rows[0] else 0.0
    roll_rms = (sum(float(r["roll_y"]) ** 2 for r in rows) / n) ** 0.5 * 180 / 3.14159265
    yaw_err = max(abs(float(r.get("hip_yaw_common_error_rad", 0)))) if "hip_yaw_common_error_rad" in rows[0] else 0.0
    mode_tau_max = max(abs(float(r["mode_hip_yaw_div_tau_left"])) for r in rows)
    tau_final_max = max(abs(float(r.get("l_hip_yaw_tau_shape_final", 0)))) if "l_hip_yaw_tau_shape_final" in rows[0] else 0.0
    sat_rows = sum(1 for r in rows if r.get("mode_hip_yaw_div_tau_left_sat", "False") == "True")
    # Check sign (mode-div torque opposes divergence_error)
    sign_ok = sum(
        1 for r in rows
        if abs(float(r["mode_hip_yaw_div_error"])) < 1e-9
        or float(r["mode_hip_yaw_div_error"]) * float(r["mode_hip_yaw_div_tau_left"]) <= 0
    )

    print(f"done ({elapsed:.0f}s) hy={hy:.4f} rows={n} sat={sat_rows}")
    return (tele_path, n, hy, pitch, sup, roll_rms, yaw_err, mode_tau_max, tau_final_max, sat_rows, sign_ok)


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    for case_id, height_label, steps, push_mag, push_dur, push_int in PUSH_CASES:
        print(f"\n=== {case_id} ===")
        for cand_name, kp, kd, max_torque in CANDIDATES:
            result = _run_candidate(
                case_id=case_id,
                height_label=height_label,
                steps=steps,
                push_mag=push_mag,
                push_dur=push_dur,
                push_int=push_int,
                candidate_name=cand_name,
                kp=kp,
                kd=kd,
                max_torque=max_torque,
            )
            tele_path, n_rows, hy, pitch, sup, roll_rms, yaw_err, mode_tau, tau_final, sat, sign_ok = result
            all_metrics.append({
                "case": case_id,
                "candidate": cand_name,
                "kp": kp,
                "kd": kd,
                "max_torque": max_torque,
                "rows": n_rows,
                "hip_yaw_abs_max": hy,
                "pitch_max_deg": pitch,
                "support_max_m": sup,
                "roll_rms_deg": roll_rms,
                "yaw_common_err_max_rad": yaw_err,
                "mode_div_tau_max": mode_tau,
                "tau_final_max": tau_final,
                "sat_rows": sat_rows,
                "sign_ok": sign_ok,
                "sign_ok_pct": round(100.0 * sign_ok / n_rows, 1) if n_rows > 0 else 0.0,
                "telemetry_path": str(tele_path) if tele_path else "",
            })
            print(f"    hy={hy:.4f}  pitch={pitch:.2f}  sup={sup:.4f}  mode_tau={mode_tau:.4f}  sat={sat}/{n_rows}  sign={round(100.0*sign_ok/n_rows,1) if n_rows>0 else 0}%")

    # Write aggregate metrics CSV
    metrics_csv = OUT_BASE / "sweep_metrics.csv"
    with open(metrics_csv, "w", newline="") as f:
        fieldnames = [
            "case", "candidate", "kp", "kd", "max_torque",
            "rows", "hip_yaw_abs_max", "pitch_max_deg", "support_max_m",
            "roll_rms_deg", "yaw_common_err_max_rad",
            "mode_div_tau_max", "tau_final_max", "sat_rows", "sign_ok", "sign_ok_pct",
            "telemetry_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)

    print(f"\nMetrics: {metrics_csv}")

    # Summary table
    print("\n" + "=" * 120)
    print(f"{'Case':<25} {'Candidate':<25} {'hy_max':<10} {'pitch':<8} {'sup':<8} {'mode_tau':<10} {'sat':<6} {'sign%':<8} {'rows':<6}")
    print("=" * 120)
    for m in all_metrics:
        print(f"{m['case']:<25} {m['candidate']:<25} {m['hip_yaw_abs_max']:<10.4f} {m['pitch_max_deg']:<8.2f} {m['support_max_m']:<8.4f} {m['mode_div_tau_max']:<10.4f} {m['sat_rows']:<6} {m['sign_ok_pct']:<8.1f} {m['rows']:<6}")


if __name__ == "__main__":
    main()
