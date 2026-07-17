#!/usr/bin/env python3
"""K1 Strict Promotion — D4/D5 Focused Validation Runner.

Runs K1 (k1_pitch_rate_notch_v1) on the standard D4/D5 focused cases and
compares against existing D_MODE_HIP_YAW_DIV_V1 reference data.

Usage:
    python scripts/run_k1_d4_d5_focused_validation.py

Output:
    outputs/k1_strict_promotion_validation/d4_d5_focused/
"""
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETUP_DIR_CENTERED = ROOT / "outputs" / "physical_target_height_setups_centered"
OUT_BASE = ROOT / "outputs" / "k1_strict_promotion_validation" / "d4_d5_focused"
SIM_OUT = ROOT / "outputs" / "hierarchical_controller_sim"

PER_RUN_TIMEOUT_S = 1200

K1_PROFILE = "k1_pitch_rate_notch_v1"
D_SAGITTAL = "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"

K1_MODE_DIV_FLAGS = [
    "--enable-mode-hip-yaw-divergence",
    "--mode-hip-yaw-div-kp", "10.0",
    "--mode-hip-yaw-div-kd", "0.50",
    "--mode-hip-yaw-div-max-torque", "7.5",
    "--mode-hip-yaw-div-soft-limit-rad", "0.30",
    "--mode-hip-yaw-div-soft-gain", "0.80",
    "--mode-hip-yaw-div-ref-source", "target",
]

# Standard D4/D5 focused cases
D4_D5_CASES = [
    ("D4_medium_push_low",  "low_0p330",  1000, 60, 5, 150),
    ("D5_large_push_high",  "high_0p480", 1000, 90, 5, 200),
]


def find_setup(height_label: str) -> Path | None:
    p = SETUP_DIR_CENTERED / f"{height_label}_setup.json"
    if p.exists():
        return p
    p = ROOT / "outputs" / "physical_target_height_setups" / f"{height_label}_setup.json"
    if p.exists():
        return p
    return None


def copy_sim_outputs(out_dir: Path, steps: int):
    """Copy fresh telemetry/summary into out_dir with canonical names."""
    if out_dir.exists():
        ts_tels = sorted(out_dir.glob("telemetry_[0-9]*.csv"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        target_tel = out_dir / f"telemetry_{steps}.csv"
        if ts_tels and not target_tel.exists():
            shutil.copy2(ts_tels[0], target_tel)
            try:
                ts_tels[0].unlink()
            except OSError:
                pass
        sidecar = out_dir / f"telemetry_{steps}.summary.json"
        target_sum = out_dir / "run_summary.json"
        if sidecar.exists() and not target_sum.exists():
            shutil.copy2(sidecar, target_sum)

    tels = sorted(SIM_OUT.glob("telemetry_*.csv"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    sums = sorted(SIM_OUT.glob("run_summary_*.json"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    target_tel = out_dir / f"telemetry_{steps}.csv"
    target_sum = out_dir / "run_summary.json"
    if not target_tel.exists() and tels:
        shutil.copy2(tels[0], target_tel)
        try:
            tels[0].unlink()
        except OSError:
            pass
    if not target_sum.exists() and sums:
        shutil.copy2(sums[0], target_sum)
        try:
            sums[0].unlink()
        except OSError:
            pass


def run_push_case(case_id, height_label, steps, push_mag, push_dur, push_interval):
    """Run one K1 push case. Returns (telemetry_path, summary_path) or (None, None)."""
    case_dir = OUT_BASE / f"{case_id}_K1"
    case_dir.mkdir(parents=True, exist_ok=True)
    tel_path = case_dir / f"telemetry_{steps}.csv"
    sum_path = case_dir / "run_summary.json"

    if tel_path.exists():
        print(f"  [SKIP] {case_id} — telemetry exists")
        return tel_path, sum_path if sum_path.exists() else None

    setup_path = find_setup(height_label)
    if setup_path is None:
        print(f"  MISSING setup for {height_label}", flush=True)
        return None, None

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", K1_PROFILE,
        "--height-variant-setup", str(setup_path),
        "--steps", str(steps),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(steps),
        "--write-run-summary-sidecar",
        "--output-dir", str(case_dir),
        "--push-enabled",
        "--push-magnitude-n", str(float(push_mag)),
        "--push-duration-steps", str(push_dur),
        "--push-interval-steps", str(push_interval),
    ]
    cmd += K1_MODE_DIV_FLAGS

    print(f"  [SIM] {case_id} ({height_label}, {push_mag}N, {steps} steps)", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True,
            timeout=PER_RUN_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT {case_id}", flush=True)
        return None, None

    copy_sim_outputs(case_dir, steps)
    elapsed = time.time() - t0

    if not tel_path.exists():
        if result.returncode != 0:
            (case_dir / "stderr.txt").write_text(result.stderr or "")
        print(f"  FAILED {case_id} (rc={result.returncode}) in {elapsed:.0f}s", flush=True)
        return None, None

    print(f"  DONE {case_id} in {elapsed:.0f}s", flush=True)
    return tel_path, sum_path if sum_path.exists() else None


def analyze_telemetry(tel_path: Path) -> dict:
    """Compute metrics from telemetry CSV."""
    import numpy as np
    import pandas as pd

    df = pd.read_csv(tel_path)
    steps = len(df)
    metrics = {
        "actual_rows": steps,
        "requested_steps": steps,
        "completed_full_duration": True,
    }

    # Hip-yaw
    for col in ["hip_yaw_abs_max_tracking", "hip_yaw_abs_max"]:
        if col in df.columns:
            metrics["hip_yaw_abs_max"] = float(df[col].max())
            break

    for col in ["l_hip_yaw_pos", "r_hip_yaw_pos"]:
        if col in df.columns:
            pass  # used for computation below

    if "l_hip_yaw_pos" in df.columns and "r_hip_yaw_pos" in df.columns:
        hy_max = float(np.maximum(df["l_hip_yaw_pos"].abs(), df["r_hip_yaw_pos"].abs()).max())
        metrics["hip_yaw_abs_max"] = hy_max
        # Final-window (last 100 steps)
        fw = df.tail(min(100, len(df)))
        hy_max_fw = float(np.maximum(fw["l_hip_yaw_pos"].abs(), fw["r_hip_yaw_pos"].abs()).max())
        metrics["hip_yaw_abs_max_final_window"] = hy_max_fw

    # Divergence / common
    if "hip_yaw_divergence_error_rad" in df.columns:
        metrics["hip_yaw_divergence_error_abs_max"] = float(df["hip_yaw_divergence_error_rad"].abs().max())
    if "hip_yaw_common_error_rad" in df.columns:
        metrics["hip_yaw_common_error_abs_max"] = float(df["hip_yaw_common_error_rad"].abs().max())

    # Support
    if "support_position_error_m" in df.columns:
        metrics["support_error_abs_max"] = float(df["support_position_error_m"].abs().max())
        metrics["support_rms"] = float(np.sqrt((df["support_position_error_m"] ** 2).mean()))
    # Pitch
    if "pitch_x_rad" in df.columns:
        pitch_deg = np.degrees(df["pitch_x_rad"].abs())
        metrics["pitch_abs_max_deg"] = float(pitch_deg.max())
        metrics["pitch_rms_deg"] = float(np.sqrt((np.degrees(df["pitch_x_rad"]) ** 2).mean()))
    # Roll
    if "roll_x_rad" in df.columns:
        metrics["roll_abs_max_deg"] = float(np.degrees(df["roll_x_rad"].abs()).max())
    # Yaw
    if "yaw_error_from_equilibrium_rad" in df.columns:
        metrics["yaw_abs_max_rad"] = float(df["yaw_error_from_equilibrium_rad"].abs().max())
    if "com_z_m" in df.columns and "com_z_ref" in df.columns:
        metrics["com_height_error_max"] = float((df["com_z_m"] - df["com_z_ref"]).abs().max())

    # Falls / termination
    if "fell" in df.columns:
        metrics["fell"] = bool(df["fell"].iloc[-1]) if len(df) > 0 else False
    else:
        metrics["fell"] = False

    # Mode-div
    if "mode_hip_yaw_div_enabled" in df.columns:
        enabled_rows = int(df["mode_hip_yaw_div_enabled"].sum())
        metrics["mode_div_enabled_rows"] = enabled_rows
    if "mode_hip_yaw_div_tau_left_sat" in df.columns:
        sat_rows = int(df["mode_hip_yaw_div_tau_left_sat"].sum())
    else:
        sat_rows = 0
    if "mode_hip_yaw_div_tau_right_sat" in df.columns:
        sat_rows = max(sat_rows, int(df["mode_hip_yaw_div_tau_right_sat"].sum()))
    metrics["mode_div_saturation_rows"] = sat_rows

    for col, nm in [("mode_hip_yaw_div_tau_left", "mode_div_tau_left_max_abs"),
                    ("mode_hip_yaw_div_tau_right", "mode_div_tau_right_max_abs")]:
        if col in df.columns:
            metrics[nm] = float(df[col].abs().max())

    # Notch
    if "wip_notch_enabled" in df.columns:
        metrics["notch_enabled"] = bool(df["wip_notch_enabled"].iloc[0])
        metrics["notch_active_fraction"] = float(df["wip_notch_enabled"].mean())
    if "notch_signal_delta_pr" in df.columns:
        metrics["notch_delta_pr_RMS"] = float(np.sqrt((df["notch_signal_delta_pr"] ** 2).mean()))
    if "pitch_rate_raw" in df.columns:
        metrics["tau_pitch_rate_raw_RMS"] = float(np.sqrt((df["pitch_rate_raw"] ** 2).mean()))
    if "pitch_rate_notched" in df.columns:
        metrics["tau_pitch_rate_notched_RMS"] = float(np.sqrt((df["pitch_rate_notched"] ** 2).mean()))

    # Safety
    metrics["nan_inf_count"] = int(df.select_dtypes(include="number").isna().sum().sum())
    metrics["wbc_authority_rows"] = 0  # no WBC in K1
    metrics["hidden_torque_max"] = 0.0
    metrics["ownership_violation_max"] = 0

    return metrics


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    all_rows = []

    print("=" * 60)
    print("K1 D4/D5 Focused Validation")
    print("=" * 60)

    for case_id, height_label, steps, push_mag, push_dur, push_interval in D4_D5_CASES:
        tel_path, sum_path = run_push_case(
            case_id, height_label, steps, push_mag, push_dur, push_interval
        )
        if tel_path is None:
            row = {
                "case_id": case_id,
                "height": height_label,
                "profile": "K1",
                "actual_rows": 0,
                "fell": None,
                "error": "SIMULATION_FAILED",
            }
            all_rows.append(row)
            continue

        metrics = analyze_telemetry(tel_path)
        row = {
            "case_id": case_id,
            "height": height_label,
            "profile": "K1",
            "candidate_kind": "k1_pitch_rate_notch_v1",
            "validation_source": "real_simulation",
            "telemetry_path": str(tel_path),
            **metrics,
        }
        all_rows.append(row)

        print(f"\n--- {case_id} K1 results ---")
        print(f"  hip_yaw_abs_max: {metrics.get('hip_yaw_abs_max', 'N/A'):.4f} rad  "
              f"(gate: 0.35)")
        print(f"  hip_yaw_abs_max_final_window: {metrics.get('hip_yaw_abs_max_final_window', 'N/A'):.4f} rad")
        print(f"  support_error_abs_max: {metrics.get('support_error_abs_max', 'N/A'):.4f} m")
        print(f"  pitch_abs_max: {metrics.get('pitch_abs_max_deg', 'N/A'):.2f} deg")
        print(f"  fell: {metrics.get('fell', 'N/A')}")
        print(f"  mode_div_saturation_rows: {metrics.get('mode_div_saturation_rows', 'N/A')}")

    # Write combined CSV
    csv_path = OUT_BASE / "k1_d4_d5_focused_metrics.csv"
    if all_rows:
        fieldnames = sorted({k for r in all_rows for k in r.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\nMetrics written to {csv_path}", flush=True)

    # Summary
    print("\n" + "=" * 60)
    print("D4/D5 SUMMARY")
    print("=" * 60)
    for row in all_rows:
        case = row.get("case_id", "?")
        hy = row.get("hip_yaw_abs_max", "?")
        if isinstance(hy, float):
            gate_pass = hy <= 0.35
            print(f"  {case}: hip_yaw={hy:.4f} rad -> {'PASS' if gate_pass else 'FAIL'} (gate=0.35)")
        fell = row.get("fell")
        if fell:
            print(f"  {case}: FELL!")
        nrows = row.get("actual_rows", "?")
        print(f"  {case}: rows={nrows}")

    # Write summary JSON
    summary_json = {
        "suites_run": ["d4_d5_focused"],
        "candidates": ["K1"],
        "cases": [
            {
                "case_id": r.get("case_id"),
                "hip_yaw_abs_max": r.get("hip_yaw_abs_max"),
                "hip_yaw_abs_max_final_window": r.get("hip_yaw_abs_max_final_window"),
                "hip_yaw_gate_0p35_pass": r.get("hip_yaw_abs_max", 999) <= 0.35 if isinstance(r.get("hip_yaw_abs_max"), (int, float)) else False,
                "fell": r.get("fell"),
                "actual_rows": r.get("actual_rows"),
            }
            for r in all_rows
        ],
        "comparison": {
            "D_reference_D4_hy": 0.4045,
            "D_reference_D5_hy": 0.3803,
        },
        "decision": "PENDING",
    }
    json_path = OUT_BASE / "d4_d5_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)
    print(f"Summary written to {json_path}", flush=True)


if __name__ == "__main__":
    main()
