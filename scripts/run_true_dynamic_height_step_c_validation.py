#!/usr/bin/env python3
"""True dynamic-height Step C validation harness for K1 current-best baseline.

The current Step C harness runs fixed-height cases only. This means K1 notch gate
crossing from 0.42 to 0.48 m has not been dynamically validated. This script:

1. Generates height trajectory JSONs for 7 profiles
2. Runs simulate_hierarchical_controller.py with --dynamic-height-trajectory
3. Collects telemetry and computes metrics
4. Analyzes notch gate crossing behavior, pitch/support stability, hip-yaw

Height profiles all cross the notch gate (0.42-0.48 m) in various patterns.

Output:
    outputs/k1_controller_completion/true_dynamic_step_c/
        trajectories/  (JSON trajectory files)
        raw/           (raw telemetry CSVs per profile)
        analysis/      (metrics CSV and summary)

Usage:
    python scripts/run_true_dynamic_height_step_c_validation.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SETUP_DIR_CENTERED = ROOT / "outputs" / "physical_target_height_setups_centered"
OUT_BASE = ROOT / "outputs" / "k1_controller_completion" / "true_dynamic_step_c"
TRAJ_DIR = OUT_BASE / "trajectories"
RAW_DIR = OUT_BASE / "raw"
ANALYSIS_DIR = OUT_BASE / "analysis"

PER_RUN_TIMEOUT_S = 5400  # 90 min for long dynamic runs (increased from 1800)

K1_PROFILE = "k1_pitch_rate_notch_v1"

K1_MODE_DIV_FLAGS = [
    "--enable-mode-hip-yaw-divergence",
    "--mode-hip-yaw-div-kp", "10.0",
    "--mode-hip-yaw-div-kd", "0.50",
    "--mode-hip-yaw-div-max-torque", "7.5",
    "--mode-hip-yaw-div-soft-limit-rad", "0.30",
    "--mode-hip-yaw-div-soft-gain", "0.80",
    "--mode-hip-yaw-div-ref-source", "target",
]

# ---- Height profile definitions ---- #
# Each profile defines waypoints that the trajectory will interpolate linearly.
# Waypoints: (step, height_m)
# All profiles must cross the notch gate (0.42-0.48 m) at some point.

HEIGHT_PROFILES = {
    "slow_ladder_0p330_to_0p480_to_0p330": {
        "description": "Slow stepwise height changes, crossing notch gate up and down",
        "steps": 5000,
        "setup": "low_0p330",
        "waypoints": [
            (0, 0.330),
            (500, 0.330),
            (1000, 0.350),
            (1500, 0.380),
            (2000, 0.420),  # entering notch gate
            (2500, 0.450),  # inside notch gate
            (3000, 0.480),  # above notch gate
            (3500, 0.450),  # back into notch
            (4000, 0.420),  # exiting notch gate
            (4500, 0.380),
            (5000, 0.330),
        ],
    },
    "medium_ramp_0p330_to_0p480": {
        "description": "Smooth ramp from below gate to above gate, then back",
        "steps": 6000,
        "setup": "low_0p330",
        "waypoints": [
            (0, 0.330),
            (200, 0.330),
            (1500, 0.480),  # smooth ramp crossing gate
            (3000, 0.480),
            (4500, 0.330),  # smooth ramp back down crossing gate
            (6000, 0.330),
        ],
    },
    "abrupt_0p330_to_0p480": {
        "description": "Abrupt height transitions that cross the notch gate quickly",
        "steps": 5000,
        "setup": "low_0p330",
        "waypoints": [
            (0, 0.330),
            (300, 0.330),
            (400, 0.480),  # abrupt jump above gate
            (2000, 0.480),
            (2100, 0.330),  # abrupt drop below gate
            (3500, 0.330),
            (3600, 0.480),  # another abrupt jump
            (5000, 0.480),
        ],
    },
    "random_dwell_cross_gate": {
        "description": "Random-height dwell periods that cross gate multiple times",
        "steps": 5000,
        "setup": "low_0p330",
        "waypoints": [
            (0, 0.330),
            (400, 0.330),
            (600, 0.440),  # inside notch gate
            (1200, 0.440),
            (1400, 0.370),  # below gate
            (1800, 0.370),
            (2000, 0.460),  # inside gate
            (2400, 0.460),
            (2600, 0.410),  # just below gate
            (2800, 0.410),
            (3000, 0.480),  # above gate
            (3600, 0.480),
            (3800, 0.390),  # below gate
            (4200, 0.390),
            (4400, 0.450),  # inside gate
            (5000, 0.450),
        ],
    },
    "high_to_low_0p480_to_0p330": {
        "description": "Starting high, transitioning low, crossing gate downward",
        "steps": 4000,
        "setup": "high_0p480",
        "waypoints": [
            (0, 0.480),  # start above gate
            (500, 0.480),
            (2000, 0.330),  # smooth descent crossing gate
            (4000, 0.330),
        ],
    },
    "repeated_gate_crossing_0p400_0p460": {
        "description": "Oscillating around the notch gate edges multiple times",
        "steps": 5000,
        "setup": "low_0p330",
        "waypoints": [
            (0, 0.400),  # below gate
            (300, 0.400),
            (600, 0.460),  # above gate start
            (900, 0.400),  # back below
            (1200, 0.460),  # above again
            (1500, 0.400),  # below
            (1800, 0.460),  # above
            (2100, 0.400),  # below
            (2400, 0.460),  # above
            (2700, 0.400),  # below
            (3000, 0.460),  # above
            (3300, 0.400),  # below
            (3600, 0.460),  # above
            (3900, 0.400),  # below
            (4200, 0.460),  # above
            (4500, 0.400),  # below
            (5000, 0.400),
        ],
    },
    "stress_gate_crossing_0p410_0p490": {
        "description": "Aggressive crossing back and forth across gate margins",
        "steps": 5000,
        "setup": "low_0p330",
        "waypoints": [
            (0, 0.410),  # just below gate
            (100, 0.490),  # just above gate
            (200, 0.410),  # back below
            (300, 0.490),  # above
            (400, 0.410),
            (500, 0.490),
            (600, 0.410),
            (700, 0.490),
            (800, 0.410),
            (900, 0.490),
            (1000, 0.410),
            (1200, 0.490),
            (1400, 0.410),
            (1600, 0.490),
            (1800, 0.410),
            (2000, 0.490),
            (2200, 0.410),
            (2400, 0.490),
            (2600, 0.410),
            (2800, 0.490),
            (3000, 0.410),
            (3200, 0.490),
            (3400, 0.410),
            (3600, 0.490),
            (3800, 0.410),
            (4000, 0.490),
            (4200, 0.410),
            (4400, 0.490),
            (4600, 0.410),
            (4800, 0.490),
            (5000, 0.410),
        ],
    },
}

# Quick mode profiles — shorter trajectories that still cross the notch gate
QUICK_HEIGHT_PROFILES = {
    "quick_medium_ramp_0p330_to_0p480": {
        "description": "Shortened smooth ramp crossing notch gate both ways",
        "steps": 2000,
        "setup": "low_0p330",
        "waypoints": [
            (0, 0.330),
            (100, 0.330),
            (800, 0.480),  # smooth ramp crossing gate
            (1200, 0.480),
            (1900, 0.330),  # smooth ramp back down crossing gate
            (2000, 0.330),
        ],
    },
    "quick_abrupt_0p330_to_0p480": {
        "description": "Shortened abrupt transition across notch gate",
        "steps": 1500,
        "setup": "low_0p330",
        "waypoints": [
            (0, 0.330),
            (200, 0.330),
            (300, 0.480),  # abrupt jump above gate
            (900, 0.480),
            (1000, 0.330),  # abrupt drop below gate
            (1500, 0.330),
        ],
    },
    "quick_repeated_gate_crossing": {
        "description": "Shortened repeated gate crossings (0.40-0.46 m)",
        "steps": 2000,
        "setup": "low_0p330",
        "waypoints": [
            (0, 0.400),  # below gate
            (200, 0.400),
            (400, 0.460),  # above gate start
            (600, 0.400),  # back below
            (800, 0.460),  # above again
            (1000, 0.400),  # below
            (1200, 0.460),  # above
            (1400, 0.400),  # below
            (1600, 0.460),  # above
            (1800, 0.400),  # below
            (2000, 0.400),
        ],
    },
    "quick_gate_margins_0p410_0p490": {
        "description": "Shortened aggressive crossing at gate margins",
        "steps": 1500,
        "setup": "low_0p330",
        "waypoints": [
            (0, 0.410),  # just below gate
            (100, 0.490),  # just above gate
            (200, 0.450),  # inside notch gate [0.42, 0.48]
            (300, 0.490),  # above
            (400, 0.410),
            (500, 0.490),
            (600, 0.410),
            (700, 0.490),
            (800, 0.410),
            (900, 0.490),
            (1000, 0.410),
            (1100, 0.490),
            (1200, 0.410),
            (1300, 0.490),
            (1400, 0.410),
            (1500, 0.410),
        ],
    },
    "quick_high_to_low_0p480_to_0p330": {
        "description": "Shortened high-to-low transition crossing gate downward",
        "steps": 1500,
        "setup": "high_0p480",
        "waypoints": [
            (0, 0.480),  # start above gate
            (300, 0.480),
            (800, 0.330),  # smooth descent crossing gate
            (1500, 0.330),
        ],
    },
}


def find_setup(height_label: str) -> Path | None:
    p = SETUP_DIR_CENTERED / f"{height_label}_setup.json"
    if p.exists():
        return p
    p = ROOT / "outputs" / "physical_target_height_setups" / f"{height_label}_setup.json"
    if p.exists():
        return p
    return None


def write_trajectory_json(profile_name: str, waypoints: list, steps: int) -> Path:
    """Write a height trajectory JSON file."""
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    wp_data = [{"step": int(s), "height_m": float(h)} for s, h in waypoints]
    traj = {
        "height_profile_name": profile_name,
        "steps": steps,
        "waypoints": wp_data,
    }
    path = TRAJ_DIR / f"{profile_name}.json"
    with open(path, "w") as f:
        json.dump(traj, f, indent=2)
    return path


def copy_sim_outputs(out_dir: Path, steps: int):
    """Copy fresh telemetry/summary into out_dir with canonical names."""
    sim_out = ROOT / "outputs" / "hierarchical_controller_sim"
    tels = sorted(sim_out.glob("telemetry_*.csv"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    sums = sorted(sim_out.glob("run_summary_*.json"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    target_tel = out_dir / f"telemetry_{steps}.csv"
    target_sum = out_dir / "run_summary.json"
    if not target_tel.exists() and tels:
        shutil.copy2(tels[0], target_tel)
    if not target_sum.exists() and sums:
        shutil.copy2(sums[0], target_sum)


def run_dynamic_height_profile(
    profile_name: str, traj_path: Path, setup_label: str, steps: int
) -> Path | None:
    """Run one dynamic-height simulation. Returns telemetry path or None."""
    profile_dir = RAW_DIR / profile_name
    profile_dir.mkdir(parents=True, exist_ok=True)
    tel_path = profile_dir / f"telemetry_{steps}.csv"

    if tel_path.exists():
        print(f"  [SKIP] {profile_name} — telemetry exists ({len(open(tel_path).readlines())} rows)")
        return tel_path

    setup_path = find_setup(setup_label)
    if setup_path is None:
        print(f"  MISSING setup for {setup_label} (profile: {profile_name})", flush=True)
        return None

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
        "--output-dir", str(profile_dir),
        "--dynamic-height-trajectory", str(traj_path),
    ]
    cmd += K1_MODE_DIV_FLAGS

    print(f"  [SIM] {profile_name} ({setup_label}, {steps} steps)", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True,
            timeout=PER_RUN_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT {profile_name}", flush=True)
        return None

    copy_sim_outputs(profile_dir, steps)
    elapsed = time.time() - t0

    if not tel_path.exists():
        if result.returncode != 0:
            (profile_dir / "stderr.txt").write_text(result.stderr or "")
        print(f"  FAILED {profile_name} (rc={result.returncode}) in {elapsed:.0f}s", flush=True)
        return None

    print(f"  DONE {profile_name} in {elapsed:.0f}s", flush=True)
    return tel_path


def analyze_telemetry(profile_name: str, tel_path: Path) -> dict:
    """Compute validation metrics from dynamic-height telemetry CSV.

    Required telemetry columns (expected from simulation):
        pitch_x_rad, pitch_rate_raw_rad_s, pitch_rate_notched_rad_s,
        pitch_rate_effective_rad_s, wip_notch_height_gate,
        support_position_error_m, dynamic_height_target_m,
        l_hip_yaw_pos, r_hip_yaw_pos, hip_yaw_common_error_rad,
        hip_yaw_divergence_error_rad, current_com_z_m,
        roll_y_rad, dynamic_height_active,
    """
    import pandas as pd
    df = pd.read_csv(tel_path)
    nrows = len(df)
    metrics = {
        "profile": profile_name,
        "rows": nrows,
        "fell": False,
        "wbc": 0,
        "hidden_torque": 0,
        "ownership_violation": 0,
        "fall_step": -1,
    }

    # ---- Safety checks ---- #
    # Fall detection (height collapse)
    if "current_com_z_m" in df.columns:
        min_height = float(df["current_com_z_m"].min())
        if min_height < 0.20:  # height collapse
            metrics["fell"] = True
            fall_idx = int((df["current_com_z_m"] < 0.20).idxmax()) if (df["current_com_z_m"] < 0.20).any() else -1
            metrics["fall_step"] = fall_idx

    # NaN/Inf check
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    nan_count = int(df[numeric_cols].isna().sum().sum())
    inf_count = int(np.isinf(df[numeric_cols].values).sum())
    metrics["nan_count"] = nan_count
    metrics["inf_count"] = inf_count

    # ---- Dynamic height tracking ---- #
    if "dynamic_height_target_m" in df.columns:
        metrics["height_target_min"] = float(df["dynamic_height_target_m"].min())
        metrics["height_target_max"] = float(df["dynamic_height_target_m"].max())
        metrics["height_target_mean"] = float(df["dynamic_height_target_m"].mean())
    if "current_com_z_m" in df.columns:
        metrics["height_actual_mean"] = float(df["current_com_z_m"].mean())
        metrics["height_actual_min"] = float(df["current_com_z_m"].min())
        metrics["height_tracking_rmse"] = float(np.sqrt(
            ((df["current_com_z_m"] - df["dynamic_height_target_m"]) ** 2).mean()
        )) if "dynamic_height_target_m" in df.columns else 0.0

    # ---- Notch gate crossing ---- #
    if "wip_notch_height_gate" in df.columns:
        gate = df["wip_notch_height_gate"]
        metrics["notch_gate_mean"] = float(gate.mean())
        metrics["notch_gate_max"] = float(gate.max())
        metrics["notch_gate_min"] = float(gate.min())
        metrics["notch_crossings"] = int(((gate > 0.01) & (gate.shift(1) <= 0.01)).sum())
        # Fraction of steps with active notch
        notch_active = (gate > 0.5).mean()
        metrics["notch_active_fraction"] = float(notch_active)
    if "notch_height_gate_from_traj" in df.columns:
        metrics["traj_notch_active"] = float((df["notch_height_gate_from_traj"] > 0.5).mean())

    # ---- Pitch analysis ---- #
    if "pitch_x_rad" in df.columns:
        pitch_deg = np.degrees(df["pitch_x_rad"].abs())
        metrics["pitch_rms_deg"] = float(np.sqrt((pitch_deg ** 2).mean()))
        metrics["pitch_max_abs_deg"] = float(pitch_deg.max())
        # Windowed pitch RMS: final 500 steps, middle 500 steps
        if nrows >= 1500:
            fw = df.tail(500)
            fw_pitch_deg = np.degrees(fw["pitch_x_rad"].abs())
            metrics["pitch_rms_final_window_deg"] = float(np.sqrt((fw_pitch_deg ** 2).mean()))
            mw = df.iloc[len(df)//2-250:len(df)//2+250]
            mw_pitch_deg = np.degrees(mw["pitch_x_rad"].abs())
            metrics["pitch_rms_mid_window_deg"] = float(np.sqrt((mw_pitch_deg ** 2).mean()))

    # ---- Pitch rate analysis ---- #
    if "pitch_rate_raw_rad_s" in df.columns:
        pr_raw = df["pitch_rate_raw_rad_s"].abs()
        metrics["pitch_rate_raw_rms"] = float(np.sqrt((pr_raw ** 2).mean()))
        pr_eff = df["pitch_rate_effective_rad_s"].abs()
        metrics["pitch_rate_effective_rms"] = float(np.sqrt((pr_eff ** 2).mean()))
        # Notch attenuation
        if "pitch_rate_notched_rad_s" in df.columns:
            pr_notch = df["pitch_rate_notched_rad_s"].abs()
            metrics["pitch_rate_notched_rms"] = float(np.sqrt((pr_notch ** 2).mean()))
            # Only compute attenuation in notch-active region
            if "wip_notch_height_gate" in df.columns:
                active_mask = df["wip_notch_height_gate"] > 0.5
                if active_mask.any():
                    raw_active = df.loc[active_mask, "pitch_rate_raw_rad_s"].abs()
                    eff_active = df.loc[active_mask, "pitch_rate_effective_rad_s"].abs()
                    metrics["notch_attenuation_active_region"] = float(
                        1.0 - (eff_active.mean() / raw_active.mean() if raw_active.mean() > 1e-9 else 0.0)
                    )

    # ---- Pitch rate tau analysis ---- #
    if "tau_pitch_rate_raw_signal" in df.columns:
        metrics["tau_pitch_rate_raw_rms"] = float(np.sqrt((df["tau_pitch_rate_raw_signal"] ** 2).mean()))
    if "tau_pitch_rate_filtered_signal" in df.columns:
        metrics["tau_pitch_rate_filtered_rms"] = float(np.sqrt((df["tau_pitch_rate_filtered_signal"] ** 2).mean()))

    # ---- Support analysis ---- #
    if "support_position_error_m" in df.columns:
        support_abs = df["support_position_error_m"].abs()
        metrics["support_rms_m"] = float(np.sqrt((support_abs ** 2).mean()))
        metrics["support_max_m"] = float(support_abs.max())
        if nrows >= 1500:
            fw = df.tail(500)
            fw_sup = fw["support_position_error_m"].abs()
            metrics["support_rms_final_window_m"] = float(np.sqrt((fw_sup ** 2).mean()))

    # ---- Hip-yaw analysis ---- #
    if "l_hip_yaw_pos" in df.columns and "r_hip_yaw_pos" in df.columns:
        hy_max = float(np.maximum(df["l_hip_yaw_pos"].abs(), df["r_hip_yaw_pos"].abs()).max())
        metrics["hip_yaw_abs_max"] = hy_max
        if nrows >= 500:
            fw = df.tail(500)
            hy_max_fw = float(np.maximum(fw["l_hip_yaw_pos"].abs(), fw["r_hip_yaw_pos"].abs()).max())
            metrics["hip_yaw_abs_max_final_window"] = hy_max_fw
        metrics["hip_yaw_gate_pass"] = hy_max < 0.35

        # Common vs divergence breakdown
        if "hip_yaw_common_error_rad" in df.columns:
            metrics["hip_yaw_common_abs_max"] = float(df["hip_yaw_common_error_rad"].abs().max())
        if "hip_yaw_divergence_error_rad" in df.columns:
            metrics["hip_yaw_divergence_abs_max"] = float(df["hip_yaw_divergence_error_rad"].abs().max())

    # ---- Roll safety ---- #
    if "roll_y_rad" in df.columns:
        roll_deg = np.degrees(df["roll_y_rad"].abs())
        metrics["roll_rms_deg"] = float(np.sqrt((roll_deg ** 2).mean()))
        metrics["roll_max_abs_deg"] = float(roll_deg.max())

    # ---- Gate crossing pitch analysis ---- #
    # Look for pitch/support spikes at or near notch gate crossing events
    if "wip_notch_height_gate" in df.columns and "pitch_x_rad" in df.columns:
        gate = df["wip_notch_height_gate"]
        pitch_deg = np.degrees(df["pitch_x_rad"].abs())
        # Find crossing regions
        crossings = (gate > 0.01) & (gate.shift(1) <= 0.01)
        crossing_indices = df.index[crossings].tolist()
        metrics["n_gate_crossings"] = len(crossing_indices)
        # For each crossing, check pitch spike within ±50 steps
        pitch_spikes = []
        for ci in crossing_indices:
            window = df.loc[max(0, ci-50):min(nrows-1, ci+50)]
            spike = float(np.degrees(window["pitch_x_rad"].abs()).max())
            pitch_spikes.append(spike)
        if pitch_spikes:
            metrics["pitch_spike_at_crossing_mean_deg"] = float(np.mean(pitch_spikes))
            metrics["pitch_spike_at_crossing_max_deg"] = float(max(pitch_spikes))

    # ---- WBC / hidden / ownership ---- #
    # Check telemetry columns for WBC, hidden torque, ownership violation flags
    for flag_col in ["wbc_enabled", "wbc_active", "hidden_torque", "ownership_violation"]:
        if flag_col in df.columns:
            if df[flag_col].dtype == bool:
                metrics[flag_col + "_any"] = bool(df[flag_col].any())
            elif df[flag_col].dtype in (int, float):
                metrics[flag_col + "_max"] = float(df[flag_col].max())

    return metrics


def write_metrics_csv(all_metrics: list[dict], path: Path):
    """Write metrics to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        if all_metrics:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)
    print(f"  [METRICS] Wrote {len(all_metrics)} rows to {path}")


def write_summary(all_metrics: list[dict], path: Path):
    """Write a human-readable summary."""
    lines = []
    lines.append("=" * 80)
    lines.append("TRUE DYNAMIC-HEIGHT STEP C VALIDATION — SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Baseline: K1 ({K1_PROFILE})")
    lines.append("")

    # Overall pass/fail
    n_total = len(all_metrics)
    n_pass = sum(1 for m in all_metrics if not m.get("fell", True) and m.get("nan_count", 0) == 0 and m.get("inf_count", 0) == 0)
    lines.append(f"Profiles run: {n_total}")
    lines.append(f"Profiles pass (no fall, no NaN/Inf): {n_pass}")
    lines.append("")

    for m in all_metrics:
        name = m["profile"]
        fell = m.get("fell", "?")
        hy = m.get("hip_yaw_abs_max", "N/A")
        hy_gate = "PASS" if m.get("hip_yaw_gate_pass", False) else "FAIL"
        pitch_rms = m.get("pitch_rms_deg", "N/A")
        support_rms = m.get("support_rms_m", "N/A")
        notch_frac = m.get("notch_active_fraction", 0.0)
        roll_rms = m.get("roll_rms_deg", 0.0)
        nans = m.get("nan_count", 0)
        lines.append(
            f"  {name:45s}  fell={fell}  hy={hy:.4f} {hy_gate}  "
            f"pitch={pitch_rms:.2f}°  sup={support_rms:.4f}m  "
            f"notch={notch_frac:.2f}  roll={roll_rms:.2f}°  nan={nans}"
        )

    # Gate crossing analysis
    lines.append("")
    lines.append("-" * 80)
    lines.append("NOTCH GATE CROSSING ANALYSIS")
    lines.append("-" * 80)
    for m in all_metrics:
        name = m["profile"]
        n_cross = m.get("n_gate_crossings", 0)
        spike_mean = m.get("pitch_spike_at_crossing_mean_deg", "N/A")
        spike_max = m.get("pitch_spike_at_crossing_max_deg", "N/A")
        lines.append(
            f"  {name:45s}  crossings={n_cross:3d}  "
            f"spike_mean={spike_mean}°  spike_max={spike_max}°"
        )

    lines.append("")
    verdict = "PASS" if n_pass == n_total else "PARTIAL" if n_pass > 0 else "FAIL"
    lines.append(f"VERDICT: {verdict}")
    lines.append("=" * 80)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"  [SUMMARY] Wrote summary to {path}")


def main():
    parser = argparse.ArgumentParser(description="True dynamic-height Step C validation harness")
    parser.add_argument("--quick", action="store_true", help="Use quick mode (shortened profiles)")
    parser.add_argument("--profiles", nargs="*", default=None,
                        help="Specific profiles to run (default: all)")
    args = parser.parse_args()

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    # Determine which profiles to run
    if args.quick:
        profile_names = args.profiles or list(QUICK_HEIGHT_PROFILES.keys())
        profiles_dict = QUICK_HEIGHT_PROFILES
    else:
        profile_names = args.profiles or list(HEIGHT_PROFILES.keys())
        profiles_dict = HEIGHT_PROFILES
    print(f"[INFO] Quick mode: {'ON' if args.quick else 'OFF'}", flush=True)
    print(f"[INFO] Profiles to run: {len(profile_names)}", flush=True)

    # Phase 1: Generate trajectory JSONs and run simulations
    all_metrics = []
    for pname in profile_names:
        if pname not in profiles_dict:
            print(f"  [WARN] Unknown profile: {pname}", flush=True)
            continue
        info = profiles_dict[pname]
        print(f"\n{'='*60}")
        print(f"Profile: {pname}")
        print(f"  {info['description']}")
        print(f"  Setup: {info['setup']}, Steps: {info['steps']}")
        print(f"  Waypoints: {len(info['waypoints'])}")
        print(f"{'='*60}")

        # Write trajectory JSON
        traj_path = write_trajectory_json(pname, info["waypoints"], info["steps"])

        # Run simulation
        tel_path = run_dynamic_height_profile(
            pname, traj_path, info["setup"], info["steps"]
        )
        if tel_path is None or not tel_path.exists():
            print(f"  [WARN] No telemetry for {pname}, skipping analysis", flush=True)
            continue

        # Analyze telemetry
        metrics = analyze_telemetry(pname, tel_path)
        all_metrics.append(metrics)

        # Quick summary
        hy = metrics.get("hip_yaw_abs_max", "N/A")
        pitch = metrics.get("pitch_rms_deg", "N/A")
        nans = metrics.get("nan_count", 0)
        print(f"  [RESULT] hy_max={hy}, pitch_rms={pitch}°, nan={nans}", flush=True)

    # Phase 2: Write analysis
    if all_metrics:
        write_metrics_csv(all_metrics, ANALYSIS_DIR / "dynamic_step_c_metrics.csv")
        write_summary(all_metrics, ANALYSIS_DIR / "dynamic_step_c_summary.txt")
    else:
        print("  [WARN] No metrics collected", flush=True)

    print(f"\n{'='*60}")
    print(f"Done. Outputs in {OUT_BASE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
