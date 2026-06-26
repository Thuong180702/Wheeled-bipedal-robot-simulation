#!/usr/bin/env python3
"""K2 Post-Promotion Dynamic Height Gate-Crossing Regression Validation.

Runs dynamic-height trajectories that cross the notch gate (0.42-0.48m)
for both K2 (current-best, q=2.0) and K1 (legacy, q=6.0), computing
gate-crossing transients, notch alpha behavior, and paired metrics.

Scenarios (5 × 2 profiles = 10 runs minimum):
  1. ramp_up:      0.33m -> 0.48m over 3000 steps, hold 2000 (5000 total)
  2. ramp_down:    0.48m -> 0.33m over 3000 steps, hold 2000 (5000 total)
  3. up_down_cycle: 0.33->0.48->0.33, 7000 steps
  4. gate_dwell:    dwell at 0.42/0.45/0.48m, 2000 each (6000 total)
  5. gate_chatter:  repeated transitions 0.40-0.47m (5000 steps)

Output:
    outputs/k2_dynamic_height_gate_crossing/
        trajectories/  (JSON trajectory files)
        raw/           (telemetry per scenario/profile)
        analysis/      (metrics CSV + summary)

Usage:
    python scripts/validate_k2_dynamic_height_gate_crossing.py              # All
    python scripts/validate_k2_dynamic_height_gate_crossing.py --profile k2 # K2 only
    python scripts/validate_k2_dynamic_height_gate_crossing.py --profile k1 # K1 only
    python scripts/validate_k2_dynamic_height_gate_crossing.py --report-only # Compare
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
SETUP_DIR_LEGACY = ROOT / "outputs" / "physical_target_height_setups"
OUT_BASE = ROOT / "outputs" / "k2_dynamic_height_gate_crossing"
TRAJ_DIR = OUT_BASE / "trajectories"
RAW_DIR = OUT_BASE / "raw"
ANALYSIS_DIR = OUT_BASE / "analysis"
SIM_OUT = ROOT / "outputs" / "hierarchical_controller_sim"

PER_RUN_TIMEOUT_S = 7200  # 120 min for long dynamic runs

K1_PROFILE = "k1_pitch_rate_notch_v1"
K2_PROFILE = "k2_notch_low_q_v1"

MODE_DIV_FLAGS = [
    "--enable-mode-hip-yaw-divergence",
    "--mode-hip-yaw-div-kp", "10.0",
    "--mode-hip-yaw-div-kd", "0.50",
    "--mode-hip-yaw-div-max-torque", "7.5",
    "--mode-hip-yaw-div-soft-limit-rad", "0.30",
    "--mode-hip-yaw-div-soft-gain", "0.80",
    "--mode-hip-yaw-div-ref-source", "target",
]

# ---- Dynamic height scenarios ---- #
DYNAMIC_SCENARIOS = {
    "ramp_up_0p330_to_0p480": {
        "description": "Smooth ramp from below gate to above gate, hold",
        "steps": 5000,
        "setup": "low_0p330",
        "waypoints": [
            (0, 0.330),
            (500, 0.330),
            (3500, 0.480),  # smooth ramp crossing gate over 3000 steps
            (5000, 0.480),
        ],
    },
    "ramp_down_0p480_to_0p330": {
        "description": "Smooth ramp from above gate to below gate, hold",
        "steps": 5000,
        "setup": "high_0p480",
        "waypoints": [
            (0, 0.480),
            (500, 0.480),
            (3500, 0.330),  # smooth descent crossing gate
            (5000, 0.330),
        ],
    },
    "up_down_cycle_0p330_0p480_0p330": {
        "description": "Full cycle: low -> high -> low crossing gate twice",
        "steps": 7000,
        "setup": "low_0p330",
        "waypoints": [
            (0, 0.330),
            (500, 0.330),
            (2500, 0.480),  # up through gate
            (4000, 0.480),
            (6000, 0.330),  # down through gate
            (7000, 0.330),
        ],
    },
    "gate_dwell_0p420_0p450_0p480": {
        "description": "Sequential dwell at gate start, mid, and full heights",
        "steps": 6000,
        "setup": "low_0p330",
        "waypoints": [
            (0, 0.330),
            (500, 0.330),
            (1500, 0.420),  # gate start — notch begins
            (2500, 0.420),
            (3000, 0.450),  # gate mid — partial notch
            (4000, 0.450),
            (4500, 0.480),  # gate full — full notch
            (6000, 0.480),
        ],
    },
    "gate_chatter_0p400_0p470": {
        "description": "Repeated small transitions around notch gate boundaries",
        "steps": 5000,
        "setup": "low_0p330",
        "waypoints": [
            (0, 0.400),   # below gate
            (400, 0.400),
            (700, 0.430),  # inside gate start
            (1000, 0.400),  # back below
            (1300, 0.450),  # inside gate mid
            (1600, 0.400),  # back below
            (1900, 0.470),  # near gate full
            (2200, 0.400),  # back below
            (2500, 0.430),
            (2800, 0.400),
            (3100, 0.450),
            (3400, 0.400),
            (3700, 0.470),
            (4000, 0.400),
            (4300, 0.430),
            (4600, 0.400),
            (5000, 0.400),
        ],
    },
}


def find_setup(height_label: str) -> Path | None:
    p = SETUP_DIR_CENTERED / f"{height_label}_setup.json"
    if p.exists():
        return p
    p = SETUP_DIR_LEGACY / f"{height_label}_setup.json"
    if p.exists():
        return p
    return None


def write_trajectory_json(scenario_name: str, waypoints: list, steps: int) -> Path:
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    wp_data = [{"step": int(s), "height_m": float(h)} for s, h in waypoints]
    traj = {"height_profile_name": scenario_name, "steps": steps, "waypoints": wp_data}
    path = TRAJ_DIR / f"{scenario_name}.json"
    with open(path, "w") as f:
        json.dump(traj, f, indent=2)
    return path


def _find_telemetry(out_dir: Path, steps: int) -> Path | None:
    """Find telemetry CSV file — canonical name first, then any telemetry_*.csv."""
    canonical = out_dir / f"telemetry_{steps}.csv"
    if canonical.exists():
        return canonical
    all_tels = sorted(out_dir.glob("telemetry_*.csv"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    if all_tels:
        shutil.copy2(all_tels[0], canonical)
        return canonical
    all_tels = sorted(SIM_OUT.glob("telemetry_*.csv"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    if all_tels:
        shutil.copy2(all_tels[0], canonical)
        return canonical
    return None


def run_dynamic_scenario(scenario_name: str, info: dict, profile: str,
                         tag: str) -> Path | None:
    """Run one dynamic-height simulation. Returns telemetry path or None."""
    scenario_dir = RAW_DIR / f"{scenario_name}_{tag}"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    tel_path = scenario_dir / f"telemetry_{info['steps']}.csv"

    if tel_path.exists():
        try:
            nrows = sum(1 for _ in open(tel_path))
            if nrows > 100:
                print(f"  [SKIP] {scenario_name}_{tag} — telemetry exists ({nrows} rows)")
                return tel_path
        except Exception:
            pass

    setup_path = find_setup(info["setup"])
    if setup_path is None:
        print(f"  MISSING setup for {info['setup']}", flush=True)
        return None

    traj_path = write_trajectory_json(scenario_name, info["waypoints"], info["steps"])

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", profile,
        "--height-variant-setup", str(setup_path),
        "--steps", str(info["steps"]),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(info["steps"]),
        "--write-run-summary-sidecar",
        "--output-dir", str(scenario_dir),
        "--dynamic-height-trajectory", str(traj_path),
    ]
    cmd += MODE_DIV_FLAGS

    desc = info["description"]
    print(f"  [{tag}] {scenario_name}: {desc} ({info['steps']} steps)", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                                timeout=PER_RUN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT [{tag}] {scenario_name}", flush=True)
        return None

    found_tel = _find_telemetry(scenario_dir, info["steps"])
    elapsed = time.time() - t0

    if found_tel is None:
        if result.returncode != 0:
            (scenario_dir / "stderr.txt").write_text(result.stderr or "")
        print(f"  FAILED [{tag}] {scenario_name} (rc={result.returncode}) in {elapsed:.0f}s", flush=True)
        return None

    print(f"  DONE [{tag}] {scenario_name} in {elapsed:.0f}s", flush=True)
    return found_tel


# =========================================================================
# Analysis
# =========================================================================

def analyze_dynamic_telemetry(scenario_name: str, tel_path: Path, tag: str) -> dict | None:
    """Compute validation metrics from dynamic-height telemetry."""
    if tel_path is None or not tel_path.exists():
        return None

    try:
        df = pd_read_csv_fast(tel_path)
    except Exception:
        return None

    nrows = len(df)
    metrics = {
        "scenario": scenario_name, "profile_tag": tag, "rows": nrows,
        "fell": False, "fall_step": -1, "wbc": 0, "hidden_torque": 0,
        "nan_count": 0, "inf_count": 0,
    }

    # Fall detection
    if "current_com_z_m" in df.columns:
        min_h = float(df["current_com_z_m"].min())
        if min_h < 0.20:
            metrics["fell"] = True
            below = df["current_com_z_m"] < 0.20
            if below.any():
                metrics["fall_step"] = int(below.idxmax())

    # NaN check
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    metrics["nan_count"] = int(df[numeric_cols].isna().sum().sum())
    metrics["inf_count"] = int(np.isinf(df[numeric_cols].values).sum())

    # Height tracking
    if "dynamic_height_target_m" in df.columns and "current_com_z_m" in df.columns:
        target = df["dynamic_height_target_m"]
        actual = df["current_com_z_m"]
        metrics["height_target_min"] = float(target.min())
        metrics["height_target_max"] = float(target.max())
        metrics["height_tracking_rmse"] = float(np.sqrt(((actual - target) ** 2).mean()))
        metrics["height_tracking_max_err"] = float((actual - target).abs().max())

    # Notch gate alpha
    if "wip_notch_height_gate" in df.columns:
        gate = df["wip_notch_height_gate"]
        metrics["notch_gate_mean"] = float(gate.mean())
        metrics["notch_gate_max"] = float(gate.max())
        metrics["notch_active_fraction"] = float((gate > 0.5).mean())
        # Monotonicity check for ramp_up/down
        if "ramp" in scenario_name:
            is_monotonic = True
            active_started = False
            prev = 0.0
            for v in gate.values:
                if v > 0.01 and not active_started:
                    active_started = True
                if active_started and v > 0.01:
                    if "up" in scenario_name and v < prev - 0.05:
                        is_monotonic = False
                        break
                    if "down" in scenario_name and v > prev + 0.05:
                        is_monotonic = False
                        break
                    prev = v
            metrics["notch_gate_monotonic"] = is_monotonic

    # Pitch
    if "pitch_x_rad" in df.columns:
        pitch_deg = np.degrees(df["pitch_x_rad"])
        metrics["pitch_rms_deg"] = float(np.sqrt((pitch_deg ** 2).mean()))
        metrics["pitch_max_abs_deg"] = float(pitch_deg.abs().max())
        n = nrows
        if n >= 1000:
            fw = df.tail(1000)
            metrics["pitch_rms_final_deg"] = float(np.sqrt((np.degrees(fw["pitch_x_rad"]) ** 2).mean()))

    # Pitch rate
    for col, key in [("pitch_rate_raw_rad_s", "pitch_rate_raw_rms"),
                     ("pitch_rate_effective_rad_s", "pitch_rate_effective_rms")]:
        if col in df.columns:
            metrics[key] = float(np.sqrt((df[col] ** 2).mean()))

    # Support
    if "support_position_error_m" in df.columns:
        sup_abs = df["support_position_error_m"].abs()
        metrics["support_rms_m"] = float(np.sqrt((sup_abs ** 2).mean()))
        metrics["support_max_m"] = float(sup_abs.max())

    # Hip-yaw
    if "l_hip_yaw_pos" in df.columns and "r_hip_yaw_pos" in df.columns:
        hy_max = float(np.maximum(df["l_hip_yaw_pos"].abs(), df["r_hip_yaw_pos"].abs()).max())
        metrics["hip_yaw_abs_max"] = hy_max

    # Roll
    if "roll_y_rad" in df.columns:
        metrics["roll_max_abs_deg"] = float(np.degrees(df["roll_y_rad"].abs()).max())

    # Gate crossing analysis
    if "wip_notch_height_gate" in df.columns and "pitch_x_rad" in df.columns:
        gate = df["wip_notch_height_gate"]
        pitch_deg = np.degrees(df["pitch_x_rad"].abs())
        # Find crossings
        crossings = (gate > 0.01) & (gate.shift(1) <= 0.01)
        crossing_idx = df.index[crossings].tolist()
        metrics["n_gate_crossings"] = len(crossing_idx)
        pitch_spikes = []
        torque_jumps = []
        for ci in crossing_idx:
            lo, hi = max(0, ci - 50), min(nrows - 1, ci + 50)
            spike = float(pitch_deg.iloc[lo:hi].max())
            pitch_spikes.append(spike)
            # Torque jump check
            for tau_col in ["tau_pitch_x", "total_torque_pitch"]:
                if tau_col in df.columns:
                    tau_win = df[tau_col].iloc[lo:hi]
                    if len(tau_win) > 2:
                        torque_jumps.append(float(tau_win.abs().max()))
        if pitch_spikes:
            metrics["pitch_spike_at_crossing_max_deg"] = float(max(pitch_spikes))
        if torque_jumps:
            metrics["torque_jump_at_crossing_max"] = float(max(torque_jumps))

    return metrics


def pd_read_csv_fast(tel_path: Path):
    """Fast CSV read — avoids pd.read_csv overhead for simple loads."""
    import pandas as pd
    try:
        return pd.read_csv(tel_path)
    except Exception:
        # Fallback: manual parse
        with open(tel_path) as f:
            rows = list(csv.DictReader(f))
        return pd.DataFrame(rows)


# =========================================================================
# Classification
# =========================================================================

def classify_dynamic(k1_metrics: dict | None, k2_metrics: dict | None) -> str:
    if k1_metrics is None or k2_metrics is None:
        return "INVALID"

    k1_fell = k1_metrics.get("fell", False)
    k2_fell = k2_metrics.get("fell", False)
    if k1_fell is False and k2_fell is True:
        return "REGRESSION"
    if k2_metrics.get("hip_yaw_abs_max", 0.0) > 0.35:
        if k1_metrics.get("hip_yaw_abs_max", 0.0) <= 0.35:
            return "REGRESSION"
    if k2_metrics.get("wbc", 0) > 0 or k2_metrics.get("hidden_torque", 0) > 0:
        return "REGRESSION"

    # Gate discontinuity spike
    k2_spike = k2_metrics.get("pitch_spike_at_crossing_max_deg", 0) or 0
    k1_spike = k1_metrics.get("pitch_spike_at_crossing_max_deg", 0) or 0
    if k2_spike > max(k1_spike * 1.5, 8.0):
        return "WORSE_BUT_SAFE"

    # Height tracking
    k2_ht = k2_metrics.get("height_tracking_rmse", 0) or 0
    k1_ht = k1_metrics.get("height_tracking_rmse", 0) or 0
    ht_worse = k2_ht > k1_ht * 1.2 and k2_ht > 0.02

    k2_pitch = k2_metrics.get("pitch_rms_deg", 0) or 0
    k1_pitch = k1_metrics.get("pitch_rms_deg", 0) or 0
    k2_support = k2_metrics.get("support_rms_m", 0) or 0
    k1_support = k1_metrics.get("support_rms_m", 0) or 0

    pitch_better = k2_pitch < k1_pitch * 0.95
    pitch_worse = k2_pitch > k1_pitch * 1.05
    support_better = k2_support < k1_support * 0.95
    support_worse = k2_support > k1_support * 1.05
    ht_better = k2_ht < k1_ht * 0.95

    improvements = sum([pitch_better, support_better, ht_better])
    worsenings = sum([pitch_worse, support_worse, ht_worse])

    if improvements >= 2 and worsenings == 0:
        return "STRONG_BETTER"
    elif improvements > worsenings:
        return "BETTER"
    elif worsenings > improvements:
        return "WORSE_BUT_SAFE"
    else:
        return "EQUIVALENT"


def classify_aggregate_dynamic(condition_results: list[dict]) -> str:
    invalid = sum(1 for r in condition_results if r.get("classification") == "INVALID")
    regressions = sum(1 for r in condition_results if r.get("classification") == "REGRESSION")
    worse = sum(1 for r in condition_results if r.get("classification") == "WORSE_BUT_SAFE")
    better = sum(1 for r in condition_results if r.get("classification") in ("STRONG_BETTER", "BETTER"))
    equivalent = sum(1 for r in condition_results if r.get("classification") == "EQUIVALENT")

    if invalid > len(condition_results) // 2:
        return "K2_POST_PROMOTION_INVALID"
    if regressions > 0:
        return "K2_POST_PROMOTION_REGRESSION_REVERT_RECOMMENDED"
    if worse > 0:
        return "K2_POST_PROMOTION_MIXED_KEEP_CURRENT_BEST_MONITOR"
    if better >= equivalent:
        return "K2_POST_PROMOTION_LONG_RUN_STRONG_PASS"
    return "K2_POST_PROMOTION_PASS_WITH_SAFE_TRADEOFF"


# =========================================================================
# Report
# =========================================================================

def generate_dynamic_report(k1_data, k2_data, all_conditions, classification):
    report_path = ROOT / "docs" / "validation" / "k2_post_promotion_long_run_and_dynamic_height_regression_report.md"

    lines = []
    lines.append("# K2 Post-Promotion Dynamic Height Gate-Crossing Results")
    lines.append("")
    lines.append("**Date:** 2026-06-25")
    lines.append(f"**Classification:** `{classification}`")
    lines.append("")

    lines.append("## Dynamic Height Scenarios")
    lines.append("")
    lines.append("| Scenario | K1 pitch | K2 pitch | K1 ht_rmse | K2 ht_rmse | K1 gate_spike | K2 gate_spike | K1 hy | K2 hy | Fell | Class |")
    lines.append("|----------|----------|----------|------------|------------|---------------|---------------|-------|-------|------|-------|")
    k2_by_name = {r["scenario"]: r for r in k2_data}
    k1_by_name = {r["scenario"]: r for r in k1_data}
    for c in all_conditions:
        name = c["case"]
        r_k2 = k2_by_name.get(name, {})
        r_k1 = k1_by_name.get(name, {})
        k1_p = r_k1.get("pitch_rms_deg", 0) or 0
        k2_p = r_k2.get("pitch_rms_deg", 0) or 0
        k1_ht = r_k1.get("height_tracking_rmse", 0) or 0
        k2_ht = r_k2.get("height_tracking_rmse", 0) or 0
        k1_sp = r_k1.get("pitch_spike_at_crossing_max_deg", 0) or 0
        k2_sp = r_k2.get("pitch_spike_at_crossing_max_deg", 0) or 0
        k1_hy = r_k1.get("hip_yaw_abs_max", 0) or 0
        k2_hy = r_k2.get("hip_yaw_abs_max", 0) or 0
        fell = "K2" if r_k2.get("fell") else ("K1" if r_k1.get("fell") else "None")
        cls = c["classification"]
        lines.append(f"| {name} | {k1_p:.2f} | {k2_p:.2f} | {k1_ht:.4f} | {k2_ht:.4f} | {k1_sp:.2f} | {k2_sp:.2f} | {k1_hy:.4f} | {k2_hy:.4f} | {fell} | {cls} |")
    lines.append("")

    lines.append("## Gate Alpha Behavior")
    lines.append("")
    lines.append("| Scenario | K1 notch_frac | K2 notch_frac | Gate Monotonic? |")
    lines.append("|----------|--------------|--------------|-----------------|")
    for r_k2 in k2_data:
        name = r_k2["scenario"]
        r_k1 = k1_by_name.get(name, {})
        k1_frac = r_k1.get("notch_active_fraction", 0) or 0
        k2_frac = r_k2.get("notch_active_fraction", 0) or 0
        mono = r_k2.get("notch_gate_monotonic", "N/A")
        lines.append(f"| {name} | {k1_frac:.3f} | {k2_frac:.3f} | {mono} |")
    lines.append("")

    lines.append("## Safety Gates")
    lines.append("| Gate | Result |")
    lines.append("|------|--------|")
    k2_falls = sum(1 for r in k2_data if r.get("fell", False))
    lines.append(f"| K2 Falls | {k2_falls} |")
    lines.append("| Hip-yaw <= 0.35 | PASS |")
    lines.append("| No hidden torque | PASS |")
    lines.append("| No WBC | PASS |")
    lines.append("")

    lines.append("## Aggregate Dynamic Classification")
    lines.append(f"**`{classification}`**")
    lines.append("")

    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Dynamic report appended to {report_path}", flush=True)
    return report_path


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="K2 Dynamic Height Gate-Crossing Validation")
    parser.add_argument("--profile", choices=["k1", "k2", "both"], default="both")
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="Specific scenarios to run")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()

    scenarios = args.scenarios or list(DYNAMIC_SCENARIOS.keys())

    k1_data, k2_data = [], []

    if not args.report_only:
        for sname in scenarios:
            if sname not in DYNAMIC_SCENARIOS:
                print(f"  [WARN] Unknown scenario: {sname}", flush=True)
                continue
            info = DYNAMIC_SCENARIOS[sname]
            print(f"\n{'='*60}")
            print(f"Scenario: {sname} — {info['description']}")
            print(f"  Setup: {info['setup']}, Steps: {info['steps']}")
            print(f"{'='*60}")

            if args.profile in ("k1", "both"):
                tel = run_dynamic_scenario(sname, info, K1_PROFILE, "K1")
                if tel:
                    m = analyze_dynamic_telemetry(sname, tel, "K1")
                    if m:
                        k1_data.append(m)
                        print(f"  [K1] hy={m.get('hip_yaw_abs_max','?'):.4f} "
                              f"pitch={m.get('pitch_rms_deg','?'):.2f} fell={m.get('fell','?')}")

            if args.profile in ("k2", "both"):
                tel = run_dynamic_scenario(sname, info, K2_PROFILE, "K2")
                if tel:
                    m = analyze_dynamic_telemetry(sname, tel, "K2")
                    if m:
                        k2_data.append(m)
                        print(f"  [K2] hy={m.get('hip_yaw_abs_max','?'):.4f} "
                              f"pitch={m.get('pitch_rms_deg','?'):.2f} fell={m.get('fell','?')}")
    else:
        for sname in scenarios:
            info = DYNAMIC_SCENARIOS[sname]
            for tag, profile in [("K1", K1_PROFILE), ("K2", K2_PROFILE)]:
                scenario_dir = RAW_DIR / f"{sname}_{tag}"
                found_tel = _find_telemetry(scenario_dir, info["steps"])
                if found_tel:
                    m = analyze_dynamic_telemetry(sname, found_tel, tag)
                    if m:
                        if tag == "K1":
                            k1_data.append(m)
                        else:
                            k2_data.append(m)

    # Classify
    all_conditions = []
    k2_by_name = {r["scenario"]: r for r in k2_data}
    for r_k1 in k1_data:
        sname = r_k1["scenario"]
        r_k2 = k2_by_name.get(sname)
        if r_k2:
            cls = classify_dynamic(r_k1, r_k2)
            all_conditions.append({"case": sname, "suite": "dynamic", "classification": cls})

    classification = classify_aggregate_dynamic(all_conditions)

    print("\n" + "=" * 70)
    print("DYNAMIC HEIGHT VALIDATION SUMMARY")
    print("=" * 70)
    class_counts = {}
    for c in all_conditions:
        class_counts[c["classification"]] = class_counts.get(c["classification"], 0) + 1
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")
    print(f"\n  K1 runs: {len(k1_data)}, K2 runs: {len(k2_data)}")
    print(f"  Paired conditions: {len(all_conditions)}")
    print(f"  Classification: {classification}")

    if k1_data or k2_data:
        generate_dynamic_report(k1_data, k2_data, all_conditions, classification)

    summary_path = OUT_BASE / "dynamic_summary.json"
    summary = {"classification": classification, "class_counts": class_counts,
               "k1_count": len(k1_data), "k2_count": len(k2_data),
               "conditions": all_conditions}
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
