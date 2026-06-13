#!/usr/bin/env python3
"""Analyze E2 500-step telemetry and compare with D2/E1 baselines."""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Paths
e2_telemetry = Path("outputs/step_e_extreme_support_fix_eval/e2_low_0p300_500/e2_low_0p300_500_telemetry.csv")
d2_telemetry = Path("outputs/step_e_extreme_height_d2_official_check/low_0p300_5000_telemetry.csv")

# Load E2 telemetry
print("Loading E2 telemetry...")
e2_df = pd.read_csv(e2_telemetry)
print(f"E2 rows: {len(e2_df)}")

# Load D2 telemetry (first 500 rows)
print("Loading D2 telemetry...")
d2_df = pd.read_csv(d2_telemetry, nrows=500)
print(f"D2 rows: {len(d2_df)}")

# Extract key columns (use hip_yaw_abs_max directly)
support_col = "support_position_error_m"
tau_pos_col = "tau_position"
tau_pos_raw_col = "tau_position_raw"
tau_pos_int_col = "tau_position_integral"
integral_active_col = "integral_active"
hip_yaw_abs_col = "hip_yaw_abs_max"  # Direct column
wheel_vel_l_col = "wheel_vel_left_rad_s"
wheel_vel_r_col = "wheel_vel_right_rad_s"
pitch_col = "robot_pitch_x"
roll_col = "robot_roll_y"
com_z_col = "com_z"
height_error_col = "height_error_m"
hidden_torque_col = "hidden_torque_norm"
ownership_col = "ownership_violation_count"
effective_max_pos_tau_col = "effective_max_position_tau"
contact_valid_col = "contact_force_valid"

# Compute E2 statistics
e2_stats = {
    "label": "E2",
    "support_position_error_max": float(e2_df[support_col].abs().max()),
    "support_position_error_mean": float(e2_df[support_col].abs().mean()),
    "support_position_error_final": float(e2_df[support_col].abs().iloc[-1]),
    "support_position_error_rms": float(np.sqrt((e2_df[support_col]**2).mean())),
    "hip_yaw_abs_max": float(e2_df[hip_yaw_abs_col].max()),
    "hip_yaw_abs_mean": float(e2_df[hip_yaw_abs_col].mean()),
    "wheel_vel_mean_max": float(np.maximum(
        e2_df[wheel_vel_l_col].abs().max(),
        e2_df[wheel_vel_r_col].abs().max()
    )),
    "wheel_vel_mean_mean": float((e2_df[wheel_vel_l_col].abs().mean() + e2_df[wheel_vel_r_col].abs().mean()) / 2),
    "pitch_max": float(e2_df[pitch_col].abs().max()),
    "pitch_mean": float(e2_df[pitch_col].abs().mean()),
    "roll_max": float(e2_df[roll_col].abs().max()),
    "roll_mean": float(e2_df[roll_col].abs().mean()),
    "height_error_max": float(e2_df[height_error_col].abs().max()),
    "height_error_mean": float(e2_df[height_error_col].abs().mean()),
    "contact_valid_percent": float((e2_df[contact_valid_col] > 0).sum() / len(e2_df) * 100),
    "hidden_torque_max": float(e2_df[hidden_torque_col].max()),
    "ownership_violations_max": int(e2_df[ownership_col].max()),
    "integral_active_count": int(e2_df[integral_active_col].sum()),
    "integral_active_percent": float(e2_df[integral_active_col].sum() / len(e2_df) * 100),
    "tau_position_integral_max": float(e2_df[tau_pos_int_col].abs().max()),
    "tau_position_integral_mean": float(e2_df[tau_pos_int_col].abs().mean()),
    "tau_position_raw_max": float(e2_df[tau_pos_raw_col].abs().max()),
    "tau_position_raw_mean": float(e2_df[tau_pos_raw_col].abs().mean()),
    "tau_position_final": float(e2_df[tau_pos_col].iloc[-1]),
    "effective_max_position_tau": float(e2_df[effective_max_pos_tau_col].iloc[-1]) if effective_max_pos_tau_col in e2_df.columns else None,
}

# Compute crossing count for support_position_error > 0.15 m
e2_support_abs = e2_df[support_col].abs()
e2_crossings = (e2_support_abs > 0.15).sum()
e2_first_crossing = e2_support_abs[e2_support_abs > 0.15].index[0] if e2_crossings > 0 else None
e2_stats["support_error_gt_0p15_count"] = int(e2_crossings)
e2_stats["support_error_gt_0p15_first_step"] = int(e2_first_crossing) if e2_first_crossing is not None else None

# Compute D2 statistics (first 500 rows)
d2_stats = {
    "label": "D2",
    "support_position_error_max": float(d2_df[support_col].abs().max()),
    "support_position_error_mean": float(d2_df[support_col].abs().mean()),
    "support_position_error_final": float(d2_df[support_col].abs().iloc[-1]),
    "support_position_error_rms": float(np.sqrt((d2_df[support_col]**2).mean())),
    "hip_yaw_abs_max": float(d2_df[hip_yaw_abs_col].max()),
    "hip_yaw_abs_mean": float(d2_df[hip_yaw_abs_col].mean()),
    "wheel_vel_mean_max": float(np.maximum(
        d2_df[wheel_vel_l_col].abs().max(),
        d2_df[wheel_vel_r_col].abs().max()
    )),
    "wheel_vel_mean_mean": float((d2_df[wheel_vel_l_col].abs().mean() + d2_df[wheel_vel_r_col].abs().mean()) / 2),
    "pitch_max": float(d2_df[pitch_col].abs().max()),
    "pitch_mean": float(d2_df[pitch_col].abs().mean()),
    "roll_max": float(d2_df[roll_col].abs().max()),
    "roll_mean": float(d2_df[roll_col].abs().mean()),
    "height_error_max": float(d2_df[height_error_col].abs().max()),
    "height_error_mean": float(d2_df[height_error_col].abs().mean()),
    "contact_valid_percent": float((d2_df[contact_valid_col] > 0).sum() / len(d2_df) * 100),
    "hidden_torque_max": float(d2_df[hidden_torque_col].max()),
    "ownership_violations_max": int(d2_df[ownership_col].max()),
    "integral_active_count": 0,  # D2 has no integral
    "integral_active_percent": 0.0,
    "tau_position_integral_max": 0.0,
    "tau_position_integral_mean": 0.0,
    "tau_position_raw_max": float(d2_df[tau_pos_raw_col].abs().max()),
    "tau_position_raw_mean": float(d2_df[tau_pos_raw_col].abs().mean()),
    "tau_position_final": float(d2_df[tau_pos_col].iloc[-1]),
}

# Compute crossing count for D2
d2_support_abs = d2_df[support_col].abs()
d2_crossings = (d2_support_abs > 0.15).sum()
d2_first_crossing = d2_support_abs[d2_support_abs > 0.15].index[0] if d2_crossings > 0 else None
d2_stats["support_error_gt_0p15_count"] = int(d2_crossings)
d2_stats["support_error_gt_0p15_first_step"] = int(d2_first_crossing) if d2_first_crossing is not None else None

# Load E1 corrected comparison
e1_comparison_path = Path("outputs/step_e_extreme_support_fix_eval/e1_500_corrected_metric_comparison.json")
if e1_comparison_path.exists():
    with open(e1_comparison_path) as f:
        e1_data = json.load(f)
    e1_before = e1_data.get("e1_before_500", {})
    e1_after = e1_data.get("e1_after_500", {})
    e1_d2 = e1_data.get("d2_500", {})
else:
    e1_before = {}
    e1_after = {}
    e1_d2 = {}

# Print comparison table
print("\n" + "="*80)
print("E2 500-STEP COMPARISON: D2 vs E1 vs E2")
print("="*80)

print("\n### OFFICIAL SUPPORT METRIC ###")
print(f"{'Metric':<35} {'D2':>15} {'E1_before':>15} {'E1_after':>15} {'E2':>15}")
print("-"*95)
print(f"{'support_position_error_max (m)':<35} {d2_stats['support_position_error_max']:>15.6f} {e1_before.get('support_position_error_max', 0):>15.6f} {e1_after.get('support_position_error_max', 0):>15.6f} {e2_stats['support_position_error_max']:>15.6f}")
print(f"{'support_position_error_mean (m)':<35} {d2_stats['support_position_error_mean']:>15.6f} {e1_before.get('support_position_error_mean', 0):>15.6f} {e1_after.get('support_position_error_mean', 0):>15.6f} {e2_stats['support_position_error_mean']:>15.6f}")
print(f"{'support_position_error_final (m)':<35} {d2_stats['support_position_error_final']:>15.6f} {e1_before.get('support_position_error_final', 0):>15.6f} {e1_after.get('support_position_error_final', 0):>15.6f} {e2_stats['support_position_error_final']:>15.6f}")
print(f"{'support_error_gt_0p15_count':<35} {d2_stats['support_error_gt_0p15_count']:>15} {e1_before.get('support_error_gt_0p15_count', 0):>15} {e1_after.get('support_error_gt_0p15_count', 0):>15} {e2_stats['support_error_gt_0p15_count']:>15}")
print(f"{'support_error_gt_0p15_first_step':<35} {d2_stats['support_error_gt_0p15_first_step'] if d2_stats['support_error_gt_0p15_first_step'] is not None else 'N/A':>15} {e1_before.get('support_error_gt_0p15_first_step', 'N/A'):>15} {e1_after.get('support_error_gt_0p15_first_step', 'N/A'):>15} {e2_stats['support_error_gt_0p15_first_step'] if e2_stats['support_error_gt_0p15_first_step'] is not None else 'N/A':>15}")

print("\n### POSITION AUTHORITY ###")
print(f"{'Metric':<35} {'D2':>15} {'E1_before':>15} {'E1_after':>15} {'E2':>15}")
print("-"*95)
print(f"{'effective_max_position_tau (Nm)':<35} {'4.0 (default)':>15} {'4.0':>15} {'4.0':>15} {e2_stats.get('effective_max_position_tau', 'N/A'):>15}")
print(f"{'tau_position_raw_max (Nm)':<35} {d2_stats['tau_position_raw_max']:>15.4f} {e1_before.get('tau_position_raw_max', 0):>15.4f} {e1_after.get('tau_position_raw_max', 0):>15.4f} {e2_stats['tau_position_raw_max']:>15.4f}")
print(f"{'tau_position_integral_max (Nm)':<35} {d2_stats['tau_position_integral_max']:>15.4f} {e1_before.get('tau_position_integral_max', 0):>15.4f} {e1_after.get('tau_position_integral_max', 0):>15.4f} {e2_stats['tau_position_integral_max']:>15.4f}")
print(f"{'integral_active_count':<35} {d2_stats['integral_active_count']:>15} {e1_before.get('integral_active_count', 0):>15} {e1_after.get('integral_active_count', 0):>15} {e2_stats['integral_active_count']:>15}")
print(f"{'integral_active_percent':<35} {d2_stats['integral_active_percent']:>14.1f}% {e1_before.get('integral_active_percent', 0):>14.1f}% {e1_after.get('integral_active_percent', 0):>14.1f}% {e2_stats['integral_active_percent']:>14.1f}%")

print("\n### OTHER STEP E GATES ###")
print(f"{'Metric':<35} {'D2':>15} {'E1_before':>15} {'E1_after':>15} {'E2':>15}")
print("-"*95)
print(f"{'hip_yaw_abs_max (rad)':<35} {d2_stats['hip_yaw_abs_max']:>15.6f} {e1_before.get('hip_yaw_abs_max', 0):>15.6f} {e1_after.get('hip_yaw_abs_max', 0):>15.6f} {e2_stats['hip_yaw_abs_max']:>15.6f}")
print(f"{'wheel_vel_mean_max (rad/s)':<35} {d2_stats['wheel_vel_mean_max']:>15.4f} {e1_before.get('wheel_vel_mean_max', 0):>15.4f} {e1_after.get('wheel_vel_mean_max', 0):>15.4f} {e2_stats['wheel_vel_mean_max']:>15.4f}")
print(f"{'contact_valid_percent':<35} {d2_stats['contact_valid_percent']:>14.1f}% {e1_before.get('contact_valid_percent', 0):>14.1f}% {e1_after.get('contact_valid_percent', 0):>14.1f}% {e2_stats['contact_valid_percent']:>14.1f}%")
print(f"{'height_error_max (m)':<35} {d2_stats['height_error_max']:>15.6f} {e1_before.get('height_error_max', 0):>15.6f} {e1_after.get('height_error_max', 0):>15.6f} {e2_stats['height_error_max']:>15.6f}")
print(f"{'roll_max (rad)':<35} {d2_stats['roll_max']:>15.6f} {e1_before.get('roll_max', 0):>15.6f} {e1_after.get('roll_max', 0):>15.6f} {e2_stats['roll_max']:>15.6f}")
print(f"{'pitch_max (rad)':<35} {d2_stats['pitch_max']:>15.6f} {e1_before.get('pitch_max', 0):>15.6f} {e1_after.get('pitch_max', 0):>15.6f} {e2_stats['pitch_max']:>15.6f}")
print(f"{'hidden_torque_max':<35} {d2_stats['hidden_torque_max']:>15.4f} {e1_before.get('hidden_torque_max', 0):>15.4f} {e1_after.get('hidden_torque_max', 0):>15.4f} {e2_stats['hidden_torque_max']:>15.4f}")
print(f"{'ownership_violations_max':<35} {d2_stats['ownership_violations_max']:>15} {e1_before.get('ownership_violations_max', 0):>15} {e1_after.get('ownership_violations_max', 0):>15} {e2_stats['ownership_violations_max']:>15}")

# Determine classification
print("\n" + "="*80)
print("CLASSIFICATION DECISION")
print("="*80)

# Check if support improved
e2_support_max = e2_stats['support_position_error_max']
d2_support_max = d2_stats['support_position_error_max']
e1_after_support_max = e1_after.get('support_position_error_max', d2_support_max)

support_improved = e2_support_max < d2_support_max
support_worsened = e2_support_max > d2_support_max

e2_crossings = e2_stats['support_error_gt_0p15_count']
d2_crossings = d2_stats['support_error_gt_0p15_count']
crossings_improved = e2_crossings < d2_crossings

# Check other gates
hip_yaw_ok = e2_stats['hip_yaw_abs_max'] <= d2_stats['hip_yaw_abs_max'] * 1.05
wheel_vel_ok = e2_stats['wheel_vel_mean_max'] <= d2_stats['wheel_vel_mean_max'] * 1.05
contact_ok = e2_stats['contact_valid_percent'] >= 95.0
height_ok = e2_stats['height_error_max'] <= d2_stats['height_error_max'] * 1.1
roll_ok = e2_stats['roll_max'] <= d2_stats['roll_max'] * 1.1
hidden_ok = e2_stats['hidden_torque_max'] == 0.0
ownership_ok = e2_stats['ownership_violations_max'] == 0

# Check if higher cap was reached
cap_increased = e2_stats.get('effective_max_position_tau', 0) > 4.0

print(f"\nSupport metric comparison:")
print(f"  D2 max:   {d2_support_max:.6f} m")
print(f"  E1 max:   {e1_after_support_max:.6f} m")
print(f"  E2 max:   {e2_support_max:.6f} m")
print(f"  Delta:    {e2_support_max - d2_support_max:.6f} m")

print(f"\nCrossings > 0.15 m:")
print(f"  D2: {d2_crossings}")
print(f"  E2: {e2_crossings}")

print(f"\nPosition cap:")
print(f"  D2/E1: 4.0 Nm")
print(f"  E2 effective: {e2_stats.get('effective_max_position_tau', 'N/A')} Nm")
print(f"  Cap increased: {cap_increased}")

print(f"\nIntegral activity:")
print(f"  D2: {d2_stats['integral_active_count']} steps")
print(f"  E1 after: {e1_after.get('integral_active_count', 0)} steps")
print(f"  E2: {e2_stats['integral_active_count']} steps")

print(f"\nOther gates:")
print(f"  hip_yaw_ok: {hip_yaw_ok}")
print(f"  wheel_vel_ok: {wheel_vel_ok}")
print(f"  contact_ok: {contact_ok}")
print(f"  height_ok: {height_ok}")
print(f"  roll_ok: {roll_ok}")
print(f"  hidden_ok: {hidden_ok}")
print(f"  ownership_ok: {ownership_ok}")

# Classification
if support_improved and all([hip_yaw_ok, wheel_vel_ok, contact_ok, height_ok, roll_ok, hidden_ok, ownership_ok]):
    classification = "E2_500_IMPROVES_SUPPORT"
elif support_worsened:
    classification = "E2_500_WORSE_SUPPORT"
elif not all([hip_yaw_ok, wheel_vel_ok, contact_ok, height_ok, roll_ok, hidden_ok, ownership_ok]):
    classification = "E2_500_REGRESSES_OTHER_GATES"
else:
    classification = "E2_500_NO_EFFECT"

print(f"\n{'='*40}")
print(f"CLASSIFICATION: {classification}")
print(f"{'='*40}")

# Check if cap was reached
if cap_increased and e2_stats['tau_position_raw_max'] < 4.5:
    print("\nNOTE: Position cap increased to 5.0 Nm but tau_position_raw max was only {:.4f} Nm".format(e2_stats['tau_position_raw_max']))
    print("      The higher cap was NOT reaching saturation level.")

# Save comparison JSON
comparison_json = {
    "classification": classification,
    "d2_500": d2_stats,
    "e1_before_500": e1_before,
    "e1_after_500": e1_after,
    "e2_500": e2_stats,
    "support_improved": support_improved,
    "support_worsened": support_worsened,
    "crossings_improved": crossings_improved,
    "cap_increased": cap_increased,
    "all_gates_pass": all([hip_yaw_ok, wheel_vel_ok, contact_ok, height_ok, roll_ok, hidden_ok, ownership_ok]),
}

with open("outputs/step_e_extreme_support_fix_eval/e2_low_0p300_500_comparison.json", "w") as f:
    json.dump(comparison_json, f, indent=2)
print(f"\nSaved: outputs/step_e_extreme_support_fix_eval/e2_low_0p300_500_comparison.json")

# Save CSV summary
csv_rows = []
metrics = [
    ("support_position_error_max_m", "Official Support Metric", "m"),
    ("support_position_error_mean_m", "Official Support Metric", "m"),
    ("support_error_gt_0p15_count", "Official Support Metric", "steps"),
    ("hip_yaw_abs_max_rad", "Hip Yaw Gate", "rad"),
    ("wheel_vel_mean_max_rad_s", "Wheel Velocity Gate", "rad/s"),
    ("contact_valid_percent", "Contact Gate", "%"),
    ("height_error_max_m", "Height Gate", "m"),
    ("roll_max_rad", "Roll Gate", "rad"),
    ("pitch_max_rad", "Pitch (record)", "rad"),
    ("hidden_torque_max", "WBC Gate", "Nm"),
    ("ownership_violations_max", "WBC Gate", "count"),
    ("tau_position_raw_max_Nm", "Position Authority", "Nm"),
    ("tau_position_integral_max_Nm", "Position Authority", "Nm"),
    ("integral_active_count", "Integral Activity", "steps"),
    ("effective_max_position_tau_Nm", "Position Cap", "Nm"),
]

for metric, group, unit in metrics:
    row = {"metric": metric, "group": group, "unit": unit}
    for label, data in [("D2", d2_stats), ("E1_before", e1_before), ("E1_after", e1_after), ("E2", e2_stats)]:
        # Map metric names
        key = metric
        if key not in data:
            key = metric.replace("_m", "_m").replace("_Nm", "_Nm")
        if key not in data:
            key = metric.replace("_max_m", "_max").replace("_Nm", "_Nm")
        val = data.get(key, "N/A")
        row[label] = val
    csv_rows.append(row)

import csv
with open("outputs/step_e_extreme_support_fix_eval/e2_low_0p300_500_comparison.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["metric", "group", "unit", "D2", "E1_before", "E1_after", "E2"])
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"Saved: outputs/step_e_extreme_support_fix_eval/e2_low_0p300_500_comparison.csv")