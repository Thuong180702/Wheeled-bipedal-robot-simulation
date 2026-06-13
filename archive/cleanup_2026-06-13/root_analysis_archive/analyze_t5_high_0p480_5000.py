#!/usr/bin/env python3
"""Analyze T5 high_0p480 5000-step validation."""

import json
import pandas as pd
import numpy as np
import sys

# Load telemetry
csv_path = "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t5_high_0p480_5000/telemetry_t5_high_0p480_5000.csv"
print(f"Loading telemetry from: {csv_path}")
df = pd.read_csv(csv_path)
print(f"[OK] Loaded {len(df)} rows, {len(df.columns)} columns\n")

# ========== PHASE 4: Drift and Band Analysis ==========
print("=" * 80)
print("PHASE 4: Drift and Band Analysis")
print("=" * 80)

drift_col = "active_pitch_crossing_signed_error_m"
e = df[drift_col].values
abs_e = np.abs(e)

drift_metrics = {
    "survived_steps": len(df),
    "min_e": float(e.min()),
    "max_e": float(e.max()),
    "max_abs_e": float(abs_e.max()),
    "peak_to_peak": float(e.max() - e.min()),
    "mean_e": float(e.mean()),
    "mean_abs_e": float(abs_e.mean()),
    "final_e": float(e[-1]),
    "positive_pct": float((e > 0).sum() / len(e) * 100),
    "negative_pct": float((e < 0).sum() / len(e) * 100),
    "zero_crossings": int(np.sum(np.diff(np.sign(e)) != 0)),
}

band_metrics = {}
thresholds = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]
for thresh in thresholds:
    outside_count = int((abs_e > thresh).sum())
    outside_pct = float(outside_count / len(e) * 100)
    band_metrics[f"outside_{thresh:.2f}"] = {
        "count": outside_count,
        "pct": outside_pct
    }

print(f"\nSurvival: {drift_metrics['survived_steps']}/5000")
print(f"Min e: {drift_metrics['min_e']:.4f} m")
print(f"Max e: {drift_metrics['max_e']:.4f} m")
print(f"Max |e|: {drift_metrics['max_abs_e']:.4f} m")
print(f"Mean |e|: {drift_metrics['mean_abs_e']:.4f} m")
print(f"Final e: {drift_metrics['final_e']:.4f} m")
print(f"Positive: {drift_metrics['positive_pct']:.1f}%, Negative: {drift_metrics['negative_pct']:.1f}%")

print(f"\nBand Metrics:")
print(f"  Outside ±0.08 m: {band_metrics['outside_0.08']['pct']:.1f}%")
print(f"  Outside ±0.10 m: {band_metrics['outside_0.10']['pct']:.1f}%")
print(f"  Outside ±0.15 m: {band_metrics['outside_0.15']['pct']:.1f}%")

# ========== PHASE 5: Window and Accumulation Analysis ==========
print("\n" + "=" * 80)
print("PHASE 5: Window and Accumulation Analysis")
print("=" * 80)

window_size = 500
num_windows = 10
window_metrics = []

for i in range(num_windows):
    start = i * window_size
    end = (i + 1) * window_size
    e_window = e[start:end]
    abs_e_window = np.abs(e_window)

    window_metrics.append({
        "window": i + 1,
        "steps": f"{start}-{end}",
        "max_abs_e": float(abs_e_window.max()),
        "mean_abs_e": float(abs_e_window.mean()),
        "final_e": float(e_window[-1]),
        "outside_0p08_pct": float((abs_e_window > 0.08).sum() / len(e_window) * 100),
        "outside_0p10_pct": float((abs_e_window > 0.10).sum() / len(e_window) * 100),
        "outside_0p15_pct": float((abs_e_window > 0.15).sum() / len(e_window) * 100),
    })

# Accumulation ratio
first_1000_mean = float(abs_e[:1000].mean())
last_1000_mean = float(abs_e[-1000:].mean())
accumulation_ratio = last_1000_mean / first_1000_mean if first_1000_mean > 1e-9 else 1.0

accumulation_metrics = {
    "first_1000_mean_abs_e": first_1000_mean,
    "last_1000_mean_abs_e": last_1000_mean,
    "ratio": accumulation_ratio,
    "classification": "stable" if accumulation_ratio < 1.2 else ("monitor" if accumulation_ratio < 1.5 else "accumulating")
}

print(f"\nAccumulation Ratio: {accumulation_ratio:.3f}")
print(f"Classification: {accumulation_metrics['classification'].upper()}")
print(f"First 1000 mean |e|: {first_1000_mean:.4f} m")
print(f"Last 1000 mean |e|: {last_1000_mean:.4f} m")

worst_window = max(window_metrics, key=lambda w: w["outside_0p08_pct"])
best_window = min(window_metrics, key=lambda w: w["outside_0p08_pct"])
print(f"\nWorst window: {worst_window['window']} ({worst_window['steps']}) - {worst_window['outside_0p08_pct']:.1f}% outside ±0.08")
print(f"Best window: {best_window['window']} ({best_window['steps']}) - {best_window['outside_0p08_pct']:.1f}% outside ±0.08")

# ========== PHASE 6: Tuned Feature Activation Analysis ==========
print("\n" + "=" * 80)
print("PHASE 6: Tuned Feature Activation Analysis")
print("=" * 80)

tuned_active = df["tuned_recenter_active"].sum()
tuned_active_pct = float(tuned_active / len(df) * 100)

band_states = df["tuned_band_state"].value_counts().to_dict()
band_state_ids = df["tuned_band_state_id"].value_counts().sort_index().to_dict()

print(f"\nTuned recenter active: {tuned_active}/{len(df)} steps ({tuned_active_pct:.1f}%)")
print(f"\nBand state distribution:")
for state, count in sorted(band_states.items(), key=lambda x: x[1], reverse=True):
    pct = count / len(df) * 100
    print(f"  {state}: {count} steps ({pct:.1f}%)")

# ========== PHASE 7: Stability and Safety Analysis ==========
print("\n" + "=" * 80)
print("PHASE 7: Stability and Safety Analysis")
print("=" * 80)

contact_left_pct = float(df["left_wheel_contact"].mean() * 100)
contact_right_pct = float(df["right_wheel_contact"].mean() * 100)
contact_both_pct = float((df["left_wheel_contact"] & df["right_wheel_contact"]).mean() * 100)

com_z_min = float(df["com_z_m"].min())
com_z_max = float(df["com_z_m"].max())
com_z_mean = float(df["com_z_m"].mean())

pitch_rms = float(np.sqrt((df["pitch_x_rad"]**2).mean()) * 180/np.pi)
roll_rms = float(np.sqrt((df["roll_y_rad"]**2).mean()) * 180/np.pi)

wheel_vel_mean = df["wheel_vel_mean_rad_s"].abs()
wheel_vel_max = float(wheel_vel_mean.max())
wheel_vel_rms = float(np.sqrt((wheel_vel_mean**2).mean()))
wheel_vel_gt_5 = int((wheel_vel_mean > 5.0).sum())
wheel_vel_gt_6 = int((wheel_vel_mean > 6.0).sum())
wheel_vel_gt_7 = int((wheel_vel_mean > 7.0).sum())

ownership_violations = int(df["ownership_violation_count"].max())

stability_metrics = {
    "contact_left_pct": contact_left_pct,
    "contact_right_pct": contact_right_pct,
    "contact_both_pct": contact_both_pct,
    "com_z_min": com_z_min,
    "com_z_max": com_z_max,
    "com_z_mean": com_z_mean,
    "pitch_rms_deg": pitch_rms,
    "roll_rms_deg": roll_rms,
    "wheel_vel_max_rad_s": wheel_vel_max,
    "wheel_vel_rms_rad_s": wheel_vel_rms,
    "wheel_vel_gt_5_count": wheel_vel_gt_5,
    "wheel_vel_gt_6_count": wheel_vel_gt_6,
    "wheel_vel_gt_7_count": wheel_vel_gt_7,
    "ownership_violations": ownership_violations,
}

print(f"\nContact: L={contact_left_pct:.1f}%, R={contact_right_pct:.1f}%, Both={contact_both_pct:.1f}%")
print(f"CoM Z: {com_z_min:.3f} - {com_z_max:.3f} m (mean={com_z_mean:.3f})")
print(f"Pitch RMS: {pitch_rms:.3f} deg")
print(f"Roll RMS: {roll_rms:.3f} deg")
print(f"Wheel vel: max={wheel_vel_max:.2f} rad/s, RMS={wheel_vel_rms:.2f} rad/s")
print(f"Wheel vel > 5 rad/s: {wheel_vel_gt_5} steps")
print(f"Wheel vel > 7 rad/s: {wheel_vel_gt_7} steps")
print(f"Ownership violations: {ownership_violations}")

# ========== Save Results ==========
output = {
    "phase_4_drift": drift_metrics,
    "phase_4_bands": band_metrics,
    "phase_5_accumulation": accumulation_metrics,
    "phase_5_window_metrics": window_metrics,
    "phase_6_tuned_activation": {
        "recenter_active_steps": int(tuned_active),
        "recenter_active_pct": tuned_active_pct,
        "band_states": band_states,
        "band_state_ids": {int(k): int(v) for k, v in band_state_ids.items()},
    },
    "phase_7_stability": stability_metrics,
}

output_path = "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t5_high_0p480_5000_analysis.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n[OK] Analysis saved to: {output_path}")

# Save window metrics CSV
window_df = pd.DataFrame(window_metrics)
window_csv_path = "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t5_high_0p480_5000_window_metrics.csv"
window_df.to_csv(window_csv_path, index=False)
print(f"[OK] Window metrics saved to: {window_csv_path}")

print("\n[OK] Phases 4-7 analysis complete")
