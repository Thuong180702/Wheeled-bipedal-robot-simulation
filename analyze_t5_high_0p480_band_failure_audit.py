#!/usr/bin/env python3
"""Phase 1: Audit T5 high_0p480 band failure during windows 2-7."""

import json
import pandas as pd
import numpy as np

# Load T5 high_0p480 5000-step telemetry
csv_path = "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t5_high_0p480_5000/telemetry_t5_high_0p480_5000.csv"
print(f"Loading T5 high_0p480 5000-step telemetry from: {csv_path}")
df = pd.read_csv(csv_path)
print(f"[OK] Loaded {len(df)} rows, {len(df.columns)} columns\n")

# Use correct physical drift column
drift_col = "active_pitch_crossing_signed_error_m"
e = df[drift_col].values
abs_e = np.abs(e)

print("=" * 80)
print("PHASE 1: T5 High_0p480 Band Failure Audit (Windows 2-7)")
print("=" * 80)

# ========== Part 1: Threshold Crossing Events ==========
print("\n=== Part 1: Threshold Crossing Events ===\n")

thresholds = [0.05, 0.08, 0.10, 0.12, 0.15]
crossing_events = []

for thresh in thresholds:
    # Find first crossing
    outside = abs_e > thresh
    if outside.any():
        first_idx = np.where(outside)[0][0]

        event = {
            "threshold_m": thresh,
            "first_crossing_step": int(first_idx),
            "first_crossing_e_m": float(e[first_idx]),
            "band_state": df.loc[first_idx, "tuned_band_state"],
            "band_state_id": int(df.loc[first_idx, "tuned_band_state_id"]),
            "position_cap_current": float(df.loc[first_idx, "tuned_position_cap_current"]),
            "wheel_damping_scale": float(df.loc[first_idx, "tuned_wheel_damping_scale"]),
            "recenter_active": bool(df.loc[first_idx, "tuned_recenter_active"]),
            "pitch_rad": float(df.loc[first_idx, "pitch_x_rad"]),
            "pitch_deg": float(df.loc[first_idx, "pitch_x_rad"] * 180 / np.pi),
            "wheel_vel_mean_rad_s": float(df.loc[first_idx, "wheel_vel_mean_rad_s"]),
        }
        crossing_events.append(event)

        print(f"Threshold +/-{thresh:.2f} m crossed at step {first_idx}")
        print(f"  Error: {event['first_crossing_e_m']:.4f} m")
        print(f"  Band: {event['band_state']} (ID={event['band_state_id']})")
        print(f"  Position cap: {event['position_cap_current']:.1f} Nm")
        print(f"  Wheel damping scale: {event['wheel_damping_scale']:.2f}")
        print(f"  Recenter active: {event['recenter_active']}")
        print(f"  Pitch: {event['pitch_deg']:.3f} deg")
        print(f"  Wheel velocity: {event['wheel_vel_mean_rad_s']:.3f} rad/s")
        print()

# ========== Part 2: Windows 2-7 Analysis (steps 500-3500) ==========
print("\n=== Part 2: Windows 2-7 Deep Dive (Steps 500-3500) ===\n")

window_start = 500
window_end = 3500
df_problem = df.iloc[window_start:window_end].copy()
e_problem = e[window_start:window_end]
abs_e_problem = abs_e[window_start:window_end]

# Overall statistics
outside_0p08 = (abs_e_problem > 0.08).sum()
outside_0p08_pct = outside_0p08 / len(e_problem) * 100

print(f"Windows 2-7 ({window_start}-{window_end}):")
print(f"  Steps outside +/-0.08 m: {outside_0p08}/{len(e_problem)} ({outside_0p08_pct:.1f}%)")
print(f"  Max |e|: {abs_e_problem.max():.4f} m")
print(f"  Mean |e|: {abs_e_problem.mean():.4f} m")
print()

# Band state distribution during problem window
band_state_dist = df_problem["tuned_band_state"].value_counts().to_dict()
print("Band state distribution during problem window:")
for state, count in sorted(band_state_dist.items(), key=lambda x: x[1], reverse=True):
    pct = count / len(df_problem) * 100
    print(f"  {state}: {count} steps ({pct:.1f}%)")
print()

# Emergency band analysis
emergency_steps = df_problem[df_problem["tuned_band_state"] == "emergency"]
if len(emergency_steps) > 0:
    first_emergency = window_start + emergency_steps.index[0] - df_problem.index[0]
    print(f"Emergency band first entered at step {first_emergency} (global)")
    print(f"  Error at entry: {e[first_emergency]:.4f} m")
    print(f"  Emergency active: {len(emergency_steps)} / {len(df_problem)} steps ({len(emergency_steps)/len(df_problem)*100:.1f}%)")
    print()
else:
    print("Emergency band never entered during windows 2-7")
    print()

# ========== Part 3: Root Cause Analysis ==========
print("\n=== Part 3: Root Cause Analysis ===\n")

# Check if emergency entered too late
emergency_threshold = 0.12  # T5 emergency threshold
first_above_emergency = np.where(abs_e > emergency_threshold)[0]
if len(first_above_emergency) > 0:
    first_above_step = first_above_emergency[0]
    band_at_first_above = df.loc[first_above_step, "tuned_band_state"]
    print(f"First time |e| > {emergency_threshold:.2f} m: step {first_above_step}")
    print(f"  Band state at that moment: {band_at_first_above}")
    if band_at_first_above != "emergency":
        print(f"  ==> Emergency band NOT active when threshold crossed")
        print(f"  ==> Diagnosis: EMERGENCY_TOO_LATE")
    else:
        print(f"  ==> Emergency band WAS active")
    print()

# Check authority during problem window
print("Authority analysis during windows 2-7:")
print(f"  Position cap mean: {df_problem['tuned_position_cap_current'].mean():.2f} Nm")
print(f"  Position cap max: {df_problem['tuned_position_cap_current'].max():.2f} Nm")
print(f"  Wheel damping scale mean: {df_problem['tuned_wheel_damping_scale'].mean():.3f}")
print(f"  Wheel damping scale min: {df_problem['tuned_wheel_damping_scale'].min():.3f}")
print()

# Check if authority too weak
emergency_only = df_problem[df_problem["tuned_band_state"] == "emergency"]
if len(emergency_only) > 0:
    emergency_cap = emergency_only["tuned_position_cap_current"].mean()
    emergency_damping = emergency_only["tuned_wheel_damping_scale"].mean()
    print(f"During emergency band:")
    print(f"  Position cap: {emergency_cap:.1f} Nm")
    print(f"  Wheel damping scale: {emergency_damping:.3f}")

    if emergency_cap < 7.5:
        print(f"  ==> Emergency cap {emergency_cap:.1f} Nm may be too weak for high_0p480")
        print(f"  ==> Diagnosis: AUTHORITY_TOO_WEAK")
    print()

# Check pitch coupling
pitch_problem = df_problem["pitch_x_rad"].values * 180 / np.pi
pitch_rms_problem = np.sqrt((pitch_problem**2).mean())
print(f"Pitch during windows 2-7:")
print(f"  Pitch RMS: {pitch_rms_problem:.3f} deg")
print(f"  Pitch max: {pitch_problem.max():.3f} deg")
print(f"  Pitch min: {pitch_problem.min():.3f} deg")

if pitch_rms_problem > 5.0:
    print(f"  ==> Pitch RMS {pitch_rms_problem:.3f} deg is elevated")
    print(f"  ==> High-height gravitational torque may be dominating")
    print(f"  ==> Diagnosis contributes: PITCH_COUPLING_DOMINATES")
print()

# ========== Part 4: Window 7 vs Window 10 Comparison ==========
print("\n=== Part 4: Window 7 vs Window 10 Comparison ===\n")

# Window 7: steps 3000-3500 (worst)
w7_start, w7_end = 3000, 3500
df_w7 = df.iloc[w7_start:w7_end]
e_w7 = e[w7_start:w7_end]
abs_e_w7 = abs_e[w7_start:w7_end]

# Window 10: steps 4500-5000 (best)
w10_start, w10_end = 4500, 5000
df_w10 = df.iloc[w10_start:w10_end]
e_w10 = e[w10_start:w10_end]
abs_e_w10 = abs_e[w10_start:w10_end]

print("Window 7 (3000-3500) - WORST:")
print(f"  Outside +/-0.08 m: {(abs_e_w7 > 0.08).sum()}/{len(e_w7)} ({(abs_e_w7 > 0.08).sum()/len(e_w7)*100:.1f}%)")
print(f"  Mean |e|: {abs_e_w7.mean():.4f} m")
print(f"  Band state mode: {df_w7['tuned_band_state'].mode()[0]}")
print(f"  Position cap mean: {df_w7['tuned_position_cap_current'].mean():.2f} Nm")
print(f"  Wheel damping scale mean: {df_w7['tuned_wheel_damping_scale'].mean():.3f}")
print(f"  Pitch RMS: {np.sqrt((df_w7['pitch_x_rad']**2).mean()) * 180/np.pi:.3f} deg")
print()

print("Window 10 (4500-5000) - BEST:")
print(f"  Outside +/-0.08 m: {(abs_e_w10 > 0.08).sum()}/{len(e_w10)} ({(abs_e_w10 > 0.08).sum()/len(e_w10)*100:.1f}%)")
print(f"  Mean |e|: {abs_e_w10.mean():.4f} m")
print(f"  Band state mode: {df_w10['tuned_band_state'].mode()[0]}")
print(f"  Position cap mean: {df_w10['tuned_position_cap_current'].mean():.2f} Nm")
print(f"  Wheel damping scale mean: {df_w10['tuned_wheel_damping_scale'].mean():.3f}")
print(f"  Pitch RMS: {np.sqrt((df_w10['pitch_x_rad']**2).mean()) * 180/np.pi:.3f} deg")
print()

print("What changed:")
delta_mean_e = abs_e_w10.mean() - abs_e_w7.mean()
delta_pitch_rms = (np.sqrt((df_w10['pitch_x_rad']**2).mean()) - np.sqrt((df_w7['pitch_x_rad']**2).mean())) * 180/np.pi
print(f"  Mean |e| change: {delta_mean_e:.4f} m")
print(f"  Pitch RMS change: {delta_pitch_rms:.3f} deg")

if delta_mean_e < -0.05:
    print(f"  ==> Drift reduced significantly in window 10")
if delta_pitch_rms < -1.0:
    print(f"  ==> Pitch reduced in window 10, helping recovery")
print()

# ========== Part 5: Final Classification ==========
print("\n=== Part 5: Final Classification ===\n")

# Determine root cause
causes = []

# Check 1: Emergency too late
if len(first_above_emergency) > 0 and band_at_first_above != "emergency":
    causes.append("EMERGENCY_TOO_LATE")

# Check 2: Authority too weak
if len(emergency_only) > 0 and emergency_cap <= 7.0:
    causes.append("AUTHORITY_TOO_WEAK")

# Check 3: Damping too strong
if len(emergency_only) > 0 and emergency_damping >= 0.10:
    causes.append("DAMPING_TOO_STRONG")

# Check 4: Pitch coupling dominates
if pitch_rms_problem > 5.0:
    causes.append("PITCH_COUPLING_DOMINATES")

if len(causes) == 0:
    classification = "T5_HIGH_FAIL_INCONCLUSIVE"
elif len(causes) == 1:
    classification = f"T5_HIGH_FAIL_{causes[0]}"
else:
    classification = "T5_HIGH_FAIL_MIXED_CAUSES"

print(f"Root causes detected: {causes}")
print(f"Classification: {classification}")
print()

# ========== Save Results ==========
output_dir = "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing"

# Save JSON
audit_summary = {
    "classification": classification,
    "root_causes": causes,
    "threshold_crossings": crossing_events,
    "windows_2_7_stats": {
        "steps": f"{window_start}-{window_end}",
        "outside_0p08_pct": float(outside_0p08_pct),
        "max_abs_e_m": float(abs_e_problem.max()),
        "mean_abs_e_m": float(abs_e_problem.mean()),
        "band_state_distribution": {k: int(v) for k, v in band_state_dist.items()},
        "emergency_steps": int(len(emergency_steps)) if len(emergency_steps) > 0 else 0,
        "emergency_pct": float(len(emergency_steps) / len(df_problem) * 100) if len(emergency_steps) > 0 else 0.0,
    },
    "authority_analysis": {
        "position_cap_mean_Nm": float(df_problem['tuned_position_cap_current'].mean()),
        "position_cap_max_Nm": float(df_problem['tuned_position_cap_current'].max()),
        "wheel_damping_scale_mean": float(df_problem['tuned_wheel_damping_scale'].mean()),
        "wheel_damping_scale_min": float(df_problem['tuned_wheel_damping_scale'].min()),
        "emergency_cap_Nm": float(emergency_cap) if len(emergency_only) > 0 else None,
        "emergency_damping": float(emergency_damping) if len(emergency_only) > 0 else None,
    },
    "pitch_coupling": {
        "pitch_rms_deg": float(pitch_rms_problem),
        "pitch_max_deg": float(pitch_problem.max()),
        "pitch_min_deg": float(pitch_problem.min()),
    },
    "window_comparison": {
        "window_7": {
            "steps": f"{w7_start}-{w7_end}",
            "outside_0p08_pct": float((abs_e_w7 > 0.08).sum() / len(e_w7) * 100),
            "mean_abs_e_m": float(abs_e_w7.mean()),
            "pitch_rms_deg": float(np.sqrt((df_w7['pitch_x_rad']**2).mean()) * 180/np.pi),
        },
        "window_10": {
            "steps": f"{w10_start}-{w10_end}",
            "outside_0p08_pct": float((abs_e_w10 > 0.08).sum() / len(e_w10) * 100),
            "mean_abs_e_m": float(abs_e_w10.mean()),
            "pitch_rms_deg": float(np.sqrt((df_w10['pitch_x_rad']**2).mean()) * 180/np.pi),
        },
    }
}

json_path = f"{output_dir}/t5_high_0p480_band_failure_audit.json"
with open(json_path, "w") as f:
    json.dump(audit_summary, f, indent=2)
print(f"[OK] Audit summary saved to: {json_path}")

# Save events CSV
events_df = pd.DataFrame(crossing_events)
csv_out_path = f"{output_dir}/t5_high_0p480_band_failure_events.csv"
events_df.to_csv(csv_out_path, index=False)
print(f"[OK] Crossing events saved to: {csv_out_path}")

print("\n[OK] Phase 1 audit complete")
