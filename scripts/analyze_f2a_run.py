"""Analyze F2a hysteresis recenter results."""
import pandas as pd
import numpy as np

# Load F2a telemetry
f2a = pd.read_csv("outputs/step_e_extreme_support_fix_eval/f2a_low_0p300_500/telemetry.csv")

# Check signed support error (using the compensated sagittal error)
signed_error_col = "yaw_aware_sagittal_error_compensated_m"  # hip_yaw_comp_support_error_m equivalent
if signed_error_col not in f2a.columns:
    # Fallback to sagittal_position_error_m
    signed_error_col = "sagittal_position_error_m"

signed_error = f2a[signed_error_col].values

print("=" * 60)
print("F2a Hysteresis Recenter Analysis - 500-step Simulation")
print("=" * 60)

# Basic stats
positive_mask = signed_error > 0
negative_mask = signed_error < 0
zero_mask = signed_error == 0

print(f"\nSigned Support Error Statistics:")
print(f"  Mean: {np.mean(signed_error):.6f} m")
print(f"  Median: {np.median(signed_error):.6f} m")
print(f"  Std: {np.std(signed_error):.6f} m")
print(f"  Min: {np.min(signed_error):.6f} m")
print(f"  Max: {np.max(signed_error):.6f} m")
print(f"  RMS: {np.sqrt(np.mean(signed_error**2)):.6f} m")
print(f"  MAE: {np.mean(np.abs(signed_error)):.6f} m")

print(f"\nSign Distribution:")
print(f"  Positive: {positive_mask.sum()} ({100*positive_mask.mean():.1f}%)")
print(f"  Negative: {negative_mask.sum()} ({100*negative_mask.mean():.1f}%)")
print(f"  Zero: {zero_mask.sum()} ({100*zero_mask.mean():.1f}%)")

# Zero crossings
sign_changes = np.diff(np.sign(signed_error))
zero_crossings = np.sum(np.abs(sign_changes) >= 2)
print(f"\nZero Crossings: {zero_crossings}")

# Outside ±0.15 threshold
outside_positive = (signed_error > 0.15).sum()
outside_negative = (signed_error < -0.15).sum()
outside_total = ((signed_error > 0.15) | (signed_error < -0.15)).sum()
print(f"\nOutside ±0.15 m threshold:")
print(f"  Outside +0.15: {outside_positive} steps")
print(f"  Outside -0.15: {outside_negative} steps")
print(f"  Total outside: {outside_total} steps ({100*outside_total/len(signed_error):.1f}%)")

# Hysteresis state analysis
hyst_state = f2a["hysteresis_recenter_state"].values if "hysteresis_recenter_state" in f2a.columns else None
hyst_active = f2a["hysteresis_recenter_active"].values if "hysteresis_recenter_active" in f2a.columns else None
hyst_gate_reason = f2a["hysteresis_recenter_gate_reason"].values if "hysteresis_recenter_gate_reason" in f2a.columns else None
hyst_tau = f2a["hysteresis_recenter_tau"].values if "hysteresis_recenter_tau" in f2a.columns else None
hyst_state_entry = f2a["hysteresis_recenter_state_entry_count"].values if "hysteresis_recenter_state_entry_count" in f2a.columns else None
hyst_state_exit = f2a["hysteresis_recenter_state_exit_count"].values if "hysteresis_recenter_state_exit_count" in f2a.columns else None

print(f"\nHysteresis State Analysis:")
if hyst_state is not None:
    neutral_count = (hyst_state == "NEUTRAL").sum()
    pos_count = (hyst_state == "RECENTER_FROM_POSITIVE").sum()
    neg_count = (hyst_state == "RECENTER_FROM_NEGATIVE").sum()
    print(f"  NEUTRAL: {neutral_count} ({100*neutral_count/len(hyst_state):.1f}%)")
    print(f"  RECENTER_FROM_POSITIVE: {pos_count} ({100*pos_count/len(hyst_state):.1f}%)")
    print(f"  RECENTER_FROM_NEGATIVE: {neg_count} ({100*neg_count/len(hyst_state):.1f}%)")

if hyst_active is not None:
    active_count = hyst_active.sum()
    print(f"\nHysteresis Active: {active_count} steps ({100*active_count/len(hyst_active):.1f}%)")

if hyst_tau is not None:
    print(f"\nHysteresis Torque:")
    print(f"  Max: {np.max(hyst_tau):.6f} Nm")
    print(f"  Min: {np.min(hyst_tau):.6f} Nm")
    print(f"  RMS: {np.sqrt(np.mean(hyst_tau**2)):.6f} Nm")

if hyst_state_entry is not None:
    final_entry = hyst_state_entry[-1]
    final_exit = hyst_state_exit[-1] if hyst_state_exit is not None else 0
    print(f"\nState Transitions:")
    print(f"  Total Entries: {final_entry}")
    print(f"  Total Exits: {final_exit}")

# Phase recenter comparison (F1b-style)
phase_active = f2a["phase_recenter_active"].values if "phase_recenter_active" in f2a.columns else None
if phase_active is not None:
    phase_active_count = phase_active.sum()
    print(f"\nPhase Recenter (F1-style) Active: {phase_active_count} steps ({100*phase_active_count/len(phase_active):.1f}%)")

# Longest same-sign intervals
def longest_same_sign_interval(arr):
    if len(arr) == 0:
        return 0, 0, 0
    signs = np.sign(arr)
    if np.all(signs == 0):
        return 0, 0, 0
    # Replace 0 with previous non-zero sign for continuity
    signs_nonzero = signs[signs != 0]
    if len(signs_nonzero) == 0:
        return 0, 0, 0
    first_sign = signs_nonzero[0]
    longest_pos = 0
    longest_neg = 0
    current_pos = 0
    current_neg = 0
    for s in signs:
        if s > 0:
            current_pos += 1
            current_neg = 0
            longest_pos = max(longest_pos, current_pos)
        elif s < 0:
            current_neg += 1
            current_pos = 0
            longest_neg = max(longest_neg, current_neg)
        else:
            current_pos = 0
            current_neg = 0
    return longest_pos, longest_neg, max(longest_pos, longest_neg)

longest_pos, longest_neg, longest_total = longest_same_sign_interval(signed_error)
print(f"\nLongest Same-Sign Intervals:")
print(f"  Longest Positive: {longest_pos} steps")
print(f"  Longest Negative: {longest_neg} steps")
print(f"  Longest Total: {longest_total} steps")

# Stability checks
print(f"\nStability Checks:")
print(f"  Survived 500 steps: {len(f2a) == 500}")
contact_valid = f2a["contact_force_valid"].values if "contact_force_valid" in f2a.columns else None
if contact_valid is not None:
    print(f"  Contact Valid: {contact_valid.mean()*100:.1f}%")

com_z = f2a["com_z"].values if "com_z" in f2a.columns else None
if com_z is not None:
    print(f"  Height: {np.min(com_z):.3f} - {np.max(com_z):.3f} m")

print(f"\nComparison with D2 and F1b:")
print(f"  D2 positive%: 93.0%")
print(f"  F1b positive%: 82.8%")
print(f"  F2a positive%: {100*positive_mask.mean():.1f}%")
print(f"\n  D2 outside +0.15: 96 steps")
print(f"  F1b outside +0.15: 81 steps")
print(f"  F2a outside +0.15: {outside_positive} steps")
print(f"\n  D2 zero crossings: 4")
print(f"  F1b zero crossings: 5")
print(f"  F2a zero crossings: {zero_crossings}")
print(f"\n  F1b recenter active: 65.8%")
if hyst_active is not None:
    print(f"  F2a hysteresis active: {100*active_count/len(hyst_active):.1f}%")

# Save to JSON
import json
results = {
    "profile": "F2a_hysteresis_recenter_moderate",
    "steps": 500,
    "survived": len(f2a) == 500,
    "signed_support": {
        "mean": float(np.mean(signed_error)),
        "median": float(np.median(signed_error)),
        "std": float(np.std(signed_error)),
        "min": float(np.min(signed_error)),
        "max": float(np.max(signed_error)),
        "positive_percent": float(100*positive_mask.mean()),
        "negative_percent": float(100*negative_mask.mean()),
        "zero_crossings": int(zero_crossings),
        "outside_positive_0p15": int(outside_positive),
        "outside_negative_0p15": int(outside_negative),
        "outside_total_0p15": int(outside_total),
        "longest_positive_interval": int(longest_pos),
        "longest_negative_interval": int(longest_neg),
    },
    "hysteresis": {
        "enabled": bool(f2a["hysteresis_recenter_enabled"].iloc[0]) if "hysteresis_recenter_enabled" in f2a.columns else None,
        "active_percent": float(100*active_count/len(hyst_active)) if hyst_active is not None else None,
        "neutral_percent": float(100*neutral_count/len(hyst_state)) if hyst_state is not None else None,
        "recenter_from_positive_percent": float(100*pos_count/len(hyst_state)) if hyst_state is not None else None,
        "recenter_from_negative_percent": float(100*neg_count/len(hyst_state)) if hyst_state is not None else None,
        "state_entries": int(final_entry) if hyst_state_entry is not None else None,
        "state_exits": int(final_exit) if hyst_state_exit is not None else None,
        "tau_max": float(np.max(hyst_tau)) if hyst_tau is not None else None,
        "tau_min": float(np.min(hyst_tau)) if hyst_tau is not None else None,
        "tau_rms": float(np.sqrt(np.mean(hyst_tau**2))) if hyst_tau is not None else None,
    },
    "comparison": {
        "d2_positive_percent": 93.0,
        "f1b_positive_percent": 82.8,
        "f2a_positive_percent": float(100*positive_mask.mean()),
        "d2_outside_positive_0p15": 96,
        "f1b_outside_positive_0p15": 81,
        "f2a_outside_positive_0p15": int(outside_positive),
        "d2_zero_crossings": 4,
        "f1b_zero_crossings": 5,
        "f2a_zero_crossings": int(zero_crossings),
        "f1b_recenter_active_percent": 65.8,
    }
}

with open("outputs/step_e_extreme_support_fix_eval/f2a_low_0p300_500/results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to outputs/step_e_extreme_support_fix_eval/f2a_low_0p300_500/results.json")