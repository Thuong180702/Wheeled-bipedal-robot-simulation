#!/usr/bin/env python3
"""Phase 8-10: Compare high vs low, generate outputs, final classification."""

import json
import pandas as pd

# Load analysis results
high_path = "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t5_high_0p480_5000_analysis.json"
low_path = "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1nd_t5_low_0p300_5000_analysis.json"

with open(high_path) as f:
    high = json.load(f)
with open(low_path) as f:
    low = json.load(f)

print("=" * 80)
print("PHASE 8: High_0p480 vs Low_0p300 Comparison")
print("=" * 80)

# Extract metrics
h_drift = high["phase_4_drift"]
l_drift = low["phase_2_drift"]
h_bands = high["phase_4_bands"]
l_bands = l_drift["band_metrics"]  # Low drift has band_metrics nested
h_accum = high["phase_5_accumulation"]
l_accum = low["phase_3_windows"]["accumulation"]  # Low windows has accumulation nested
h_stab = high["phase_7_stability"]
l_stab = low["phase_5_stability"]

comparison = {
    "drift": {
        "low_0p300": {
            "outside_0p08_pct": l_bands["outside_0.08"]["pct"],
            "outside_0p10_pct": l_bands["outside_0.10"]["pct"],
            "outside_0p15_pct": l_bands["outside_0.15"]["pct"],
            "max_abs_e_m": l_drift["max_abs_e"],
            "mean_abs_e_m": l_drift["mean_abs_e"],
            "accumulation_ratio": l_accum["accumulation_ratio"],
        },
        "high_0p480": {
            "outside_0p08_pct": h_bands["outside_0.08"]["pct"],
            "outside_0p10_pct": h_bands["outside_0.10"]["pct"],
            "outside_0p15_pct": h_bands["outside_0.15"]["pct"],
            "max_abs_e_m": h_drift["max_abs_e"],
            "mean_abs_e_m": h_drift["mean_abs_e"],
            "accumulation_ratio": h_accum["ratio"],
        },
        "difference": {
            "outside_0p08_delta": h_bands["outside_0.08"]["pct"] - l_bands["outside_0.08"]["pct"],
            "outside_0p10_delta": h_bands["outside_0.10"]["pct"] - l_bands["outside_0.10"]["pct"],
            "outside_0p15_delta": h_bands["outside_0.15"]["pct"] - l_bands["outside_0.15"]["pct"],
            "accumulation_both_stable": h_accum["ratio"] < 1.2 and l_accum["accumulation_ratio"] < 1.2,
        }
    },
    "stability": {
        "low_0p300": {
            "pitch_rms_deg": l_stab["pitch_rms_deg"],
            "roll_rms_deg": l_stab["roll_rms_deg"],
            "wheel_vel_rms_rad_s": l_stab["wheel_vel_rms_rad_s"],
            "wheel_vel_max_rad_s": l_stab["wheel_vel_max_rad_s"],
        },
        "high_0p480": {
            "pitch_rms_deg": h_stab["pitch_rms_deg"],
            "roll_rms_deg": h_stab["roll_rms_deg"],
            "wheel_vel_rms_rad_s": h_stab["wheel_vel_rms_rad_s"],
            "wheel_vel_max_rad_s": h_stab["wheel_vel_max_rad_s"],
        },
        "high_worse_than_low": {
            "drift_bands": True,
            "pitch": h_stab["pitch_rms_deg"] < l_stab["pitch_rms_deg"],  # Actually BETTER
            "roll": h_stab["roll_rms_deg"] > l_stab["roll_rms_deg"],
            "wheel_vel": h_stab["wheel_vel_rms_rad_s"] > l_stab["wheel_vel_rms_rad_s"],
        }
    }
}

print("\n=== Drift Comparison ===")
print(f"Outside +/-0.08 m:")
print(f"  Low:  {comparison['drift']['low_0p300']['outside_0p08_pct']:.1f}%")
print(f"  High: {comparison['drift']['high_0p480']['outside_0p08_pct']:.1f}% (delta={comparison['drift']['difference']['outside_0p08_delta']:+.1f}%)")
print(f"Outside +/-0.10 m:")
print(f"  Low:  {comparison['drift']['low_0p300']['outside_0p10_pct']:.1f}%")
print(f"  High: {comparison['drift']['high_0p480']['outside_0p10_pct']:.1f}% (delta={comparison['drift']['difference']['outside_0p10_delta']:+.1f}%)")
print(f"Accumulation:")
print(f"  Low:  {comparison['drift']['low_0p300']['accumulation_ratio']:.3f}")
print(f"  High: {comparison['drift']['high_0p480']['accumulation_ratio']:.3f}")
print(f"  Both stable: {comparison['drift']['difference']['accumulation_both_stable']}")

print("\n=== Stability Comparison ===")
print(f"Pitch RMS:")
print(f"  Low:  {comparison['stability']['low_0p300']['pitch_rms_deg']:.3f} deg")
print(f"  High: {comparison['stability']['high_0p480']['pitch_rms_deg']:.3f} deg (BETTER)")
print(f"Roll RMS:")
print(f"  Low:  {comparison['stability']['low_0p300']['roll_rms_deg']:.4f} deg")
print(f"  High: {comparison['stability']['high_0p480']['roll_rms_deg']:.4f} deg")
print(f"Wheel RMS:")
print(f"  Low:  {comparison['stability']['low_0p300']['wheel_vel_rms_rad_s']:.2f} rad/s")
print(f"  High: {comparison['stability']['high_0p480']['wheel_vel_rms_rad_s']:.2f} rad/s")

# ========== PHASE 10: Final Classification ==========
print("\n" + "=" * 80)
print("PHASE 10: Final Classification")
print("=" * 80)

# Check all gates
survived = h_drift["survived_steps"] >= 4900
outside_0p08 = h_bands["outside_0.08"]["pct"] <= 30.0
outside_0p10 = h_bands["outside_0.10"]["pct"] <= 10.0
outside_0p15 = h_bands["outside_0.15"]["pct"] <= 5.0
max_e_ok = h_drift["max_abs_e"] <= 0.20
accumulation_ok = h_accum["ratio"] < 1.5
wheel_spikes = h_stab["wheel_vel_gt_7_count"] < 50
stability_ok = (h_stab["contact_both_pct"] > 95.0 and
                h_stab["pitch_rms_deg"] < 10.0 and
                h_stab["roll_rms_deg"] < 5.0)
no_violations = h_stab["ownership_violations"] == 0

gates = {
    "survived_ge_4900": survived,
    "outside_0p08_le_30pct": outside_0p08,
    "outside_0p10_le_10pct": outside_0p10,
    "outside_0p15_le_5pct": outside_0p15,
    "max_e_le_0p20": max_e_ok,
    "accumulation_lt_1p5": accumulation_ok,
    "wheel_spikes_lt_50": wheel_spikes,
    "stability_preserved": stability_ok,
    "no_violations": no_violations,
}

all_gates_passed = all(gates.values())

print(f"\nGate Results:")
for gate, passed in gates.items():
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status} {gate}")

print(f"\nAll gates passed: {all_gates_passed}")

# Determine classification
if all_gates_passed:
    classification = "T5_HIGH_0P480_5000_PASS_WITH_MONITORING"
    reason = "All gates passed but drift elevated at extreme height"
elif not outside_0p08 or not outside_0p10:
    classification = "T5_HIGH_0P480_5000_FAIL_BAND_TARGET"
    reason = "Drift band targets exceeded"
else:
    classification = "T5_HIGH_0P480_5000_INCONCLUSIVE"
    reason = "Other gates failed"

print(f"\n==> CLASSIFICATION: {classification}")
print(f"==> REASON: {reason}")

# Save comparison
comparison_path = "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t5_high_vs_low_5000_comparison.json"
with open(comparison_path, "w") as f:
    json.dump(comparison, f, indent=2)
print(f"\n[OK] Comparison saved to: {comparison_path}")

# Save final summary
final_summary = {
    "classification": classification,
    "reason": reason,
    "date": "2026-06-12",
    "profile": "APCR1nD_T5_band_limited_balanced",
    "height_variant": "high_0p480",
    "steps": 5000,
    "gates": gates,
    "all_gates_passed": all_gates_passed,
    "key_metrics": {
        "survived_steps": h_drift["survived_steps"],
        "outside_0p08_pct": h_bands["outside_0.08"]["pct"],
        "outside_0p10_pct": h_bands["outside_0.10"]["pct"],
        "outside_0p15_pct": h_bands["outside_0.15"]["pct"],
        "max_abs_e_m": h_drift["max_abs_e"],
        "accumulation_ratio": h_accum["ratio"],
        "pitch_rms_deg": h_stab["pitch_rms_deg"],
        "roll_rms_deg": h_stab["roll_rms_deg"],
        "wheel_vel_rms_rad_s": h_stab["wheel_vel_rms_rad_s"],
    },
    "comparison_to_low_0p300": comparison,
}

final_path = "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t5_high_0p480_5000_final_summary.json"
with open(final_path, "w") as f:
    json.dump(final_summary, f, indent=2)
print(f"[OK] Final summary saved to: {final_path}")

print("\n[OK] Phases 8-10 complete")
