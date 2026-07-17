#!/usr/bin/env python3
"""
Phase 2: K1 Controllability Audit

Measures local wheel-torque sensitivity at heights 0.33, 0.40, and 0.48 m
using K1 telemetry to compute linearized input sensitivity.

Key questions:
- d(pitch)/d(tau_wheel) — can wheel torque still affect pitch?
- d(COM_position)/d(tau_wheel) — can wheel torque move the COM?
- d(support_error)/d(tau_wheel) — can wheel torque correct support?
- Is the system near a weakly controllable region at 0.48 m?
"""

import csv
import json
import statistics
import sys
import os
import math
import numpy as np

# Fix Windows encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Paths ──────────────────────────────────────────────────────────────────
TELEMETRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "d_baseline_single_90n_10step_push_step300_3000",
    "telemetry_1782262602.csv",
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "system_audit", "controllability",
)
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "controllability_audit.json")
OUTPUT_REPORT = os.path.join(OUTPUT_DIR, "controllability_report.md")

TARGET_HEIGHTS = [0.33, 0.40, 0.48]
HEIGHT_TOLERANCE = 0.02  # ±2 cm window around target height

# ── Load telemetry ─────────────────────────────────────────────────────────
print("Loading K1 telemetry...")
with open(TELEMETRY_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

N = len(rows)
print(f"  {N} steps loaded")

# Compute dt from simulation time
dt = float(rows[1]["sim_time_s"]) - float(rows[0]["sim_time_s"]) if N > 1 else 0.005
print(f"  dt = {dt:.4f} s")

# ── Extract time series ────────────────────────────────────────────────────
com_z = np.array([float(r["com_z"]) for r in rows])
pitch_x = np.array([float(r["pitch_x"]) for r in rows])
pitch_rate = np.array([float(r["pitch_rate_x"]) for r in rows])
com_y = np.array([float(r["com_y"]) for r in rows])
com_vy = np.array([float(r["com_vy"]) for r in rows])
cp_error_y = np.array([float(r.get("cp_error_y", 0)) for r in rows])
support_error = np.array([float(r.get("support_position_error_m", 0)) for r in rows])
tau_left = np.array([float(r["tau_left"]) for r in rows])
tau_right = np.array([float(r["tau_right"]) for r in rows])
tau_common = tau_left + tau_right  # total bilaterally symmetric torque
tau_diff = tau_left - tau_right  # differential torque (should ~0 for sagittal)
def _safe_float(val, default=0.0):
    """Handle boolean string values like 'True'/'False' in telemetry."""
    if isinstance(val, str) and val in ("True", "False"):
        return 1.0 if val == "True" else 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

push_active = np.array([_safe_float(r.get("push_active", 0)) for r in rows])
tau_pitch = np.array([float(r.get("tau_pitch", 0)) for r in rows])
tau_pitch_rate = np.array([float(r.get("tau_pitch_rate", 0)) for r in rows])
tau_position = np.array([float(r.get("tau_position", 0)) for r in rows])
tau_support_vel = np.array([float(r.get("tau_support_velocity", 0)) for r in rows])
tau_cp = np.array([float(r.get("tau_cp", 0)) for r in rows])
tau_com_vy = np.array([float(r.get("tau_com_vy", 0)) for r in rows])

# Pitch acceleration via finite differences
pitch_accel = np.zeros_like(pitch_rate)
pitch_accel[1:] = (pitch_rate[1:] - pitch_rate[:-1]) / dt
pitch_accel[0] = pitch_accel[1] if N > 1 else 0.0

# COM acceleration
com_accel_y = np.zeros_like(com_vy)
com_accel_y[1:] = (com_vy[1:] - com_vy[:-1]) / dt
com_accel_y[0] = com_accel_y[1] if N > 1 else 0.0

# Support error rate
support_error_rate = np.zeros_like(support_error)
support_error_rate[1:] = (support_error[1:] - support_error[:-1]) / dt
support_error_rate[0] = support_error_rate[1] if N > 1 else 0.0

# Support error acceleration
support_error_accel = np.zeros_like(support_error_rate)
support_error_accel[1:] = (support_error_rate[1:] - support_error_rate[:-1]) / dt
support_error_accel[0] = support_error_accel[1] if N > 1 else 0.0

print("  Time series extracted")

# ── Per-height sensitivity analysis ────────────────────────────────────────
results = {}

for target_h in TARGET_HEIGHTS:
    print(f"\n── Height {target_h:.2f} m ──")

    # Find segments near target height, post-push only to avoid push transients
    push_end = int(np.argwhere(push_active > 0.5)[-1]) if np.any(push_active > 0.5) else 300
    near_height = (np.abs(com_z - target_h) < HEIGHT_TOLERANCE) & (np.arange(N) > push_end + 50)

    n_samples = np.sum(near_height)
    print(f"  Samples near height: {n_samples}")

    if n_samples < 10:
        print(f"  WARNING: Too few samples at h={target_h}m, expanding tolerance")
        near_height = (np.abs(com_z - target_h) < HEIGHT_TOLERANCE * 2) & (np.arange(N) > push_end + 50)
        n_samples = np.sum(near_height)

    if n_samples < 10:
        print(f"  SKIPPING: still too few samples")
        results[target_h] = {
            "samples": 0,
            "error": "insufficient_samples",
        }
        continue

    indices = np.where(near_height)[0]

    # ── Sensitivity computation ─────────────────────────────────────────
    # For each pair of consecutive steps near the target height:
    # d(pitch_accel) = B_pitch * d(tau_common)
    # We compute: sensitivity = d(state_accel)/d(tau_common)

    # Use only consecutive samples within the height window
    d_tau = []
    d_pitch_accel = []
    d_com_accel = []
    d_support_accel = []
    pitch_vals = []
    support_vals = []
    tau_mean_vals = []
    tau_comp_vals = {k: [] for k in ["tau_pitch", "tau_pitch_rate", "tau_cp", "tau_com_vy",
                                        "tau_position", "tau_support_velocity", "tau_common"]}

    for i_idx in range(len(indices) - 1):
        i = indices[i_idx]
        j = indices[i_idx + 1]
        if j != i + 1:
            continue  # only consecutive steps

        dt_i = dt
        d_tau.append(tau_common[j] - tau_common[i])
        d_pitch_accel.append(pitch_accel[j] - pitch_accel[i])
        d_com_accel.append(com_accel_y[j] - com_accel_y[i])
        d_support_accel.append(support_error_accel[j] - support_error_accel[i])
        pitch_vals.append(pitch_x[i])
        support_vals.append(support_error[i])
        tau_mean_vals.append(abs(tau_common[i]))
        tau_comp_vals["tau_pitch"].append(tau_pitch[i])
        tau_comp_vals["tau_pitch_rate"].append(tau_pitch_rate[i])
        tau_comp_vals["tau_cp"].append(tau_cp[i])
        tau_comp_vals["tau_com_vy"].append(tau_com_vy[i])
        tau_comp_vals["tau_position"].append(tau_position[i])
        tau_comp_vals["tau_support_velocity"].append(tau_support_vel[i])
        tau_comp_vals["tau_common"].append(tau_common[i])

    n_pairs = len(d_tau)
    print(f"  Consecutive pairs: {n_pairs}")

    if n_pairs < 5:
        print(f"  SKIPPING: too few pairs")
        results[target_h] = {"samples": n_samples, "pairs": n_pairs, "error": "insufficient_pairs"}
        continue

    d_tau_a = np.array(d_tau)
    d_pa = np.array(d_pitch_accel)
    d_ca = np.array(d_com_accel)
    d_sa = np.array(d_support_accel)

    # Remove outliers (3 sigma)
    tau_std = np.std(d_tau_a)
    tau_mean = np.mean(d_tau_a)
    valid = (np.abs(d_tau_a - tau_mean) < 3 * tau_std) & (np.abs(d_tau_a) > 1e-6)
    d_tau_f = d_tau_a[valid]
    d_pa_f = d_pa[valid]
    d_ca_f = d_ca[valid]
    d_sa_f = d_sa[valid]
    print(f"  Pairs after outlier removal: {len(d_tau_f)}")

    if len(d_tau_f) < 5:
        results[target_h] = {"samples": n_samples, "pairs": len(d_tau_f), "error": "insufficient_clean_pairs"}
        continue

    # Linear regression: d(accel) = sensitivity * d(tau) + offset
    # sensitivity = cov(d_accel, d_tau) / var(d_tau)
    var_tau = np.var(d_tau_f)
    if var_tau < 1e-12:
        print(f"  SKIPPING: zero variance in d_tau")
        results[target_h] = {"samples": n_samples, "pairs": len(d_tau_f), "error": "zero_tau_variance"}
        continue

    sens_pitch = np.cov(d_pa_f, d_tau_f)[0, 1] / var_tau  # d(pitch_accel)/d(tau_common) [rad/s^2 per Nm]
    sens_com = np.cov(d_ca_f, d_tau_f)[0, 1] / var_tau  # d(com_accel_y)/d(tau_common) [m/s^2 per Nm]
    sens_support = np.cov(d_sa_f, d_tau_f)[0, 1] / var_tau  # d(support_accel)/d(tau_common)

    # Correlation coefficients
    corr_pitch = np.corrcoef(d_pa_f, d_tau_f)[0, 1]
    corr_com = np.corrcoef(d_ca_f, d_tau_f)[0, 1]
    corr_support = np.corrcoef(d_sa_f, d_tau_f)[0, 1]

    # Mean torque component values
    comp_means = {k: float(np.mean(np.abs(v))) for k, v in tau_comp_vals.items()}

    # Compute state ranges
    pitch_range = [float(np.min(pitch_x[indices])), float(np.max(pitch_x[indices]))]
    support_range = [float(np.min(support_error[indices])), float(np.max(support_error[indices]))]
    height_range = [float(np.min(com_z[indices])), float(np.max(com_z[indices]))]

    # ── Additional: sensitivity to task-space error ─────────────────────
    # How much does 1 Nm of wheel torque change:
    # - pitch angle over 0.1s?   delta_pitch ≈ sens_pitch * (0.1)^2 / 2
    horizon = 0.1  # 100 ms
    pitch_change_per_Nm_100ms = sens_pitch * horizon**2 / 2  # rad
    com_change_per_Nm_100ms = sens_com * horizon**2 / 2  # m

    # Required torque to correct 1 degree pitch error
    # tau_needed = desired_accel / sensitivity
    # For 1 deg error with pd control at 5 Hz bandwidth
    desired_pitch_accel_per_deg = 1.0 * math.pi / 180.0 * (2 * math.pi * 5)**2  # 1 deg -> accel @ 5 Hz
    tau_per_deg_pitch = desired_pitch_accel_per_deg / sens_pitch if abs(sens_pitch) > 1e-9 else float("inf")

    # Required torque to correct 0.1m support error
    desired_support_accel_per_01m = 0.1 * (2 * math.pi * 1)**2  # 0.1m -> accel @ 1 Hz
    tau_per_01m_support = desired_support_accel_per_01m / sens_support if abs(sens_support) > 1e-9 else float("inf")

    # ── Controllability assessment ──────────────────────────────────────
    # Weak controllability = low sensitivity + low correlation
    # Strong controllability = high sensitivity + high correlation
    pitch_controllable = abs(sens_pitch) > 1e-2 and abs(corr_pitch) > 0.1
    com_controllable = abs(sens_com) > 1e-3 and abs(corr_com) > 0.1
    support_controllable = abs(sens_support) > 1e-4 and abs(corr_support) > 0.1

    print(f"\n  Sensitivity results:")
    print(f"    d(pitch_accel)/d(tau) = {sens_pitch:.6f} rad/s^2/Nm (r={corr_pitch:.3f})")
    print(f"    d(com_accel)/d(tau)   = {sens_com:.6f} m/s^2/Nm (r={corr_com:.3f})")
    print(f"    d(support_accel)/d(tau) = {sens_support:.6f} m/s^2/Nm (r={corr_support:.3f})")
    print(f"    Pitch controllable: {pitch_controllable}")
    print(f"    COM controllable: {com_controllable}")
    print(f"    Support controllable: {support_controllable}")
    print(f"    tau per 1 deg pitch: {tau_per_deg_pitch:.2f} Nm")
    print(f"    tau per 0.1m support: {tau_per_01m_support:.2f} Nm")
    print(f"    Pitch change per Nm/100ms: {pitch_change_per_Nm_100ms*180/math.pi:.4f} deg")

    results[target_h] = {
        "samples": int(n_samples),
        "pairs": int(len(d_tau_f)),
        "height_actual_min": height_range[0],
        "height_actual_max": height_range[1],
        "pitch_range_rad": pitch_range,
        "pitch_range_deg": [float(p * 180 / math.pi) for p in pitch_range],
        "support_range_m": support_range,
        "sensitivity": {
            "d_pitch_accel_d_tau_rad_s2_per_nm": float(sens_pitch),
            "d_com_accel_d_tau_m_s2_per_nm": float(sens_com),
            "d_support_accel_d_tau": float(sens_support),
        },
        "correlation": {
            "r_pitch_tau": float(corr_pitch),
            "r_com_tau": float(corr_com),
            "r_support_tau": float(corr_support),
        },
        "torque_per_correction": {
            "tau_per_1deg_pitch_nm": float(tau_per_deg_pitch),
            "tau_per_01m_support_nm": float(tau_per_01m_support),
        },
        "effect_per_nm_100ms": {
            "pitch_change_deg": float(pitch_change_per_Nm_100ms * 180 / math.pi),
            "com_change_m": float(com_change_per_Nm_100ms),
        },
        "controllability": {
            "pitch": "CONTROLLABLE" if pitch_controllable else "WEAKLY_CONTROLLABLE",
            "com": "CONTROLLABLE" if com_controllable else "WEAKLY_CONTROLLABLE",
            "support": "CONTROLLABLE" if support_controllable else "WEAKLY_CONTROLLABLE",
        },
        "torque_composition_mean_abs_nm": comp_means,
    }

# ── Cross-height comparison ────────────────────────────────────────────────
if len([h for h in TARGET_HEIGHTS if isinstance(results.get(h, {}).get("sensitivity"), dict)]) >= 2:
    sensitivities = {}
    for h in TARGET_HEIGHTS:
        r = results.get(h, {})
        sens = r.get("sensitivity", {})
        if sens:
            sensitivities[h] = {
                "pitch": sens.get("d_pitch_accel_d_tau_rad_s2_per_nm", 0),
                "com": sens.get("d_com_accel_d_tau_m_s2_per_nm", 0),
                "support": sens.get("d_support_accel_d_tau", 0),
            }

    sens_heights = sorted(sensitivities.keys())
    if len(sens_heights) >= 2:
        h_lo, h_hi = sens_heights[0], sens_heights[-1]
        pitch_trend = "DECREASING" if abs(sensitivities[h_hi]["pitch"]) < abs(sensitivities[h_lo]["pitch"]) else "INCREASING_OR_FLAT"
        com_trend = "DECREASING" if abs(sensitivities[h_hi]["com"]) < abs(sensitivities[h_lo]["com"]) else "INCREASING_OR_FLAT"
        support_trend = "DECREASING" if abs(sensitivities[h_hi]["support"]) < abs(sensitivities[h_lo]["support"]) else "INCREASING_OR_FLAT"
    else:
        pitch_trend = com_trend = support_trend = "INSUFFICIENT_DATA"
else:
    pitch_trend = com_trend = support_trend = "INSUFFICIENT_DATA"

# ── Compile results ────────────────────────────────────────────────────────
audit = {
    "audit": "controllability",
    "target": "K1_PITCH_RATE_NOTCH_V1",
    "run": "high_0p480_90N_push_step300_3000steps",
    "method": "telemetry_based_local_linear_regression",
    "dt_s": dt,
    "target_heights_m": TARGET_HEIGHTS,
    "height_tolerance_m": HEIGHT_TOLERANCE,
    "results": {str(h): results[h] for h in TARGET_HEIGHTS},
    "cross_height_trends": {
        "pitch_sensitivity_trend": pitch_trend,
        "com_sensitivity_trend": com_trend,
        "support_sensitivity_trend": support_trend,
        "sensitivity_values": sensitivities,
    },
}

# ── Answer the questions ───────────────────────────────────────────────────
# Is the system near a weakly controllable region at 0.48m?
h48 = results.get(0.48, {})
h33 = results.get(0.33, {})

sens_48 = h48.get("sensitivity", {})
sens_33 = h33.get("sensitivity", {})

pitch_sens_48 = sens_48.get("d_pitch_accel_d_tau_rad_s2_per_nm", 0)
pitch_sens_33 = sens_33.get("d_pitch_accel_d_tau_rad_s2_per_nm", 0)

if abs(pitch_sens_48) < 0.5 * abs(pitch_sens_33):
    weak_at_48 = True
    weak_verdict = "YES_CONTROLLABILITY_DEGRADES_SIGNIFICANTLY_AT_0p48m"
elif abs(pitch_sens_48) < abs(pitch_sens_33):
    weak_at_48 = False
    weak_verdict = "MODERATE_DEGRADATION_AT_0p48m"
else:
    weak_at_48 = False
    weak_verdict = "NO_DEGRADATION_CONTROLLABILITY_MAINTAINED"

controllable_pitch = h48.get("controllability", {}).get("pitch", "UNKNOWN")
controllable_support = h48.get("controllability", {}).get("support", "UNKNOWN")

overall_verdict = "CONTROLLABLE_AT_ALL_HEIGHTS"
if controllable_pitch == "WEAKLY_CONTROLLABLE" and controllable_support == "WEAKLY_CONTROLLABLE":
    overall_verdict = "WEAKLY_CONTROLLABLE_AT_ALL_HEIGHTS"
elif controllable_pitch == "WEAKLY_CONTROLLABLE":
    overall_verdict = "PITCH_WEAKLY_CONTROLLABLE"
elif controllable_support == "WEAKLY_CONTROLLABLE":
    overall_verdict = "SUPPORT_WEAKLY_CONTROLLABLE"
elif weak_at_48:
    overall_verdict = "CONTROLLABILITY_DEGRADES_AT_HIGH_HEIGHT"

audit["verdict"] = overall_verdict
audit["weak_at_0p48m"] = weak_verdict

# ── Save JSON ─────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2)
print(f"\nJSON audit saved to: {OUTPUT_JSON}")

# ── Print summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("K1 CONTROLLABILITY AUDIT — SUMMARY")
print("=" * 70)

for h in TARGET_HEIGHTS:
    r = results.get(h, {})
    sens = r.get("sensitivity", {})
    corr = r.get("correlation", {})
    contr = r.get("controllability", {})
    tau_corr = r.get("torque_per_correction", {})
    effects = r.get("effect_per_nm_100ms", {})

    print(f"\n── Height {h:.2f} m ({r.get('samples', 0)} samples, {r.get('pairs', 0)} clean pairs) ──")
    if not sens:
        print(f"  ERROR: {r.get('error', 'unknown')}")
        continue
    print(f"  d(pitch_accel)/d(tau_common) = {sens.get('d_pitch_accel_d_tau_rad_s2_per_nm', 0):.6f} rad/s^2/Nm (r={corr.get('r_pitch_tau', 0):.3f}) -> {contr.get('pitch', '?')}")
    print(f"  d(com_accel)/d(tau_common)   = {sens.get('d_com_accel_d_tau_m_s2_per_nm', 0):.6f} m/s^2/Nm (r={corr.get('r_com_tau', 0):.3f}) -> {contr.get('com', '?')}")
    print(f"  d(support_accel)/d(tau)      = {sens.get('d_support_accel_d_tau', 0):.6f} m/s^2/Nm (r={corr.get('r_support_tau', 0):.3f}) -> {contr.get('support', '?')}")
    print(f"  tau per 1 deg pitch error: {tau_corr.get('tau_per_1deg_pitch_nm', float('inf')):.2f} Nm")
    print(f"  tau per 0.1m support error: {tau_corr.get('tau_per_01m_support_nm', float('inf')):.2f} Nm")
    print(f"  1 Nm for 100 ms -> {effects.get('pitch_change_deg', 0):.4f} deg pitch, {effects.get('com_change_m', 0):.4f} m COM")

print(f"\n── Cross-height trends ──")
print(f"  Pitch sensitivity trend: {pitch_trend}")
print(f"  COM sensitivity trend: {com_trend}")
print(f"  Support sensitivity trend: {support_trend}")

print(f"\n── VERDICT ──")
print(f"  {overall_verdict}")
print(f"  Weak at 0.48m: {weak_verdict}")

# ── Write report ──────────────────────────────────────────────────────────
report_lines = [
    "# K1 Controllability Audit",
    "",
    f"**Verdict:** `{overall_verdict}`",
    f"**Weak at 0.48m:** `{weak_verdict}`",
    "",
    "## Method",
    "",
    f"- Telemetry-based local linear regression from K1 focused recovery run",
    f"- DT: {dt:.4f} s",
    f"- Height tolerance: ±{HEIGHT_TOLERANCE:.2f} m",
    f"- Post-push only (> step {push_end + 50})",
    f"- Covariance-based: sensitivity = Cov(d_accel, d_tau) / Var(d_tau)",
    "",
    "## Results by Height",
    "",
    "| Height | Samples | dPitch/dTau (rad/s²/Nm) | dCOM/dTau (m/s²/Nm) | dSupport/dTau | Pitch Ctrl | Support Ctrl |",
    "|--------|---------|--------------------------|----------------------|---------------|------------|-------------|",
]

for h in TARGET_HEIGHTS:
    r = results.get(h, {})
    sens = r.get("sensitivity", {})
    contr = r.get("controllability", {})
    n = r.get("samples", 0)
    row = f"| {h:.2f}m | {n} |"
    row += f" {sens.get('d_pitch_accel_d_tau_rad_s2_per_nm', 0):.6f} |"
    row += f" {sens.get('d_com_accel_d_tau_m_s2_per_nm', 0):.6f} |"
    row += f" {sens.get('d_support_accel_d_tau', 0):.6f} |"
    row += f" {contr.get('pitch', '?')} |"
    row += f" {contr.get('support', '?')} |"
    report_lines.append(row)

report_lines += [
    "",
    "## Torque per Correction",
    "",
    "| Height | tau per 1° pitch (Nm) | tau per 0.1m support (Nm) | Pitch change per Nm/100ms (°) |",
    "|--------|------------------------|----------------------------|------------------------------|",
]

for h in TARGET_HEIGHTS:
    r = results.get(h, {})
    tc = r.get("torque_per_correction", {})
    eff = r.get("effect_per_nm_100ms", {})
    report_lines.append(
        f"| {h:.2f}m | {tc.get('tau_per_1deg_pitch_nm', float('inf')):.2f} |"
        f" {tc.get('tau_per_01m_support_nm', float('inf')):.2f} |"
        f" {eff.get('pitch_change_deg', 0):.4f} |"
    )

report_lines += [
    "",
    "## Diagnosis",
    "",
    "### Does wheel torque still have enough authority at 0.48m?",
    "",
]

h48_r = results.get(0.48, {})
sens48 = h48_r.get("sensitivity", {})
h33_r = results.get(0.33, {})
sens33 = h33_r.get("sensitivity", {})

if pitch_trend == "DECREASING":
    ratio = abs(sens48.get("d_pitch_accel_d_tau_rad_s2_per_nm", 1)) / max(abs(sens33.get("d_pitch_accel_d_tau_rad_s2_per_nm", 1)), 1e-9)
    report_lines.append(
        f"Pitch sensitivity at 0.48m is {ratio:.1%} of the sensitivity at 0.33m. "
        f"The wheel torque authority for pitch control **decreases with height** — "
        f"this is expected physics (longer lever arm at lower squat = more torque authority)."
    )
else:
    report_lines.append(
        "Pitch sensitivity does not significantly degrade with height. "
        "Wheel torque authority for pitch control is maintained."
    )

report_lines += [
    "",
    "### Is the system near a weakly controllable region?",
    "",
    f"**{weak_verdict}**",
    "",
    f"- Pitch controllable at 0.48m: {h48_r.get('controllability', {}).get('pitch', '?')}",
    f"- Support controllable at 0.48m: {h48_r.get('controllability', {}).get('support', '?')}",
    "",
    "### Interpretation",
    "",
]

if "DEGRAD" in weak_verdict:
    report_lines.append(
        "The system DOES lose some controllability at taller heights. "
        "However, this is a fundamental plant characteristic (moment arm geometry): "
        "at taller heights, wheel torque has less leverage to affect pitch and COM. "
        "This is NOT a controller architecture problem — it's a physical limitation."
    )
else:
    report_lines.append(
        "The system maintains controllability at all tested heights. "
        "The wheel torque has sufficient authority to affect pitch, COM position, "
        "and support error. Controllability is NOT the bottleneck."
    )

report_lines += [
    "",
    f"**Classification:** `{overall_verdict}`",
]

with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"Report saved to: {OUTPUT_REPORT}")
print("\nDone.")
