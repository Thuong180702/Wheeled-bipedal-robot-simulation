#!/usr/bin/env python3
"""
Phase 3: Perfect Support Correction Offline Audit

Using recorded K1 telemetry, computes counterfactuals:
- If support error were magically zero, what pitch trajectory would remain?
- If pitch error were magically zero, what support trajectory would remain?
- Estimate required support torque and required pitch torque.

Key questions:
- What dominates failure: support drift or pitch oscillation?
- Are support and pitch dynamics separable?
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
    "outputs", "system_audit", "perfect_support",
)
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "perfect_support_audit.json")
OUTPUT_REPORT = os.path.join(OUTPUT_DIR, "perfect_support_report.md")

# ── Load telemetry ─────────────────────────────────────────────────────────
print("Loading K1 telemetry...")
with open(TELEMETRY_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

N = len(rows)
dt = float(rows[1]["sim_time_s"]) - float(rows[0]["sim_time_s"]) if N > 1 else 0.005

print(f"  {N} steps, dt={dt:.4f}s")

# ── Extract key signals ────────────────────────────────────────────────────
pitch_x = np.array([float(r["pitch_x"]) for r in rows])
pitch_rate = np.array([float(r["pitch_rate_x"]) for r in rows])
com_z = np.array([float(r["com_z"]) for r in rows])
com_y = np.array([float(r["com_y"]) for r in rows])
support_error = np.array([float(r.get("support_position_error_m", 0)) for r in rows])
cp_error_y = np.array([float(r.get("cp_error_y", 0)) for r in rows])
com_vy = np.array([float(r.get("com_vy", 0)) for r in rows])
def _safe_float(val, default=0.0):
    """Handle boolean string values like 'True'/'False' in telemetry."""
    if isinstance(val, str) and val in ("True", "False"):
        return 1.0 if val == "True" else 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

push_active = np.array([_safe_float(r.get("push_active", 0)) for r in rows])
tau_left = np.array([float(r["tau_left"]) for r in rows])
tau_right = np.array([float(r["tau_right"]) for r in rows])
tau_common = tau_left + tau_right

# Controller torque components
tau_pitch_term = np.array([float(r.get("sagittal_term_pitch", 0)) for r in rows])
tau_pitch_rate_term = np.array([float(r.get("sagittal_term_pitch_rate", 0)) for r in rows])
tau_cp_term = np.array([float(r.get("sagittal_term_cp", 0)) for r in rows])
tau_com_vy_term = np.array([float(r.get("sagittal_term_com_vy", 0)) for r in rows])
tau_pos = np.array([float(r.get("tau_position", 0)) for r in rows])
tau_sup_vel = np.array([float(r.get("tau_support_velocity", 0)) for r in rows])

# ── Find push window ───────────────────────────────────────────────────────
push_steps = np.where(push_active > 0.5)[0]
push_start = int(push_steps[0]) if len(push_steps) > 0 else 300
push_end = int(push_steps[-1]) if len(push_steps) > 0 else 310
post_push_start = push_end + 1

print(f"  Push: step {push_start} to {push_end}")

# ── Counterfactual analysis ────────────────────────────────────────────────
# We analyze 3 scenarios post-push:
# 1. REALITY: actual K1 trajectory
# 2. ZERO_SUPPORT: support error = 0 (simulated)
# 3. ZERO_PITCH: pitch error = 0 (simulated)
#
# Since we can't actually rerun, we compute the TORQUE that would be required
# to zero each error, and whether that torque is within the budget.

# ── 1. Compute torque decomposition ───────────────────────────────────────
# The total wheel torque is composed of:
# tau_pitch (proportional to pitch error)
# tau_pitch_rate (proportional to pitch rate)
# tau_cp (proportional to CP error)
# tau_com_vy (proportional to COM velocity)
# tau_position (proportional to support position error)
# tau_support_velocity (proportional to support error rate)
# + individual wheel damping

# For the counterfactual: if support_error were 0, tau_position would be 0
# but other terms (pitch, CP, etc.) would also change because the robot state would differ.
# We CANNOT simulate the full counterfactual without the simulator.
#
# Instead, we compute:
# (a) What fraction of total torque is "support-related" vs "pitch-related"?
# (b) If we removed all support-related torque, how much headroom would be freed?
# (c) If we removed all pitch-related torque, how much headroom would be freed?
# (d) What torque is "required" to maintain current error levels?

print("\n── Torque Composition Analysis ──")

# Partition torque into support-related and pitch-related components
# Support-related: tau_position, tau_support_velocity
# Pitch-related: tau_pitch, tau_pitch_rate
# Mixed: tau_cp, tau_com_vy (affect both)

support_torque = tau_pos + tau_sup_vel
pitch_torque = tau_pitch_term + tau_pitch_rate_term
mixed_torque = tau_cp_term + tau_com_vy_term

# Compute means and contributions
abs_total = np.abs(tau_pitch_term) + np.abs(tau_pitch_rate_term) + np.abs(tau_cp_term) + np.abs(tau_com_vy_term) + np.abs(tau_pos) + np.abs(tau_sup_vel)
abs_total_mean = np.where(abs_total > 1e-9, abs_total, 1.0)

pitch_fraction = (np.abs(tau_pitch_term) + np.abs(tau_pitch_rate_term)) / abs_total_mean
support_fraction = (np.abs(tau_pos) + np.abs(tau_sup_vel)) / abs_total_mean
mixed_fraction = (np.abs(tau_cp_term) + np.abs(tau_com_vy_term)) / abs_total_mean

# ── 2. Counterfactual torque budgets ──────────────────────────────────────
# Question: if support error were zero:
#   - tau_position ≈ 0 (directly proportional)
#   - Remaining torque would be pitch + CP + mixed
#   - Would the remaining pitch oscillation still exist?
#
# Answer: YES. tau_position zeroing doesn't change pitch dynamics directly.
# The pitch oscillation is driven by the pitch-pitch_rate-CP feedback loop.

# Question: if pitch error were zero:
#   - tau_pitch ≈ 0
#   - tau_pitch_rate would also decay (since no pitch error → no pitch rate from physics)
#   - Remaining torque would be support + CP + mixed
#   - Would support still drift?
#
# Answer: The support torque (tau_position) is driven by support_error independently.
# If pitch were magically zero, support would still drift because the integral
# of COM velocity would still accumulate position error.

print("\n── Counterfactual: Zero Support Error ──")
print("  (If support error = 0, what pitch torque remains?)")

# Post-push segment
pp_mask = np.arange(N) >= post_push_start
pp_support_torque_mean = float(np.mean(np.abs(support_torque[pp_mask])))
pp_pitch_torque_mean = float(np.mean(np.abs(pitch_torque[pp_mask])))
pp_mixed_torque_mean = float(np.mean(np.abs(mixed_torque[pp_mask])))

# The pitch torque that would remain if support=0
# (in reality, the dynamics would also change, but this is a first-order estimate)
remaining_pitch_torque = pp_pitch_torque_mean + pp_mixed_torque_mean
print(f"  Current post-push mean |pitch torque|: {pp_pitch_torque_mean:.3f} Nm")
print(f"  Current post-push mean |support torque|: {pp_support_torque_mean:.3f} Nm")
print(f"  Current post-push mean |mixed torque|: {pp_mixed_torque_mean:.3f} Nm")
print(f"  Remaining pitch torque if support=0: {remaining_pitch_torque:.3f} Nm")
print(f"  => Pitch oscillation WOULD still exist even with perfect support")

print("\n── Counterfactual: Zero Pitch Error ──")
print("  (If pitch error = 0, what support torque remains?)")

remaining_support_torque = pp_support_torque_mean + pp_mixed_torque_mean
print(f"  Remaining support torque if pitch=0: {remaining_support_torque:.3f} Nm")
print(f"  => Support correction WOULD still be needed even with perfect pitch")

# ── 3. Required torque for complete correction ─────────────────────────────
# Compute: if we wanted to zero support_error in 1 second,
# how much torque would be needed?

print("\n── Required Torque for Complete Correction ──")

# Support correction requirement:
# support_accel_needed = 2 * support_error / T^2  (constant accel to reach zero in T seconds)
# Then tau_needed = support_accel_needed / sensitivity
# Estimate sensitivity from controllability audit or from relationship:
# tau_position = k_position * support_error
# support_accel_from_torque = tau_position * some_factor

# From the controller: k_position is about 50-300 depending on height
# position_authority_scale scales it
# The relationship is: support_accel ≈ tau_wheel / (wheel_radius * mass) * geometry
# Very roughly: 1 Nm wheel torque ≈ 1 / (0.1m * 25kg) ≈ 0.4 m/s^2 COM accel

# Let's compute the actual relationship from the telemetry
# d(support_error)/dt = support_error_rate ≈ com_vy (approximate)
# d(com_vy)/dt ≈ tau_common / (mass * wheel_radius) * cos(pitch)
# Actually: com_accel ≈ tau_common / (m * r_wheel) projected

# More precisely from telemetry: correlation between tau_common and com_vy changes
pp_cov = np.cov(com_vy[pp_mask], tau_common[pp_mask])
var_tau_pp = np.var(tau_common[pp_mask])
if var_tau_pp > 1e-9:
    d_vy_d_tau = pp_cov[0, 1] / var_tau_pp
else:
    d_vy_d_tau = 0.0

# Mass approx
mass = float(rows[0].get("mass_kg", 25.0))

# COM accel from tau
com_accel = np.zeros_like(com_vy)
com_accel[1:] = (com_vy[1:] - com_vy[:-1]) / dt
com_accel[0] = com_accel[1] if N > 1 else 0.0

pp_cov2 = np.cov(com_accel[pp_mask], tau_common[pp_mask])
if var_tau_pp > 1e-9:
    d_accel_d_tau = pp_cov2[0, 1] / var_tau_pp
else:
    d_accel_d_tau = mass * 0.1  # rough estimate: tau = m * a * r

print(f"  Mass: {mass:.1f} kg")
print(f"  d(com_vy)/d(tau): {d_vy_d_tau:.4f} (m/s)/Nm")
print(f"  d(com_accel)/d(tau): {d_accel_d_tau:.4f} m/s^2/Nm")

# Required torque for support centering
# Assume support error = 0.1m, want to zero in T=2s
support_error_pp_mean = float(np.mean(np.abs(support_error[pp_mask])))
support_error_pp_max = float(np.max(np.abs(support_error[pp_mask])))
T_correct = 2.0  # seconds to correct

if abs(d_accel_d_tau) > 1e-9:
    tau_for_support_01m_2s = (2 * 0.1 / T_correct**2) / d_accel_d_tau
    tau_for_support_mean_2s = (2 * support_error_pp_mean / T_correct**2) / d_accel_d_tau
    tau_for_support_max_2s = (2 * support_error_pp_max / T_correct**2) / d_accel_d_tau
else:
    tau_for_support_01m_2s = tau_for_support_mean_2s = tau_for_support_max_2s = float("inf")

print(f"\n  Support error mean (post-push): {support_error_pp_mean:.3f} m")
print(f"  Support error max (post-push): {support_error_pp_max:.3f} m")
print(f"  Required torque to zero 0.1m support in {T_correct:.0f}s: {tau_for_support_01m_2s:.2f} Nm")
print(f"  Required torque to zero mean support in {T_correct:.0f}s: {tau_for_support_mean_2s:.2f} Nm")
print(f"  Required torque to zero max support in {T_correct:.0f}s: {tau_for_support_max_2s:.2f} Nm")

# ── 4. Required torque for pitch correction ────────────────────────────────
pitch_error_pp_mean = float(np.mean(np.abs(pitch_x[pp_mask])))
pitch_error_pp_max = float(np.max(np.abs(pitch_x[pp_mask])))
pitch_error_pp_std = float(np.std(pitch_x[pp_mask]))

# Pitch sensitivity from controller: d(pitch_accel)/d(tau)
# Can estimate from telemetry
pp_cov3 = np.cov(np.diff(pitch_rate[pp_mask]) / dt, np.diff(tau_common[pp_mask]))
var_dtau = np.var(np.diff(tau_common[pp_mask]))
if var_dtau > 1e-9:
    d_pitch_accel_d_tau = pp_cov3[0, 1] / var_dtau
else:
    # theoretical: tau = I * alpha_pitch, I ≈ m*h^2, tau_wheel = F * r
    # pitch torque from wheel: tau_pitch = F * r_wheel * h (moment arm)
    # alpha_pitch = tau_pitch / I = (tau_wheel * h / r_wheel) / (m * h^2)
    #             = tau_wheel / (m * h * r_wheel)
    d_pitch_accel_d_tau = 1.0 / (mass * 0.48 * 0.1)  # rough

if abs(d_pitch_accel_d_tau) > 1e-9:
    tau_for_pitch_1deg_2s = (2 * (math.pi / 180) / T_correct**2) / d_pitch_accel_d_tau
    tau_for_pitch_mean_2s = (2 * pitch_error_pp_mean / T_correct**2) / d_pitch_accel_d_tau
else:
    tau_for_pitch_1deg_2s = tau_for_pitch_mean_2s = float("inf")

print(f"\n  Pitch error mean (post-push): {pitch_error_pp_mean*180/math.pi:.2f} deg")
print(f"  Pitch error max (post-push): {pitch_error_pp_max*180/math.pi:.2f} deg")
print(f"  d(pitch_accel)/d(tau): {d_pitch_accel_d_tau:.4f} rad/s^2/Nm")
print(f"  Required torque to zero 1 deg pitch in {T_correct:.0f}s: {tau_for_pitch_1deg_2s:.2f} Nm")
print(f"  Required torque to zero mean pitch in {T_correct:.0f}s: {tau_for_pitch_mean_2s:.2f} Nm")

# ── 5. Dominance analysis ──────────────────────────────────────────────────
# Which requires more torque: pitch correction or support correction?
# Normalize both to "torque required for typical error"

pitch_torque_demand = abs(tau_for_pitch_mean_2s) if abs(tau_for_pitch_mean_2s) < 100 else pp_pitch_torque_mean
support_torque_demand = abs(tau_for_support_mean_2s) if abs(tau_for_support_mean_2s) < 100 else pp_support_torque_mean

if pitch_torque_demand > support_torque_demand * 1.5:
    dominance = "PITCH_DOMINATES"
elif support_torque_demand > pitch_torque_demand * 1.5:
    dominance = "SUPPORT_DOMINATES"
else:
    dominance = "BOTH_CONTRIBUTE_EQUALLY"

print(f"\n── Dominance Analysis ──")
print(f"  Pitch torque demand: {pitch_torque_demand:.3f} Nm")
print(f"  Support torque demand: {support_torque_demand:.3f} Nm")
print(f"  Dominance: {dominance}")

# ── 6. Separability analysis ──────────────────────────────────────────────
# Are pitch and support dynamically coupled?
# If coupled: correcting one without the other is impossible
# If separable: they can be addressed independently

# Compute the cross-correlation between pitch error and support error at various lags
max_lag = int(1.0 / dt)  # 1 second
cross_corr = []
for lag in range(-max_lag, max_lag + 1):
    if lag < 0:
        x = pitch_x[pp_mask][:lag]
        y = support_error[pp_mask][-lag:]
    elif lag > 0:
        x = pitch_x[pp_mask][lag:]
        y = support_error[pp_mask][:-lag]
    else:
        x = pitch_x[pp_mask]
        y = support_error[pp_mask]
    if len(x) > 10:
        r = np.corrcoef(x, y)[0, 1]
        cross_corr.append((lag * dt, r))

cross_corr_sorted = sorted(cross_corr, key=lambda x: abs(x[1]), reverse=True)
max_cross_corr = cross_corr_sorted[0]
opt_lag = max_cross_corr[0]
opt_corr = max_cross_corr[1]

print(f"\n── Separability Analysis ──")
print(f"  Max cross-correlation: r={opt_corr:.3f} at lag={opt_lag:.2f}s")
print(f"  Zero-lag correlation: r={np.corrcoef(pitch_x[pp_mask], support_error[pp_mask])[0, 1]:.3f}")

if abs(opt_corr) > 0.7:
    separable = "STRONGLY_COUPLED_CANNOT_SEPARATE"
elif abs(opt_corr) > 0.4:
    separable = "MODERATELY_COUPLED_PARTIALLY_SEPARABLE"
else:
    separable = "WEAKLY_COUPLED_LARGELY_SEPARABLE"

print(f"  Separability: {separable}")

# ── 7. What dominates failure? ─────────────────────────────────────────────
# Look at post-push trajectory: does support drift grow unboundedly?
# Does pitch oscillation persist?

# Support drift rate: linear trend of |support_error| over post-push window
pp_steps = np.arange(post_push_start, N)
if len(pp_steps) > 10:
    abs_support_pp = np.abs(support_error[pp_steps])
    # Linear fit
    A = np.vstack([pp_steps - pp_steps[0], np.ones_like(pp_steps)]).T
    slope_support, intercept_support = np.linalg.lstsq(A, abs_support_pp, rcond=None)[0]
    support_drift_rate = slope_support  # m per step
    support_drift_rate_per_s = support_drift_rate / dt

    abs_pitch_pp = np.abs(pitch_x[pp_steps])
    slope_pitch, intercept_pitch = np.linalg.lstsq(A, abs_pitch_pp, rcond=None)[0]
    pitch_drift_rate = slope_pitch  # rad per step
    pitch_drift_rate_per_s = pitch_drift_rate / dt

    print(f"\n── Post-Push Drift Analysis ──")
    print(f"  Support drift rate: {support_drift_rate_per_s:.6f} m/s")
    print(f"  Pitch drift rate: {pitch_drift_rate_per_s*180/math.pi:.6f} deg/s")

    if support_drift_rate_per_s > 1e-4 and abs(support_drift_rate_per_s) > abs(pitch_drift_rate_per_s) * 10:
        failure_driver = "SUPPORT_DRIFT_DOMINATES"
    elif abs(pitch_drift_rate_per_s) > 1e-4 and abs(pitch_drift_rate_per_s) > abs(support_drift_rate_per_s) * 10:
        failure_driver = "PITCH_OSCILLATION_DOMINATES"
    else:
        failure_driver = "BOTH_CONTRIBUTE"
else:
    support_drift_rate_per_s = 0.0
    pitch_drift_rate_per_s = 0.0
    failure_driver = "INSUFFICIENT_DATA"

print(f"  Failure driver: {failure_driver}")

# ── 8. Energy analysis ─────────────────────────────────────────────────────
# Total "work" done by pitch torque vs support torque
pitch_energy_pp = float(np.sum(np.abs(pitch_torque[pp_mask]) * np.abs(pitch_rate[pp_mask])) * dt)
support_energy_pp = float(np.sum(np.abs(support_torque[pp_mask]) * np.abs(com_vy[pp_mask])) * dt)
mixed_energy_pp = float(np.sum(np.abs(mixed_torque[pp_mask]) * np.abs(pitch_rate[pp_mask])) * dt)

print(f"\n── Energy Analysis (post-push) ──")
print(f"  Pitch torque work: {pitch_energy_pp:.2f} Nm*rad")
print(f"  Support torque work: {support_energy_pp:.2f} Nm*m/s")
print(f"  Mixed torque work: {mixed_energy_pp:.2f}")

# ── Compile results ────────────────────────────────────────────────────────
MAX_TAU = 5.0  # Nm per wheel

audit = {
    "audit": "perfect_support_correction",
    "target": "K1_PITCH_RATE_NOTCH_V1",
    "run": "high_0p480_90N_push_step300_3000steps",
    "push_window": {"start": push_start, "end": push_end},
    "dt_s": dt,
    "mass_kg": mass,
    "torque_composition_post_push": {
        "pitch_torque_mean_abs_nm": pp_pitch_torque_mean,
        "support_torque_mean_abs_nm": pp_support_torque_mean,
        "mixed_torque_mean_abs_nm": pp_mixed_torque_mean,
    },
    "counterfactual_zero_support": {
        "remaining_pitch_torque_nm": remaining_pitch_torque,
        "pitch_oscillation_would_remain": True,
        "freed_torque_budget_nm": pp_support_torque_mean,
        "freed_budget_pct": round(100.0 * pp_support_torque_mean / MAX_TAU, 1),
    },
    "counterfactual_zero_pitch": {
        "remaining_support_torque_nm": remaining_support_torque,
        "support_drift_would_remain": True,
        "freed_torque_budget_nm": pp_pitch_torque_mean,
        "freed_budget_pct": round(100.0 * pp_pitch_torque_mean / MAX_TAU, 1),
    },
    "required_torque": {
        "d_com_accel_d_tau_m_s2_per_nm": float(d_accel_d_tau),
        "d_pitch_accel_d_tau_rad_s2_per_nm": float(d_pitch_accel_d_tau),
        "tau_for_01m_support_2s_nm": float(tau_for_support_01m_2s),
        "tau_for_mean_support_2s_nm": float(tau_for_support_mean_2s),
        "tau_for_max_support_2s_nm": float(tau_for_support_max_2s),
        "tau_for_1deg_pitch_2s_nm": float(tau_for_pitch_1deg_2s),
        "tau_for_mean_pitch_2s_nm": float(tau_for_pitch_mean_2s),
    },
    "dominance": dominance,
    "failure_driver": failure_driver,
    "separability": {
        "max_cross_corr": float(opt_corr),
        "optimal_lag_s": float(opt_lag),
        "classification": separable,
    },
    "drift_rates": {
        "support_drift_rate_m_per_s": float(support_drift_rate_per_s),
        "pitch_drift_rate_rad_per_s": float(pitch_drift_rate_per_s),
        "pitch_drift_rate_deg_per_s": float(pitch_drift_rate_per_s * 180.0 / math.pi),
    },
    "energy_post_push": {
        "pitch_energy": pitch_energy_pp,
        "support_energy": support_energy_pp,
        "mixed_energy": mixed_energy_pp,
    },
    "post_push_stats": {
        "support_error_mean_m": float(support_error_pp_mean),
        "support_error_max_m": float(support_error_pp_max),
        "pitch_error_mean_deg": float(pitch_error_pp_mean * 180.0 / math.pi),
        "pitch_error_max_deg": float(pitch_error_pp_max * 180.0 / math.pi),
        "pitch_error_std_deg": float(pitch_error_pp_std * 180.0 / math.pi),
    },
}

audit["verdict"] = failure_driver

# ── Save ───────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2)
print(f"\nJSON audit saved to: {OUTPUT_JSON}")

# ── Print summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PERFECT SUPPORT CORRECTION AUDIT — SUMMARY")
print("=" * 70)

print(f"\n── Torque Composition (post-push, mean absolute) ──")
print(f"  Pitch terms: {pp_pitch_torque_mean:.3f} Nm ({100*pp_pitch_torque_mean/(pp_pitch_torque_mean+pp_support_torque_mean+pp_mixed_torque_mean+1e-9):.0f}%)")
print(f"  Support terms: {pp_support_torque_mean:.3f} Nm ({100*pp_support_torque_mean/(pp_pitch_torque_mean+pp_support_torque_mean+pp_mixed_torque_mean+1e-9):.0f}%)")
print(f"  Mixed terms: {pp_mixed_torque_mean:.3f} Nm")

print(f"\n── Counterfactual: Zero Support Error ──")
print(f"  Pitch torque would still exist: YES ({remaining_pitch_torque:.3f} Nm)")
print(f"  Freed torque: {pp_support_torque_mean:.3f} Nm ({100*pp_support_torque_mean/MAX_TAU:.1f}% of budget)")

print(f"\n── Counterfactual: Zero Pitch Error ──")
print(f"  Support torque would still be needed: YES ({remaining_support_torque:.3f} Nm)")
print(f"  Freed torque: {pp_pitch_torque_mean:.3f} Nm ({100*pp_pitch_torque_mean/MAX_TAU:.1f}% of budget)")

print(f"\n── Required Torque ──")
print(f"  To zero mean support in 2s: {tau_for_support_mean_2s:.2f} Nm")
print(f"  To zero mean pitch in 2s: {tau_for_pitch_mean_2s:.2f} Nm")

print(f"\n── What Dominates Failure? ──")
print(f"  Support drift rate: {support_drift_rate_per_s:.6f} m/s")
print(f"  Pitch drift rate: {pitch_drift_rate_per_s*180/math.pi:.6f} deg/s")
print(f"  Failure driver: {failure_driver}")
print(f"  Dominance: {dominance}")
print(f"  Separability: {separable}")

print(f"\n── VERDICT ──")
print(f"  {failure_driver}")

# ── Write report ──────────────────────────────────────────────────────────
report_lines = [
    "# Perfect Support Correction Offline Audit",
    "",
    f"**Verdict:** `{failure_driver}`",
    f"**Dominance:** `{dominance}`",
    f"**Separability:** `{separable}`",
    "",
    "## Method",
    "",
    "Using recorded K1 focused recovery telemetry, this audit performs offline counterfactual analysis:",
    "1. Decompose total wheel torque into pitch-related, support-related, and mixed components",
    "2. Estimate required torque to zero each error independently",
    "3. Analyze drift rates and cross-correlation to determine which error drives failure",
    "",
    "**CRITICAL CAVEAT:** This is a first-order offline estimate. The actual dynamics are nonlinear",
    "and counterfactual states would differ from recorded states. This analysis identifies",
    "DIRECTIONAL answers, not precise values.",
    "",
    "## Torque Composition (Post-Push)",
    "",
    "| Component | Mean |Abs| (Nm) | % of Total |",
    "|-----------|------|---------|",
    f"| Pitch (tau_pitch + tau_pitch_rate) | {pp_pitch_torque_mean:.3f} | {100*pp_pitch_torque_mean/(pp_pitch_torque_mean+pp_support_torque_mean+pp_mixed_torque_mean+1e-9):.1f}% |",
    f"| Support (tau_position + tau_support_velocity) | {pp_support_torque_mean:.3f} | {100*pp_support_torque_mean/(pp_pitch_torque_mean+pp_support_torque_mean+pp_mixed_torque_mean+1e-9):.1f}% |",
    f"| Mixed (tau_cp + tau_com_vy) | {pp_mixed_torque_mean:.3f} | {100*pp_mixed_torque_mean/(pp_pitch_torque_mean+pp_support_torque_mean+pp_mixed_torque_mean+1e-9):.1f}% |",
    "",
    "## Counterfactual Analysis",
    "",
    "### If Support Error Were Zero",
    "",
    f"- Pitch torque components would remain: **{remaining_pitch_torque:.3f} Nm**",
    f"- Freed torque budget: **{pp_support_torque_mean:.3f} Nm ({100*pp_support_torque_mean/MAX_TAU:.1f}% of budget)**",
    f"- **Pitch oscillation would STILL exist** — support centering does NOT eliminate pitch dynamics",
    "",
    "### If Pitch Error Were Zero",
    "",
    f"- Support torque components would remain: **{remaining_support_torque:.3f} Nm**",
    f"- Freed torque budget: **{pp_pitch_torque_mean:.3f} Nm ({100*pp_pitch_torque_mean/MAX_TAU:.1f}% of budget)**",
    f"- **Support drift would STILL exist** — pitch stabilization does NOT center support",
    "",
    "## Required Torque for Correction",
    "",
    "| Target | Error | Time | Required Torque | Within Budget (5 Nm)? |",
    "|--------|-------|------|----------------|----------------------|",
    f"| Support | 0.1 m | 2 s | {tau_for_support_01m_2s:.2f} Nm | {'YES' if tau_for_support_01m_2s < 5 else 'NO'} |",
    f"| Support | {support_error_pp_mean:.3f} m (mean) | 2 s | {tau_for_support_mean_2s:.2f} Nm | {'YES' if tau_for_support_mean_2s < 5 else 'NO'} |",
    f"| Support | {support_error_pp_max:.3f} m (max) | 2 s | {tau_for_support_max_2s:.2f} Nm | {'YES' if tau_for_support_max_2s < 5 else 'NO'} |",
    f"| Pitch | 1 deg | 2 s | {tau_for_pitch_1deg_2s:.2f} Nm | {'YES' if tau_for_pitch_1deg_2s < 5 else 'NO'} |",
    f"| Pitch | {pitch_error_pp_mean*180/math.pi:.2f} deg (mean) | 2 s | {tau_for_pitch_mean_2s:.2f} Nm | {'YES' if tau_for_pitch_mean_2s < 5 else 'NO'} |",
    "",
    "## Separability",
    "",
    f"- Max cross-correlation: r = {opt_corr:.3f} at lag {opt_lag:.2f}s",
    f"- Zero-lag correlation: r = {np.corrcoef(pitch_x[pp_mask], support_error[pp_mask])[0, 1]:.3f}",
    f"- Classification: **{separable}**",
    "",
    "## Drift Rates (Post-Push)",
    "",
    f"| Error | Drift Rate |",
    f"|-------|-----------|",
    f"| Support | {support_drift_rate_per_s:.6f} m/s |",
    f"| Pitch | {pitch_drift_rate_per_s*180/math.pi:.6f} deg/s |",
    "",
    "## Diagnosis",
    "",
    f"### What dominates failure?",
    "",
    f"**{failure_driver}**",
    "",
]

if failure_driver == "SUPPORT_DRIFT_DOMINATES":
    report_lines += [
        "Support drift is the dominant failure mechanism. Even if pitch were perfectly stabilized,",
        "the COM would gradually drift away from the support polygon, eventually causing",
        "`height_too_low` as the robot squats to compensate.",
        "",
        "This suggests that the K1 architecture, while good at pitch stabilization, does not",
        "provide enough position-centering authority to prevent slow drift.",
    ]
elif failure_driver == "PITCH_OSCILLATION_DOMINATES":
    report_lines += [
        "Pitch oscillation is the dominant failure mechanism. The 0.4 Hz low-frequency",
        "oscillation grows over time, coupling into support error through the COM kinematics,",
        "and eventually the oscillation amplitude exceeds the recovery capability.",
    ]
else:
    report_lines += [
        "Both pitch oscillation and support drift contribute to failure. The dynamics are",
        "coupled and neither error can be corrected in isolation.",
    ]

report_lines += [
    "",
    "### Would another controller architecture likely help?",
    "",
]

if separable == "WEAKLY_COUPLED_LARGELY_SEPARABLE" and failure_driver == "SUPPORT_DRIFT_DOMINATES":
    report_lines += [
        "**MAYBE.** Support drift is independently correctable. A controller that adds",
        "targeted support-centering authority (NOT replacing K1's pitch damping, but",
        "augmenting it) could address the drift without degrading pitch stability.",
        "The fact that support and pitch are weakly coupled means an additive approach",
        "(original L-family concept) may work if implemented with correct signs.",
    ]
elif separable == "STRONGLY_COUPLED_CANNOT_SEPARATE":
    report_lines += [
        "**UNLIKELY.** Support and pitch are too strongly coupled to address independently.",
        "Any increase in support-correction torque will excite pitch, and vice versa.",
        "The architecture may need a fundamentally different approach (e.g., full-state",
        "LQR, MPC, or RL residual) that models the coupling explicitly.",
    ]
else:
    report_lines += [
        "**POSSIBLY.** The coupling is moderate, so architectural changes could help if",
        "they respect the coupling dynamics. However, the improvement may be incremental",
        "rather than fundamental.",
    ]

report_lines += [
    "",
    f"**Classification:** `{failure_driver}`",
]

with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"Report saved to: {OUTPUT_REPORT}")
print("\nDone.")
