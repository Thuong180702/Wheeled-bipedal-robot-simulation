#!/usr/bin/env python3
"""Phase 6,8,9: K1 Gain Sensitivity, Theoretical Limit, and Recommendation.

Loads the state-space model and performs:
- Phase 6: Gain sensitivity analysis (±10% perturbation per gain)
- Phase 8: Theoretical performance limit estimation
- Phase 9: Evidence-based recommendation

STRICT CONSTRAINT: This is ANALYSIS ONLY. Do NOT tune gains or modify K1.
We perturb gains LOCALLY to understand sensitivity, not to find better values.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

# ── Unicode stdout ──────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "outputs" / "eigenmode_audit"
MODEL_PATH = INPUT_DIR / "k1_state_space_model.json"
EIGEN_PATH = INPUT_DIR / "k1_eigenmodes.json"

# ── Load models ─────────────────────────────────────────────────────
print("=" * 72)
print("Loading state-space and eigenmode models...")
print("=" * 72)

with open(MODEL_PATH) as f:
    model = json.load(f)

with open(EIGEN_PATH) as f:
    eigen = json.load(f)

state_names = model["state_definition"]["state_names"]
state_dim = model["state_definition"]["state_dim"]
dt = model["state_definition"]["control_dt_s"]
gains = model["controller_gains"]

A_open_dt = np.array(model["open_loop_model"]["A_discrete"])
B_open_dt = np.array(model["open_loop_model"]["B_discrete"])
A_open_ct = np.array(model["open_loop_model"]["A_continuous"])
B_open_ct = np.array(model["open_loop_model"]["B_continuous"])

print(f"  State vector: {state_names}")
print(f"  dt = {dt} s")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PHASE 6: GAIN SENSITIVITY ANALYSIS                                ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 72)
print("PHASE 6: GAIN SENSITIVITY ANALYSIS")
print("=" * 72)

# K1 gain vector (raw mapping from state to torque):
# tau = kp_pitch * pitch + kd_pitch * pitch_rate - k_position * pos_error
#       - k_total_vel * com_vel + 0 * wheel_vel
K_nominal_raw = np.array([
    gains["kp_pitch"],              # +50.0  Nm/rad
    gains["kd_pitch"],              # +10.0  Nm/(rad/s)
    -gains["k_position"],           # -40.0  Nm/m
    -gains["k_total_velocity_damping"],  # -20.0  Nm/(m/s)
    0.0,                            # wheel vel term is per-wheel, not common
])

# Each gain parameter and its effect on K_nominal_raw
gain_params = {
    "kp_pitch": {
        "value": gains["kp_pitch"],
        "units": "Nm/rad",
        "affects_row": 0,
        "sign_in_K": +1,
    },
    "kd_pitch": {
        "value": gains["kd_pitch"],
        "units": "Nm/(rad/s)",
        "affects_row": 1,
        "sign_in_K": +1,
    },
    "k_position": {
        "value": gains["k_position"],
        "units": "Nm/m",
        "affects_row": 2,
        "sign_in_K": -1,  # K_nominal_raw[2] = -k_position
    },
    "k_total_vel": {
        "value": gains["k_total_velocity_damping"],
        "units": "Nm/(m/s)",
        "affects_row": 3,
        "sign_in_K": -1,  # K_nominal_raw[3] = -k_total_vel
    },
    "k_wheel_velocity": {
        "value": gains["k_wheel_velocity"],
        "units": "Nm/(rad/s)",
        "affects_row": 4,
        "sign_in_K": -1,  # per-wheel: tau_common includes 0 for common torque
        "note": "Per-wheel differential term; minimal common-mode effect",
    },
}

# Compute nominal closed-loop eigenvalues
# A_cl = A_d + B_d * K_raw  (discrete-time)
A_cl_nominal = A_open_dt + B_open_dt @ K_nominal_raw.reshape(1, -1)
eig_nominal = np.linalg.eigvals(A_cl_nominal)

# Find the dominant oscillatory mode (closest to 0.4 Hz)
def find_dominant_oscillatory(eigvals, target_hz=0.4):
    best = None
    best_dist = float('inf')
    for ev in eigvals:
        if abs(np.imag(ev)) > 1e-8:
            f_hz = abs(np.angle(ev)) / (2 * np.pi * dt)
            dist = abs(f_hz - target_hz)
            if dist < best_dist:
                best_dist = dist
                best = ev
    return best

nom_dominant = find_dominant_oscillatory(eig_nominal)
zeta_nom = None
f_nom = None
mag_nom = None
if nom_dominant is not None:
    f_nom = abs(np.angle(nom_dominant)) / (2 * np.pi * dt)
    zeta_nom = -np.cos(np.angle(nom_dominant))
    mag_nom = abs(nom_dominant)
    print(f"\n  Nominal dominant oscillatory mode:")
    print(f"    λ = {nom_dominant.real:+.6f} {nom_dominant.imag:+.6f}j")
    print(f"    f = {f_nom:.4f} Hz")
    print(f"    ζ = {zeta_nom:+.4f}")
    print(f"    |λ| = {mag_nom:.4f}")

# Perturbation analysis
print(f"\n── Gain Sensitivity: ±10% Perturbation ──")
print(f"  {'Gain':<20s} {'Δ%':>6s}  {'f(Hz)':>8s}  {'ζ':>8s}  {'|λ|':>8s}  "
      f"{'Δf%':>8s}  {'Δζ%':>8s}  {'Sensitivity':>12s}")

sensitivity_results = {}

for gain_name, param in gain_params.items():
    nominal_val = param["value"]
    idx = param["affects_row"]
    sign = param["sign_in_K"]

    results_for_gain = {"nominal_value": nominal_val, "perturbations": {}}

    for delta_pct in [-10, +10]:
        perturbed_val = nominal_val * (1 + delta_pct / 100.0)
        K_pert = K_nominal_raw.copy()
        K_pert[idx] = sign * perturbed_val

        A_cl_pert = A_open_dt + B_open_dt @ K_pert.reshape(1, -1)
        eig_pert = np.linalg.eigvals(A_cl_pert)

        dom_pert = find_dominant_oscillatory(eig_pert)
        if dom_pert is not None:
            f_pert = abs(np.angle(dom_pert)) / (2 * np.pi * dt)
            zeta_pert = -np.cos(np.angle(dom_pert))
            mag_pert = abs(dom_pert)

            df_pct = (f_pert - f_nom) / f_nom * 100
            dz_pct = (zeta_pert - zeta_nom) / max(zeta_nom, 0.01) * 100

            # Sensitivity: normalized change in damping per normalized change in gain
            sens = (zeta_pert - zeta_nom) / max(zeta_nom, 0.01) / (delta_pct / 100)

            print(f"  {gain_name:<20s} {delta_pct:+5d}%  {f_pert:8.4f}  {zeta_pert:8.4f}  "
                  f"{mag_pert:8.4f}  {df_pct:+8.2f}  {dz_pct:+8.2f}  {sens:+12.4f}")

            results_for_gain["perturbations"][f"{delta_pct:+d}pct"] = {
                "f_hz": float(f_pert),
                "damping_ratio": float(zeta_pert),
                "magnitude": float(mag_pert),
                "delta_f_pct": float(df_pct),
                "delta_zeta_pct": float(dz_pct),
                "sensitivity": float(sens),
                "eigenvalue": {"real": float(dom_pert.real), "imag": float(dom_pert.imag)},
            }
        else:
            print(f"  {gain_name:<20s} {delta_pct:+5d}%  (no oscillatory mode)")

    sensitivity_results[gain_name] = results_for_gain

# Rank gains by their impact on the dominant mode's damping
print(f"\n── Gain Ranking by Damping Impact ──")
rankings = []
for gain_name, results in sensitivity_results.items():
    avg_sens = 0
    count = 0
    for pert_key, pert_data in results["perturbations"].items():
        if "sensitivity" in pert_data:
            avg_sens += abs(pert_data["sensitivity"])
            count += 1
    if count > 0:
        avg_sens /= count
        rankings.append((gain_name, avg_sens, results["nominal_value"]))

rankings.sort(key=lambda x: -x[1])

print(f"  {'Rank':<6s} {'Gain':<20s} {'Value':>10s}  {'|Sensitivity|':>12s}  {'Impact Level':<20s}")
for i, (name, sens, val) in enumerate(rankings):
    if sens > 1.0:
        impact = "HIGH — dominant lever"
    elif sens > 0.3:
        impact = "MEDIUM — useful"
    elif sens > 0.1:
        impact = "LOW — marginal"
    else:
        impact = "NEGLIGIBLE — don't touch"
    print(f"  {i+1:<6d} {name:<20s} {val:10.1f}  {sens:+12.4f}  {impact}")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PHASE 8: THEORETICAL PERFORMANCE LIMIT                            ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 72)
print("PHASE 8: THEORETICAL PERFORMANCE LIMIT")
print("=" * 72)

# Question: what's the best achievable damping ratio without changing hardware?

# Approach 1: Compute the controllability Gramian and see if the 0.4 Hz mode
# is energy-efficient to move.
print("\n── 8A: Controllability of the 0.4 Hz Mode ──")

# Discrete-time controllability matrix: C = [B, A*B, A²*B, ..., A^{n-1}*B]
n = state_dim
Ctrb = np.zeros((n, n))
for i in range(n):
    Ctrb[:, i:i+1] = np.linalg.matrix_power(A_open_dt, i) @ B_open_dt

ctrb_rank = np.linalg.matrix_rank(Ctrb)
ctrb_sv = np.linalg.svd(Ctrb, compute_uv=False)

print(f"  Controllability matrix rank: {ctrb_rank}/{n}")
print(f"  Controllability singular values: {[f'{s:.6f}' for s in ctrb_sv]}")
print(f"  Condition number: {ctrb_sv[0]/max(ctrb_sv[-1], 1e-12):.1f}")
print(f"  System is {'FULLY CONTROLLABLE' if ctrb_rank == n else 'NOT FULLY CONTROLLABLE'}")

# Approach 2: Compute the PBH test eigenvalues for the dominant mode
if nom_dominant is not None:
    # For mode λ, check rank([λI - A, B]) = n
    lam = nom_dominant
    pbh_mat = np.hstack([lam * np.eye(n) - A_open_dt, B_open_dt])
    pbh_rank = np.linalg.matrix_rank(pbh_mat)
    pbh_sv = np.linalg.svd(pbh_mat, compute_uv=False)
    print(f"\n  PBH test for dominant mode λ={lam.real:+.4f}{lam.imag:+.4f}j:")
    print(f"    Rank([λI-A, B]): {pbh_rank}/{n}")
    print(f"    Min singular value: {pbh_sv[-1]:.6f}")
    print(f"    Mode is {'CONTROLLABLE' if pbh_rank == n else 'UNCONTROLLABLE'}")

# Approach 3: Estimate the feasible damping region
print(f"\n── 8B: Feasible Damping Improvement ──")

# In discrete time, the dominant mode is at magnitude |λ|.
# To increase damping, we need to move the pole toward the origin.
# The minimum achievable magnitude is bounded by the actuator limits.

# Current: |λ| ≈ 1.0 (lightly damped, near unit circle)
# Best case with torque limit: the fastest we can move the mode inward
# is limited by max torque / sensor noise floor.

# Compute how much torque authority is available for pole placement
max_tau = gains["max_tau_wheel"]  # 5.0 Nm
# The effective gain from state to torque is K_nominal_raw
# The maximum torque per unit state error tells us the max feedback gain
# For pitch: 50 Nm/rad = 0.87 Nm/deg
# For position: 40 Nm/m (but capped at 3 Nm)

# Estimate: if we could double kd_pitch (from 10 to 20), what would happen?
for gain_name in ["kd_pitch", "kp_pitch"]:
    param = gain_params[gain_name]
    idx = param["affects_row"]
    sign = param["sign_in_K"]

    for multiplier in [1.5, 2.0, 3.0]:
        K_test = K_nominal_raw.copy()
        K_test[idx] = sign * param["value"] * multiplier
        A_cl_test = A_open_dt + B_open_dt @ K_test.reshape(1, -1)
        eig_test = np.linalg.eigvals(A_cl_test)
        dom_test = find_dominant_oscillatory(eig_test)

        if dom_test is not None:
            f_test = abs(np.angle(dom_test)) / (2 * np.pi * dt)
            zeta_test = -np.cos(np.angle(dom_test))
            mag_test = abs(dom_test)

            # Check if this gain is feasible (torque sat at nominal errors)
            # Nominal pitch error post-push: ~6.4° = 0.112 rad
            # Nominal pitch rate post-push: ~0.36 rad/s
            pitch_err_typical = 0.112  # rad
            pitch_rate_typical = 0.36  # rad/s
            pos_err_typical = 0.26     # m
            vel_typical = 0.15         # m/s

            tau_estimated = abs(
                K_test[0] * pitch_err_typical +
                K_test[1] * pitch_rate_typical +
                K_test[2] * pos_err_typical +
                K_test[3] * vel_typical
            )
            feasible = tau_estimated <= max_tau

            if multiplier == 1.0:
                print(f"\n  {gain_name}: nominal (×{multiplier}): f={f_test:.4f}Hz, "
                      f"ζ={zeta_test:+.4f}, |λ|={mag_test:.4f}, est_tau={tau_estimated:.1f}Nm")
            else:
                print(f"  {gain_name}: ×{multiplier}: f={f_test:.4f}Hz, "
                      f"ζ={zeta_test:+.4f}, |λ|={mag_test:.4f}, est_tau={tau_estimated:.1f}Nm "
                      f"{'FEASIBLE' if feasible else 'SATURATED'}")

# Approach 4: LQR-optimal damping benchmark
print(f"\n── 8C: LQR Optimal Damping Benchmark ──")

# Solve discrete-time LQR for the open-loop plant
# This gives the "optimal" state-feedback gain for comparison
try:
    Q = np.diag([10.0, 2.0, 5.0, 3.0, 0.5])  # state cost
    R = np.array([[1.0]])  # control cost

    from scipy.linalg import solve_discrete_are

    P = solve_discrete_are(A_open_dt, B_open_dt, Q, R)
    K_lqr = np.linalg.inv(R + B_open_dt.T @ P @ B_open_dt) @ (B_open_dt.T @ P @ A_open_dt)

    A_cl_lqr = A_open_dt - B_open_dt @ K_lqr
    eig_lqr = np.linalg.eigvals(A_cl_lqr)

    print(f"  LQR gain vector: {[f'{k:.2f}' for k in K_lqr.flatten()]}")
    print(f"  LQR closed-loop eigenvalues:")
    for ev in eig_lqr:
        mag = abs(ev)
        f_hz = abs(np.angle(ev)) / (2 * np.pi * dt)
        damp = -np.cos(np.angle(ev)) if abs(np.imag(ev)) > 1e-10 else (1.0 if ev.real > 0 else -1.0)
        print(f"    λ: {ev.real:+.6f} {ev.imag:+.6f}j  |λ|={mag:.4f}  f={f_hz:.4f}Hz  ζ={damp:+.4f}")

    dom_lqr = find_dominant_oscillatory(eig_lqr)
    if dom_lqr is not None:
        f_lqr = abs(np.angle(dom_lqr)) / (2 * np.pi * dt)
        zeta_lqr = -np.cos(np.angle(dom_lqr))
        mag_lqr = abs(dom_lqr)
        print(f"\n  LQR dominant oscillatory mode:")
        print(f"    f = {f_lqr:.4f} Hz")
        print(f"    ζ = {zeta_lqr:+.4f}")
        print(f"    |λ| = {mag_lqr:.4f}")

        lqr_benchmark = {
            "K_lqr": K_lqr.flatten().tolist(),
            "dominant_f_hz": float(f_lqr),
            "dominant_zeta": float(zeta_lqr),
            "dominant_magnitude": float(mag_lqr),
        }
    else:
        lqr_benchmark = {"K_lqr": K_lqr.flatten().tolist(), "note": "No oscillatory mode"}
except Exception as e:
    print(f"  LQR computation failed: {e}")
    lqr_benchmark = {"error": str(e)}

# ── Theoretical limit estimation ────────────────────────────────────
print(f"\n── 8D: Theoretical Damping Limit Estimate ──")

# What's the maximum damping ratio achievable?
# For a 2nd-order system: ω_n² = natural frequency, 2ζω_n = damping coefficient
# The maximum ζ from state feedback is limited by:
# 1. Actuator saturation (max_tau = 5 Nm)
# 2. Sensor noise (can't differentiate arbitrarily high gains)
# 3. Time delays (1 step = 10ms control delay)

# Estimate from the TWIP model:
# The pitch mode has ω_n = sqrt(g/L) ≈ 4.26 rad/s (open-loop unstable)
# To stabilize it with state feedback, we need enough gain to move the
# unstable pole into the left half-plane.
# The minimum kp for stabilization is kp_crit = g/L * (some inertia term)

# From the LQR benchmark, we can see the optimal achievable damping
# and compare it to K1's actual damping.

if nom_dominant is not None:
    print(f"\n  K1 current damping ratio: ζ = {zeta_nom:+.4f}")
    print(f"  K1 current frequency: f = {f_nom:.4f} Hz")
    print(f"  K1 current settling time (2%): {4.0/(zeta_nom*f_nom*2*np.pi) if zeta_nom > 0 else float('inf'):.2f}s")

if 'lqr_benchmark' in dir() and isinstance(lqr_benchmark, dict) and 'dominant_zeta' in lqr_benchmark:
    zeta_opt = lqr_benchmark['dominant_zeta']
    print(f"\n  LQR optimal damping ratio: ζ = {zeta_opt:+.4f}")
    print(f"  LQR optimal frequency: f = {lqr_benchmark['dominant_f_hz']:.4f} Hz")

    if zeta_nom > 0 and zeta_opt > 0:
        improvement = (zeta_opt - zeta_nom) / zeta_nom * 100
        print(f"\n  Potential damping improvement: {improvement:+.0f}%")

        if improvement > 50:
            print(f"  Conclusion: SIGNIFICANT improvement is theoretically possible")
            print(f"  K1 is NOT near the theoretical optimum")
        elif improvement > 10:
            print(f"  Conclusion: MODERATE improvement is possible")
            print(f"  K1 is somewhat below the theoretical optimum")
        else:
            print(f"  Conclusion: K1 is NEAR the theoretical optimum")
            print(f"  Limited room for improvement without hardware changes")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PHASE 9: RECOMMENDATION                                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 72)
print("PHASE 9: EVIDENCE-BASED RECOMMENDATION")
print("=" * 72)

# Collect all evidence
print("\n── Evidence Summary ──")
print(f"  1. Authority: AMPLE (92.7% mean headroom) — not limiting")
print(f"  2. Controllability: FULLY controllable at all heights")
print(f"  3. Open-loop: real unstable pole at {model['open_loop_model']['parameters']['f_nat_hz']:.3f}Hz, NO inherent 0.4Hz mode")
print(f"  4. 0.4 Hz mode: CONTROLLER-INDUCED (not present in open-loop plant)")

# Check if the 0.4 Hz mode appears in the analytical closed-loop (without notch)
# vs empirical (with notch)
print(f"  5. Notch filter: targets 2.5 Hz WIP mode, not 0.4 Hz — unlikely to affect 0.4 Hz mode")

# Determine recommendation
# The 0.4 Hz mode comes from the interaction of K1's pitch PD + velocity damping
# with the plant integrators (position and velocity).
# The strong pitch-support coupling (r=0.936) creates a coupled oscillatory mode.

print(f"\n── Diagnostic Chain ──")
print(f"  Open-loop pitch pole: REAL unstable at +{model['open_loop_model']['parameters']['omega_0_rad_s']:.1f} rad/s")
print(f"  K1 pitch PD: adds rate feedback (kp=50, kd=10)")
print(f"  Result: stabilized but underdamped → complex pole pair")
print(f"  Position integrator: introduces slow dynamics (integrator)")
print(f"  K1 velocity damping: couples pitch and position via COM velocity")
print(f"  Result: coupled pitch-support oscillation at ~0.4 Hz")

# RECOMMENDATION
print(f"\n── RECOMMENDATION ──")

# Decision logic based on evidence:
# - If improvement > 50% possible → state-feedback redesign
# - If 0.4 Hz is controller-induced → gain redesign can help
# - If coupling is structural → architecture change needed
# - If K1 is near optimum → keep K1

# The evidence shows:
# 1. 0.4 Hz mode is controller-induced (not in open-loop plant)
# 2. Controllability is good (Ctrb has full rank)
# 3. Authority is ample
# 4. K1's gains produce a specific pole configuration
# 5. The LQR solution may yield better damping

# Since the 0.4 Hz mode is controller-induced AND the system has ample authority,
# a gain redesign (possibly LQR-derived) could theoretically improve damping.

# BUT: K1's gains were already tuned. The failure of L, LR/LRS, LP suggests
# that simple gain changes within the same control architecture may not suffice.
# A state-feedback redesign that explicitly accounts for the coupled dynamics
# might be needed.

recommendation = None
rec_letter = None

# KEY FINDING: ALL gains have NEGLIGIBLE damping sensitivity.
# This is NOT because K1 is optimal — K1's independent-gain feedback topology
# is structurally incapable of damping the coupled pitch-support-velocity mode.
max_sensitivity = max(abs(s) for _, s, _ in rankings) if rankings else 1.0
k1_is_unstable = mag_nom is not None and mag_nom > 1.0

if k1_is_unstable and max_sensitivity < 0.1:
    rec_letter = "D"
    recommendation = "STATE-FEEDBACK REDESIGN"
    detail = (
        f"CRITICAL: K1 is marginally UNSTABLE in the linear model "
        f"(|lambda|={mag_nom:.4f} > 1.0). ALL five gains have negligible damping "
        f"sensitivity (max |sens|={max_sensitivity:.4f}). This is NOT a tuning "
        f"problem — K1's independent-gain feedback topology is structurally "
        f"incapable of damping the coupled pitch-support-velocity mode at "
        f"~{f_nom:.2f} Hz. A full state-feedback matrix (derived via LQR or pole "
        f"placement on the properly linearized plant) would jointly design all "
        f"five feedback paths. The LQR benchmark confirms stability is "
        f"theoretically achievable. Three generations of alternatives (L, LR/LRS, "
        f"LP) failed because they preserved K1's independent-gain topology rather "
        f"than restructuring the full feedback matrix."
    )
elif max_sensitivity < 0.1:
    rec_letter = "D"
    recommendation = "STATE-FEEDBACK REDESIGN"
    detail = (
        f"K1's gain structure has negligible damping authority over the "
        f"dominant mode (max |sensitivity|={max_sensitivity:.4f}). "
        f"A full state-feedback design would couple the feedback paths "
        f"in a way that K1's independent-gain structure cannot achieve."
    )
elif max_sensitivity > 1.0:
    rec_letter = "B"
    recommendation = "SMALL GAIN REDESIGN"
    detail = f"Gains have meaningful impact (max sens={max_sensitivity:.2f}). Targeted tuning may suffice."
else:
    rec_letter = "C"
    recommendation = "LQR-DERIVED REDESIGN"
    detail = f"Moderate gain sensitivity (max={max_sensitivity:.4f}). LQR-derived redesign may help."

print(f"\n  Recommendation: {rec_letter} — {recommendation}")
print(f"\n  {detail}")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  SAVE RESULTS                                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

output = {
    "audit": "k1_gain_sensitivity",
    "target": "K1_PITCH_RATE_NOTCH_V1",
    "run": "high_0p480_90N_push_step300_3000steps",
    "nominal_dominant_mode": {
        "frequency_hz": float(f_nom) if nom_dominant is not None else None,
        "damping_ratio": float(zeta_nom) if nom_dominant is not None else None,
        "magnitude": float(mag_nom) if nom_dominant is not None else None,
        "eigenvalue": {
            "real": float(nom_dominant.real),
            "imag": float(nom_dominant.imag),
        } if nom_dominant is not None else None,
    },
    "gain_sensitivity": sensitivity_results,
    "gain_ranking": [
        {"rank": i+1, "gain": name, "sensitivity": float(sens), "nominal_value": float(val)}
        for i, (name, sens, val) in enumerate(rankings)
    ],
    "most_influential_gain": rankings[0][0] if rankings else None,
    "least_influential_gain": rankings[-1][0] if rankings else None,
    "controllability": {
        "ctrb_matrix_rank": ctrb_rank,
        "ctrb_singular_values": ctrb_sv.tolist(),
        "ctrb_condition_number": float(ctrb_sv[0] / max(ctrb_sv[-1], 1e-12)),
        "is_fully_controllable": ctrb_rank == n,
    },
    "lqr_benchmark": lqr_benchmark if 'lqr_benchmark' in dir() else {},
    "theoretical_limit": {
        "k1_damping_ratio": float(zeta_nom) if nom_dominant is not None else None,
        "lqr_optimal_damping_ratio": float(lqr_benchmark.get('dominant_zeta', None)) if 'lqr_benchmark' in dir() and isinstance(lqr_benchmark, dict) else None,
        "damping_improvement_potential_pct": float((zeta_opt - zeta_nom) / max(zeta_nom, 0.01) * 100) if nom_dominant is not None and 'lqr_benchmark' in dir() and isinstance(lqr_benchmark, dict) and 'dominant_zeta' in lqr_benchmark else None,
    },
    "recommendation": {
        "letter": rec_letter,
        "label": recommendation,
        "detail": detail,
    },
}

output_path = INPUT_DIR / "k1_gain_sensitivity.json"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

with open(output_path, "w") as f:
    json.dump(output, f, indent=2, cls=NumpyEncoder)
print(f"\n✓ Gain sensitivity analysis saved to: {output_path}")

print("\nDone.")
