#!/usr/bin/env python3
"""Phase 3-5,7: K1 Eigenmode Analysis — eigenvalues, participation factors, mode classification.

Loads the state-space model from Phase 1-2 and performs:
- Phase 3: Open-loop mode analysis
- Phase 4: Closed-loop mode analysis
- Phase 5: Participation factor analysis
- Phase 7: Mode classification

STRICT CONSTRAINT: This is ANALYSIS ONLY. Do NOT tune gains or modify K1.
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
OUTPUT_DIR = INPUT_DIR
INPUT_PATH = INPUT_DIR / "k1_state_space_model.json"

# ── Load model ──────────────────────────────────────────────────────
print("=" * 72)
print("Loading state-space model...")
print("=" * 72)

with open(INPUT_PATH) as f:
    model = json.load(f)

state_names = model["state_definition"]["state_names"]
state_dim = model["state_definition"]["state_dim"]
dt = model["state_definition"]["control_dt_s"]

A_open_ct = np.array(model["open_loop_model"]["A_continuous"])
A_open_dt = np.array(model["open_loop_model"]["A_discrete"])
B_open_ct = np.array(model["open_loop_model"]["B_continuous"])
B_open_dt = np.array(model["open_loop_model"]["B_discrete"])

print(f"  State vector: {state_names}")
print(f"  State dimension: {state_dim}")
print(f"  dt = {dt} s")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PHASE 3: OPEN-LOOP MODE ANALYSIS                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 72)
print("PHASE 3: OPEN-LOOP MODE ANALYSIS")
print("=" * 72)

# Continuous-time eigenvalues
eig_ct = np.linalg.eigvals(A_open_ct)
eigvecs_ct = np.linalg.eig(A_open_ct)[1]

print(f"\n── Continuous-Time Open-Loop Modes ──")
print(f"  A_open shape: {A_open_ct.shape}")
print(f"  Characteristic polynomial coefficients:")
char_poly = np.poly(A_open_ct)
print(f"    {char_poly}")

open_loop_modes = []
for i, ev in enumerate(eig_ct):
    freq_hz = np.abs(ev.imag) / (2 * np.pi) if abs(ev.imag) > 1e-10 else 0.0
    damp = -ev.real / np.sqrt(ev.real**2 + ev.imag**2) if abs(ev) > 1e-10 else 0.0
    if np.isnan(damp):
        damp = 0.0
    mode_type = "UNSTABLE_REAL" if ev.real > 0 and abs(ev.imag) < 1e-8 else (
        "STABLE_REAL" if ev.real < 0 and abs(ev.imag) < 1e-8 else (
        "OSCILLATORY_UNSTABLE" if ev.real > 0 else "OSCILLATORY_STABLE"
    ))
    if abs(ev.real) < 1e-8 and abs(ev.imag) < 1e-8:
        mode_type = "INTEGRATOR"

    # Mode shape: which state dominates?
    evec = eigvecs_ct[:, i]
    evec_norm = np.abs(evec) / (np.sum(np.abs(evec)) + 1e-12)

    print(f"\n  Mode {i}:")
    print(f"    Eigenvalue: {ev.real:+.6f} {ev.imag:+.6f}j")
    print(f"    Frequency: {freq_hz:.4f} Hz")
    print(f"    Damping ratio: {damp:+.4f}")
    print(f"    Type: {mode_type}")
    print(f"    Mode shape (normalized |eigenvector|):")
    for j, name in enumerate(state_names):
        bar = "█" * int(evec_norm[j] * 50)
        print(f"      {name:>25s}: {evec_norm[j]:.4f} {bar}")

    open_loop_modes.append({
        "index": i,
        "eigenvalue": {"real": float(ev.real), "imag": float(ev.imag)},
        "frequency_hz": freq_hz,
        "damping_ratio": float(damp),
        "type": mode_type,
        "mode_shape": {state_names[j]: float(evec_norm[j]) for j in range(state_dim)},
        "time_constant_s": float(-1.0 / ev.real) if ev.real < -1e-10 else (
            float('inf') if ev.real > -1e-10 and ev.real < 1e-10 else None
        ),
    })

# Discrete-time eigenvalues for comparison
eig_dt = np.linalg.eigvals(A_open_dt)
eig_ct_from_dt = np.log(eig_dt) / dt  # inverse ZOH

print(f"\n── Discrete-Time Open-Loop Modes (dt={dt}s) ──")
for i, ev in enumerate(eig_dt):
    ev_ct = eig_ct_from_dt[i]
    freq_hz = np.abs(np.angle(ev)) / (2 * np.pi * dt)
    mag = np.abs(ev)
    stability = "UNSTABLE" if mag > 1.0 else ("STABLE" if mag < 1.0 else "MARGINALLY_STABLE")
    print(f"  Mode {i}: λ_d={ev.real:+.6f}{ev.imag:+.6f}j  |λ|={mag:.4f}  f={freq_hz:.4f}Hz  "
          f"λ_c={ev_ct.real:+.4f}{ev_ct.imag:+.4f}j  {stability}")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PHASE 4: CLOSED-LOOP MODE ANALYSIS                                ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 72)
print("PHASE 4: CLOSED-LOOP MODE ANALYSIS")
print("=" * 72)

cl_models = model["closed_loop_models"]

# 4A: Analytical closed-loop (A_cl = A_open_dt - B_open_dt * K)
print("\n── 4A: Analytical Closed-Loop (K1 gains, no notch) ──")

# K1 gain vector: maps state x to torque u
# u = -K * x  where:
#   u = -(kp_pitch * pitch + kd_pitch * pitch_rate
#         + k_total_vel * (-com_velocity) + k_position * (-position_error)
#         + 0 * wheel_vel_mean)
# Note: position centering torque is clipped at ±3 Nm, but for linearization
# we use the unclipped gain.
#
# Sign conventions:
#   tau = kp_pitch * pitch_x + kd_pitch * pitch_rate
#       - k_total_vel * com_velocity
#       - k_position * position_error
#       - k_wheel_vel * wheel_vel_mean * 0.5 (per wheel = -0.25 mean effect)
#
# So: u = -K * x where K = [-kp_pitch, -kd_pitch, +k_position, +k_total_vel, ...]
# Wait, let me be more careful.
#
# K1 produces:
#   tau_common ≈ kp_pitch * pitch + kd_pitch * pitch_rate_filt
#                - k_total_vel_damping * com_vel
#                - k_position * position_error
#                - 0 * wheel_vel_mean  (wheel velocity damping is per-wheel, differential)
#
# For the common torque (ignoring per-wheel differential):
#   u_common = 50*pitch + 10*pitch_rate -20*com_vel -40*pos_error + 0*wheel_vel
#
# In state-feedback form: u = -K*x where:
#   u = -( -50*pitch - 10*pitch_rate + 20*com_vel + 40*pos_error + 0*wheel_vel )
#   K = [-50, -10, 40, 20, 0]
#
# Wait, that's u = -Kx: Kx = -u = -50*pitch - 10*rate + 40*pos + 20*vel
# So K = [-50, -10, 40, 20, 0]
#
# But wait - position torque is CAPPED at ±3 Nm. For small errors, the gain is 40 Nm/m.
# For the linearization, I'll use the uncapped gain to see the "ideal" pole locations,
# then note that the cap may affect the actual dynamics.

gains = model["controller_gains"]
K_k1 = np.array([
    -gains["kp_pitch"],          # -50: negative feedback on pitch
    -gains["kd_pitch"],          # -10: negative feedback on pitch rate
    gains["k_position"],         # +40: NEGATIVE feedback on position_error (u = -Kx, Kx includes -40*pos_err → actual u = +40*pos_err, but pos_err sign convention?)
    gains["k_total_velocity_damping"],  # +20: NEGATIVE feedback on velocity (u opposes velocity)
    0.0,                         # wheel velocity damping is per-wheel only
])

# Actually let me double-check the sign convention.
# K1 computes: tau = kp_pitch * pitch + kd_pitch * pitch_rate - k_total * com_vel - k_pos * pos_error
# In state-space: u = [kp_pitch, kd_pitch, -k_pos, -k_total, 0] @ x
# So: u = K_raw @ x where K_raw = [kp_pitch, kd_pitch, -k_pos, -k_total, 0]
# In feedback form: u = -K @ x → K = [-kp_pitch, -kd_pitch, k_pos, k_total, 0]
K_k1_raw = np.array([
    gains["kp_pitch"],           # +50
    gains["kd_pitch"],           # +10
    -gains["k_position"],        # -40
    -gains["k_total_velocity_damping"],  # -20
    0.0,                         # 0
])

# For linear analysis, we use: u = K_k1_raw @ x
# So A_cl = A_d + B_d @ K_k1_raw  (discrete-time)

A_cl_analytical_dt = A_open_dt + B_open_dt @ K_k1_raw.reshape(1, -1)
eig_cl_analytical = np.linalg.eigvals(A_cl_analytical_dt)

print(f"  K1 gain vector (raw): {K_k1_raw}")
print(f"  K1 gain vector (feedback form K): {K_k1}")
print(f"\n  Analytical closed-loop eigenvalues:")
for i, ev in enumerate(eig_cl_analytical):
    mag = np.abs(ev)
    freq_hz = np.abs(np.angle(ev)) / (2 * np.pi * dt)
    damp = -np.cos(np.angle(ev)) if abs(np.imag(ev)) > 1e-10 else (
        1.0 if ev.real > 0 else -1.0
    )
    stability = "STABLE" if mag < 1.0 else ("UNSTABLE" if mag > 1.0 else "MARGINAL")
    print(f"    λ{i}: {ev.real:+.6f} {ev.imag:+.6f}j  |λ|={mag:.4f}  f={freq_hz:.4f}Hz  ζ={damp:+.4f}  {stability}")

# 4B: Empirical closed-loop from telemetry
print(f"\n── 4B: Empirical Closed-Loop (from telemetry) ──")

all_cl_modes = {}
for height_key, cl_data in cl_models.get("by_height", {}).items():
    h_m = cl_data["height_m"]
    A_cl_dt = np.array(cl_data["A_closed_discrete"])
    eig_cl = np.linalg.eigvals(A_cl_dt)
    eigvecs_cl = np.linalg.eig(A_cl_dt)[1]

    print(f"\n  Height {h_m:.2f}m ({cl_data['n_pairs']} pairs):")
    print(f"    R²: {[f'{r:.4f}' for r in cl_data['r2_scores']]}")

    modes_this_height = []
    for i, ev in enumerate(eig_cl):
        mag = np.abs(ev)
        freq_hz = np.abs(np.angle(ev)) / (2 * np.pi * dt)
        damp = -np.cos(np.angle(ev)) if abs(np.imag(ev)) > 1e-10 else (
            1.0 if ev.real > 0 else -1.0
        )
        is_oscillatory = abs(np.imag(ev)) > 1e-8
        stability = "STABLE" if mag < 1.0 else ("UNSTABLE" if mag > 1.0 else "MARGINAL")

        # Mode shape
        evec = eigvecs_cl[:, i]
        evec_norm = np.abs(evec) / (np.sum(np.abs(evec)) + 1e-12)

        # Time constant (discrete → continuous)
        if abs(ev) > 1e-10 and ev.real != 0:
            tau_s = -dt / np.log(mag) if mag > 0 and mag != 1.0 else float('inf')
        else:
            tau_s = float('inf')

        print(f"    λ{i}: {ev.real:+.6f} {ev.imag:+.6f}j  |λ|={mag:.4f}  "
              f"f={freq_hz:.4f}Hz  ζ={damp:+.4f}  τ={tau_s:.4f}s  {stability}")
        print(f"      Mode shape: ", end="")
        for j, name in enumerate(state_names):
            print(f"{name}={evec_norm[j]:.3f}  ", end="")
        print()

        modes_this_height.append({
            "index": i,
            "eigenvalue_real": float(ev.real),
            "eigenvalue_imag": float(ev.imag),
            "magnitude": float(mag),
            "frequency_hz": float(freq_hz),
            "damping_ratio": float(damp),
            "time_constant_s": float(tau_s) if tau_s != float('inf') else None,
            "is_oscillatory": is_oscillatory,
            "is_stable": mag < 1.0,
            "stability": stability,
            "mode_shape": {state_names[j]: float(evec_norm[j]) for j in range(state_dim)},
        })

    all_cl_modes[height_key] = modes_this_height

# Global model
if "global_post_push" in cl_models:
    A_cl_global = np.array(cl_models["global_post_push"]["A_closed_discrete"])
    eig_cl_global = np.linalg.eigvals(A_cl_global)
    eigvecs_cl_global = np.linalg.eig(A_cl_global)[1]

    print(f"\n  Global model ({cl_models['global_post_push']['n_pairs']} pairs):")
    global_modes = []
    for i, ev in enumerate(eig_cl_global):
        mag = np.abs(ev)
        freq_hz = np.abs(np.angle(ev)) / (2 * np.pi * dt)
        damp = -np.cos(np.angle(ev)) if abs(np.imag(ev)) > 1e-10 else (
            1.0 if ev.real > 0 else -1.0
        )
        is_oscillatory = abs(np.imag(ev)) > 1e-8
        stability = "STABLE" if mag < 1.0 else ("UNSTABLE" if mag > 1.0 else "MARGINAL")
        evec = eigvecs_cl_global[:, i]
        evec_norm = np.abs(evec) / (np.sum(np.abs(evec)) + 1e-12)

        tau_s = -dt / np.log(mag) if mag > 0 and abs(mag - 1.0) > 1e-8 and mag > 0 else float('inf')

        print(f"    λ{i}: {ev.real:+.6f} {ev.imag:+.6f}j  |λ|={mag:.4f}  "
              f"f={freq_hz:.4f}Hz  ζ={damp:+.4f}  τ={tau_s:.4f}s  {stability}")
        print(f"      Mode shape: ", end="")
        for j, name in enumerate(state_names):
            print(f"{name}={evec_norm[j]:.3f}  ", end="")
        print()

        global_modes.append({
            "index": i,
            "eigenvalue_real": float(ev.real),
            "eigenvalue_imag": float(ev.imag),
            "magnitude": float(mag),
            "frequency_hz": float(freq_hz),
            "damping_ratio": float(damp),
            "time_constant_s": float(tau_s) if tau_s != float('inf') else None,
            "is_oscillatory": is_oscillatory,
            "is_stable": mag < 1.0,
            "stability": stability,
            "mode_shape": {state_names[j]: float(evec_norm[j]) for j in range(state_dim)},
        })

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PHASE 5: PARTICIPATION FACTOR ANALYSIS                            ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 72)
print("PHASE 5: PARTICIPATION FACTOR ANALYSIS")
print("=" * 72)

# Participation factor p_ki = |v_ki| * |w_ki| / (|v_k| * |w_k|)
# where v_k = k-th right eigenvector, w_k = k-th left eigenvector

def compute_participation_factors(A, state_names, dt_val):
    """Compute participation factors for all modes of A."""
    n = A.shape[0]
    eigvals, right_eigvecs = np.linalg.eig(A)
    left_eigvecs = np.linalg.inv(right_eigvecs).T  # left eigenvectors = rows of inv(V)

    results = []
    for k in range(n):
        ev = eigvals[k]
        rv = right_eigvecs[:, k]   # right eigenvector
        lv = left_eigvecs[:, k]    # left eigenvector

        # Participation factors: p_{ki} = lv_i * rv_i
        # (product form; sum over k of p_{ki} = 1 for each mode)
        pf_raw = lv * rv  # elementwise
        pf_abs = np.abs(pf_raw)
        pf_norm = pf_abs / (np.sum(pf_abs) + 1e-12)

        freq_hz = np.abs(np.angle(ev)) / (2 * np.pi * dt_val)
        damp = -np.cos(np.angle(ev)) if abs(np.imag(ev)) > 1e-10 else (
            1.0 if ev.real > 0 else -1.0
        )
        mag = np.abs(ev)

        results.append({
            "mode_index": k,
            "eigenvalue": {"real": float(ev.real), "imag": float(ev.imag)},
            "magnitude": float(mag),
            "frequency_hz": float(freq_hz),
            "damping_ratio": float(damp),
            "participation_factors": {
                state_names[i]: {
                    "pf_normalized": float(pf_norm[i]),
                    "pf_raw_real": float(pf_raw[i].real),
                    "pf_raw_imag": float(pf_raw[i].imag),
                }
                for i in range(n)
            },
            "dominant_state": state_names[int(np.argmax(pf_norm))],
            "top_states": sorted(
                [(state_names[i], float(pf_norm[i])) for i in range(n)],
                key=lambda x: -x[1]
            ),
        })
    return results

# For each height-specific closed-loop model
participation_results = {}
for height_key, cl_data in cl_models.get("by_height", {}).items():
    A_cl = np.array(cl_data["A_closed_discrete"])
    h_m = cl_data["height_m"]
    print(f"\n── Height {h_m:.2f}m ──")

    pf_results = compute_participation_factors(A_cl, state_names, dt)
    participation_results[height_key] = pf_results

    for pf in pf_results:
        print(f"  Mode {pf['mode_index']}: f={pf['frequency_hz']:.4f}Hz, ζ={pf['damping_ratio']:+.4f}")
        print(f"    Dominant state: {pf['dominant_state']}")
        print(f"    Participation factors:")
        for state, val in pf["top_states"]:
            bar = "█" * int(val * 50)
            print(f"      {state:>25s}: {val:.4f} {bar}")

# Also compute for the global model
if "global_post_push" in cl_models:
    A_cl_global = np.array(cl_models["global_post_push"]["A_closed_discrete"])
    print(f"\n── Global Model ──")
    global_pf = compute_participation_factors(A_cl_global, state_names, dt)
    participation_results["global"] = global_pf
    for pf in global_pf:
        print(f"  Mode {pf['mode_index']}: f={pf['frequency_hz']:.4f}Hz, ζ={pf['damping_ratio']:+.4f}")
        print(f"    Dominant state: {pf['dominant_state']}")
        for state, val in pf["top_states"]:
            bar = "█" * int(val * 50)
            print(f"      {state:>25s}: {val:.4f} {bar}")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PHASE 7: MODE CLASSIFICATION                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 72)
print("PHASE 7: MODE CLASSIFICATION")
print("=" * 72)

def classify_mode(pf_result, is_open_loop=False):
    """Classify a mode based on its participation factors and frequency."""
    pf_dict = pf_result["participation_factors"]
    f_hz = pf_result["frequency_hz"]
    damp = pf_result["damping_ratio"]

    # Compute state group contributions
    pitch_contrib = pf_dict["pitch_x"]["pf_normalized"] + pf_dict["pitch_rate_x"]["pf_normalized"]
    support_contrib = pf_dict["position_error"]["pf_normalized"]
    vel_contrib = pf_dict["com_velocity"]["pf_normalized"]
    wheel_contrib = pf_dict["wheel_vel_mean"]["pf_normalized"]

    # Classification logic
    if is_open_loop:
        if abs(pf_result["eigenvalue"]["imag"]) < 1e-8 and pf_result["eigenvalue"]["real"] > 0:
            return "PLANT_UNSTABLE_REAL_POLE"
        elif abs(pf_result["eigenvalue"]["imag"]) < 1e-8 and abs(pf_result["eigenvalue"]["real"]) < 1e-8:
            return "INTEGRATOR"
        else:
            return "PLANT_MODE"

    # Closed-loop classification
    if f_hz < 0.01:
        return "STEADY_STATE_INTEGRATOR"

    if pitch_contrib > 0.5 and support_contrib > 0.2:
        mode_type = "COUPLED_PITCH_SUPPORT_MODE"
    elif pitch_contrib > 0.5:
        mode_type = "PITCH_DOMINANT_MODE"
    elif support_contrib > 0.5:
        mode_type = "SUPPORT_DOMINANT_MODE"
    elif vel_contrib > 0.5:
        mode_type = "VELOCITY_DAMPING_MODE"
    elif wheel_contrib > 0.5:
        mode_type = "WHEEL_DYNAMICS_MODE"
    elif pitch_contrib > 0.3 and vel_contrib > 0.3:
        mode_type = "PITCH_VELOCITY_HYBRID_MODE"
    elif support_contrib > 0.3 and vel_contrib > 0.3:
        mode_type = "SUPPORT_VELOCITY_HYBRID_MODE"
    else:
        mode_type = "HYBRID_MODE"

    # Refine with frequency
    if 0.3 < f_hz < 0.5:
        mode_type = "OBSERVED_0P4HZ_" + mode_type
    elif f_hz > 2.0:
        mode_type = "WIP_" + mode_type

    # Refine with damping
    if damp < 0.1:
        mode_type += "_UNDERDAMPED"
    elif damp < 0.5:
        mode_type += "_MODERATELY_DAMPED"
    else:
        mode_type += "_WELL_DAMPED"

    # Source classification
    if isinstance(mode_type, str) and "PLANT" in mode_type:
        source = "PLANT_STRUCTURAL_MODE"
    elif isinstance(mode_type, str) and "COUPLED" in mode_type:
        source = "COUPLED_PITCH_SUPPORT_MODE"
    elif isinstance(mode_type, str) and ("PITCH_DOMINANT" in mode_type or "SUPPORT_DOMINANT" in mode_type):
        source = "CONTROLLER_INDUCED_MODE"
    elif isinstance(mode_type, str) and "VELOCITY" in mode_type:
        source = "VELOCITY_DAMPING_MODE"
    else:
        source = "HYBRID_MODE"

    return mode_type

# Classify open-loop modes
print("\n── Open-Loop Mode Classification ──")
for olm in open_loop_modes:
    # Build pseudo participation factor from mode shape
    pseudo_pf = {
        "participation_factors": {
            name: {"pf_normalized": olm["mode_shape"][name]}
            for name in state_names
        },
        "frequency_hz": olm["frequency_hz"],
        "damping_ratio": olm["damping_ratio"],
        "eigenvalue": olm["eigenvalue"],
    }
    cls = classify_mode(pseudo_pf, is_open_loop=True)
    print(f"  Mode {olm['index']}: {olm['frequency_hz']:.4f}Hz → {cls}")
    olm["classification"] = cls

# Classify closed-loop modes
print("\n── Closed-Loop Mode Classification ──")
mode_classifications = {}
for height_key, modes in all_cl_modes.items():
    print(f"\n  Height {height_key}m:")
    mode_classifications[height_key] = []
    for mode_info in modes:
        # Find matching participation factor result
        pf_match = None
        if height_key in participation_results:
            for pf in participation_results[height_key]:
                if pf["mode_index"] == mode_info["index"]:
                    pf_match = pf
                    break
        if pf_match is None:
            pf_match = {
                "participation_factors": {
                    name: {"pf_normalized": mode_info["mode_shape"][name]}
                    for name in state_names
                },
                "frequency_hz": mode_info["frequency_hz"],
                "damping_ratio": mode_info["damping_ratio"],
                "eigenvalue": {
                    "real": mode_info["eigenvalue_real"],
                    "imag": mode_info["eigenvalue_imag"],
                },
            }
        cls = classify_mode(pf_match, is_open_loop=False)
        mode_info["classification"] = cls
        mode_classifications[height_key].append({
            "mode_index": mode_info["index"],
            "frequency_hz": mode_info["frequency_hz"],
            "damping_ratio": mode_info["damping_ratio"],
            "classification": cls,
            "dominant_state": pf_match.get("dominant_state", "N/A"),
        })
        print(f"    Mode {mode_info['index']}: {mode_info['frequency_hz']:.4f}Hz, "
              f"ζ={mode_info['damping_ratio']:+.4f} → {cls}")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  KEY FINDING: 0.4 Hz Mode Identification                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 72)
print("KEY FINDING: 0.4 Hz OSCILLATION IDENTIFICATION")
print("=" * 72)

# Find the mode closest to 0.4 Hz
target_freq = 0.4
best_match = None
best_distance = float('inf')

for height_key, modes in all_cl_modes.items():
    for mode_info in modes:
        if mode_info["is_oscillatory"]:
            dist = abs(mode_info["frequency_hz"] - target_freq)
            if dist < best_distance:
                best_distance = dist
                best_match = {
                    "height": height_key,
                    "mode": mode_info,
                }

if best_match:
    m = best_match["mode"]
    print(f"\n  Best match for 0.4 Hz oscillation:")
    print(f"    Height: {best_match['height']}m")
    print(f"    Frequency: {m['frequency_hz']:.4f} Hz")
    print(f"    Damping ratio: {m['damping_ratio']:+.4f}")
    print(f"    Magnitude: {m['magnitude']:.4f}")
    print(f"    Stability: {m['stability']}")
    print(f"    Classification: {m['classification']}")
    print(f"    Time constant: {m['time_constant_s']:.4f}s" if m['time_constant_s'] else "    Time constant: N/A")
    print(f"    Mode shape:")
    for state, val in sorted(m["mode_shape"].items(), key=lambda x: -x[1]):
        bar = "█" * int(val * 50)
        print(f"      {state:>25s}: {val:.4f} {bar}")

    # Compute continuous-time equivalent
    ev_ct = np.log(m["magnitude"]) / dt + 1j * m["frequency_hz"] * 2 * np.pi
    print(f"\n    Continuous-time equivalent:")
    print(f"      λ_ct ≈ {ev_ct.real:+.4f} ± {ev_ct.imag:+.4f}j")
    print(f"      ω_n = {abs(ev_ct):.4f} rad/s")
    print(f"      ζ = {m['damping_ratio']:+.4f}")
    # Settling time to 2%: t_s = 4 / (ζ * ω_n)
    if abs(ev_ct.real) > 1e-10:
        t_settle = 4.0 / abs(ev_ct.real)
        print(f"      Settling time (2%): {t_settle:.2f}s ({t_settle/dt:.0f} steps)")
    print(f"      Period: {1.0/m['frequency_hz']:.3f}s" if m['frequency_hz'] > 0 else "")
else:
    print("\n  No oscillatory mode found near 0.4 Hz!")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  SAVE RESULTS                                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

output = {
    "audit": "k1_eigenmodes",
    "target": "K1_PITCH_RATE_NOTCH_V1",
    "run": "high_0p480_90N_push_step300_3000steps",
    "state_names": state_names,
    "dt_s": dt,
    "open_loop_modes": open_loop_modes,
    "analytical_closed_loop": {
        "A_closed_discrete": A_cl_analytical_dt.tolist(),
        "eigenvalues": [
            {"real": float(ev.real), "imag": float(ev.imag),
             "magnitude": float(np.abs(ev)),
             "frequency_hz": float(np.abs(np.angle(ev)) / (2 * np.pi * dt)),
             "damping_ratio": float(
                 -np.cos(np.angle(ev)) if abs(np.imag(ev)) > 1e-10
                 else (1.0 if ev.real > 0 else -1.0)
             )}
            for ev in eig_cl_analytical
        ],
    },
    "empirical_closed_loop": {
        "by_height": {
            hk: {
                "height_m": cl_data["height_m"],
                "n_pairs": cl_data["n_pairs"],
                "r2_scores": cl_data["r2_scores"],
                "modes": all_cl_modes.get(hk, []),
            }
            for hk, cl_data in cl_models.get("by_height", {}).items()
        },
        "global": {
            "modes": global_modes if "global_post_push" in cl_models else [],
        },
    },
    "participation_factors": {
        hk: participation_results[hk]
        for hk in participation_results
    },
    "mode_classifications": mode_classifications,
    "zero_point_four_hz_identification": best_match,
}

output_path = OUTPUT_DIR / "k1_eigenmodes.json"

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
print(f"\n✓ Eigenmode analysis saved to: {output_path}")

# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("SUMMARY: Eigenmode Analysis")
print("=" * 72)
print(f"  Open-loop modes: {len(open_loop_modes)}")
print(f"  Open-loop has oscillatory mode: {any(abs(m['eigenvalue']['imag']) > 1e-8 for m in open_loop_modes)}")
print(f"  Closed-loop (analytical): {len(eig_cl_analytical)} eigenvalues")
print(f"  Closed-loop (empirical): heights analyzed = {len(all_cl_modes)}")
if best_match:
    print(f"  0.4 Hz mode identified: f={best_match['mode']['frequency_hz']:.4f}Hz, "
          f"ζ={best_match['mode']['damping_ratio']:+.4f}")
    print(f"  Classification: {best_match['mode']['classification']}")

print("\nDone.")
