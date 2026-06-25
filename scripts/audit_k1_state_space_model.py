#!/usr/bin/env python3
"""Phase 1-2: K1 State-Space Model — state definition, data extraction, linearization.

System Audit Task: Determine the closed-loop dynamics of K1_PITCH_RATE_NOTCH_V1.
This script defines the sagittal state vector, extracts telemetry data,
builds open-loop (analytical TWIP) and closed-loop (regression-based)
linear state-space models, and saves all matrices for downstream analysis.

STRICT CONSTRAINT: This is ANALYSIS ONLY. Do NOT tune gains or modify K1.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Unicode stdout ──────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TELEMETRY_PATH = (
    PROJECT_ROOT / "outputs"
    / "d_baseline_single_90n_10step_push_step300_3000"
    / "telemetry_1782262602.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eigenmode_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Helper ──────────────────────────────────────────────────────────
def _safe_float(val, default=0.0):
    if isinstance(val, str) and val in ("True", "False"):
        return 1.0 if val == "True" else 0.0
    try:
        result = float(val)
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PHASE 1: STATE VECTOR DEFINITION                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("=" * 72)
print("PHASE 1: STATE VECTOR DEFINITION")
print("=" * 72)

STATE_DEFINITION = {
    "state_names": [
        "pitch_x",              # 0: body pitch angle [rad]
        "pitch_rate_x",         # 1: body pitch rate [rad/s] (raw, not notch-filtered)
        "position_error",       # 2: support-center sagittal position error [m]
        "com_velocity",         # 3: COM sagittal velocity [m/s]
        "wheel_vel_mean",       # 4: mean wheel angular velocity [rad/s]
    ],
    "state_dim": 5,
    "input_names": [
        "tau_wheel_common",     # 0: common wheel torque [Nm]
    ],
    "input_dim": 1,
    "state_units": ["rad", "rad/s", "m", "m/s", "rad/s"],
    "input_units": ["Nm"],
    "control_dt_s": 0.01,       # 100 Hz control loop
    "physics_dt_s": 0.002,      # 500 Hz MuJoCo physics
    "n_physics_substeps": 5,
}

print(f"\nState dimension: {STATE_DEFINITION['state_dim']}")
print(f"Input dimension: {STATE_DEFINITION['input_dim']}")
print(f"Control dt: {STATE_DEFINITION['control_dt_s']} s")
print(f"\nState vector x = [")
for i, (name, unit) in enumerate(zip(STATE_DEFINITION["state_names"], STATE_DEFINITION["state_units"])):
    print(f"    x[{i}] = {name}  [{unit}]")
print("]")
print(f"\nInput vector u = [tau_wheel_common]  [Nm]")

# ── Justification ────────────────────────────────────────────────────
print("\n── State Selection Justification ──")
print("""
1. pitch_x: The primary balance variable. K1's largest feedback term
   (kp_pitch=50.0, contributing ~27% of post-push torque). This is the
   inverted pendulum angle that must be stabilized.

2. pitch_rate_x: K1 applies rate damping (kd_pitch=10.0, ~12% of torque).
   Using RAW (unfiltered) pitch rate captures plant dynamics; the notch
   filter is treated as part of the controller transfer function.

3. position_error: Support-center sagittal offset. K1 applies position
   centering (k_position=40.0, capped at ±3 Nm, ~26% of torque). This
   is the state that couples with pitch via the 0.936 correlation.

4. com_velocity: COM sagittal velocity. K1 applies velocity damping
   (k_velocity=15.0 + kd_com_vy=5.0 = 20.0 total, ~34% of torque).
   Critical for post-push energy dissipation.

5. wheel_vel_mean: Mean wheel angular velocity. K1 applies per-wheel
   velocity damping (k_wheel_velocity=0.5). Captures the wheel dynamics
   that convert torque into support motion.

WHY NOT include:
- roll_y / yaw_z: Lateral/transverse dynamics — separate from sagittal.
- cp_error: K1 disables CP feedback (kp_cp=0), so not relevant.
- support_velocity: K1 disables this (k_support_velocity=0).
- wheel_vel_left/right separately: Mean captures sagittal dynamics;
  differential component is small and off-axis.
- COM height: K1 doesn't directly control height; posture PD handles it.
- Notch filter states: The 2.5 Hz notch is treated as part of the
  controller dynamics; closed-loop modes from telemetry capture its
  effective behavior.
""")

# ── Telemetry column mapping ─────────────────────────────────────────
TELEM_COLS = {
    "pitch_x":                   "pitch_x",           # rad
    "pitch_rate_x":              "pitch_rate_x_rad_s",  # rad/s (raw, unfiltered)
    "position_error":            "sagittal_position_error_m",    # m
    "com_velocity":              "sagittal_velocity_m_s",        # m/s (projected onto sagittal axis)
    "wheel_vel_mean":            "wheel_vel_mean_rad_s",         # rad/s
    "tau_left":                  "tau_left",          # Nm
    "tau_right":                 "tau_right",         # Nm
    "tau_common_unclipped":      "tau_common_unclipped",  # Nm
    "tau_pitch":                 "tau_pitch",
    "tau_pitch_rate":            "tau_pitch_rate",
    "tau_sagittal_velocity":     "tau_sagittal_velocity",
    "tau_position":              "tau_position",
    "tau_cp":                    "tau_cp",
    "tau_com_vy":                "tau_com_vy",
    "contact_valid":             "contact_valid",
    "terminated":                "terminated",
    "push_active":               "push_active",
    "height":                    "com_z",             # m, for height binning
    "time":                      "time",
    "step":                      "step",
}


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PHASE 2: LOCAL LINEARIZATION                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 72)
print("PHASE 2: LOCAL LINEARIZATION")
print("=" * 72)

# ── Load telemetry ───────────────────────────────────────────────────
print(f"\nLoading telemetry: {TELEMETRY_PATH}")
df = pd.read_csv(TELEMETRY_PATH)
print(f"  Total steps: {len(df)}")

# Extract state columns
X_raw = np.zeros((len(df), STATE_DEFINITION["state_dim"]))
for i, name in enumerate(STATE_DEFINITION["state_names"]):
    col = TELEM_COLS[name]
    if col in df.columns:
        X_raw[:, i] = np.array([_safe_float(v) for v in df[col]])
    else:
        print(f"  WARNING: column '{col}' not found in telemetry!")

# Extract input (common wheel torque)
u_cols_present = []
tau_left = np.zeros(len(df))
tau_right = np.zeros(len(df))
if TELEM_COLS["tau_left"] in df.columns:
    tau_left = np.array([_safe_float(v) for v in df[TELEM_COLS["tau_left"]]])
    u_cols_present.append("tau_left")
if TELEM_COLS["tau_right"] in df.columns:
    tau_right = np.array([_safe_float(v) for v in df[TELEM_COLS["tau_right"]]])
    u_cols_present.append("tau_right")

# Common wheel torque = mean of left and right
U_raw = (tau_left + tau_right) / 2.0

# Also extract component torques for reference
component_torques = {}
for key in ["tau_pitch", "tau_pitch_rate", "tau_sagittal_velocity",
            "tau_position", "tau_cp", "tau_com_vy", "tau_common_unclipped"]:
    col = TELEM_COLS[key]
    if col in df.columns:
        component_torques[key] = np.array([_safe_float(v) for v in df[col]])

# Extract contact and push flags
contact_valid = np.ones(len(df), dtype=bool)
if TELEM_COLS["contact_valid"] in df.columns:
    cv = np.array([_safe_float(v) for v in df[TELEM_COLS["contact_valid"]]])
    contact_valid = cv > 0.5

push_active = np.zeros(len(df), dtype=bool)
if TELEM_COLS["push_active"] in df.columns:
    pa = np.array([_safe_float(v) for v in df[TELEM_COLS["push_active"]]])
    push_active = pa > 0.5

# Extract height
height = np.zeros(len(df))
if TELEM_COLS["height"] in df.columns:
    height = np.array([_safe_float(v) for v in df[TELEM_COLS["height"]]])

print(f"  State columns extracted: {STATE_DEFINITION['state_names']}")
print(f"  Input columns found: {u_cols_present}")
print(f"  Contact valid steps: {contact_valid.sum()}/{len(df)}")
print(f"  Push active steps: {push_active.sum()}/{len(df)}")

# ── Equilibrium Analysis ─────────────────────────────────────────────
print("\n── Equilibrium Analysis ──")

# Find equilibrium: pre-push, both feet in contact, stable height
pre_push_mask = (np.arange(len(df)) < 300) & contact_valid
equil_X = X_raw[pre_push_mask]
equil_U = U_raw[pre_push_mask]

x_eq = np.mean(equil_X, axis=0)
x_std = np.std(equil_X, axis=0)
u_eq = np.mean(equil_U)

print(f"  Pre-push equilibrium (steps 0-299, contact only):")
for i, name in enumerate(STATE_DEFINITION["state_names"]):
    print(f"    {name:>25s}: mean={x_eq[i]:+.6f}, std={x_std[i]:.6f} [{STATE_DEFINITION['state_units'][i]}]")
print(f"    {'tau_wheel_common':>25s}: mean={u_eq:+.6f} [Nm]")

# ── Data Segmentation by Height ──────────────────────────────────────
print("\n── Height Binning ──")

# Bin the post-push data by height for height-specific linearization
POST_PUSH_START = 309  # step after 10-step push window (push at 300-309)
post_push_mask = (np.arange(len(df)) >= POST_PUSH_START) & contact_valid
post_push_height = height[post_push_mask]

target_heights = [0.48, 0.40, 0.33]
height_bins = {}
tolerance = 0.02

for h_target in target_heights:
    h_mask = np.abs(post_push_height - h_target) < tolerance
    h_indices = np.where(post_push_mask)[0][h_mask]
    if len(h_indices) >= 10:
        height_bins[h_target] = {"indices": h_indices, "count": len(h_indices)}
        print(f"  Height {h_target:.2f} m: {len(h_indices)} samples")
    else:
        print(f"  Height {h_target:.2f} m: {len(h_indices)} samples (INSUFFICIENT)")

# ── 2A: Analytical Open-Loop TWIP Model ──────────────────────────────
print("\n── 2A: Analytical Open-Loop TWIP Model ──")

# Physics parameters
g = 9.81
M_total = 7.7       # kg (from MJCF body mass sum)
L_com = 0.54         # m (CoM height above wheel axis, from LQR model)
r_wheel = 0.06        # m (wheel radius)
I_wheel = 0.00012247  # kg.m² (single wheel inertia)
m_wheel = 0.1         # kg (single wheel mass)
dt = STATE_DEFINITION["control_dt_s"]  # 0.01 s

# TWIP dynamics linearized around upright equilibrium:
# The critical insight: the open-loop plant has a REAL unstable pole
# at sqrt(g/L_com), NOT an oscillatory mode.

omega_0 = np.sqrt(g / L_com)  # natural frequency of the inverted pendulum
f_nat_hz = omega_0 / (2 * np.pi)

print(f"  g = {g} m/s²")
print(f"  L_com = {L_com} m (CoM above wheel axis)")
print(f"  r_wheel = {r_wheel} m")
print(f"  M_total = {M_total} kg")
print(f"  omega_0 = sqrt(g/L) = {omega_0:.3f} rad/s")
print(f"  f_nat = {f_nat_hz:.3f} Hz")

# State: [pitch, pitch_rate, position_error, com_velocity, wheel_vel_mean]
# Open-loop continuous-time A matrix (no controller):
#
# From inverted pendulum + cart dynamics:
#   d(pitch)/dt = pitch_rate
#   d(pitch_rate)/dt = (g/L)*pitch + alpha*tau
#   d(pos_error)/dt = -com_vel  (position error decreases as COM moves forward)
#   d(com_vel)/dt = beta*tau
#   d(wheel_vel)/dt = gamma*tau
#
# Where alpha, beta, gamma come from the rigid-body dynamics.
# We can derive approximate values from the torque sensitivity
# measured in the Phase 2 controllability audit:
#   d(pitch_accel)/d(tau) = 4.16 rad/s²/Nm at 0.48m (from Phase 2 audit)
#   d(com_accel)/d(tau) = 0.17 m/s²/Nm
#   d(support_accel)/d(tau) = 1.38 m/s²/Nm

# Use measured sensitivities at 0.48m from prior audit
alpha_pitch = 4.164   # rad/s² per Nm: pitch acceleration per unit torque
beta_com = 0.173      # m/s² per Nm: COM acceleration per unit torque
gamma_support = 1.38  # m/s² per Nm: support acceleration per unit torque

# Relationship: support motion ~= r_wheel * wheel_accel
# d(support)/dt ≈ r * wheel_vel (for small angles)
# d²(support)/dt² ≈ r * d(wheel_vel)/dt

# So: d(wheel_vel)/dt = gamma_support / r_wheel  per Nm of torque
gamma_wheel = gamma_support / r_wheel  # rad/s² per Nm

print(f"\n  Using measured sensitivities from Phase 2 audit (0.48m):")
print(f"    d(pitch_accel)/d(tau) = {alpha_pitch:.3f} rad/s²/Nm")
print(f"    d(com_accel)/d(tau) = {beta_com:.3f} m/s²/Nm")
print(f"    d(support_accel)/d(tau) = {gamma_support:.3f} m/s²/Nm")
print(f"    d(wheel_accel)/d(tau) = {gamma_wheel:.3f} rad/s²/Nm")

# Build A_open continuous
# The position_error derivative convention:
#   position_error = support_center_x - reference_x
#   When COM moves forward (positive com_velocity), the robot pitches forward
#   and the support center moves forward, reducing the position error.
#   So: d(position_error)/dt ≈ -com_velocity (approximately)
#
# But actually: the position error changes because:
#   1. The wheels rotate (translational motion of the support)
#   2. The COM tilts (geometric effect of pitch)
# For small angles, d(position_error)/dt ≈ r_wheel * wheel_vel - com_vel
#
# Let me use: d(position_error)/dt = r_wheel * wheel_vel_mean - com_velocity
# (wheel rotation moves support forward, COM velocity changes relative position)

A_open_ct = np.zeros((5, 5))
# Row 0: d(pitch)/dt = pitch_rate
A_open_ct[0, 1] = 1.0
# Row 1: d(pitch_rate)/dt = omega_0² * pitch = (g/L) * pitch (inverted pendulum)
A_open_ct[1, 0] = omega_0**2  # g/L ≈ 18.17
# Row 2: d(position_error)/dt = r_wheel * wheel_vel - com_velocity
A_open_ct[2, 3] = -1.0       # -com_velocity
A_open_ct[2, 4] = r_wheel     # +r * wheel_vel
# Row 3: d(com_velocity)/dt = 0 (no restoring force on COM without torque)
# Row 4: d(wheel_vel)/dt = 0 (no torque → constant velocity)

# B_open continuous (maps torque [Nm] to state derivatives)
B_open_ct = np.zeros((5, 1))
B_open_ct[1, 0] = alpha_pitch   # pitch acceleration per Nm
B_open_ct[3, 0] = beta_com      # COM acceleration per Nm
B_open_ct[4, 0] = gamma_wheel   # wheel acceleration per Nm

print(f"\n  A_open (continuous-time):")
print(f"    [[{A_open_ct[0,0]:8.3f}, {A_open_ct[0,1]:8.3f}, {A_open_ct[0,2]:8.3f}, {A_open_ct[0,3]:8.3f}, {A_open_ct[0,4]:8.3f}]")
for i in range(1, 5):
    print(f"     [{A_open_ct[i,0]:8.3f}, {A_open_ct[i,1]:8.3f}, {A_open_ct[i,2]:8.3f}, {A_open_ct[i,3]:8.3f}, {A_open_ct[i,4]:8.3f}]")
print(f"    ]")

print(f"\n  B_open (continuous-time):")
for i in range(5):
    print(f"    [{B_open_ct[i,0]:8.4f}]")

# ── Discretize Open-Loop Model ─────────────────────────────────────
# Using zero-order hold: A_d = expm(A * dt), B_d = ∫expm(A*s)*B ds
from scipy.linalg import expm

# Build augmented matrix for ZOH discretization
n = 5
m = 1
A_aug = np.zeros((n + m, n + m))
A_aug[:n, :n] = A_open_ct
A_aug[:n, n:] = B_open_ct
# A_aug[n:, :] = 0 (last row stays zero)

exp_A_aug = expm(A_aug * dt)
A_open_dt = exp_A_aug[:n, :n]
B_open_dt = exp_A_aug[:n, n:]

print(f"\n  A_open (discrete-time, dt={dt}s):")
for i in range(5):
    vals = ", ".join(f"{A_open_dt[i,j]:10.6f}" for j in range(5))
    print(f"    [{vals}]")

print(f"\n  B_open (discrete-time, dt={dt}s):")
for i in range(5):
    print(f"    [{B_open_dt[i,0]:10.6f}]")

# Check stability of A_open_dt
eig_open_dt = np.linalg.eigvals(A_open_dt)
print(f"\n  Open-loop discrete-time eigenvalues:")
for i, ev in enumerate(eig_open_dt):
    mag = np.abs(ev)
    freq_hz = np.abs(np.angle(ev)) / (2 * np.pi * dt) if mag > 0 else 0
    damp = -np.cos(np.angle(ev)) if mag > 0 and not np.isclose(np.abs(np.imag(ev)), 0) else None
    print(f"    λ{i}: {ev.real:+.6f} {ev.imag:+.6f}j  |λ|={mag:.4f}  f={freq_hz:.4f} Hz")

# ── 2B: Closed-Loop Identification from Telemetry ────────────────────
print("\n── 2B: Closed-Loop Identification from Telemetry ──")

# For closed-loop system identification, we use the post-push data
# (richer excitation) to fit: x_{k+1} = A_cl * x_k + B_cl * u_k
# Or, since u_k is itself a function of x_k through K1's feedback,
# we can fit: x_{k+1} = A_cl * x_k  (absorbing the feedback)

# But better: model x_{k+1} - x_k*tau_open terms = A_cl * x_k
# Actually, for the closed-loop, u = -K*x (approximately), so:
# x_{k+1} = (A_d - B_d*K) * x_k = A_cl * x_k
# We can fit A_cl directly from data.

# Let's fit at each height separately
cl_results = {}

for h_target, bin_info in height_bins.items():
    idx = bin_info["indices"]
    count = bin_info["count"]

    if count < 20:
        print(f"\n  Height {h_target:.2f}m: {count} samples (SKIP - insufficient)")
        continue

    # Build X_k and X_{k+1}
    # Use only consecutive pairs within the height bin
    X_k = []
    X_next = []
    U_k = []

    for i in range(len(idx) - 1):
        if idx[i+1] == idx[i] + 1:  # consecutive steps
            X_k.append(X_raw[idx[i]])
            X_next.append(X_raw[idx[i+1]])
            U_k.append(U_raw[idx[i]])

    X_k = np.array(X_k)
    X_next = np.array(X_next)
    U_k = np.array(U_k).reshape(-1, 1)

    n_pairs = len(X_k)
    print(f"\n  Height {h_target:.2f}m: {n_pairs} consecutive pairs")

    if n_pairs < 20:
        print(f"    SKIP - insufficient pairs")
        continue

    # Center data at the local mean (de-mean for linearization around equilibrium)
    x_mean = np.mean(X_k, axis=0)
    u_mean = np.mean(U_k)
    X_k_ctr = X_k - x_mean
    X_next_ctr = X_next - x_mean
    U_k_ctr = U_k - u_mean

    # Safety: remove any NaN/Inf
    nan_mask = np.any(np.isnan(X_k_ctr), axis=1) | np.any(np.isnan(X_next_ctr), axis=1)
    nan_mask |= np.any(np.isinf(X_k_ctr), axis=1) | np.any(np.isinf(X_next_ctr), axis=1)
    if np.any(nan_mask):
        print(f"    Removing {nan_mask.sum()}/{len(X_k_ctr)} pairs with NaN/Inf")
        X_k_ctr = X_k_ctr[~nan_mask]
        X_next_ctr = X_next_ctr[~nan_mask]
        U_k_ctr = U_k_ctr[~nan_mask]
        n_pairs = len(X_k_ctr)

    if n_pairs < 20:
        print(f"    SKIP - insufficient clean pairs")
        continue

    # Fit A_cl via least squares: X_{k+1} = A_cl * X_k
    # Regularize for stability
    try:
        A_cl_dt = X_next_ctr.T @ X_k_ctr @ np.linalg.inv(
            X_k_ctr.T @ X_k_ctr + 1e-6 * np.eye(STATE_DEFINITION["state_dim"])
        )
    except np.linalg.LinAlgError:
        A_cl_dt = X_next_ctr.T @ np.linalg.pinv(X_k_ctr.T)

    # Also fit the full model including input: X_{k+1} = A_d * X_k + B_d * u_k
    XU = np.hstack([X_k_ctr, U_k_ctr])  # [N x 6]
    try:
        AB = X_next_ctr.T @ XU @ np.linalg.inv(XU.T @ XU + 1e-6 * np.eye(6))
        A_full_dt = AB[:, :5]
        B_full_dt = AB[:, 5:]
    except np.linalg.LinAlgError:
        AB = X_next_ctr.T @ np.linalg.pinv(XU.T)
        A_full_dt = AB[:, :5]
        B_full_dt = AB[:, 5:]

    # Fit quality: R² for each state
    X_pred = X_k_ctr @ A_cl_dt.T
    ss_res = np.sum((X_next_ctr - X_pred)**2, axis=0)
    ss_tot = np.sum((X_next_ctr - np.mean(X_next_ctr, axis=0))**2, axis=0)
    r2_scores = 1 - ss_res / (ss_tot + 1e-12)

    # Condition number of X_k^T * X_k
    XtX = X_k_ctr.T @ X_k_ctr
    try:
        cond_XtX = np.linalg.cond(XtX)
    except np.linalg.LinAlgError:
        cond_XtX = 1e10
    try:
        singular_values = np.linalg.svd(X_k_ctr, compute_uv=False)
    except np.linalg.LinAlgError:
        singular_values = np.linalg.svd(X_k_ctr + 1e-6 * np.eye(n_pairs, 5)[:,:5], compute_uv=False)

    # Compute eigenvalues
    eig_cl_dt = np.linalg.eigvals(A_cl_dt)

    print(f"    R² scores: {[f'{r:.4f}' for r in r2_scores]}")
    print(f"    Condition number of X^T X: {cond_XtX:.1f}")
    print(f"    Singular values of X: {[f'{s:.4f}' for s in singular_values]}")
    print(f"    Closed-loop discrete-time eigenvalues:")
    for j, ev in enumerate(eig_cl_dt):
        mag = np.abs(ev)
        freq_hz = np.abs(np.angle(ev)) / (2 * np.pi * dt)
        damp = -np.cos(np.angle(ev)) if not np.isclose(np.abs(np.imag(ev)), 0) else (
            1.0 if ev.real > 0 else -1.0
        )
        print(f"      λ{j}: {ev.real:+.6f} {ev.imag:+.6f}j  |λ|={mag:.4f}  f={freq_hz:.4f} Hz  ζ={damp:+.4f}")

    cl_results[h_target] = {
        "height_m": h_target,
        "n_pairs": n_pairs,
        "x_mean": x_mean.tolist(),
        "u_mean": float(u_mean),
        "A_closed_discrete": A_cl_dt.tolist(),
        "A_full_discrete": A_full_dt.tolist(),
        "B_full_discrete": B_full_dt.tolist(),
        "r2_scores": r2_scores.tolist(),
        "condition_number_XtX": float(cond_XtX),
        "singular_values": singular_values.tolist(),
        "eigenvalues_closed_discrete": [
            {"real": float(ev.real), "imag": float(ev.imag),
             "magnitude": float(np.abs(ev)),
             "frequency_hz": float(np.abs(np.angle(ev)) / (2 * np.pi * dt)),
             "damping_ratio": float(
                 -np.cos(np.angle(ev)) if not np.isclose(np.abs(np.imag(ev)), 0)
                 else (1.0 if ev.real > 0 else -1.0)
             )}
            for ev in eig_cl_dt
        ],
    }

# Also fit a global model (all post-push data)
print(f"\n── Global Post-Push Model ──")
post_push_idx = np.where(post_push_mask)[0]
X_pp = X_raw[post_push_idx]

X_k_global = []
X_next_global = []
for i in range(len(post_push_idx) - 1):
    if post_push_idx[i+1] == post_push_idx[i] + 1:
        X_k_global.append(X_pp[i])
        X_next_global.append(X_pp[i+1])

X_k_global = np.array(X_k_global)
X_next_global = np.array(X_next_global)
x_mean_global = np.mean(X_k_global, axis=0)
X_k_ctr = X_k_global - x_mean_global
X_next_ctr = X_next_global - x_mean_global

A_cl_global = X_next_ctr.T @ X_k_ctr @ np.linalg.inv(
    X_k_ctr.T @ X_k_ctr + 1e-6 * np.eye(5)
)
eig_cl_global = np.linalg.eigvals(A_cl_global)

print(f"  Global pairs: {len(X_k_global)}")
print(f"  Global closed-loop eigenvalues:")
for j, ev in enumerate(eig_cl_global):
    mag = np.abs(ev)
    freq_hz = np.abs(np.angle(ev)) / (2 * np.pi * dt)
    damp = -np.cos(np.angle(ev)) if not np.isclose(np.abs(np.imag(ev)), 0) else (
        1.0 if ev.real > 0 else -1.0
    )
    print(f"    λ{j}: {ev.real:+.6f} {ev.imag:+.6f}j  |λ|={mag:.4f}  f={freq_hz:.4f} Hz  ζ={damp:+.4f}")

# ── Save Results ────────────────────────────────────────────────────
output = {
    "audit": "k1_state_space_model",
    "target": "K1_PITCH_RATE_NOTCH_V1",
    "run": "high_0p480_90N_push_step300_3000steps",
    "state_definition": STATE_DEFINITION,
    "equilibrium": {
        "x_eq": x_eq.tolist(),
        "x_std": x_std.tolist(),
        "u_eq": float(u_eq),
    },
    "open_loop_model": {
        "method": "analytical_twip_with_measured_sensitivities",
        "parameters": {
            "g_m_s2": g,
            "L_com_m": L_com,
            "r_wheel_m": r_wheel,
            "M_total_kg": M_total,
            "omega_0_rad_s": omega_0,
            "f_nat_hz": f_nat_hz,
            "alpha_pitch_rad_s2_per_nm": alpha_pitch,
            "beta_com_m_s2_per_nm": beta_com,
            "gamma_support_m_s2_per_nm": gamma_support,
            "gamma_wheel_rad_s2_per_nm": gamma_wheel,
            "dt_s": dt,
        },
        "A_continuous": A_open_ct.tolist(),
        "B_continuous": B_open_ct.tolist(),
        "A_discrete": A_open_dt.tolist(),
        "B_discrete": B_open_dt.tolist(),
        "eigenvalues_discrete": [
            {"real": float(ev.real), "imag": float(ev.imag),
             "magnitude": float(np.abs(ev)),
             "frequency_hz": float(np.abs(np.angle(ev)) / (2 * np.pi * dt)),
             "damping_ratio": float(
                 -np.cos(np.angle(ev)) if not np.isclose(np.abs(np.imag(ev)), 0)
                 else (1.0 if ev.real > 0 else -1.0)
             )}
            for ev in eig_open_dt
        ],
        "dominant_pole_frequency_hz": f_nat_hz,
        "has_oscillatory_mode": False,
        "has_unstable_real_pole": True,
        "note": "Open-loop plant has real unstable pole at sqrt(g/L) ≈ 4.26 rad/s (0.678 Hz). No inherent oscillatory mode — the 0.4 Hz oscillation must be controller-induced or from leg kinematics not captured by the rigid TWIP model.",
    },
    "closed_loop_models": {
        "method": "linear_regression_on_telemetry",
        "dt_s": dt,
        "by_height": {
            f"{h:.2f}": cl_results[h]
            for h in target_heights if h in cl_results
        },
        "global_post_push": {
            "n_pairs": len(X_k_global),
            "x_mean": x_mean_global.tolist(),
            "A_closed_discrete": A_cl_global.tolist(),
            "eigenvalues": [
                {"real": float(ev.real), "imag": float(ev.imag),
                 "magnitude": float(np.abs(ev)),
                 "frequency_hz": float(np.abs(np.angle(ev)) / (2 * np.pi * dt)),
                 "damping_ratio": float(
                     -np.cos(np.angle(ev)) if not np.isclose(np.abs(np.imag(ev)), 0)
                     else (1.0 if ev.real > 0 else -1.0)
                 )}
                for ev in eig_cl_global
            ],
        },
    },
    "controller_gains": {
        "kp_pitch": 50.0,
        "kd_pitch": 10.0,
        "k_velocity": 15.0,
        "kd_com_vy": 5.0,
        "k_total_velocity_damping": 20.0,
        "k_position": 40.0,
        "max_position_tau": 3.0,
        "k_wheel_velocity": 0.5,
        "k_support_velocity": 0.0,
        "kp_cp": 0.0,
        "max_tau_wheel": 5.0,
        "notch_filter_hz": 2.5,
        "notch_filter_q": 6.0,
        "notch_active_at_0p48m": True,
    },
}

output_path = OUTPUT_DIR / "k1_state_space_model.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n✓ State-space model saved to: {output_path}")

# ── Print Summary ────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("SUMMARY: State-Space Model")
print("=" * 72)
print(f"  State dimension: {STATE_DEFINITION['state_dim']}")
print(f"  Input dimension: {STATE_DEFINITION['input_dim']}")
print(f"  Open-loop dominant pole: {f_nat_hz:.3f} Hz (real unstable)")
print(f"  Open-loop has oscillatory mode: NO")
print(f"  Number of height-specific closed-loop models: {len(cl_results)}")
print(f"  Global closed-loop model pairs: {len(X_k_global)}")

# Find the dominant closed-loop oscillatory mode
for h_target in target_heights:
    if h_target in cl_results:
        evs = cl_results[h_target]["eigenvalues_closed_discrete"]
        osc_modes = [ev for ev in evs if abs(ev["imag"]) > 0.01]
        osc_modes.sort(key=lambda ev: abs(ev["imag"]))
        if osc_modes:
            dominant = osc_modes[-1]  # highest frequency oscillatory mode
            print(f"\n  Height {h_target:.2f}m dominant oscillatory mode:")
            print(f"    Frequency: {dominant['frequency_hz']:.4f} Hz")
            print(f"    Damping ratio: {dominant['damping_ratio']:+.4f}")
            print(f"    Magnitude: {dominant['magnitude']:.4f}")

print("\nDone.")
