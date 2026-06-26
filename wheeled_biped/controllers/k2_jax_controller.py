"""K2 JAX JIT controller — Stage 2: notch filter + torque composer parity.

This module provides pure JAX-compatible functions for the K2 controller
computation path. Stage 2 implements only the notch filter and torque composer
components. Later stages will add sagittal, posture, support, and full-step
integration.

Design rules:
- All functions are pure (no mutable state, no class instances).
- Flat arrays for state, params, and diagnostics.
- Fixed field order defined by constant tuples.
- Python pack/unpack helpers at the boundary.
- float64 for parity (JAX x64 enabled at import).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from wheeled_biped.controllers.signal_filters import (
    biquad_notch_coefficients as _python_biquad_notch_coefficients,
    biquad_notch_update as _python_biquad_notch_update,
    smoothstep_gate_jax,
)

# ===========================================================================
# State layout (Stage 2: notch + prev_tau only)
# ===========================================================================

K2_JAX_STATE_FIELDS_STAGE2: tuple[str, ...] = (
    # Notch filter state (4)
    "notch_x1",
    "notch_x2",
    "notch_y1",
    "notch_y2",
    # Previous torque for rate limiting (10)
    "prev_tau_0",   # l_hip_roll
    "prev_tau_1",   # l_hip_yaw
    "prev_tau_2",   # l_hip_pitch
    "prev_tau_3",   # l_knee
    "prev_tau_4",   # l_wheel
    "prev_tau_5",   # r_hip_roll
    "prev_tau_6",   # r_hip_yaw
    "prev_tau_7",   # r_hip_pitch
    "prev_tau_8",   # r_knee
    "prev_tau_9",   # r_wheel
)
K2_JAX_STATE_SIZE_STAGE2: int = len(K2_JAX_STATE_FIELDS_STAGE2)  # 14

# Index constants for fast state access inside JIT
_IDX_NOTCH_X1 = 0
_IDX_NOTCH_X2 = 1
_IDX_NOTCH_Y1 = 2
_IDX_NOTCH_Y2 = 3
_IDX_PREV_TAU_START = 4


def pack_state_stage2(
    notch_x1: float = 0.0,
    notch_x2: float = 0.0,
    notch_y1: float = 0.0,
    notch_y2: float = 0.0,
    prev_tau: np.ndarray | jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Pack Python K2 state into flat JAX state array (Stage 2 layout).

    Args:
        notch_x1..y2: Biquad notch filter state (zero-initialized)
        prev_tau: Previous torque vector, shape (10,). Zero if None.

    Returns:
        Flat state array, shape (14,), dtype float64
    """
    state = jnp.zeros(K2_JAX_STATE_SIZE_STAGE2, dtype=jnp.float64)
    state = state.at[_IDX_NOTCH_X1].set(float(notch_x1))
    state = state.at[_IDX_NOTCH_X2].set(float(notch_x2))
    state = state.at[_IDX_NOTCH_Y1].set(float(notch_y1))
    state = state.at[_IDX_NOTCH_Y2].set(float(notch_y2))
    if prev_tau is not None:
        state = state.at[_IDX_PREV_TAU_START:_IDX_PREV_TAU_START + 10].set(
            jnp.asarray(prev_tau, dtype=jnp.float64)
        )
    return state


def unpack_state_stage2(state_flat: jnp.ndarray) -> dict:
    """Unpack flat JAX state array into Python dict (Stage 2 layout).

    Args:
        state_flat: Flat state array, shape (14,), dtype float64

    Returns:
        Dict mapping field names to Python float/array values
    """
    state_np = np.asarray(state_flat, dtype=np.float64)
    return {
        "notch_x1": float(state_np[_IDX_NOTCH_X1]),
        "notch_x2": float(state_np[_IDX_NOTCH_X2]),
        "notch_y1": float(state_np[_IDX_NOTCH_Y1]),
        "notch_y2": float(state_np[_IDX_NOTCH_Y2]),
        "prev_tau": state_np[_IDX_PREV_TAU_START:_IDX_PREV_TAU_START + 10].copy(),
    }


# ===========================================================================
# Params layout (Stage 2: notch + composer only)
# ===========================================================================

K2_JAX_PARAMS_FIELDS_STAGE2: tuple[str, ...] = (
    # Notch coefficients (5)
    "notch_b0",
    "notch_b1",
    "notch_b2",
    "notch_a1",
    "notch_a2",
    # Notch metadata for telemetry (3)
    "notch_fs_hz",
    "notch_fc_hz",
    "notch_Q",
    # Torque limits per joint (10)
    "torque_limit_0", "torque_limit_1", "torque_limit_2", "torque_limit_3", "torque_limit_4",
    "torque_limit_5", "torque_limit_6", "torque_limit_7", "torque_limit_8", "torque_limit_9",
    # Max torque rate per joint (10)
    "max_torque_rate_0", "max_torque_rate_1", "max_torque_rate_2", "max_torque_rate_3", "max_torque_rate_4",
    "max_torque_rate_5", "max_torque_rate_6", "max_torque_rate_7", "max_torque_rate_8", "max_torque_rate_9",
    # Control timestep (1)
    "control_dt",
)
K2_JAX_PARAMS_SIZE_STAGE2: int = len(K2_JAX_PARAMS_FIELDS_STAGE2)  # 29

# Index constants for fast params access inside JIT
_IDX_NOTCH_B0 = 0
_IDX_NOTCH_B1 = 1
_IDX_NOTCH_B2 = 2
_IDX_NOTCH_A1 = 3
_IDX_NOTCH_A2 = 4
_IDX_NOTCH_FS = 5
_IDX_NOTCH_FC = 6
_IDX_NOTCH_Q = 7
_IDX_TORQUE_LIMIT_START = 8
_IDX_MAX_TORQUE_RATE_START = 18
_IDX_CONTROL_DT = 28


def pack_params_stage2(
    fs_hz: float = 100.0,
    fc_hz: float = 2.5,
    Q: float = 2.0,
    torque_limit: np.ndarray | jnp.ndarray | None = None,
    max_torque_rate: np.ndarray | jnp.ndarray | None = None,
    control_dt: float = 0.01,
) -> jnp.ndarray:
    """Pack K2 controller params into flat JAX params array (Stage 2 layout).

    Computes biquad coefficients from (fs_hz, fc_hz, Q) automatically.

    Args:
        fs_hz: Sample rate (default 100 Hz)
        fc_hz: Notch centre frequency (default 2.5 Hz)
        Q: Notch quality factor (default 2.0 for K2)
        torque_limit: Per-joint torque limits, shape (10,)
        max_torque_rate: Per-joint max torque rate, shape (10,)
        control_dt: Control timestep in seconds

    Returns:
        Flat params array, shape (29,), dtype float64
    """
    b0, b1, b2, a1, a2 = _python_biquad_notch_coefficients(fs_hz, fc_hz, Q)

    params = jnp.zeros(K2_JAX_PARAMS_SIZE_STAGE2, dtype=jnp.float64)
    params = params.at[_IDX_NOTCH_B0].set(float(b0))
    params = params.at[_IDX_NOTCH_B1].set(float(b1))
    params = params.at[_IDX_NOTCH_B2].set(float(b2))
    params = params.at[_IDX_NOTCH_A1].set(float(a1))
    params = params.at[_IDX_NOTCH_A2].set(float(a2))
    params = params.at[_IDX_NOTCH_FS].set(float(fs_hz))
    params = params.at[_IDX_NOTCH_FC].set(float(fc_hz))
    params = params.at[_IDX_NOTCH_Q].set(float(Q))

    if torque_limit is not None:
        params = params.at[_IDX_TORQUE_LIMIT_START:_IDX_TORQUE_LIMIT_START + 10].set(
            jnp.asarray(torque_limit, dtype=jnp.float64)
        )
    if max_torque_rate is not None:
        params = params.at[_IDX_MAX_TORQUE_RATE_START:_IDX_MAX_TORQUE_RATE_START + 10].set(
            jnp.asarray(max_torque_rate, dtype=jnp.float64)
        )
    params = params.at[_IDX_CONTROL_DT].set(float(control_dt))
    return params


def unpack_params_stage2(params_flat: jnp.ndarray) -> dict:
    """Unpack flat JAX params array into Python dict (Stage 2 layout)."""
    p = np.asarray(params_flat, dtype=np.float64)
    return {
        "notch_b0": float(p[_IDX_NOTCH_B0]),
        "notch_b1": float(p[_IDX_NOTCH_B1]),
        "notch_b2": float(p[_IDX_NOTCH_B2]),
        "notch_a1": float(p[_IDX_NOTCH_A1]),
        "notch_a2": float(p[_IDX_NOTCH_A2]),
        "notch_fs_hz": float(p[_IDX_NOTCH_FS]),
        "notch_fc_hz": float(p[_IDX_NOTCH_FC]),
        "notch_Q": float(p[_IDX_NOTCH_Q]),
        "torque_limit": p[_IDX_TORQUE_LIMIT_START:_IDX_TORQUE_LIMIT_START + 10].copy(),
        "max_torque_rate": p[_IDX_MAX_TORQUE_RATE_START:_IDX_MAX_TORQUE_RATE_START + 10].copy(),
        "control_dt": float(p[_IDX_CONTROL_DT]),
    }


# ===========================================================================
# JAX notch filter step (pure function, JIT-compatible)
# ===========================================================================

def k2_jax_notch_step(
    pitch_rate: jnp.ndarray,
    state_flat: jnp.ndarray,
    params_flat: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply K2 biquad notch filter to pitch_rate signal.

    Pure JAX function — no mutable state. Reads notch state and coefficients
    from flat arrays.

    Args:
        pitch_rate: Scalar pitch rate input (JAX array, shape ())
        state_flat: Flat state array, shape (STATE_SIZE,)
        params_flat: Flat params array, shape (PARAMS_SIZE,)

    Returns:
        (filtered_pitch_rate, new_state_flat)
        - filtered_pitch_rate: scalar JAX array
        - new_state_flat: updated state with new notch x1,x2,y1,y2
    """
    b0 = params_flat[_IDX_NOTCH_B0]
    b1 = params_flat[_IDX_NOTCH_B1]
    b2 = params_flat[_IDX_NOTCH_B2]
    a1 = params_flat[_IDX_NOTCH_A1]
    a2 = params_flat[_IDX_NOTCH_A2]

    x = pitch_rate
    x1 = state_flat[_IDX_NOTCH_X1]
    x2 = state_flat[_IDX_NOTCH_X2]
    y1 = state_flat[_IDX_NOTCH_Y1]
    y2 = state_flat[_IDX_NOTCH_Y2]

    # Direct Form II Transposed
    y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2

    # Update state in-place
    new_state = state_flat.at[_IDX_NOTCH_X1].set(x)
    new_state = new_state.at[_IDX_NOTCH_X2].set(x1)
    new_state = new_state.at[_IDX_NOTCH_Y1].set(y)
    new_state = new_state.at[_IDX_NOTCH_Y2].set(y1)

    return y, new_state


# ===========================================================================
# JAX torque composer (pure function, JIT-compatible)
# ===========================================================================

def k2_jax_torque_composer_step(
    tau_sum: jnp.ndarray,       # shape (10,)
    tau_prev: jnp.ndarray,      # shape (10,)
    params_flat: jnp.ndarray,   # shape (PARAMS_SIZE,)
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compose torque with clipping and rate limiting — JAX pure function.

    Matches BalanceCoreTorqueComposer.compose() logic exactly:
    1. Clip summed torque to torque_limit
    2. Rate-limit vs previous torque
    3. Return final torque + saturation masks

    Args:
        tau_sum: Summed torque from all sources, shape (10,)
        tau_prev: Previous final torque, shape (10,)
        params_flat: Flat params array

    Returns:
        (tau_final, tau_clipped, saturation_mask, rate_saturation_mask)
        - tau_final: shape (10,) — rate-limited torque
        - tau_clipped: shape (10,) — clipped but not rate-limited
        - saturation_mask: shape (10,) — bool, where clipping occurred
        - rate_saturation_mask: shape (10,) — bool, where rate limiting occurred
    """
    torque_limit = params_flat[_IDX_TORQUE_LIMIT_START:_IDX_TORQUE_LIMIT_START + 10]
    max_torque_rate = params_flat[_IDX_MAX_TORQUE_RATE_START:_IDX_MAX_TORQUE_RATE_START + 10]
    control_dt = params_flat[_IDX_CONTROL_DT]

    # Clip to actuator limits
    tau_clipped = jnp.clip(tau_sum, -torque_limit, torque_limit)

    # Detect clipping
    saturation_mask = jnp.abs(tau_sum - tau_clipped) > 1e-9

    # Rate limit: tau_final = tau_prev + clip((tau_clipped - tau_prev)/dt, -max_rate, max_rate) * dt
    delta_desired = tau_clipped - tau_prev
    delta_rate = delta_desired / control_dt
    delta_rate_limited = jnp.clip(delta_rate, -max_torque_rate, max_torque_rate)
    tau_final = tau_prev + delta_rate_limited * control_dt

    # Detect rate saturation
    rate_saturation_mask = jnp.abs(delta_rate - delta_rate_limited) > 1e-9

    # Update prev_tau in state for next step
    new_state_prev_tau = tau_final

    return tau_final, tau_clipped, saturation_mask, rate_saturation_mask


# ===========================================================================
# Python-reference wrappers (for parity testing)
# ===========================================================================

def python_biquad_notch_update(
    x: float, x1: float, x2: float, y1: float, y2: float,
    b0: float, b1: float, b2: float, a1: float, a2: float,
) -> tuple[float, float, float, float, float]:
    """Pure Python biquad notch update — wraps the signal_filters function.

    Used as the reference oracle for parity tests.
    """
    return _python_biquad_notch_update(x, x1, x2, y1, y2, b0, b1, b2, a1, a2)


def python_torque_composer(
    tau_sum: np.ndarray,
    tau_prev: np.ndarray,
    torque_limit: np.ndarray,
    max_torque_rate: np.ndarray,
    control_dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure Python torque composer — matches BalanceCoreTorqueComposer.compose().

    Used as the reference oracle for parity tests.
    """
    # Clip
    tau_clipped = np.clip(tau_sum, -torque_limit, torque_limit)
    saturation_mask = np.abs(tau_sum - tau_clipped) > 1e-9

    # Rate limit
    delta_desired = tau_clipped - tau_prev
    delta_rate = delta_desired / control_dt
    delta_rate_limited = np.clip(delta_rate, -max_torque_rate, max_torque_rate)
    tau_final = tau_prev + delta_rate_limited * control_dt
    rate_saturation_mask = np.abs(delta_rate - delta_rate_limited) > 1e-9

    return tau_final, tau_clipped, saturation_mask, rate_saturation_mask


# ===========================================================================
# Stage 3: JAX math utilities (pure functions)
# ===========================================================================

def _jax_smoothstep01(u):
    """JAX-compatible smoothstep: s(0)=0, s(1)=1, s'(0)=s'(1)=0."""
    u_c = jnp.clip(u, 0.0, 1.0)
    return u_c * u_c * (3.0 - 2.0 * u_c)


def _jax_apply_rate_limit(prev, target, max_delta):
    """Limit per-step change from prev toward target to max_delta."""
    delta = target - prev
    return jnp.where(max_delta <= 0.0, target,
           jnp.where(delta > max_delta, prev + max_delta,
           jnp.where(delta < -max_delta, prev - max_delta, target)))


def _jax_apply_lowpass(prev, target, alpha):
    """First-order low-pass: (1-alpha)*prev + alpha*target."""
    return jnp.where(alpha <= 0.0, prev,
           jnp.where(alpha >= 1.0, target, (1.0 - alpha) * prev + alpha * target))


# ===========================================================================
# Stage 3: Height scheduling (JAX-compatible)
# ===========================================================================

def k2_jax_scheduled_k_position(z_ref, k_nominal, k_low_max, z_low, z_high):
    """Smooth k_position: increases at LOW heights."""
    u = (z_high - z_ref) / (z_high - z_low)
    s = _jax_smoothstep01(u)
    return k_nominal + (k_low_max - k_nominal) * s


def k2_jax_scheduled_k_wheel_velocity(z_ref, k_nominal, k_high_max, z_low, z_high):
    """Smooth k_wheel_velocity: increases at HIGH heights (inverse of k_position)."""
    u = (z_high - z_ref) / (z_high - z_low)
    s = _jax_smoothstep01(u)
    return k_high_max + (k_nominal - k_high_max) * s


# ===========================================================================
# Stage 3: Pitch reference offset interpolation
# ===========================================================================

def k2_jax_interpolate_pitch_ref_offset(height_m, heights_m, offsets_deg, clamp=True):
    """Piecewise-linear lookup of scheduled pitch_ref offset.

    Args:
        height_m: scalar query height
        heights_m: (N,) JAX array of strictly ascending breakpoints
        offsets_deg: (N,) JAX array of offsets at each breakpoint
        clamp: if True, hold endpoint value outside range

    Returns:
        Interpolated offset in degrees. Returns 0.0 if schedule is empty.
    """
    n = heights_m.shape[0]

    def _interior(h):
        idx = jnp.searchsorted(heights_m, h, side='right')
        idx = jnp.clip(idx, 1, n - 1)
        h0 = heights_m[idx - 1]
        h1 = heights_m[idx]
        o0 = offsets_deg[idx - 1]
        o1 = offsets_deg[idx]
        t = (h - h0) / (h1 - h0)
        return o0 + t * (o1 - o0)

    def _extrap_low(h):
        t = (h - heights_m[0]) / (heights_m[1] - heights_m[0])
        return offsets_deg[0] + t * (offsets_deg[1] - offsets_deg[0])

    def _extrap_high(h):
        t = (h - heights_m[-2]) / (heights_m[-1] - heights_m[-2])
        return offsets_deg[-2] + t * (offsets_deg[-1] - offsets_deg[-2])

    below = height_m <= heights_m[0]
    above = height_m >= heights_m[-1]

    if clamp:
        result = jnp.where(below, offsets_deg[0],
                  jnp.where(above, offsets_deg[-1], _interior(height_m)))
    else:
        result = jnp.where(below, _extrap_low(height_m),
                  jnp.where(above, _extrap_high(height_m), _interior(height_m)))
    return jnp.where(n <= 1, offsets_deg[0] if n == 1 else 0.0, result)


# ===========================================================================
# Stage 3: Support-position outer loop pitch reference
# ===========================================================================

def k2_jax_compute_outer_loop_pitch_ref(
    support_error_m, support_error_rate_m_s, integral_error_m_s,
    kp_deg_per_m, kd_deg_per_mps, ki_deg_per_m_s,
    deadband_m, theta_ref_max_deg,
):
    """PD(+I) dynamic pitch_ref offset (deg) for Phase B support-position outer loop."""
    error_p = jnp.where(jnp.abs(support_error_m) < deadband_m, 0.0, support_error_m)
    dynamic = kp_deg_per_m * error_p + kd_deg_per_mps * support_error_rate_m_s + ki_deg_per_m_s * integral_error_m_s
    return jnp.clip(dynamic, -theta_ref_max_deg, theta_ref_max_deg)


# ===========================================================================
# Stage 3: Pre-evaluated grid interpolation (for PCHIP functions)
# ===========================================================================

def k2_jax_grid_interpolate(height_m, grid_heights, grid_values):
    """Linear interpolation on a pre-evaluated fine grid.

    Args:
        height_m: scalar query height
        grid_heights: (G,) JAX array of strictly ascending grid points
        grid_values: (G,) JAX array of pre-evaluated function values

    Returns:
        Linearly interpolated value, clamped to grid endpoints
    """
    g = grid_heights.shape[0]
    h = jnp.clip(height_m, grid_heights[0], grid_heights[-1])
    idx_f = (h - grid_heights[0]) / (grid_heights[-1] - grid_heights[0]) * (g - 1)
    idx_lo = jnp.clip(jnp.floor(idx_f).astype(jnp.int32), 0, g - 2)
    idx_hi = idx_lo + 1
    t = idx_f - idx_lo
    return grid_values[idx_lo] + t * (grid_values[idx_hi] - grid_values[idx_lo])


def build_calibrated_grid_params(height_min=0.30, height_max=0.48, n_points=20000):
    """Pre-evaluate all calibrated outer loop PCHIP functions on a fine grid.

    Returns dict with grid_heights, kp_grid, kd_grid, ki_grid, theta_max_grid,
    deadband_grid, rate_limit_grid, lowpass_grid.

    n_points=20000 ensures linear interpolation error < 1e-6 for all functions
    (empirically verified at 10000 random test points).
    """
    from wheeled_biped.controllers.calibrated_outer_loop_functions import (
        calibrated_kp_deg_per_m,
        calibrated_kd_deg_per_mps,
        calibrated_ki_deg_per_m_s,
        calibrated_theta_ref_max_deg,
        calibrated_deadband_m,
        calibrated_rate_limit_deg_per_step,
        calibrated_lowpass_alpha,
    )
    hs = np.linspace(height_min, height_max, n_points, dtype=np.float64)
    return {
        "grid_heights": jnp.array(hs),
        "kp_grid": jnp.array([calibrated_kp_deg_per_m(float(h)) for h in hs]),
        "kd_grid": jnp.array([calibrated_kd_deg_per_mps(float(h)) for h in hs]),
        "ki_grid": jnp.array([calibrated_ki_deg_per_m_s(float(h)) for h in hs]),
        "theta_max_grid": jnp.array([calibrated_theta_ref_max_deg(float(h)) for h in hs]),
        "deadband_grid": jnp.array([calibrated_deadband_m(float(h)) for h in hs]),
        "rate_limit_grid": jnp.array([calibrated_rate_limit_deg_per_step(float(h)) for h in hs]),
        "lowpass_grid": jnp.array([calibrated_lowpass_alpha(float(h)) for h in hs]),
    }


def build_physics_ff_grid_params(height_min=0.30, height_max=0.48, n_points=100000):
    """Pre-evaluate physics equilibrium feedforward PCHIP functions on a fine grid.

    n_points=100000 ensures linear interpolation error < 1e-6 for the
    high-curvature physics FF functions (empirically verified).
    """
    from wheeled_biped.controllers.physics_equilibrium_feedforward import (
        physics_equilibrium_feedforward_tau_each_wheel_nm,
        physics_equilibrium_pitch_eq_no_off_deg,
    )
    hs = np.linspace(height_min, height_max, n_points, dtype=np.float64)
    return {
        "grid_heights": jnp.array(hs),
        "tau_eq_ff_grid": jnp.array([physics_equilibrium_feedforward_tau_each_wheel_nm(float(h)) for h in hs]),
        "pitch_eq_grid": jnp.array([physics_equilibrium_pitch_eq_no_off_deg(float(h)) for h in hs]),
    }


# ===========================================================================
# Stage 3: Low-band support (Gaussian gate, K2-active)
# ===========================================================================

def k2_jax_low_band_support_gate(height_m, center_m, sigma_m):
    """Gaussian height gate for low-band support correction."""
    z = (height_m - center_m) / sigma_m
    return jnp.exp(-0.5 * z * z)


def k2_jax_low_band_support_pitch_ref(
    height_m, support_error_m,
    center_m, sigma_m, kp_peak_deg_per_m,
    theta_ref_max_peak_deg, pitch_ref_offset_peak_deg,
):
    """Low-band support outer-loop pitch reference contribution."""
    gate = k2_jax_low_band_support_gate(height_m, center_m, sigma_m)
    kp = kp_peak_deg_per_m * gate
    theta_max = theta_ref_max_peak_deg * gate
    raw = kp * support_error_m
    pitch_ref = jnp.clip(raw, -theta_max, theta_max)
    pitch_ref_offset_deg = pitch_ref_offset_peak_deg * gate
    return pitch_ref + pitch_ref_offset_deg, theta_max


# ===========================================================================
# Stage 3: Sagittal legacy torque assembly (K2-active terms only)
# ===========================================================================

def k2_jax_sagittal_torque_assembly(
    pitch_x_rad, pitch_rate_rad_s,
    sagittal_velocity_m_s, sagittal_position_error_m,
    wheel_vel_left_rad_s, wheel_vel_right_rad_s,
    support_velocity_m_s,
    kp_pitch, effective_pitch_scale, effective_pitch_tau_cap,
    effective_kd_pitch,
    effective_k_velocity, effective_velocity_damping_scale,
    effective_support_velocity_gain, effective_support_velocity_scale,
    effective_k_wheel_velocity,
    effective_k_position, effective_max_position_tau,
    kp_cp, kd_com_vy,
    wheel_torque_sign,
    pitch_bias_comp_tau=0.0,
    position_integral_tau=0.0,
    pitch_soft_start_rad=0.30, pitch_hard_limit_rad=0.60, min_pitch_scale=0.0,
    enable_pitch_aware_position_scaling=False,
    enable_torque_budget_aware_position=False,
    position_tau_budget_cap=7.0,
    max_tau_wheel=5.0,
):
    """K2-active sagittal torque assembly — pure JAX function.

    Only includes terms active for K2_NOTCH_LOW_Q_V1.
    Disabled terms (recenter, hysteresis, bias_cancel, APC, L_feedback) are omitted.
    """
    tau_pitch_raw = kp_pitch * pitch_x_rad
    tau_pitch_scheduled = tau_pitch_raw * effective_pitch_scale
    tau_pitch = jnp.where(
        effective_pitch_tau_cap > 0.0,
        jnp.clip(tau_pitch_scheduled, -effective_pitch_tau_cap, effective_pitch_tau_cap),
        tau_pitch_scheduled,
    )
    tau_pitch = tau_pitch - pitch_bias_comp_tau

    tau_pitch_rate = effective_kd_pitch * pitch_rate_rad_s
    tau_sagittal_velocity = -effective_k_velocity * effective_velocity_damping_scale * sagittal_velocity_m_s
    tau_support_velocity = -effective_support_velocity_gain * effective_support_velocity_scale * support_velocity_m_s
    tau_cp = -kp_cp * sagittal_position_error_m
    tau_com_vy = -kd_com_vy * sagittal_velocity_m_s

    tau_position_p = -effective_k_position * sagittal_position_error_m
    tau_position = tau_position_p + position_integral_tau

    def _apply_pitch_aware(tau_pos):
        abs_pitch = jnp.abs(pitch_x_rad)
        u_pitch = (abs_pitch - pitch_soft_start_rad) / (pitch_hard_limit_rad - pitch_soft_start_rad)
        s = _jax_smoothstep01(u_pitch)
        scale = 1.0 - s * (1.0 - min_pitch_scale)
        return tau_pos * scale

    tau_position = jnp.where(
        enable_pitch_aware_position_scaling,
        _apply_pitch_aware(tau_position),
        tau_position,
    )

    tau_balance_before_pos = (
        tau_pitch + tau_pitch_rate + tau_sagittal_velocity + tau_support_velocity
        + tau_cp + tau_com_vy
        + 0.5 * (-effective_k_wheel_velocity * wheel_vel_left_rad_s
                 - effective_k_wheel_velocity * wheel_vel_right_rad_s)
    )
    pos_lower = -max_tau_wheel - tau_balance_before_pos
    pos_upper = max_tau_wheel - tau_balance_before_pos
    tau_position_budget_clipped = jnp.clip(tau_position, pos_lower, pos_upper)
    tau_position = jnp.where(
        enable_torque_budget_aware_position,
        tau_position_budget_clipped,
        tau_position,
    )
    tau_position = jnp.clip(tau_position, -effective_max_position_tau, effective_max_position_tau)

    tau_wheel_vel_left = -effective_k_wheel_velocity * wheel_vel_left_rad_s
    tau_wheel_vel_right = -effective_k_wheel_velocity * wheel_vel_right_rad_s

    tau_common_unclipped = (
        tau_pitch + tau_pitch_rate + tau_sagittal_velocity + tau_support_velocity
        + tau_position + tau_cp + tau_com_vy
    )
    tau_common = wheel_torque_sign * tau_common_unclipped
    tau_left = tau_common + tau_wheel_vel_left
    tau_right = tau_common + tau_wheel_vel_right

    tau = jnp.zeros(10, dtype=jnp.float64)
    tau = tau.at[4].set(tau_left)
    tau = tau.at[9].set(tau_right)

    diag = {
        "tau_pitch": tau_pitch, "tau_pitch_rate": tau_pitch_rate,
        "tau_sagittal_velocity": tau_sagittal_velocity,
        "tau_support_velocity": tau_support_velocity,
        "tau_position": tau_position, "tau_cp": tau_cp, "tau_com_vy": tau_com_vy,
        "tau_wheel_vel_left": tau_wheel_vel_left,
        "tau_wheel_vel_right": tau_wheel_vel_right,
        "tau_common_unclipped": tau_common_unclipped,
    }
    return tau, diag


# ===========================================================================
# Stage 3: Shape posture, lateral roll, yaw, mode-div, support FF
# ===========================================================================

def k2_jax_shape_posture_compute(
    q_ref, joint_pos, joint_vel,
    kp_hip_yaw=5.0, kd_hip_yaw=1.0,
    kp_hip_pitch=30.0, kd_hip_pitch=4.0,
    kp_knee=40.0, kd_knee=5.0,
    kp_hip_roll=0.0, kd_hip_roll=0.0,
    posture_weight=1.0, contact_degraded_scale=1.0,
):
    """Shape/posture PD control — pure JAX function."""
    error = q_ref - joint_pos
    authority = posture_weight * contact_degraded_scale
    tau = jnp.zeros(10, dtype=jnp.float64)
    tau = tau.at[1].set(authority * (kp_hip_yaw * error[1] - kd_hip_yaw * joint_vel[1]))
    tau = tau.at[6].set(authority * (kp_hip_yaw * error[6] - kd_hip_yaw * joint_vel[6]))
    tau = tau.at[2].set(authority * (kp_hip_pitch * error[2] - kd_hip_pitch * joint_vel[2]))
    tau = tau.at[7].set(authority * (kp_hip_pitch * error[7] - kd_hip_pitch * joint_vel[7]))
    tau = tau.at[3].set(authority * (kp_knee * error[3] - kd_knee * joint_vel[3]))
    tau = tau.at[8].set(authority * (kp_knee * error[8] - kd_knee * joint_vel[8]))
    tau = tau.at[0].set(authority * (kp_hip_roll * error[0] - kd_hip_roll * joint_vel[0]))
    tau = tau.at[5].set(authority * (kp_hip_roll * error[5] - kd_hip_roll * joint_vel[5]))
    diag = {"posture_tau_max_abs": jnp.max(jnp.abs(tau)),
            "posture_active_joint_count": jnp.sum(jnp.abs(tau) > 1e-12).astype(jnp.int32)}
    return tau, diag


def k2_jax_lateral_roll_compute(
    roll_y_rad, roll_rate_y_rad_s,
    hip_roll_pos_left=0.0, hip_roll_pos_right=0.0,
    hip_roll_vel_left=0.0, hip_roll_vel_right=0.0,
    hip_roll_ref_left=0.0, hip_roll_ref_right=0.0,
    kp_roll=40.0, kd_roll=8.0, max_roll_moment=50.0, hip_roll_torque_sign=1.0,
    enable_stance_regularization=False,
    kp_stance=5.0, kd_stance=1.0, max_stance_torque=5.0, stance_weight=0.4,
):
    """Lateral roll balance — pure JAX function."""
    m_roll = kp_roll * roll_y_rad + kd_roll * roll_rate_y_rad_s
    m_roll_clipped = jnp.clip(m_roll, -max_roll_moment, max_roll_moment)
    tau_roll_left = hip_roll_torque_sign * m_roll_clipped
    tau_roll_right = -hip_roll_torque_sign * m_roll_clipped

    def _stance(pos, vel, ref):
        err = ref - pos
        return jnp.clip(kp_stance * err - kd_stance * vel, -max_stance_torque, max_stance_torque)

    s_left = _stance(hip_roll_pos_left, hip_roll_vel_left, hip_roll_ref_left)
    s_right = _stance(hip_roll_pos_right, hip_roll_vel_right, hip_roll_ref_right)

    tau = jnp.zeros(10, dtype=jnp.float64)
    tau = tau.at[0].set(tau_roll_left + jnp.where(enable_stance_regularization, stance_weight * s_left, 0.0))
    tau = tau.at[5].set(tau_roll_right + jnp.where(enable_stance_regularization, stance_weight * s_right, 0.0))
    return tau, {"lateral_roll_tau": tau_roll_left}


def k2_jax_yaw_compute(yaw_error_rad, yaw_rate_rad_s, kp_yaw=5.0, kd_yaw=1.0, max_yaw_torque=3.0):
    """Yaw controller — pure JAX function."""
    tau_antisym = jnp.clip(kp_yaw * yaw_error_rad - kd_yaw * yaw_rate_rad_s, -max_yaw_torque, max_yaw_torque)
    tau = jnp.zeros(10, dtype=jnp.float64)
    tau = tau.at[1].set(-tau_antisym)
    tau = tau.at[6].set(tau_antisym)
    return tau


def k2_jax_mode_div_compute(
    div_error, div_rate, height_m,
    kp_div=10.0, kd_div=0.50, max_torque=7.5,
    soft_limit_rad=0.30, soft_gain=0.80,
    support_error_m=0.0, support_error_rate_m_s=0.0,
    enable_support_gate=False,
):
    """Mode-based hip-yaw divergence controller — pure JAX function."""
    raw = -(kp_div * div_error + kd_div * div_rate)
    z_low, z_high = soft_limit_rad, soft_limit_rad + soft_gain
    u_h = (z_high - height_m) / (z_high - z_low)
    height_gate = _jax_smoothstep01(u_h)
    torque = raw * height_gate
    torque_clipped = jnp.clip(torque, -max_torque, max_torque)
    tau = jnp.zeros(10, dtype=jnp.float64)
    tau = tau.at[1].set(torque_clipped)
    tau = tau.at[6].set(-torque_clipped)
    return tau


def k2_jax_support_feedforward_compute(
    support_position_error_m=0.0, target_com_height=0.45,
    k_support_hip_yaw=3.0, support_comp_sign=1.0, tau_max_support_comp=5.0,
):
    """Support feedforward — pure JAX function."""
    u_h = (0.393 - target_com_height) / (0.393 - 0.300)
    height_gate = _jax_smoothstep01(u_h)
    raw = support_comp_sign * k_support_hip_yaw * support_position_error_m * height_gate
    comp = jnp.clip(raw, -tau_max_support_comp, tau_max_support_comp)
    tau = jnp.zeros(10, dtype=jnp.float64)
    tau = tau.at[1].set(comp)
    tau = tau.at[6].set(-comp)
    return tau


# ===========================================================================
# Stage 3: Python-reference wrappers (for parity testing)
# ===========================================================================

def python_smoothstep01(u):
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


def python_scheduled_k_position(z_ref, k_nominal, k_low_max, z_low, z_high):
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        scheduled_k_position as _ref,
    )
    return _ref(z_ref, k_nominal, k_low_max, z_low, z_high)


def python_scheduled_k_wheel_velocity(z_ref, k_nominal, k_high_max, z_low, z_high):
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        scheduled_k_wheel_velocity as _ref,
    )
    return _ref(z_ref, k_nominal, k_high_max, z_low, z_high)


def python_interpolate_pitch_ref_offset(height_m, heights_m, offsets_deg, clamp=True):
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        interpolate_pitch_ref_offset as _ref,
    )
    return _ref(height_m, heights_m, offsets_deg, clamp)


def python_compute_outer_loop_pitch_ref(*args):
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        compute_outer_loop_pitch_ref as _ref,
    )
    return _ref(*args)


def python_apply_rate_limit(prev, target, max_delta):
    if max_delta <= 0.0:
        return float(target)
    delta = target - prev
    if delta > max_delta:
        return float(prev + max_delta)
    if delta < -max_delta:
        return float(prev - max_delta)
    return float(target)


def python_apply_lowpass(prev, target, alpha):
    if alpha <= 0.0:
        return float(prev)
    if alpha >= 1.0:
        return float(target)
    return float((1.0 - alpha) * prev + alpha * target)


# ===========================================================================
# Stage 4: Full K2 JAX controller step — complete layout + compose function
# ===========================================================================

# --- Complete state layout (19 fields, confirmed from Python sources) ---

K2_JAX_STATE_FIELDS: tuple[str, ...] = (
    # Notch filter (4) — BiquadNotchFilter state
    "notch_x1", "notch_x2", "notch_y1", "notch_y2",
    # Previous torque for rate limiting (10)
    "prev_tau_0", "prev_tau_1", "prev_tau_2", "prev_tau_3", "prev_tau_4",
    "prev_tau_5", "prev_tau_6", "prev_tau_7", "prev_tau_8", "prev_tau_9",
    # Height scheduling (1) — sagittal._filtered_com_z
    "filtered_com_z",
    # Previous support error (1) — sim loop prev_support_error
    "prev_support_error",
    # Outer loop state (3) — sim loop nonlocal
    "outer_loop_pitch_ref_smoothed_deg",
    "outer_loop_prev_support_error_m",
    "outer_loop_support_error_rate_smoothed",
)
K2_JAX_STATE_SIZE: int = len(K2_JAX_STATE_FIELDS)  # 19

# Index constants
_S_NOTCH_X1, _S_NOTCH_X2, _S_NOTCH_Y1, _S_NOTCH_Y2 = 0, 1, 2, 3
_S_PREV_TAU_START = 4
_S_FILTERED_COM_Z = 14
_S_PREV_SUPPORT_ERROR = 15
_S_OL_PITCH_REF_SMOOTHED = 16
_S_OL_PREV_SUPPORT_ERROR = 17
_S_OL_SUPPORT_ERROR_RATE = 18


def pack_state_k2(
    notch_x1=0.0, notch_x2=0.0, notch_y1=0.0, notch_y2=0.0,
    prev_tau=None, filtered_com_z=0.4, prev_support_error=0.0,
    ol_pitch_ref_smoothed=0.0, ol_prev_support_error=0.0, ol_support_error_rate=0.0,
):
    """Pack all K2 state into flat JAX array (19,)."""
    s = jnp.zeros(K2_JAX_STATE_SIZE, dtype=jnp.float64)
    s = s.at[_S_NOTCH_X1].set(notch_x1)
    s = s.at[_S_NOTCH_X2].set(notch_x2)
    s = s.at[_S_NOTCH_Y1].set(notch_y1)
    s = s.at[_S_NOTCH_Y2].set(notch_y2)
    if prev_tau is not None:
        s = s.at[_S_PREV_TAU_START:_S_PREV_TAU_START + 10].set(jnp.asarray(prev_tau, dtype=jnp.float64))
    s = s.at[_S_FILTERED_COM_Z].set(filtered_com_z)
    s = s.at[_S_PREV_SUPPORT_ERROR].set(prev_support_error)
    s = s.at[_S_OL_PITCH_REF_SMOOTHED].set(ol_pitch_ref_smoothed)
    s = s.at[_S_OL_PREV_SUPPORT_ERROR].set(ol_prev_support_error)
    s = s.at[_S_OL_SUPPORT_ERROR_RATE].set(ol_support_error_rate)
    return s


# --- Complete input layout ---

K2_JAX_INPUT_FIELDS: tuple[str, ...] = (
    "pitch_x_rad", "pitch_rate_x_rad_s",
    "roll_y_rad", "roll_rate_y_rad_s",
    "yaw_error_rad", "yaw_rate_rad_s",
    "com_z_m", "com_vy_m_s",
    "sagittal_velocity_m_s", "sagittal_position_error_m",
    "wheel_vel_left_rad_s", "wheel_vel_right_rad_s",
    "support_velocity_m_s",
    "commanded_height_ref_m",
    "hip_yaw_div_error", "hip_yaw_div_rate",
    "q_hip_yaw_l", "q_hip_yaw_r", "q_hip_pitch_l", "q_hip_pitch_r",
    "q_knee_l", "q_knee_r", "q_hip_roll_l", "q_hip_roll_r",
    "qd_hip_yaw_l", "qd_hip_yaw_r", "qd_hip_pitch_l", "qd_hip_pitch_r",
    "qd_knee_l", "qd_knee_r", "qd_hip_roll_l", "qd_hip_roll_r",
    "q_ref_hip_yaw_l", "q_ref_hip_yaw_r", "q_ref_hip_pitch_l", "q_ref_hip_pitch_r",
    "q_ref_knee_l", "q_ref_knee_r", "q_ref_hip_roll_l", "q_ref_hip_roll_r",
    "support_position_error_m",
)
K2_JAX_INPUT_SIZE: int = len(K2_JAX_INPUT_FIELDS)  # 42

_I_PITCH_X, _I_PITCH_RATE, _I_ROLL_Y, _I_ROLL_RATE = 0, 1, 2, 3
_I_YAW_ERR, _I_YAW_RATE, _I_COM_Z, _I_COM_VY = 4, 5, 6, 7
_I_SAG_VEL, _I_SAG_POS_ERR, _I_WHEEL_VEL_L, _I_WHEEL_VEL_R = 8, 9, 10, 11
_I_SUPPORT_VEL, _I_HEIGHT_REF = 12, 13
_I_HY_DIV_ERR, _I_HY_DIV_RATE = 14, 15
_I_Q_START, _I_QD_START = 16, 24
_I_QREF_START, _I_SUPPORT_POS_ERR = 32, 40
_I_TARGET_COM_HEIGHT = 41  # alias for height ref used by support FF


def pack_input_k2(
    pitch_x_rad, pitch_rate_x_rad_s, roll_y_rad, roll_rate_y_rad_s,
    yaw_error_rad, yaw_rate_rad_s, com_z_m, com_vy_m_s,
    sagittal_velocity_m_s, sagittal_position_error_m,
    wheel_vel_left_rad_s, wheel_vel_right_rad_s,
    support_velocity_m_s, commanded_height_ref_m,
    hip_yaw_div_error, hip_yaw_div_rate,
    joint_pos, joint_vel, q_ref,
    support_position_error_m,
):
    """Pack all K2 inputs into flat JAX array (42,)."""
    inp = jnp.zeros(K2_JAX_INPUT_SIZE, dtype=jnp.float64)
    inp = inp.at[_I_PITCH_X].set(pitch_x_rad)
    inp = inp.at[_I_PITCH_RATE].set(pitch_rate_x_rad_s)
    inp = inp.at[_I_ROLL_Y].set(roll_y_rad)
    inp = inp.at[_I_ROLL_RATE].set(roll_rate_y_rad_s)
    inp = inp.at[_I_YAW_ERR].set(yaw_error_rad)
    inp = inp.at[_I_YAW_RATE].set(yaw_rate_rad_s)
    inp = inp.at[_I_COM_Z].set(com_z_m)
    inp = inp.at[_I_COM_VY].set(com_vy_m_s)
    inp = inp.at[_I_SAG_VEL].set(sagittal_velocity_m_s)
    inp = inp.at[_I_SAG_POS_ERR].set(sagittal_position_error_m)
    inp = inp.at[_I_WHEEL_VEL_L].set(wheel_vel_left_rad_s)
    inp = inp.at[_I_WHEEL_VEL_R].set(wheel_vel_right_rad_s)
    inp = inp.at[_I_SUPPORT_VEL].set(support_velocity_m_s)
    inp = inp.at[_I_HEIGHT_REF].set(commanded_height_ref_m)
    inp = inp.at[_I_HY_DIV_ERR].set(hip_yaw_div_error)
    inp = inp.at[_I_HY_DIV_RATE].set(hip_yaw_div_rate)
    inp = inp.at[_I_Q_START:_I_Q_START + 8].set(jnp.asarray([
        joint_pos[1], joint_pos[6], joint_pos[2], joint_pos[7],
        joint_pos[3], joint_pos[8], joint_pos[0], joint_pos[5],
    ], dtype=jnp.float64))
    inp = inp.at[_I_QD_START:_I_QD_START + 8].set(jnp.asarray([
        joint_vel[1], joint_vel[6], joint_vel[2], joint_vel[7],
        joint_vel[3], joint_vel[8], joint_vel[0], joint_vel[5],
    ], dtype=jnp.float64))
    inp = inp.at[_I_QREF_START:_I_QREF_START + 8].set(jnp.asarray([
        q_ref[1], q_ref[6], q_ref[2], q_ref[7],
        q_ref[3], q_ref[8], q_ref[0], q_ref[5],
    ], dtype=jnp.float64))
    inp = inp.at[_I_SUPPORT_POS_ERR].set(support_position_error_m)
    inp = inp.at[_I_TARGET_COM_HEIGHT].set(commanded_height_ref_m)
    return inp


# --- Complete params layout (includes grid data) ---
# Params are built dynamically at init time via pack_params_k2()

# --- Diagnostics layout ---

K2_JAX_DIAG_FIELDS: tuple[str, ...] = (
    "notch_output", "notch_height_gate",
    "tau_pitch", "tau_pitch_rate", "tau_sagittal_velocity", "tau_support_velocity",
    "tau_position", "tau_wheel_vel_left", "tau_wheel_vel_right",
    "scheduled_k_position", "scheduled_k_wheel_velocity", "scheduled_kd_pitch",
    "calib_kp", "calib_kd", "calib_theta_max", "calib_deadband",
    "physics_ff_tau", "low_band_pitch_ref",
    "tau_final_0", "tau_final_1", "tau_final_2", "tau_final_3", "tau_final_4",
    "tau_final_5", "tau_final_6", "tau_final_7", "tau_final_8", "tau_final_9",
    "clip_saturation_count", "rate_limit_active_count",
)
K2_JAX_DIAG_SIZE: int = len(K2_JAX_DIAG_FIELDS)  # 30

_D_NOTCH_OUT, _D_NOTCH_GATE = 0, 1
_D_TAU_PITCH, _D_TAU_PITCH_RATE, _D_TAU_SAG_VEL = 2, 3, 4
_D_TAU_SUPPORT_VEL, _D_TAU_POSITION = 5, 6
_D_TAU_WHEEL_L, _D_TAU_WHEEL_R = 7, 8
_D_SCHED_KPOS, _D_SCHED_KWHEEL, _D_SCHED_KD = 9, 10, 11
_D_CALIB_KP, _D_CALIB_KD, _D_CALIB_THETA, _D_CALIB_DB = 12, 13, 14, 15
_D_PHYSICS_FF, _D_LOW_BAND = 16, 17
_D_TAU_FINAL_START = 18
_D_CLIP_COUNT, _D_RATE_COUNT = 28, 29


def k2_jax_diag_flat_to_dict(diag_flat):
    """Map flat JAX diagnostics to named dict."""
    d = np.asarray(diag_flat, dtype=np.float64)
    return {K2_JAX_DIAG_FIELDS[i]: float(d[i]) for i in range(K2_JAX_DIAG_SIZE)}


# --- Grid params (built at init, passed into params flat array) ---
# These are module-level singletons, built once.
_calibrated_grid_cache = None
_physics_ff_grid_cache = None


def _get_calibrated_grid():
    global _calibrated_grid_cache
    if _calibrated_grid_cache is None:
        _calibrated_grid_cache = build_calibrated_grid_params()
    return _calibrated_grid_cache


def _get_physics_ff_grid():
    global _physics_ff_grid_cache
    if _physics_ff_grid_cache is None:
        _physics_ff_grid_cache = build_physics_ff_grid_params()
    return _physics_ff_grid_cache


# ===========================================================================
# Stage 4: Full K2 JAX controller step (JIT-compatible)
# ===========================================================================

def k2_jax_controller_step(
    state_flat: jnp.ndarray,   # (19,)
    input_flat: jnp.ndarray,   # (42,)
    params_flat: jnp.ndarray,  # (P,) — contains notch coeffs + torque limits + grid refs
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Full K2 balance-core controller step — pure JAX function.

    Composes all Stage 2+3 components: notch → height schedule → sagittal →
    shape posture → lateral roll → yaw → mode-div → support FF → composer.

    Returns:
        tau: (10,) actuator torque vector
        next_state_flat: (19,) updated state
        diag_flat: (30,) diagnostics
    """
    # Unpack state
    notch_x1 = state_flat[_S_NOTCH_X1]
    notch_x2 = state_flat[_S_NOTCH_X2]
    notch_y1 = state_flat[_S_NOTCH_Y1]
    notch_y2 = state_flat[_S_NOTCH_Y2]
    prev_tau = state_flat[_S_PREV_TAU_START:_S_PREV_TAU_START + 10]
    filtered_com_z = state_flat[_S_FILTERED_COM_Z]
    prev_support_error = state_flat[_S_PREV_SUPPORT_ERROR]
    ol_pitch_ref_smoothed = state_flat[_S_OL_PITCH_REF_SMOOTHED]
    ol_prev_support_error = state_flat[_S_OL_PREV_SUPPORT_ERROR]
    ol_support_error_rate = state_flat[_S_OL_SUPPORT_ERROR_RATE]

    # Unpack inputs
    pitch_x = input_flat[_I_PITCH_X]
    pitch_rate = input_flat[_I_PITCH_RATE]
    roll_y = input_flat[_I_ROLL_Y]
    roll_rate = input_flat[_I_ROLL_RATE]
    yaw_err = input_flat[_I_YAW_ERR]
    yaw_rate = input_flat[_I_YAW_RATE]
    com_z = input_flat[_I_COM_Z]
    com_vy = input_flat[_I_COM_VY]
    sag_vel = input_flat[_I_SAG_VEL]
    sag_pos_err = input_flat[_I_SAG_POS_ERR]
    wheel_vel_l = input_flat[_I_WHEEL_VEL_L]
    wheel_vel_r = input_flat[_I_WHEEL_VEL_R]
    support_vel = input_flat[_I_SUPPORT_VEL]
    height_ref = input_flat[_I_HEIGHT_REF]
    hy_div_err = input_flat[_I_HY_DIV_ERR]
    hy_div_rate = input_flat[_I_HY_DIV_RATE]
    q_hy_l = input_flat[_I_Q_START + 0]; q_hy_r = input_flat[_I_Q_START + 1]
    q_hp_l = input_flat[_I_Q_START + 2]; q_hp_r = input_flat[_I_Q_START + 3]
    q_kn_l = input_flat[_I_Q_START + 4]; q_kn_r = input_flat[_I_Q_START + 5]
    q_hr_l = input_flat[_I_Q_START + 6]; q_hr_r = input_flat[_I_Q_START + 7]
    qd_hy_l = input_flat[_I_QD_START + 0]; qd_hy_r = input_flat[_I_QD_START + 1]
    qd_hp_l = input_flat[_I_QD_START + 2]; qd_hp_r = input_flat[_I_QD_START + 3]
    qd_kn_l = input_flat[_I_QD_START + 4]; qd_kn_r = input_flat[_I_QD_START + 5]
    qd_hr_l = input_flat[_I_QD_START + 6]; qd_hr_r = input_flat[_I_QD_START + 7]
    qref_hy_l = input_flat[_I_QREF_START + 0]; qref_hy_r = input_flat[_I_QREF_START + 1]
    qref_hp_l = input_flat[_I_QREF_START + 2]; qref_hp_r = input_flat[_I_QREF_START + 3]
    qref_kn_l = input_flat[_I_QREF_START + 4]; qref_kn_r = input_flat[_I_QREF_START + 5]
    qref_hr_l = input_flat[_I_QREF_START + 6]; qref_hr_r = input_flat[_I_QREF_START + 7]
    support_pos_err = input_flat[_I_SUPPORT_POS_ERR]

    # Unpack params
    notch_b0 = params_flat[_IDX_NOTCH_B0]
    notch_b1 = params_flat[_IDX_NOTCH_B1]
    notch_b2 = params_flat[_IDX_NOTCH_B2]
    notch_a1 = params_flat[_IDX_NOTCH_A1]
    notch_a2 = params_flat[_IDX_NOTCH_A2]
    torque_limit = params_flat[_IDX_TORQUE_LIMIT_START:_IDX_TORQUE_LIMIT_START + 10]
    max_torque_rate = params_flat[_IDX_MAX_TORQUE_RATE_START:_IDX_MAX_TORQUE_RATE_START + 10]
    control_dt = params_flat[_IDX_CONTROL_DT]

    # === Step 1: Notch filter ===
    notch_out = notch_b0 * pitch_rate + notch_b1 * notch_x1 + notch_b2 * notch_x2 - notch_a1 * notch_y1 - notch_a2 * notch_y2
    new_notch_x1 = pitch_rate
    new_notch_x2 = notch_x1
    new_notch_y1 = notch_out
    new_notch_y2 = notch_y1

    # Height gate for notch blend
    notch_gate = smoothstep_gate_jax(height_ref, 0.42, 0.48)
    pitch_rate_eff = (1.0 - notch_gate) * pitch_rate + notch_gate * notch_out

    # === Step 2: Height scheduling ===
    schedule_h = jnp.where(height_ref > 0.0, height_ref,
                 0.9 * filtered_com_z + 0.1 * com_z)
    new_filtered_com_z = schedule_h

    # K2 profile: continuous_k_position=False, continuous_k_wheel_velocity=False,
    # continuous_kd_pitch=False. Use base constructor values directly.
    # Only max_position_tau is continuous-scheduled (continuous_max_position_tau=True).
    kpos = 0.0        # base k_position for K2
    kwheel = 0.5       # base k_wheel_velocity for K2
    kd_pitch = 10.0    # base kd_pitch for K2 (not scheduled)

    # === Step 3: Calibrated outer loop + physics FF ===
    cal_grid = _get_calibrated_grid()
    cal_kp = k2_jax_grid_interpolate(schedule_h, cal_grid["grid_heights"], cal_grid["kp_grid"])
    cal_kd = k2_jax_grid_interpolate(schedule_h, cal_grid["grid_heights"], cal_grid["kd_grid"])
    cal_theta_max = k2_jax_grid_interpolate(schedule_h, cal_grid["grid_heights"], cal_grid["theta_max_grid"])
    cal_deadband = k2_jax_grid_interpolate(schedule_h, cal_grid["grid_heights"], cal_grid["deadband_grid"])
    cal_rate_limit = k2_jax_grid_interpolate(schedule_h, cal_grid["grid_heights"], cal_grid["rate_limit_grid"])
    cal_lowpass_alpha = k2_jax_grid_interpolate(schedule_h, cal_grid["grid_heights"], cal_grid["lowpass_grid"])

    ff_grid = _get_physics_ff_grid()
    physics_ff_tau = k2_jax_grid_interpolate(schedule_h, ff_grid["grid_heights"], ff_grid["tau_eq_ff_grid"])

    # Low-band support
    lb_offset, _ = k2_jax_low_band_support_pitch_ref(
        schedule_h, support_pos_err, 0.320, 0.004, 1.4, 3.0, 1.0)

    # Outer loop: update state — active K2 mechanism.
    support_error_rate_raw = jnp.where(
        ol_prev_support_error == 0.0, 0.0,
        (support_pos_err - ol_prev_support_error) / control_dt)
    new_ol_support_error_rate = _jax_apply_lowpass(
        ol_support_error_rate, support_error_rate_raw, cal_lowpass_alpha)
    new_ol_prev_support_error = support_pos_err

    ol_dynamic = k2_jax_compute_outer_loop_pitch_ref(
        support_pos_err, new_ol_support_error_rate, 0.0,
        cal_kp, cal_kd, 0.0, cal_deadband, cal_theta_max)
    ol_target = _jax_apply_rate_limit(
        ol_pitch_ref_smoothed, ol_dynamic, cal_rate_limit)
    new_ol_pitch_ref = _jax_apply_lowpass(
        ol_pitch_ref_smoothed, ol_target, cal_lowpass_alpha)

    # Total pitch ref offset (physics FF + low-band static only for comparison)
    physics_pitch_eq = k2_jax_grid_interpolate(
        schedule_h, ff_grid["grid_heights"], ff_grid["pitch_eq_grid"])
    total_pitch_ref_offset_deg = new_ol_pitch_ref + lb_offset + physics_pitch_eq

    # Apply pitch_ref_offset to effective pitch: offset reduces tau_pitch to center drift.
    # Sign: positive offset_deg → subtract from pitch_x (shift reference forward).
    pitch_ref_offset_rad = total_pitch_ref_offset_deg * (jnp.pi / 180.0)
    effective_pitch_x = pitch_x - pitch_ref_offset_rad

    # === Step 4: Sagittal torque assembly ===
    tau_sag, sag_diag = k2_jax_sagittal_torque_assembly(
        pitch_x_rad=effective_pitch_x, pitch_rate_rad_s=pitch_rate_eff,
        sagittal_velocity_m_s=sag_vel, sagittal_position_error_m=sag_pos_err,
        wheel_vel_left_rad_s=wheel_vel_l, wheel_vel_right_rad_s=wheel_vel_r,
        support_velocity_m_s=support_vel,
        kp_pitch=50.0, effective_pitch_scale=1.0, effective_pitch_tau_cap=0.0,
        effective_kd_pitch=kd_pitch,
        effective_k_velocity=0.0, effective_velocity_damping_scale=1.0,
        effective_support_velocity_gain=0.0, effective_support_velocity_scale=1.0,
        effective_k_wheel_velocity=kwheel,
        effective_k_position=kpos, effective_max_position_tau=3.0,
        kp_cp=0.0, kd_com_vy=5.0,
        wheel_torque_sign=1.0,
    )

    # === Step 5: Shape posture ===
    joint_pos_full = jnp.array([q_hr_l, q_hy_l, q_hp_l, q_kn_l, 0.0,
                                  q_hr_r, q_hy_r, q_hp_r, q_kn_r, 0.0], dtype=jnp.float64)
    joint_vel_full = jnp.array([qd_hr_l, qd_hy_l, qd_hp_l, qd_kn_l, 0.0,
                                  qd_hr_r, qd_hy_r, qd_hp_r, qd_kn_r, 0.0], dtype=jnp.float64)
    q_ref_full = jnp.array([qref_hr_l, qref_hy_l, qref_hp_l, qref_kn_l, 0.0,
                              qref_hr_r, qref_hy_r, qref_hp_r, qref_kn_r, 0.0], dtype=jnp.float64)

    tau_posture, _ = k2_jax_shape_posture_compute(q_ref_full, joint_pos_full, joint_vel_full)

    # === Step 6: Lateral roll ===
    tau_lateral, _ = k2_jax_lateral_roll_compute(roll_y, roll_rate,
        hip_roll_pos_left=q_hr_l, hip_roll_pos_right=q_hr_r,
        hip_roll_vel_left=qd_hr_l, hip_roll_vel_right=qd_hr_r,
        hip_roll_ref_left=qref_hr_l, hip_roll_ref_right=qref_hr_r)

    # === Step 7: Yaw ===
    tau_yaw = k2_jax_yaw_compute(yaw_err, yaw_rate)

    # === Step 8: Mode-div ===
    tau_mode_div = k2_jax_mode_div_compute(
        hy_div_err, hy_div_rate, schedule_h)

    # === Step 9: Support feedforward ===
    tau_support_ff = k2_jax_support_feedforward_compute(
        support_pos_err, schedule_h)

    # === Step 10: Sum and compose (active K2 mechanism) ===
    # Compose the four primary sources. Yaw and mode_div are applied POST-composer
    # (matching real simulation order where they're added to tau_smooth after).
    tau_sum = tau_sag + tau_posture + tau_lateral + tau_support_ff

    tau_final, tau_clipped, sat_mask, rate_mask = k2_jax_torque_composer_step(
        tau_sum, prev_tau, params_flat)

    # Post-composer additions: yaw and mode_div on hip-yaw joints
    tau_final = tau_final.at[1].add(tau_yaw[1])
    tau_final = tau_final.at[6].add(tau_yaw[6])
    tau_final = tau_final.at[1].add(tau_mode_div[1])
    tau_final = tau_final.at[6].add(tau_mode_div[6])

    # === Pack new state ===
    new_state = state_flat.at[_S_NOTCH_X1].set(new_notch_x1)
    new_state = new_state.at[_S_NOTCH_X2].set(new_notch_x2)
    new_state = new_state.at[_S_NOTCH_Y1].set(new_notch_y1)
    new_state = new_state.at[_S_NOTCH_Y2].set(new_notch_y2)
    new_state = new_state.at[_S_PREV_TAU_START:_S_PREV_TAU_START + 10].set(tau_final)
    new_state = new_state.at[_S_FILTERED_COM_Z].set(new_filtered_com_z)
    new_state = new_state.at[_S_PREV_SUPPORT_ERROR].set(support_pos_err)
    new_state = new_state.at[_S_OL_PITCH_REF_SMOOTHED].set(new_ol_pitch_ref)
    new_state = new_state.at[_S_OL_PREV_SUPPORT_ERROR].set(new_ol_prev_support_error)
    new_state = new_state.at[_S_OL_SUPPORT_ERROR_RATE].set(new_ol_support_error_rate)

    # === Pack diagnostics ===
    diag = jnp.zeros(K2_JAX_DIAG_SIZE, dtype=jnp.float64)
    diag = diag.at[_D_NOTCH_OUT].set(notch_out)
    diag = diag.at[_D_NOTCH_GATE].set(notch_gate)
    diag = diag.at[_D_TAU_PITCH].set(sag_diag["tau_pitch"])
    diag = diag.at[_D_TAU_PITCH_RATE].set(sag_diag["tau_pitch_rate"])
    diag = diag.at[_D_TAU_SAG_VEL].set(sag_diag["tau_sagittal_velocity"])
    diag = diag.at[_D_TAU_SUPPORT_VEL].set(sag_diag["tau_support_velocity"])
    diag = diag.at[_D_TAU_POSITION].set(sag_diag["tau_position"])
    diag = diag.at[_D_TAU_WHEEL_L].set(sag_diag["tau_wheel_vel_left"])
    diag = diag.at[_D_TAU_WHEEL_R].set(sag_diag["tau_wheel_vel_right"])
    diag = diag.at[_D_SCHED_KPOS].set(kpos)
    diag = diag.at[_D_SCHED_KWHEEL].set(kwheel)
    diag = diag.at[_D_SCHED_KD].set(kd_pitch)
    diag = diag.at[_D_CALIB_KP].set(cal_kp)
    diag = diag.at[_D_CALIB_KD].set(cal_kd)
    diag = diag.at[_D_CALIB_THETA].set(cal_theta_max)
    diag = diag.at[_D_CALIB_DB].set(cal_deadband)
    diag = diag.at[_D_PHYSICS_FF].set(physics_ff_tau)
    diag = diag.at[_D_LOW_BAND].set(lb_offset)
    diag = diag.at[_D_TAU_FINAL_START:_D_TAU_FINAL_START + 10].set(tau_final)
    diag = diag.at[_D_CLIP_COUNT].set(jnp.sum(sat_mask).astype(jnp.float64))
    diag = diag.at[_D_RATE_COUNT].set(jnp.sum(rate_mask).astype(jnp.float64))

    return tau_final, new_state, diag
