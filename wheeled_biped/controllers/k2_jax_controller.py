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
    # Mode-div hip-yaw controller config (2) — D2/D3 bugfix
    "mode_div_soft_gain",
    "mode_div_ref_source",  # 0 = "target", 1 = "zero_only_for_debug" (unsupported → error)
    # Sagittal velocity damping profile config (2) — Phase 4 parity fix
    "k_velocity",                 # effective sagittal velocity gain [Nm/(m/s)]
    "velocity_damping_scale",     # additional damping scale (1.10 in K2 via ADAPTIVE_SUPPORT_CENTERING_TRIM)
    # APCR1ND gating params (8) — Phase 4+ APCR1ND full port
    "apcr1nd_startup_guard_steps",
    "apcr1nd_safe_min_com_z",
    "apcr1nd_safe_roll_rad",
    "apcr1nd_safe_pitch_rad",
    "apcr1nd_direct_enter_m",
    "apcr1nd_release_inner_m",
    "apcr1nd_hold_outside_band",       # 0.0=False, >0=True
    "apcr1nd_converging_release_steps",
)
K2_JAX_PARAMS_SIZE_STAGE2: int = len(K2_JAX_PARAMS_FIELDS_STAGE2)  # 41 (was 33)
# Phase 4 push fix: extended params for position cap boost (7 extra scalars)
K2_JAX_PARAMS_SIZE_STAGE2_EXT = K2_JAX_PARAMS_SIZE_STAGE2 + 7  # 48 (base pos cap)
# Phase 3 standalone: standalone mode flag + equilibrium constants (1 + 5 = 6 extra)
K2_JAX_PARAMS_SIZE_STAGE2_EXT_STANDALONE = K2_JAX_PARAMS_SIZE_STAGE2_EXT + 6  # 54
_IDX_APCR1ND_POS_CAP_BOOST_ENABLED = K2_JAX_PARAMS_SIZE_STAGE2 + 0
_IDX_APCR1ND_POS_CAP_NORMAL = K2_JAX_PARAMS_SIZE_STAGE2 + 1
_IDX_APCR1ND_POS_CAP_SOFT = K2_JAX_PARAMS_SIZE_STAGE2 + 2
_IDX_APCR1ND_POS_CAP_DESIRED = K2_JAX_PARAMS_SIZE_STAGE2 + 3
_IDX_APCR1ND_POS_CAP_HARD = K2_JAX_PARAMS_SIZE_STAGE2 + 4
_IDX_APCR1ND_POS_CAP_EMERGENCY = K2_JAX_PARAMS_SIZE_STAGE2 + 5
_IDX_APCR1ND_BAND_SOFT_ENTER = K2_JAX_PARAMS_SIZE_STAGE2 + 6
# Phase 3 standalone mode flag + equilibrium constants (indices 48-53)
_IDX_STANDALONE_MODE = K2_JAX_PARAMS_SIZE_STAGE2_EXT + 0  # 48
_IDX_PITCH_X_EQ_RAD = K2_JAX_PARAMS_SIZE_STAGE2_EXT + 1  # 49
_IDX_SUPPORT_CENTER_EQ_X = K2_JAX_PARAMS_SIZE_STAGE2_EXT + 2  # 50
_IDX_SUPPORT_CENTER_EQ_Y = K2_JAX_PARAMS_SIZE_STAGE2_EXT + 3  # 51
_IDX_SAGITTAL_AXIS_X = K2_JAX_PARAMS_SIZE_STAGE2_EXT + 4  # 52
_IDX_SAGITTAL_AXIS_Y = K2_JAX_PARAMS_SIZE_STAGE2_EXT + 5  # 53

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
_IDX_MODE_DIV_SOFT_GAIN = 29
_IDX_MODE_DIV_REF_SOURCE = 30
_IDX_K_VELOCITY = 31
_IDX_VELOCITY_DAMPING_SCALE = 32
_IDX_APCR1ND_STARTUP_GUARD = 33
_IDX_APCR1ND_SAFE_COM_Z = 34
_IDX_APCR1ND_SAFE_ROLL = 35
_IDX_APCR1ND_SAFE_PITCH = 36
_IDX_APCR1ND_DIRECT_ENTER = 37
_IDX_APCR1ND_RELEASE_INNER = 38
_IDX_APCR1ND_HOLD_OUTSIDE = 39
_IDX_APCR1ND_CONVERGING_RELEASE = 40


def pack_params_stage2(
    fs_hz: float = 100.0,
    fc_hz: float = 2.5,
    Q: float = 2.0,
    torque_limit: np.ndarray | jnp.ndarray | None = None,
    max_torque_rate: np.ndarray | jnp.ndarray | None = None,
    control_dt: float = 0.01,
    mode_div_soft_gain: float = 0.80,
    mode_div_ref_source: str = "target",
    k_velocity: float = 15.0,
    velocity_damping_scale: float = 1.0,
    # APCR1ND gating params (K2 profile defaults)
    apcr1nd_startup_guard_steps: float = 40.0,
    apcr1nd_safe_min_com_z: float = 0.25,
    apcr1nd_safe_roll_rad: float = 0.30,
    apcr1nd_safe_pitch_rad: float = 0.30,
    apcr1nd_direct_enter_m: float = 0.06,
    apcr1nd_release_inner_m: float = 0.03,
    apcr1nd_hold_outside_band: bool = True,
    apcr1nd_converging_release_steps: float = 15.0,
    # Phase 3 standalone mode: equilibrium constants
    standalone_mode: bool = False,
    pitch_x_eq_rad: float = 0.0,
    support_center_eq_x_m: float = 0.0,
    support_center_eq_y_m: float = 0.0,
    sagittal_axis_x: float = 0.0,
    sagittal_axis_y: float = 0.0,
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
        mode_div_soft_gain: Mode-div soft gain height band [m] (default 0.80 — K2 runtime)
        mode_div_ref_source: "target" (0) or "zero_only_for_debug" (1)
        k_velocity: Sagittal velocity damping gain [Nm/(m/s)] (default 15.0)
        velocity_damping_scale: Additional velocity damping scale (default 1.0;
            K2 profile uses 1.10 via ADAPTIVE_SUPPORT_CENTERING_TRIM)
        apcr1nd_*: APCR1ND gating params (defaults match K2_NOTCH_LOW_Q_V1)
        standalone_mode: If True, JAX computes sag_pos_err/sag_vel/support_vel/pitch_x_error
            from raw state inputs instead of using Python-computed values.
        pitch_x_eq_rad: Equilibrium body pitch from gravity vector (rad)
        support_center_eq_x_m, support_center_eq_y_m: Equilibrium support center (m)
        sagittal_axis_x, sagittal_axis_y: Sagittal heading unit vector components

    Returns:
        Flat params array, shape (54 if standalone else 48), dtype float64
    """
    b0, b1, b2, a1, a2 = _python_biquad_notch_coefficients(fs_hz, fc_hz, Q)

    # Always allocate full standalone size to avoid JAX out-of-bounds tracing errors.
    # standalone_mode flag controls whether JAX uses the raw-input preprocessing path.
    _param_size = K2_JAX_PARAMS_SIZE_STAGE2_EXT_STANDALONE  # 54
    params = jnp.zeros(_param_size, dtype=jnp.float64)
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

    # D2/D3 bugfix: mode_div soft_gain and ref_source
    params = params.at[_IDX_MODE_DIV_SOFT_GAIN].set(float(mode_div_soft_gain))
    _ref_src_int = 0 if mode_div_ref_source == "target" else (2 if mode_div_ref_source == "disabled" else 1)
    params = params.at[_IDX_MODE_DIV_REF_SOURCE].set(float(_ref_src_int))

    # Phase 4 parity fix: sagittal velocity damping config (K2 profile values)
    params = params.at[_IDX_K_VELOCITY].set(float(k_velocity))
    params = params.at[_IDX_VELOCITY_DAMPING_SCALE].set(float(velocity_damping_scale))

    # Phase 4+ APCR1ND gating params
    params = params.at[_IDX_APCR1ND_STARTUP_GUARD].set(float(apcr1nd_startup_guard_steps))
    params = params.at[_IDX_APCR1ND_SAFE_COM_Z].set(float(apcr1nd_safe_min_com_z))
    params = params.at[_IDX_APCR1ND_SAFE_ROLL].set(float(apcr1nd_safe_roll_rad))
    params = params.at[_IDX_APCR1ND_SAFE_PITCH].set(float(apcr1nd_safe_pitch_rad))
    params = params.at[_IDX_APCR1ND_DIRECT_ENTER].set(float(apcr1nd_direct_enter_m))
    params = params.at[_IDX_APCR1ND_RELEASE_INNER].set(float(apcr1nd_release_inner_m))
    params = params.at[_IDX_APCR1ND_HOLD_OUTSIDE].set(1.0 if apcr1nd_hold_outside_band else 0.0)
    params = params.at[_IDX_APCR1ND_CONVERGING_RELEASE].set(float(apcr1nd_converging_release_steps))

    # Phase 3 standalone: always set these fields (params array is always full size)
    params = params.at[_IDX_STANDALONE_MODE].set(1.0 if standalone_mode else 0.0)
    params = params.at[_IDX_PITCH_X_EQ_RAD].set(float(pitch_x_eq_rad))
    params = params.at[_IDX_SUPPORT_CENTER_EQ_X].set(float(support_center_eq_x_m))
    params = params.at[_IDX_SUPPORT_CENTER_EQ_Y].set(float(support_center_eq_y_m))
    params = params.at[_IDX_SAGITTAL_AXIS_X].set(float(sagittal_axis_x))
    params = params.at[_IDX_SAGITTAL_AXIS_Y].set(float(sagittal_axis_y))

    return params


def unpack_params_stage2(params_flat: jnp.ndarray) -> dict:
    """Unpack flat JAX params array into Python dict (Stage 2 layout)."""
    p = np.asarray(params_flat, dtype=np.float64)
    _ref_src_int = int(p[_IDX_MODE_DIV_REF_SOURCE])
    _ref_src = "target" if _ref_src_int == 0 else ("disabled" if _ref_src_int == 2 else "zero_only_for_debug")
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
        "mode_div_soft_gain": float(p[_IDX_MODE_DIV_SOFT_GAIN]),
        "mode_div_ref_source": _ref_src,
        "k_velocity": float(p[_IDX_K_VELOCITY]),
        "velocity_damping_scale": float(p[_IDX_VELOCITY_DAMPING_SCALE]),
        "apcr1nd_startup_guard_steps": float(p[_IDX_APCR1ND_STARTUP_GUARD]),
        "apcr1nd_safe_min_com_z": float(p[_IDX_APCR1ND_SAFE_COM_Z]),
        "apcr1nd_safe_roll_rad": float(p[_IDX_APCR1ND_SAFE_ROLL]),
        "apcr1nd_safe_pitch_rad": float(p[_IDX_APCR1ND_SAFE_PITCH]),
        "apcr1nd_direct_enter_m": float(p[_IDX_APCR1ND_DIRECT_ENTER]),
        "apcr1nd_release_inner_m": float(p[_IDX_APCR1ND_RELEASE_INNER]),
        "apcr1nd_hold_outside_band": bool(p[_IDX_APCR1ND_HOLD_OUTSIDE] > 0.5),
        "apcr1nd_converging_release_steps": float(p[_IDX_APCR1ND_CONVERGING_RELEASE]),
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
    # K2_NOTCH_LOW_Q_V1 profile uses calibrated_outer_loop_function_version="v2".
    # D12 bugfix: import v2 functions (not v1) to match Python K2 runtime.
    from wheeled_biped.controllers.calibrated_outer_loop_functions_v2 import (
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
# Stage 3: APCR1ND wheel damping override (K2 profile parity fix)
# ===========================================================================
# K2_NOTCH_LOW_Q_V1 applies a band-based damping scale + minimum clamp to
# wheel velocity damping torques. Without this, JAX diverges from Python
# when sagittal position error exceeds 0.05 m (soft_enter band).
# ===========================================================================

# K2 profile constants (from K2_NOTCH_LOW_Q_V1 at module load)
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import K2_NOTCH_LOW_Q_V1 as _K2_APCR

_K2_APCR_ENABLED = bool(_K2_APCR.vd_wheel_damping_recenter_override_enabled)
_K2_APCR_TUNED = bool(_K2_APCR.apcr1nd_tuned_enabled)
_K2_APCR_SOFT_ENTER_M = float(_K2_APCR.apcr1nd_soft_enter_m)
_K2_APCR_DESIRED_BAND_M = float(_K2_APCR.apcr1nd_desired_band_m)
_K2_APCR_HARD_BAND_M = float(_K2_APCR.apcr1nd_hard_band_m)
_K2_APCR_EMERGENCY_BAND_M = float(_K2_APCR.apcr1nd_emergency_band_m)
_K2_APCR_SCALE_NORMAL = float(_K2_APCR.apcr1nd_damping_scale_normal)
_K2_APCR_SCALE_SOFT = float(_K2_APCR.apcr1nd_damping_scale_soft)
_K2_APCR_SCALE_DESIRED = float(_K2_APCR.apcr1nd_damping_scale_desired)
_K2_APCR_SCALE_HARD = float(_K2_APCR.apcr1nd_damping_scale_hard)
_K2_APCR_SCALE_EMERGENCY = float(_K2_APCR.apcr1nd_damping_scale_emergency)
_K2_APCR_MIN_DAMPING_NM = float(_K2_APCR.vd_wheel_damping_recenter_min_abs_nm)
_K2_APCR_PRESERVE_IF_HELPS = bool(_K2_APCR.apcr1nd_preserve_damping_if_helps)


def k2_jax_apcr1nd_compute_gate(
    sagittal_position_error_m,
    prev_error,
    step_counter,
    converging_steps,
    recenter_held,
    pitch_x_rad,
    roll_y_rad,
    com_z_m,
    contact_valid,
    # Params
    startup_guard_steps,
    safe_min_com_z,
    safe_roll_rad,
    safe_pitch_rad,
    soft_enter_m,
    direct_enter_m,
    desired_band_m,
    release_inner_m,
    hold_outside_band,
    converging_release_steps,
):
    """Compute APCR1ND direct recenter priority active gate — pure JAX.

    Matches Python SagittalVelocityDampedBalanceController lines 6349-6490
    (APCR1nD Tuned Variants Logic). Returns (recenter_active, new_state).

    Returns:
        recenter_active: bool — whether APCR1ND wheel damping override should apply
        new_step_counter: updated step counter
        new_converging_steps: updated converging steps counter
        new_recenter_held: updated hold/latch state
    """
    # Startup guard
    after_guard = step_counter >= startup_guard_steps
    new_step_counter = step_counter + 1.0

    # Drift detection
    signed_error = sagittal_position_error_m
    abs_error = jnp.abs(signed_error)
    e_dot = signed_error - prev_error
    new_prev_error = sagittal_position_error_m  # always update prev for next step
    moving_away = signed_error * e_dot > 0.0
    converging = (~moving_away) & (jnp.abs(e_dot) > 1e-6)

    # Safety gates — MUST include contact_valid (matches Python svdbc.py:6433)
    abs_pitch = jnp.abs(pitch_x_rad)
    abs_roll = jnp.abs(roll_y_rad)
    com_z_safe = com_z_m >= safe_min_com_z
    roll_safe = abs_roll <= safe_roll_rad
    pitch_safe = abs_pitch <= safe_pitch_rad
    safety_pass = contact_valid & com_z_safe & roll_safe & pitch_safe

    # Update converging steps counter — ONLY when safety passes (matches Python lines 6427-6430)
    # Python: converging steps update is inside the `else` (safety_pass) branch
    new_converging_steps = jnp.where(
        after_guard & safety_pass & converging,
        converging_steps + 1.0,
        jnp.where(after_guard & safety_pass, 0.0, converging_steps),
    )

    # Entry conditions
    soft_entry = (abs_error >= soft_enter_m) & (abs_error < direct_enter_m) & moving_away
    direct_entry = (abs_error >= direct_enter_m) & moving_away
    emergency_entry = abs_error >= desired_band_m

    # Hold condition
    prev_active = recenter_held > 0.5
    hold_condition = prev_active & (abs_error > release_inner_m)
    hold_outside_band_condition = (hold_outside_band > 0.5) & (abs_error > desired_band_m)

    # Release conditions
    release_by_inner_band = abs_error <= release_inner_m
    release_by_converging = (
        converging
        & (new_converging_steps >= converging_release_steps)
        & (abs_error <= desired_band_m * 0.75)
    )

    # Decision — only applicable after startup guard AND safety pass
    gated = after_guard & safety_pass
    release = gated & (release_by_inner_band | release_by_converging)
    activate = gated & (
        emergency_entry | hold_outside_band_condition | direct_entry | soft_entry | hold_condition
    )

    # IMPORTANT: release takes priority over activate (matches Python's if/elif chain)
    # When safety fails (!gated), reset recenter_held to 0 (matches Python svdbc.py:6445-6446)
    new_recenter_held = jnp.where(
        release, 0.0,
        jnp.where(activate, 1.0,
        jnp.where(after_guard & ~safety_pass, 0.0, recenter_held)),
    )
    recenter_active = new_recenter_held > 0.5

    return recenter_active, new_step_counter, new_prev_error, new_converging_steps, new_recenter_held


def k2_jax_compute_boosted_position_cap(
    abs_error, safety_gate_pass, boost_enabled, apcr1nd_tuned_enabled,
    soft_enter_m, hard_band_m, emergency_band_m, desired_band_m,
    cap_normal, cap_soft, cap_desired, cap_hard, cap_emergency,
):
    """Compute APCR1ND band-based boosted position cap — pure JAX.

    Matches Python SagittalVelocityDampedBalanceController lines 6702-6726.
    When position_cap_recenter_boost_enabled=True and safety gate passes,
    raises max_position_tau based on the APCR1ND band of abs_error.

    Returns:
        boosted_cap: float — the boosted position cap [Nm]
    """
    # Only boost when enabled AND safety passes
    _do_boost = (boost_enabled > 0.5) & safety_gate_pass

    # Determine band-based cap (matching Python's if/elif chain order)
    _is_emergency = abs_error >= emergency_band_m
    _is_hard = abs_error >= hard_band_m
    _is_desired = abs_error >= desired_band_m
    _is_soft = abs_error >= soft_enter_m

    _band_cap = jnp.where(
        _is_emergency, cap_emergency,
        jnp.where(_is_hard, cap_hard,
        jnp.where(_is_desired, cap_desired,
        jnp.where(_is_soft, cap_soft,
        cap_normal))))

    # When not boosting, return cap_normal (nominal value). When boosting, return band_cap.
    return jnp.where(_do_boost, _band_cap, cap_normal)


def k2_jax_apcr1nd_wheel_damping_override(
    tau_wheel_vel_left, tau_wheel_vel_right,
    wheel_vel_left_rad_s, wheel_vel_right_rad_s,
    sagittal_position_error_m,
    recenter_active=True,
):
    """Apply K2 APCR1ND wheel damping override — pure JAX function.

    Matches Python SagittalVelocityDampedBalanceController APCR1ND logic:
    1. Gate: only apply when recenter_active=True (from k2_jax_apcr1nd_compute_gate)
    2. Compute drift_sign from position error
    3. Compute wheel_vel_sign from mean wheel velocity
    4. Determine if damping fights drift (same sign)
    5. Band-based damping scale from position error magnitude
    6. Preserve damping if it opposes drift
    7. Scale + min-clamp wheel damping torques

    Args:
        tau_wheel_vel_left: Raw left wheel velocity damping torque [Nm]
        tau_wheel_vel_right: Raw right wheel velocity damping torque [Nm]
        wheel_vel_left_rad_s: Left wheel velocity [rad/s]
        wheel_vel_right_rad_s: Right wheel velocity [rad/s]
        sagittal_position_error_m: Sagittal position error [m]
        recenter_active: APCR1ND gating flag — only apply when True

    Returns:
        (tau_wheel_vel_left, tau_wheel_vel_right) — potentially overridden
    """
    if not _K2_APCR_ENABLED:
        return tau_wheel_vel_left, tau_wheel_vel_right

    # Drift sign from position error
    drift_sign = jnp.sign(sagittal_position_error_m)

    # Wheel velocity mean and sign
    wheel_vel_mean = 0.5 * (wheel_vel_left_rad_s + wheel_vel_right_rad_s)
    wheel_vel_sign = jnp.sign(wheel_vel_mean)

    # Damping fights drift when wheel velocity and drift have SAME sign
    damping_fights_drift = jnp.abs(drift_sign - wheel_vel_sign) < 0.5

    # Band-based damping scale
    abs_error = jnp.abs(sagittal_position_error_m)
    wheel_scale = jnp.where(
        abs_error >= _K2_APCR_EMERGENCY_BAND_M, _K2_APCR_SCALE_EMERGENCY,
        jnp.where(abs_error >= _K2_APCR_HARD_BAND_M, _K2_APCR_SCALE_HARD,
        jnp.where(abs_error >= _K2_APCR_DESIRED_BAND_M, _K2_APCR_SCALE_DESIRED,
        jnp.where(abs_error >= _K2_APCR_SOFT_ENTER_M, _K2_APCR_SCALE_SOFT,
        _K2_APCR_SCALE_NORMAL))))

    # Tuned variant: always use band scale (overridden below if preserve-if-helps)
    # Preserve damping if it helps (opposes drift)
    if _K2_APCR_PRESERVE_IF_HELPS:
        damping_opposes_drift = ~damping_fights_drift
        wheel_scale = jnp.where(damping_opposes_drift, 1.0, wheel_scale)

    # Apply override when recenter_active AND wheel_scale < 1.0 (tuned variant)
    # Gating: recenter_active must be True (from k2_jax_apcr1nd_compute_gate)
    apply_override = recenter_active & _K2_APCR_TUNED & (wheel_scale < 1.0)

    # Scale and min-clamp
    tau_l_scaled = tau_wheel_vel_left * wheel_scale
    tau_r_scaled = tau_wheel_vel_right * wheel_scale

    min_d = _K2_APCR_MIN_DAMPING_NM
    tau_l = jnp.where(
        apply_override & (jnp.abs(tau_l_scaled) < min_d),
        min_d * jnp.sign(tau_l_scaled),
        jnp.where(apply_override, tau_l_scaled, tau_wheel_vel_left),
    )
    tau_r = jnp.where(
        apply_override & (jnp.abs(tau_r_scaled) < min_d),
        min_d * jnp.sign(tau_r_scaled),
        jnp.where(apply_override, tau_r_scaled, tau_wheel_vel_right),
    )

    return tau_l, tau_r


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
    external_position_trim=0.0,  # adaptive_bias_trim contribution (Stage 4H)
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
    # Phase 0 fix: Match Python's TWO-CLIP sequence (svdbc.py:5472 + svdbc.py:5566).
    # Python clips BEFORE trim addition (line 5472), adds trim (line 5565),
    # then clips again (line 5566). JAX previously added trim first and clipped
    # once, which diverges when tau_position_raw > effective_max_position_tau:
    #   Python: pos_clipped = clip(raw, cap) → pos_clipped + trim → clip again
    #   JAX old: raw + trim → clip once
    # The first clip caps the position torque BEFORE trim, affecting the
    # post-trim value differently than a single combined clip.
    tau_position_no_trim = tau_position_p + position_integral_tau
    tau_position_no_trim = jnp.clip(tau_position_no_trim, -effective_max_position_tau, effective_max_position_tau)
    tau_position = tau_position_no_trim + external_position_trim

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
    kp_hip_yaw=15.0, kd_hip_yaw=3.0,
    # NOTE: These gains differ from Python's LegPositionController (kp=20/35, kd=3/4).
    # The JAX standalone controller has a simpler control structure than Python's
    # multi-layer (WBC + posture regularizer + leg position), so it needs higher
    # PD gains to achieve comparable effective leg stiffness. Reducing these to
    # match Python caused SAFETY_FAIL in dynamic height scenarios.
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


def k2_jax_yaw_compute(yaw_error_rad, yaw_rate_rad_s, kp_yaw=8.0, kd_yaw=2.0, max_yaw_torque=5.0):
    """Yaw controller — pure JAX function."""
    tau_antisym = jnp.clip(kp_yaw * yaw_error_rad - kd_yaw * yaw_rate_rad_s, -max_yaw_torque, max_yaw_torque)
    tau = jnp.zeros(10, dtype=jnp.float64)
    tau = tau.at[1].set(-tau_antisym)
    tau = tau.at[6].set(tau_antisym)
    return tau


def k2_jax_mode_div_compute(
    div_error, div_rate, height_m,
    kp_div=10.0, kd_div=0.50, max_torque=7.5,
    soft_limit_rad=0.30, soft_gain=0.50,
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


# Stage 7B: Empirical support feedforward vector (hip_pitch/knee fixed torques)
# Must match Python SupportFeedforwardController with scale=0.5, joint_group="hip_pitch_knee"
# Vector = [0, 0, 4.1, -15.5, 0, 0, 0, 3.2, -15.8, 0] × 0.5
_K2_EMPIRICAL_SUPPORT_FF = jnp.array(
    [0.0, 0.0, 2.05, -7.75, 0.0, 0.0, 0.0, 1.6, -7.9, 0.0],
    dtype=jnp.float64,
)


def k2_jax_empirical_support_ff():
    """Return the empirical support feedforward torque vector (hip_pitch/knee)."""
    return _K2_EMPIRICAL_SUPPORT_FF


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
# ABS ring buffer window sizes (K2 profile constants)
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import K2_NOTCH_LOW_Q_V1 as _K2_SCH_ABS
_ABS_SLOW_WINDOW_MODULE = int(_K2_SCH_ABS.adaptive_bias_window_steps)  # 300
_ABS_FAST_WINDOW_MODULE = int(_K2_SCH_ABS.adaptive_bias_fast_window_steps)  # 100
_ABS_ZC_WINDOW_MODULE = int(_K2_SCH_ABS.adaptive_bias_zero_crossing_window_steps)  # 500

# ABS ring buffer fields appended after the 6 core fields
_ABS_RING_FIELDS = tuple(f"abs_buf_{i}" for i in range(_ABS_SLOW_WINDOW_MODULE))
_ABS_CORE_FIELDS = ("abs_slow_sum", "abs_fast_sum", "abs_trim_tau",
    "abs_hold_steps", "abs_prev_err_sign", "abs_zc_count",
    "abs_slow_count", "abs_slow_ptr", "abs_guard_trigger")
# Phase 6M: Separate ZC ring buffer (500 entries) for zero-crossing parity
_ABS_ZC_HEADER_FIELDS = ("abs_zc_buf_count", "abs_zc_buf_ptr")
_ABS_ZC_BUF_FIELDS = tuple(f"abs_zc_buf_{i}" for i in range(_ABS_ZC_WINDOW_MODULE))
K2_JAX_STATE_FIELDS = K2_JAX_STATE_FIELDS + _ABS_CORE_FIELDS + _ABS_RING_FIELDS + _ABS_ZC_HEADER_FIELDS + _ABS_ZC_BUF_FIELDS
# APCR1ND gating state (4) — Phase 4+ APCR1ND full port (shifted by +502 from ZC buffer)
_APCR1ND_STATE_FIELDS = ("apcr1nd_step_counter", "apcr1nd_prev_error",
                          "apcr1nd_tuned_converging_steps", "apcr1nd_tuned_recenter_held")
K2_JAX_STATE_FIELDS = K2_JAX_STATE_FIELDS + _APCR1ND_STATE_FIELDS
# Phase 7: effective_max_position_tau — Python's runtime value (T6F/T6I-raised cap)
# Captured from sagittal controller in both-synced mode; defaults to 0.0 (use max_pos_tau)
_POS_CAP_STATE_FIELDS = ("effective_max_position_tau_py",)
K2_JAX_STATE_FIELDS = K2_JAX_STATE_FIELDS + _POS_CAP_STATE_FIELDS
# Phase 0: APCR1ND wheel damping override active flag — match Python gate parity
_APCR1ND_WD_OVERRIDE_FIELDS = ("py_wd_override_active",)
K2_JAX_STATE_FIELDS = K2_JAX_STATE_FIELDS + _APCR1ND_WD_OVERRIDE_FIELDS
K2_JAX_STATE_SIZE: int = len(K2_JAX_STATE_FIELDS)  # 836 (= 835 + 1)

# Index constants for core state (unchanged)
_S_NOTCH_X1, _S_NOTCH_X2, _S_NOTCH_Y1, _S_NOTCH_Y2 = 0, 1, 2, 3
_S_PREV_TAU_START = 4
_S_FILTERED_COM_Z = 14
_S_PREV_SUPPORT_ERROR = 15
_S_OL_PITCH_REF_SMOOTHED = 16
_S_OL_PREV_SUPPORT_ERROR = 17
_S_OL_SUPPORT_ERROR_RATE = 18
# ZC buffer indices (after ABS ring buffer: 28 + 300 = 328)
_ABS_ZC_BUF_COUNT = 328
_ABS_ZC_BUF_PTR = 329
_ABS_ZC_BUF_START = 330
_ABS_ZC_BUF_END = 830  # 330 + 500
# APCR1ND state indices (shifted by +502: was 328→830, 329→831, etc.)
_S_APCR1ND_STEP_COUNTER = 830
_S_APCR1ND_PREV_ERROR = 831
_S_APCR1ND_CONVERGING_STEPS = 832
_S_APCR1ND_RECENTER_HELD = 833
_S_EFFECTIVE_MAX_POSITION_TAU_PY = 834  # Phase 7: Python's runtime effective_max_position_tau
_S_PY_WD_OVERRIDE_ACTIVE = 835  # Phase 0: Python's APCR1ND wheel damping override active flag


def pack_state_k2(
    notch_x1=0.0, notch_x2=0.0, notch_y1=0.0, notch_y2=0.0,
    prev_tau=None, filtered_com_z=0.4, prev_support_error=0.0,
    ol_pitch_ref_smoothed=0.0, ol_prev_support_error=0.0, ol_support_error_rate=0.0,
    # APCR1ND gating state
    apcr1nd_step_counter=0.0,
    apcr1nd_prev_error=0.0,
    apcr1nd_tuned_converging_steps=0.0,
    apcr1nd_tuned_recenter_held=0.0,
):
    """Pack all K2 state into flat JAX array (now 332 with ring buffer + APCR1ND)."""
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
    # APCR1ND gating state
    s = s.at[_S_APCR1ND_STEP_COUNTER].set(apcr1nd_step_counter)
    s = s.at[_S_APCR1ND_PREV_ERROR].set(apcr1nd_prev_error)
    s = s.at[_S_APCR1ND_CONVERGING_STEPS].set(apcr1nd_tuned_converging_steps)
    s = s.at[_S_APCR1ND_RECENTER_HELD].set(apcr1nd_tuned_recenter_held)
    # ABS fields initialized to zero by default (zeros array)
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
    "contact_valid",  # Phase 6M: contact state for ABS trim safety gate parity
)
# Unified input size: always 45 elements (42 base + 3 standalone).
# Old 42-element inputs are padded with zeros at indices 42-44.
K2_JAX_INPUT_SIZE: int = 45

_I_PITCH_X, _I_PITCH_RATE, _I_ROLL_Y, _I_ROLL_RATE = 0, 1, 2, 3
_I_YAW_ERR, _I_YAW_RATE, _I_COM_Z, _I_COM_VY = 4, 5, 6, 7
_I_SAG_VEL, _I_SAG_POS_ERR, _I_WHEEL_VEL_L, _I_WHEEL_VEL_R = 8, 9, 10, 11
_I_SUPPORT_VEL, _I_HEIGHT_REF = 12, 13
_I_HY_DIV_ERR, _I_HY_DIV_RATE = 14, 15
_I_Q_START, _I_QD_START = 16, 24
_I_QREF_START, _I_SUPPORT_POS_ERR = 32, 40
_I_CONTACT_VALID = 41
# Phase 3 standalone: extended input fields (indices 42-44)
_I_COM_VX = 42         # com_vx for sagittal velocity projection
_I_SUPPORT_CENTER_X = 43  # wheel support center X (world frame)
_I_SUPPORT_CENTER_Y = 44  # wheel support center Y (world frame)
K2_JAX_INPUT_SIZE_STANDALONE = K2_JAX_INPUT_SIZE  # same unified size


def pack_input_k2(
    pitch_x_rad, pitch_rate_x_rad_s, roll_y_rad, roll_rate_y_rad_s,
    yaw_error_rad, yaw_rate_rad_s, com_z_m, com_vy_m_s,
    sagittal_velocity_m_s, sagittal_position_error_m,
    wheel_vel_left_rad_s, wheel_vel_right_rad_s,
    support_velocity_m_s, commanded_height_ref_m,
    hip_yaw_div_error, hip_yaw_div_rate,
    joint_pos, joint_vel, q_ref,
    support_position_error_m,
    contact_valid=1.0,  # Phase 6M: contact state for ABS trim safety gate parity
):
    """Pack all K2 inputs into flat JAX array (42).

    Uses NumPy intermediate to avoid per-element JAX dispatch overhead
    (~17 ms/step → ~0.01 ms/step on Windows with eager-mode .at[idx].set()).
    """
    import numpy as _np
    inp = _np.zeros(K2_JAX_INPUT_SIZE, dtype=_np.float64)
    inp[_I_PITCH_X] = float(pitch_x_rad)
    inp[_I_PITCH_RATE] = float(pitch_rate_x_rad_s)
    inp[_I_ROLL_Y] = float(roll_y_rad)
    inp[_I_ROLL_RATE] = float(roll_rate_y_rad_s)
    inp[_I_YAW_ERR] = float(yaw_error_rad)
    inp[_I_YAW_RATE] = float(yaw_rate_rad_s)
    inp[_I_COM_Z] = float(com_z_m)
    inp[_I_COM_VY] = float(com_vy_m_s)
    inp[_I_SAG_VEL] = float(sagittal_velocity_m_s)
    inp[_I_SAG_POS_ERR] = float(sagittal_position_error_m)
    inp[_I_WHEEL_VEL_L] = float(wheel_vel_left_rad_s)
    inp[_I_WHEEL_VEL_R] = float(wheel_vel_right_rad_s)
    inp[_I_SUPPORT_VEL] = float(support_velocity_m_s)
    inp[_I_HEIGHT_REF] = float(commanded_height_ref_m)
    inp[_I_HY_DIV_ERR] = float(hip_yaw_div_error)
    inp[_I_HY_DIV_RATE] = float(hip_yaw_div_rate)
    # Joint position slice: indices [1,6,2,7,3,8,0,5] (hip_yaw, hip_pitch, knee, hip_roll)
    _q_slice = _np.array([
        float(joint_pos[1]), float(joint_pos[6]), float(joint_pos[2]), float(joint_pos[7]),
        float(joint_pos[3]), float(joint_pos[8]), float(joint_pos[0]), float(joint_pos[5]),
    ], dtype=_np.float64)
    inp[_I_Q_START:_I_Q_START + 8] = _q_slice
    # Joint velocity slice
    _qd_slice = _np.array([
        float(joint_vel[1]), float(joint_vel[6]), float(joint_vel[2]), float(joint_vel[7]),
        float(joint_vel[3]), float(joint_vel[8]), float(joint_vel[0]), float(joint_vel[5]),
    ], dtype=_np.float64)
    inp[_I_QD_START:_I_QD_START + 8] = _qd_slice
    # Reference slice
    _qref_slice = _np.array([
        float(q_ref[1]), float(q_ref[6]), float(q_ref[2]), float(q_ref[7]),
        float(q_ref[3]), float(q_ref[8]), float(q_ref[0]), float(q_ref[5]),
    ], dtype=_np.float64)
    inp[_I_QREF_START:_I_QREF_START + 8] = _qref_slice
    inp[_I_SUPPORT_POS_ERR] = float(support_position_error_m)
    inp[_I_CONTACT_VALID] = float(contact_valid)
    return jnp.asarray(inp)


def pack_input_k2_standalone(
    pitch_x_rad, pitch_rate_x_rad_s, roll_y_rad, roll_rate_y_rad_s,
    yaw_error_rad, yaw_rate_rad_s, com_z_m, com_vx_m_s, com_vy_m_s,
    wheel_vel_left_rad_s, wheel_vel_right_rad_s,
    commanded_height_ref_m,
    hip_yaw_div_error, hip_yaw_div_rate,
    joint_pos, joint_vel, q_ref,
    support_center_x_m, support_center_y_m,
    contact_valid=1.0,
):
    """Pack raw-state K2 inputs into flat JAX array (45-element standalone contract).

    Unlike pack_input_k2(), this accepts ONLY raw sensor/state values.
    No Python-computed sagittal outputs (pitch_x_error, sag_pos_error,
    support_velocity, etc.) are used. JAX computes all derived quantities
    internally when STANDALONE_MODE=1.

    Input contract:
      - pitch_x_rad: RAW body pitch from centroidal state (NOT pre-adjusted)
      - pitch_rate_x_rad_s: RAW body pitch rate (NOT boosted/filtered)
      - com_vx_m_s, com_vy_m_s: RAW COM velocity for sagittal projection
      - support_center_x_m, support_center_y_m: wheel midpoint from mj_data.xpos
      - All other fields: same as pack_input_k2 (raw state/config)
    """
    import numpy as _np
    inp = _np.zeros(K2_JAX_INPUT_SIZE, dtype=_np.float64)
    inp[_I_PITCH_X] = float(pitch_x_rad)
    inp[_I_PITCH_RATE] = float(pitch_rate_x_rad_s)
    inp[_I_ROLL_Y] = float(roll_y_rad)
    inp[_I_ROLL_RATE] = float(roll_rate_y_rad_s)
    inp[_I_YAW_ERR] = float(yaw_error_rad)
    inp[_I_YAW_RATE] = float(yaw_rate_rad_s)
    inp[_I_COM_Z] = float(com_z_m)
    inp[_I_COM_VY] = float(com_vy_m_s)
    # Fields 8, 9, 12 are unused in standalone mode (JAX computes internally)
    # but populated with raw values for both-synced debug compatibility
    inp[_I_SAG_VEL] = float(com_vy_m_s)  # placeholder
    inp[_I_SAG_POS_ERR] = float(com_vx_m_s)  # placeholder (repurposed for com_vx)
    inp[_I_WHEEL_VEL_L] = float(wheel_vel_left_rad_s)
    inp[_I_WHEEL_VEL_R] = float(wheel_vel_right_rad_s)
    inp[_I_SUPPORT_VEL] = 0.0  # unused in standalone (JAX computes)
    inp[_I_HEIGHT_REF] = float(commanded_height_ref_m)
    inp[_I_HY_DIV_ERR] = float(hip_yaw_div_error)
    inp[_I_HY_DIV_RATE] = float(hip_yaw_div_rate)
    # Joint position slice
    _q_slice = _np.array([
        float(joint_pos[1]), float(joint_pos[6]), float(joint_pos[2]), float(joint_pos[7]),
        float(joint_pos[3]), float(joint_pos[8]), float(joint_pos[0]), float(joint_pos[5]),
    ], dtype=_np.float64)
    inp[_I_Q_START:_I_Q_START + 8] = _q_slice
    # Joint velocity slice
    _qd_slice = _np.array([
        float(joint_vel[1]), float(joint_vel[6]), float(joint_vel[2]), float(joint_vel[7]),
        float(joint_vel[3]), float(joint_vel[8]), float(joint_vel[0]), float(joint_vel[5]),
    ], dtype=_np.float64)
    inp[_I_QD_START:_I_QD_START + 8] = _qd_slice
    # Reference slice
    _qref_slice = _np.array([
        float(q_ref[1]), float(q_ref[6]), float(q_ref[2]), float(q_ref[7]),
        float(q_ref[3]), float(q_ref[8]), float(q_ref[0]), float(q_ref[5]),
    ], dtype=_np.float64)
    inp[_I_QREF_START:_I_QREF_START + 8] = _qref_slice
    inp[_I_SUPPORT_POS_ERR] = 0.0  # unused in standalone
    inp[_I_CONTACT_VALID] = float(contact_valid)
    # Phase 3 standalone extended fields
    inp[_I_COM_VX] = float(com_vx_m_s)
    inp[_I_SUPPORT_CENTER_X] = float(support_center_x_m)
    inp[_I_SUPPORT_CENTER_Y] = float(support_center_y_m)
    return jnp.asarray(inp)


def k2_jax_input_flat_to_dict(input_flat: jnp.ndarray) -> dict:
    """Unpack flat JAX input array into named dict for debugging."""
    d = {}
    for i, field_name in enumerate(K2_JAX_INPUT_FIELDS):
        d[field_name] = float(input_flat[i])
    return d


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
    # Phase 2 push trace: add tau_sag at wheel indices for root-cause isolation
    "tau_sag_4", "tau_sag_9",
    "tau_final_0", "tau_final_1", "tau_final_2", "tau_final_3", "tau_final_4",
    "tau_final_5", "tau_final_6", "tau_final_7", "tau_final_8", "tau_final_9",
    "clip_saturation_count", "rate_limit_active_count",
    # Phase 0: ABS trim intermediate diagnostics for state/timing parity
    "abs_slow_mean", "abs_fast_mean", "abs_sign_err", "abs_raw_target", "abs_clipped",
    "abs_is_decay", "abs_rate", "abs_trim_delta", "abs_new_trim",
    "abs_safety_pass", "external_position_trim", "abs_hold_steps",
    "tau_com_vy",  # Phase 3: COM vertical velocity damping (wheel torque divergence investigation)
    # Phase 0 APCR1ND push diagnostics: JAX-computed APCR1ND gate state
    "apcr1nd_recenter_active",   # JAX k2_jax_apcr1nd_compute_gate output
    "apcr1nd_new_step_counter",  # JAX post-update step counter
    "apcr1nd_new_prev_error",    # JAX post-update prev_error
    "apcr1nd_new_converging",    # JAX post-update converging_steps
    "apcr1nd_new_recenter_held", # JAX post-update recenter_held
    "apcr1nd_safety_pass",       # JAX safety gate (com_z & roll & pitch, NO contact_valid)
    "apcr1nd_apply_wd_override", # JAX wheel damping override applied (bool)
    "apcr1nd_wd_scale",          # JAX wheel damping scale
)
K2_JAX_DIAG_SIZE: int = len(K2_JAX_DIAG_FIELDS)  # 53

_D_NOTCH_OUT, _D_NOTCH_GATE = 0, 1
_D_TAU_PITCH, _D_TAU_PITCH_RATE, _D_TAU_SAG_VEL = 2, 3, 4
_D_TAU_SUPPORT_VEL, _D_TAU_POSITION = 5, 6
_D_TAU_WHEEL_L, _D_TAU_WHEEL_R = 7, 8
_D_SCHED_KPOS, _D_SCHED_KWHEEL, _D_SCHED_KD = 9, 10, 11
_D_CALIB_KP, _D_CALIB_KD, _D_CALIB_THETA, _D_CALIB_DB = 12, 13, 14, 15
_D_PHYSICS_FF, _D_LOW_BAND = 16, 17
_D_TAU_SAG_4, _D_TAU_SAG_9 = 18, 19
_D_TAU_FINAL_START = 20
_D_CLIP_COUNT, _D_RATE_COUNT = 30, 31
# Phase 0: ABS trim diag indices (32-43)
_D_ABS_SLOW_MEAN, _D_ABS_FAST_MEAN, _D_ABS_SIGN_ERR = 32, 33, 34
_D_ABS_RAW_TARGET, _D_ABS_CLIPPED = 35, 36
_D_ABS_IS_DECAY, _D_ABS_RATE = 37, 38
_D_ABS_TRIM_DELTA, _D_ABS_NEW_TRIM = 39, 40
_D_ABS_SAFETY_PASS, _D_EXTERNAL_POS_TRIM = 41, 42
_D_ABS_HOLD_STEPS = 43
_D_TAU_COM_VY = 44  # Phase 3: tau_com_vy for wheel torque divergence investigation
# Phase 0 APCR1ND push diagnostics (indices 45-52)
_D_APCR1ND_ACTIVE = 45
_D_APCR1ND_NEW_STEP = 46
_D_APCR1ND_NEW_PREV = 47
_D_APCR1ND_NEW_CONV = 48
_D_APCR1ND_NEW_HELD = 49
_D_APCR1ND_SAFETY = 50
_D_APCR1ND_WD_APPLY = 51
_D_APCR1ND_WD_SCALE = 52


def k2_jax_diag_flat_to_dict(diag_flat):
    """Map flat JAX diagnostics to named dict."""
    d = np.asarray(diag_flat, dtype=np.float64)
    return {K2_JAX_DIAG_FIELDS[i]: float(d[i]) for i in range(K2_JAX_DIAG_SIZE)}


# --- Grid params (pre-built at module load to avoid JIT tracer issues) ---
_calibrated_grid_cache = build_calibrated_grid_params()
_physics_ff_grid_cache = build_physics_ff_grid_params()


# ===========================================================================
# State-synced teacher-forcing: pack Python K2 internal state → JAX 328-field
# ===========================================================================

def pack_state_from_python_k2(
    notch_filter,           # BiquadNotchFilter instance (or None → zeros)
    tau_prev,               # np.ndarray or jnp.ndarray, shape (10,)
    filtered_com_z,         # float
    prev_support_error,     # float
    ol_pitch_ref_smoothed,  # float
    ol_prev_support_error,  # float
    ol_support_error_rate,  # float
    abs_trim_tau,           # float
    abs_hold_steps,         # int
    abs_prev_err_sign,      # int (-1, 0, 1)
    abs_zc_count,           # int
    abs_guard_trigger,      # int
    abs_slow_error_history, # list[float], up to 300 entries (oldest first)
    abs_fast_error_history, # list[float], up to 100 entries (oldest first)
    notch_x1=None,          # Optional: pre-snapshot notch state to avoid reference mutation (float or None)
    notch_x2=None,          # Optional: pre-snapshot notch state (float or None)
    notch_y1=None,          # Optional: pre-snapshot notch state (float or None)
    notch_y2=None,          # Optional: pre-snapshot notch state (float or None)
    # APCR1ND gating state (Phase 4+ APCR1ND full port)
    apcr1nd_step_counter=0.0,
    apcr1nd_prev_error=0.0,
    apcr1nd_tuned_converging_steps=0.0,
    apcr1nd_tuned_recenter_held=0.0,
    # Phase 7: Python's runtime effective_max_position_tau (T6F/T6I-raised cap)
    effective_max_position_tau_py=0.0,
    # Phase 0: APCR1ND wheel damping override active flag (-1=Python-skipped, 0=standalone, +1=Python-applied)
    py_wd_override_active=0.0,
    # Phase 6M: ZC error history for separate ZC buffer parity
    abs_zc_error_history=None,  # list[float], up to 500 entries (oldest first)
) -> jnp.ndarray:
    """Pack Python K2 internal controller state into JAX state array (now 836).

    Captures the complete Python K2 controller state BEFORE a control step,
    packs it into the JAX flat state layout, so JAX can compute the SAME step
    from identical state and prove formula/coefficient parity.

    State timing invariant:
        State is captured BEFORE Python computes step n.
        This state reflects the result of all previous steps (0..n-1).
        Both Python and JAX compute step n from this identical starting state.

    IMPORTANT: When notch_x1/x2/y1/y2 are provided (not None), they MUST be used
    instead of reading from notch_filter._x1 etc. This is because notch_filter is
    a mutable Python object reference — Python's compute() mutates the filter
    in-place via update() between capture and pack time. The snapshot values
    (captured BEFORE Python compute) are the correct PRE-step state.

    Args:
        notch_filter: BiquadNotchFilter instance from sagittal controller
                      (self._wip_notch_pitch_rate). None if notch disabled → zeros.
        tau_prev: Previous final torque, shape (10,). From sim loop nonlocal.
        filtered_com_z: Filtered CoM Z height. From sagittal._filtered_com_z.
        prev_support_error: Previous support position error. From sim loop nonlocal.
        ol_pitch_ref_smoothed: Outer-loop pitch ref smoothed [deg]. From sim loop.
        ol_prev_support_error: Outer-loop previous support error [m]. From sim loop.
        ol_support_error_rate: Outer-loop support error rate smoothed [m/s]. From sim loop.
        abs_trim_tau: Current ABS trim torque [Nm]. From sagittal._adaptive_bias_trim_tau.
        abs_hold_steps: Remaining sign-reversal hold steps. From sagittal.
        abs_prev_err_sign: Previous error sign (-1,0,1). From sagittal.
        abs_zc_count: Zero crossing count. From sagittal.
        abs_guard_trigger: Guard trigger counter. From sagittal.
        abs_slow_error_history: Slow window error history (list, oldest first, max 300).
        abs_fast_error_history: Fast window error history (list, oldest first, max 100).
        abs_zc_error_history: ZC window error history (list, oldest first, max 500). Phase 6M.
        notch_x1: Pre-snapshot notch x1 state (float). Overrides notch_filter._x1.
        notch_x2: Pre-snapshot notch x2 state (float). Overrides notch_filter._x2.
        notch_y1: Pre-snapshot notch y1 state (float). Overrides notch_filter._y1.
        notch_y2: Pre-snapshot notch y2 state (float). Overrides notch_filter._y2.
        apcr1nd_step_counter: APCR1ND startup guard step counter.
        apcr1nd_prev_error: APCR1ND previous sagittal position error for e_dot.
        apcr1nd_tuned_converging_steps: APCR1ND consecutive converging steps counter.
        apcr1nd_tuned_recenter_held: APCR1ND recenter held/latch state (0.0 or 1.0).

    Returns:
        Flat JAX state array, shape (834,), dtype float64.
        Ready to pass to k2_jax_controller_step().
    """
    s = jnp.zeros(K2_JAX_STATE_SIZE, dtype=jnp.float64)

    # --- Notch filter state (indices 0-3) ---
    # Phase 1 fix: use pre-snapshot values when available to avoid reference-mutation
    # bug where Python's compute() mutates notch_filter._x1 before JAX reads it.
    if notch_x1 is not None:
        # Snapshot overrides — CORRECT pre-step state (captured before Python compute)
        s = s.at[_S_NOTCH_X1].set(float(notch_x1))
        s = s.at[_S_NOTCH_X2].set(float(notch_x2))
        s = s.at[_S_NOTCH_Y1].set(float(notch_y1))
        s = s.at[_S_NOTCH_Y2].set(float(notch_y2))
    elif notch_filter is not None:
        # Legacy path: read from filter reference (may be post-mutation if caller
        # didn't snapshot — used by tests and direct calls).
        s = s.at[_S_NOTCH_X1].set(float(notch_filter._x1))
        s = s.at[_S_NOTCH_X2].set(float(notch_filter._x2))
        s = s.at[_S_NOTCH_Y1].set(float(notch_filter._y1))
        s = s.at[_S_NOTCH_Y2].set(float(notch_filter._y2))

    # --- Previous torque (indices 4-13) ---
    if tau_prev is not None:
        s = s.at[_S_PREV_TAU_START:_S_PREV_TAU_START + 10].set(
            jnp.asarray(tau_prev, dtype=jnp.float64).flatten()[:10]
        )

    # --- Filtered CoM Z (index 14) ---
    s = s.at[_S_FILTERED_COM_Z].set(float(filtered_com_z))

    # --- Previous support error (index 15) ---
    s = s.at[_S_PREV_SUPPORT_ERROR].set(float(prev_support_error))

    # --- Outer loop state (indices 16-18) ---
    s = s.at[_S_OL_PITCH_REF_SMOOTHED].set(float(ol_pitch_ref_smoothed))
    s = s.at[_S_OL_PREV_SUPPORT_ERROR].set(float(ol_prev_support_error))
    s = s.at[_S_OL_SUPPORT_ERROR_RATE].set(float(ol_support_error_rate))

    # --- ABS state (indices 19-327) ---
    # Core ABS fields (indices 19-27)
    s = s.at[_ABS_TRIM_TAU].set(float(abs_trim_tau))
    s = s.at[_ABS_HOLD_STEPS].set(float(abs_hold_steps))
    s = s.at[_ABS_PREV_ERR_SIGN].set(float(abs_prev_err_sign))
    s = s.at[_ABS_ZC_COUNT].set(float(abs_zc_count))
    s = s.at[_ABS_GUARD_TRIGGER].set(float(abs_guard_trigger))

    # Ring buffer: convert Python list (oldest first) → JAX ring buffer.
    # Phase 6M fix: pack entries starting at position 0 (not write_ptr offset).
    # This ensures _abs_sliding_mean_fast, _abs_update_ring_buffer, and
    # _abs_count_zero_crossings (which assume sequential fill from 0) work correctly.
    # The write pointer is set to (n_entries % WINDOW) for the NEXT write.
    if abs_slow_error_history is not None and len(abs_slow_error_history) > 0:
        n_entries = min(len(abs_slow_error_history), _ABS_SLOW_WINDOW)
        for i, val in enumerate(abs_slow_error_history[-n_entries:]):
            s = s.at[_ABS_SLOW_BUF_START + i].set(float(val))
        s = s.at[_ABS_SLOW_COUNT].set(float(n_entries))
        s = s.at[_ABS_SLOW_PTR].set(float(n_entries % _ABS_SLOW_WINDOW))
        slow_sum = sum(abs_slow_error_history[-n_entries:])
        s = s.at[_ABS_SLOW_SUM].set(float(slow_sum))
    else:
        s = s.at[_ABS_SLOW_COUNT].set(0.0)
        s = s.at[_ABS_SLOW_PTR].set(0.0)
        s = s.at[_ABS_SLOW_SUM].set(0.0)

    # Fast sum (computed from fast window history, matching Python's fast mean computation)
    if abs_fast_error_history is not None and len(abs_fast_error_history) > 0:
        fast_sum = sum(abs_fast_error_history[-_ABS_FAST_WINDOW:])
        s = s.at[_ABS_FAST_SUM].set(float(fast_sum))

    # Phase 6M: ZC ring buffer — pack starting at position 0 (sequential fill order).
    # Must match the address scheme used by _abs_count_zero_crossings_from_zc and
    # _abs_update_zc_buffer (which assume entries at positions 0..count-1).
    if abs_zc_error_history is not None and len(abs_zc_error_history) > 0:
        n_entries = min(len(abs_zc_error_history), _ABS_ZC_WINDOW)
        for i, val in enumerate(abs_zc_error_history[-n_entries:]):
            s = s.at[_ABS_ZC_BUF_START + i].set(float(val))
        s = s.at[_ABS_ZC_BUF_COUNT].set(float(n_entries))
        s = s.at[_ABS_ZC_BUF_PTR].set(float(n_entries % _ABS_ZC_WINDOW))

    # --- APCR1ND gating state (indices 830-833, shifted by ZC buffer) ---
    s = s.at[_S_APCR1ND_STEP_COUNTER].set(float(apcr1nd_step_counter))
    s = s.at[_S_APCR1ND_PREV_ERROR].set(float(apcr1nd_prev_error))
    s = s.at[_S_APCR1ND_CONVERGING_STEPS].set(float(apcr1nd_tuned_converging_steps))
    s = s.at[_S_APCR1ND_RECENTER_HELD].set(float(apcr1nd_tuned_recenter_held))

    # --- Phase 7: Python's runtime effective_max_position_tau (index 834) ---
    s = s.at[_S_EFFECTIVE_MAX_POSITION_TAU_PY].set(float(effective_max_position_tau_py))

    # --- Phase 0: APCR1ND wheel damping override active flag (index 835) ---
    s = s.at[_S_PY_WD_OVERRIDE_ACTIVE].set(float(py_wd_override_active))

    return s


# ===========================================================================
# Stage 4: Full K2 JAX controller step (JIT-compatible)
# ===========================================================================

def k2_jax_controller_step(
    state_flat: jnp.ndarray,   # (332,) — K2_JAX_STATE_SIZE
    input_flat: jnp.ndarray,   # (41,) — K2_JAX_INPUT_SIZE
    params_flat: jnp.ndarray,  # (41,) — K2_JAX_PARAMS_SIZE_STAGE2
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Full K2 balance-core controller step — pure JAX function.

    Composes all Stage 2+3 components: notch → height schedule → sagittal →
    shape posture → lateral roll → yaw → mode-div → support FF → composer.

    Returns:
        tau: (10,) actuator torque vector
        next_state_flat: (332,) updated state
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
    # APCR1ND gating state
    apcr1nd_step_counter = state_flat[_S_APCR1ND_STEP_COUNTER]
    apcr1nd_prev_error = state_flat[_S_APCR1ND_PREV_ERROR]
    apcr1nd_converging_steps = state_flat[_S_APCR1ND_CONVERGING_STEPS]
    apcr1nd_recenter_held = state_flat[_S_APCR1ND_RECENTER_HELD]
    # Phase 7: Python's runtime effective_max_position_tau (0.0 = use JAX-computed)
    effective_max_pos_tau_py = state_flat[_S_EFFECTIVE_MAX_POSITION_TAU_PY]

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
    # D2/D3 bugfix: mode_div params
    _mode_div_soft_gain = params_flat[_IDX_MODE_DIV_SOFT_GAIN]
    _mode_div_ref_source = params_flat[_IDX_MODE_DIV_REF_SOURCE]
    # Phase 4 parity fix: sagittal velocity damping params
    _k_velocity = params_flat[_IDX_K_VELOCITY]
    _velocity_damping_scale = params_flat[_IDX_VELOCITY_DAMPING_SCALE]
    # Phase 4+ APCR1ND gating params
    _apcr1nd_startup_guard = params_flat[_IDX_APCR1ND_STARTUP_GUARD]
    _apcr1nd_safe_com_z = params_flat[_IDX_APCR1ND_SAFE_COM_Z]
    _apcr1nd_safe_roll = params_flat[_IDX_APCR1ND_SAFE_ROLL]
    _apcr1nd_safe_pitch = params_flat[_IDX_APCR1ND_SAFE_PITCH]
    _apcr1nd_direct_enter = params_flat[_IDX_APCR1ND_DIRECT_ENTER]
    _apcr1nd_release_inner = params_flat[_IDX_APCR1ND_RELEASE_INNER]
    _apcr1nd_hold_outside = params_flat[_IDX_APCR1ND_HOLD_OUTSIDE]
    _apcr1nd_converging_release = params_flat[_IDX_APCR1ND_CONVERGING_RELEASE]
    # Phase 3 standalone: check if standalone mode is active.
    # Params array is always K2_JAX_PARAMS_SIZE_STAGE2_EXT_STANDALONE (54),
    # input is always K2_JAX_INPUT_SIZE (45). Safe to access all indices.
    _standalone_mode = params_flat[_IDX_STANDALONE_MODE] > 0.5
    _pitch_x_eq = params_flat[_IDX_PITCH_X_EQ_RAD]
    _support_center_eq_x = params_flat[_IDX_SUPPORT_CENTER_EQ_X]
    _support_center_eq_y = params_flat[_IDX_SUPPORT_CENTER_EQ_Y]
    _sag_axis_x = params_flat[_IDX_SAGITTAL_AXIS_X]
    _sag_axis_y = params_flat[_IDX_SAGITTAL_AXIS_Y]
    _com_vx_standalone = input_flat[_I_COM_VX]
    _support_center_x = input_flat[_I_SUPPORT_CENTER_X]
    _support_center_y = input_flat[_I_SUPPORT_CENTER_Y]

    # === Step 1: Height scheduling (computed FIRST — used by notch gate, gains,
    # outer loop, support FF, ABS trim, and all height-dependent components).
    # Matches Python K2: commanded_height_ref_m → schedule_height_ref,
    # fallback filtered_com_z = 0.9*filtered_com_z + 0.1*com_z when None.
    schedule_h = jnp.where(height_ref > 0.0, height_ref,
                 0.9 * filtered_com_z + 0.1 * com_z)
    new_filtered_com_z = schedule_h

    # === Step 2: Notch filter ===
    notch_out = notch_b0 * pitch_rate + notch_b1 * notch_x1 + notch_b2 * notch_x2 - notch_a1 * notch_y1 - notch_a2 * notch_y2
    new_notch_x1 = pitch_rate
    new_notch_x2 = notch_x1
    new_notch_y1 = notch_out
    new_notch_y2 = notch_y1

    # Height gate for notch blend — uses schedule_h (same as Python's schedule_height_ref),
    # not raw height_ref, so gate tracks actual/filtered height when no command is provided.
    notch_gate = smoothstep_gate_jax(schedule_h, 0.42, 0.48)
    pitch_rate_eff = (1.0 - notch_gate) * pitch_rate + notch_gate * notch_out

    # === Phase 3 standalone: compute derived sagittal quantities from raw state ===
    # When _standalone_mode is True, JAX computes sag_pos_err, sag_vel, and support_vel
    # from raw state inputs (support_center + COM velocity), matching Python's
    # compute_support_center_xy + project_sagittal_displacement + project_sagittal_velocity.
    # When False, uses Python-computed values from input_flat (backward-compatible).
    _raw_sag_pos_err = (
        (_support_center_x - _support_center_eq_x) * _sag_axis_x
        + (_support_center_y - _support_center_eq_y) * _sag_axis_y
    )
    _raw_sag_vel = _com_vx_standalone * _sag_axis_x + com_vy * _sag_axis_y
    _raw_support_vel = jnp.where(
        jnp.abs(prev_support_error) > 1e-15,
        (_raw_sag_pos_err - prev_support_error) / control_dt,
        0.0,
    )
    # Override with standalone-derived values when mode is active
    sag_pos_err = jnp.where(_standalone_mode, _raw_sag_pos_err, sag_pos_err)
    sag_vel = jnp.where(_standalone_mode, _raw_sag_vel, sag_vel)
    support_vel = jnp.where(_standalone_mode, _raw_support_vel, support_vel)
    support_pos_err = jnp.where(_standalone_mode, _raw_sag_pos_err, support_pos_err)

    # K2 profile (k2_notch_low_q_v1) continuous scheduling flags:
    #   continuous_k_position=False        → k_position = 40.0 (constructor default)
    #   continuous_k_wheel_velocity=False  → k_wheel_velocity = 0.5
    #   continuous_kd_pitch=False          → kd_pitch = 10.0
    #   continuous_k_velocity=False        → k_velocity = 15.0 (constructor, from vd_k_velocity)
    #   continuous_max_position_tau=True   → SCHEDULED: 4.0→6.0 at z=0.393→0.300
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import K2_NOTCH_LOW_Q_V1 as _k2_sch
    kpos = 40.0        # K2: vd_k_position=40.0 (constructor arg, not scheduled)
    kwheel = 0.5        # K2: base k_wheel_velocity
    kd_pitch = 10.0     # K2: base kd_pitch (not scheduled)
    # Continuous max_position_tau scheduling
    max_pos_tau = k2_jax_scheduled_k_position(
        schedule_h,
        k_nominal=float(_k2_sch.max_position_tau_nominal),   # 4.0
        k_low_max=float(_k2_sch.max_position_tau_low_max),    # 6.0
        z_low=float(_k2_sch.k_position_z_low),                # 0.300
        z_high=float(_k2_sch.k_position_z_high),              # 0.393
    )

    # === Step 3: Calibrated outer loop + physics FF ===
    # Grids pre-built at module load — safe to reference in JIT as constants
    cal_grid = _calibrated_grid_cache
    ff_grid = _physics_ff_grid_cache

    cal_kp = k2_jax_grid_interpolate(schedule_h, cal_grid["grid_heights"], cal_grid["kp_grid"])
    cal_kd = k2_jax_grid_interpolate(schedule_h, cal_grid["grid_heights"], cal_grid["kd_grid"])
    cal_theta_max = k2_jax_grid_interpolate(schedule_h, cal_grid["grid_heights"], cal_grid["theta_max_grid"])
    cal_deadband = k2_jax_grid_interpolate(schedule_h, cal_grid["grid_heights"], cal_grid["deadband_grid"])
    cal_rate_limit = k2_jax_grid_interpolate(schedule_h, cal_grid["grid_heights"], cal_grid["rate_limit_grid"])
    cal_lowpass_alpha = k2_jax_grid_interpolate(schedule_h, cal_grid["grid_heights"], cal_grid["lowpass_grid"])

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

    # D4 bugfix: Outer loop safety gate (matches Python line 6050-6093).
    # K2 profile thresholds: pitch≤12°, roll≤5°, abs_error≤0.25m, contact_required=True.
    # Contact validity is assumed True from JAX perspective (both wheels on ground
    # in all K2 two-wheel-contact scenarios). If future scenarios need wheel lift-off
    # detection, add a contact_valid input field.
    _ol_pitch_deg = jnp.abs(pitch_x) * 180.0 / jnp.pi
    _ol_roll_deg = jnp.abs(roll_y) * 180.0 / jnp.pi
    _ol_contact_ok = True  # K2: always both-wheels-on-ground
    _ol_pitch_ok = _ol_pitch_deg <= 12.0
    _ol_roll_ok = _ol_roll_deg <= 5.0
    _ol_error_ok = jnp.abs(support_pos_err) <= 0.25
    _ol_safety_pass = _ol_contact_ok & _ol_pitch_ok & _ol_roll_ok & _ol_error_ok

    ol_dynamic_raw = k2_jax_compute_outer_loop_pitch_ref(
        support_pos_err, new_ol_support_error_rate, 0.0,
        cal_kp, cal_kd, 0.0, cal_deadband, cal_theta_max)
    # When safety gate fails, zero the target (Python line 6093: target_dynamic_deg = 0.0).
    # Rate-limit + lowpass still applied → smooth decay toward zero.
    ol_dynamic = jnp.where(_ol_safety_pass, ol_dynamic_raw, 0.0)
    ol_target = _jax_apply_rate_limit(
        ol_pitch_ref_smoothed, ol_dynamic, cal_rate_limit)
    new_ol_pitch_ref = _jax_apply_lowpass(
        ol_pitch_ref_smoothed, ol_target, cal_lowpass_alpha)

    # Total pitch ref offset (physics FF + low-band static only for comparison)
    physics_pitch_eq = k2_jax_grid_interpolate(
        schedule_h, ff_grid["grid_heights"], ff_grid["pitch_eq_grid"])
    total_pitch_ref_offset_deg = new_ol_pitch_ref + lb_offset + physics_pitch_eq

    # Phase 3 standalone: compute effective pitch from raw body pitch + total offset.
    # When standalone_mode: pitch_x is RAW body pitch (NOT pre-adjusted by Python).
    #   Compute effective_pitch_x = raw_pitch_x - pitch_x_eq - offset_rad
    #   This matches Python's: pitch_x_error = body_pitch - pitch_x_eq - rad(outer_loop_total)
    # When not standalone_mode: pitch_x is already pre-adjusted by Python sim loop.
    #   Use as-is (backward compatible with existing both-synced/py-fallback paths).
    effective_pitch_x = jnp.where(
        _standalone_mode,
        pitch_x - _pitch_x_eq - jnp.deg2rad(total_pitch_ref_offset_deg),
        pitch_x,
    )

    # === Step 4a: Adaptive bias trim (active K2 strategy) — Stage 6L ring buffer ===
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import K2_NOTCH_LOW_Q_V1 as _sch

    # Safety gate: matching Python exactly (svdbc.py:5657-5681)
    # adaptive_bias_only_when_upright=True and only_when_contact_stable=True in K2 profile,
    # so upright_ok = pitch_ok AND roll_ok, and contact_ok requires contact_valid (sim-tracked,
    # always True from JAX perspective since JAX operates on Python-provided physics state).
    _abs_pitch_deg = jnp.abs(effective_pitch_x) * 180.0 / jnp.pi
    _abs_roll_deg = jnp.abs(roll_y) * 180.0 / jnp.pi
    _pitch_ok = (_abs_pitch_deg <= float(_sch.adaptive_bias_disable_if_pitch_gt_deg))
    _roll_ok = (_abs_roll_deg <= float(_sch.adaptive_bias_disable_if_roll_gt_deg))
    # Phase 6M: Use actual contact_valid from input (matches Python svdbc.py:5660)
    _contact_valid_val = input_flat[_I_CONTACT_VALID]
    _contact_ok = jnp.where(
        float(_sch.adaptive_bias_only_when_contact_stable) > 0.5,
        _contact_valid_val > 0.5,
        True,
    )
    _upright_ok = _pitch_ok & _roll_ok  # adaptive_bias_only_when_upright=True → both required
    _abs_error_ok = jnp.abs(sag_pos_err) <= float(_sch.adaptive_bias_disable_if_abs_error_gt_m)
    # Phase 0 parity fix: Python source-of-truth svdbc.py:5670-5674 reads
    # hip_yaw_abs_max_tracking with try/except NameError → fallback 0.0.
    # hip_yaw_abs_max_tracking is only a telemetry dict key, never a local
    # variable in compute() scope, so the NameError always fires → hy_val=0.0
    # → hy_ok ALWAYS True. JAX must match this effective behavior for strict
    # both-synced parity. The logically correct hip-yaw gate is deferred to a
    # separate Python controller fix (non-parity task).
    _hip_yaw_ok = True  # matches Python effective behavior (NameError fallback)
    _safety = _contact_ok & _upright_ok & _abs_error_ok & _hip_yaw_ok

    # Compute trim using sliding window ring buffer + ZC buffer (matches Python exactly)
    (_new_trim, _new_hold, _new_prev_sign, _new_zc, _trim_to_apply,
     _slow_mean, _fast_mean, state_flat, _abs_diag) = _k2_jax_adaptive_bias_trim(
        sag_pos_err, state_flat, com_z, effective_pitch_x, _safety, _contact_valid_val,
    )

    # Compute effective error for telemetry (matching Python)
    exit_th = float(_sch.adaptive_bias_exit_threshold_m)
    sign_err = jnp.sign(_slow_mean)
    _eff_error = jnp.where(
        (jnp.abs(_slow_mean) <= exit_th) | (_new_hold > 0.0),
        0.0,
        _slow_mean - sign_err * exit_th,
    )

    # === Step 4b: Sagittal torque assembly ===
    # === Step 4b: APCR1ND gating computation (Phase 4+ full port) ===
    _apcr1nd_active, _new_apcr1nd_step, _new_apcr1nd_prev, \
        _new_apcr1nd_conv, _new_apcr1nd_held = k2_jax_apcr1nd_compute_gate(
        sagittal_position_error_m=sag_pos_err,
        prev_error=apcr1nd_prev_error,
        step_counter=apcr1nd_step_counter,
        converging_steps=apcr1nd_converging_steps,
        recenter_held=apcr1nd_recenter_held,
        pitch_x_rad=effective_pitch_x,
        roll_y_rad=roll_y,
        com_z_m=com_z,
        startup_guard_steps=_apcr1nd_startup_guard,
        safe_min_com_z=_apcr1nd_safe_com_z,
        safe_roll_rad=_apcr1nd_safe_roll,
        safe_pitch_rad=_apcr1nd_safe_pitch,
        soft_enter_m=_K2_APCR_SOFT_ENTER_M,
        direct_enter_m=_apcr1nd_direct_enter,
        desired_band_m=_K2_APCR_DESIRED_BAND_M,
        release_inner_m=_apcr1nd_release_inner,
        hold_outside_band=_apcr1nd_hold_outside,
        converging_release_steps=_apcr1nd_converging_release,
        contact_valid=_contact_valid_val > 0.5,
    )

    # Phase 4 push fix: APCR1ND position cap boost (matches Python lines 6702-6726)
    # K2 profile has position_cap_recenter_boost_enabled=True → during push, raises
    # max_position_tau from 4.0 Nm up to 7.0 Nm based on sagittal position error band.
    _boost_enabled = float(_sch.position_cap_recenter_boost_enabled)
    _abs_pos_err = jnp.abs(sag_pos_err)
    # Phase 0 APCR1ND parity fix: include contact_valid in cap safety (matches Python svdbc.py:6582-6584)
    _cap_safety = (_contact_valid_val > 0.5)
    _cap_safety = _cap_safety & (com_z >= _apcr1nd_safe_com_z)
    _cap_safety = _cap_safety & (jnp.abs(roll_y) <= _apcr1nd_safe_roll)
    _cap_safety = _cap_safety & (jnp.abs(effective_pitch_x) <= _apcr1nd_safe_pitch)
    _boosted_cap = k2_jax_compute_boosted_position_cap(
        _abs_pos_err, _cap_safety, _boost_enabled,
        float(_sch.apcr1nd_tuned_enabled),
        float(_sch.apcr1nd_soft_enter_m),
        float(_sch.apcr1nd_hard_band_m),
        float(_sch.apcr1nd_emergency_band_m),
        float(_sch.apcr1nd_desired_band_m),
        float(_sch.apcr1nd_position_cap_normal_nm),
        float(_sch.apcr1nd_position_cap_soft_nm),
        float(_sch.apcr1nd_position_cap_desired_nm),
        float(_sch.apcr1nd_position_cap_hard_nm),
        float(_sch.apcr1nd_position_cap_emergency_nm),
    )
    # Phase 4 fix: Python applies TWO clips to tau_position:
    #   1. First clip to height-scheduled effective_max_position_tau (= max_pos_tau)
    #      AFTER ABS trim addition (svdbc.py:5770-5774)
    #   2. Second clip to APCR1ND boosted_cap (svdbc.py:6758)
    # JAX previously used max(max_pos_tau, boosted_cap) for a single clip,
    # which was too loose when max_pos_tau < boosted_cap.
    # Fix: pass unboosted max_pos_tau to sagittal assembly, then apply
    # APCR1ND boost clip separately.
    effective_max_pos_tau = jnp.maximum(max_pos_tau, _boosted_cap)

    tau_sag, sag_diag = k2_jax_sagittal_torque_assembly(
        pitch_x_rad=effective_pitch_x, pitch_rate_rad_s=pitch_rate_eff,
        sagittal_velocity_m_s=sag_vel, sagittal_position_error_m=sag_pos_err,
        wheel_vel_left_rad_s=wheel_vel_l, wheel_vel_right_rad_s=wheel_vel_r,
        support_velocity_m_s=support_vel,
        kp_pitch=50.0, effective_pitch_scale=1.0, effective_pitch_tau_cap=0.0,
        effective_kd_pitch=kd_pitch,
        effective_k_velocity=_k_velocity, effective_velocity_damping_scale=_velocity_damping_scale,
        effective_support_velocity_gain=0.0, effective_support_velocity_scale=1.0,
        effective_k_wheel_velocity=kwheel,
        effective_k_position=kpos,
        # Phase 7 fix: use Python's runtime effective_max_position_tau when available
        # (captured from sagittal controller in both-synced mode; includes T6F/T6I raises).
        # Falls back to height-scheduled max_pos_tau in standalone JAX mode.
        effective_max_position_tau=jnp.where(
            effective_max_pos_tau_py > 0.0, effective_max_pos_tau_py, max_pos_tau),
        kp_cp=0.0, kd_com_vy=5.0,
        wheel_torque_sign=1.0,
        external_position_trim=_trim_to_apply,
    )

    # Phase 4 fix: Apply APCR1ND position cap boost as second clip (matching Python svdbc.py:6758)
    # This re-clips tau_position at the boosted cap if higher than max_pos_tau.
    # The sagittal diag's tau_position already has the first clip (to max_pos_tau) applied.
    # GATE: Only apply when APCR1ND is active (matches Python's `if apcr1n_recenter_priority_active`)
    _pos_clip_boosted = jnp.where(
        _apcr1nd_active,
        jnp.clip(sag_diag["tau_position"], -_boosted_cap, _boosted_cap),
        sag_diag["tau_position"],
    )
    # Update tau_position in diag for telemetry accuracy
    sag_diag["tau_position"] = _pos_clip_boosted
    # Recompute tau_common and wheel torques with re-clipped position
    # wheel_torque_sign is always 1.0 for K2 profile
    _tau_common_boosted = 1.0 * (
        sag_diag["tau_pitch"] + sag_diag["tau_pitch_rate"]
        + sag_diag["tau_sagittal_velocity"] + sag_diag["tau_support_velocity"]
        + _pos_clip_boosted + sag_diag["tau_cp"] + sag_diag["tau_com_vy"]
    )
    tau_sag = tau_sag.at[4].set(_tau_common_boosted + sag_diag["tau_wheel_vel_left"])
    tau_sag = tau_sag.at[9].set(_tau_common_boosted + sag_diag["tau_wheel_vel_right"])

    # === Step 4c: APCR1ND wheel damping override (K2 parity fix) ===
    _old_tau_wvl = sag_diag["tau_wheel_vel_left"]
    _old_tau_wvr = sag_diag["tau_wheel_vel_right"]
    _new_tau_wvl, _new_tau_wvr = k2_jax_apcr1nd_wheel_damping_override(
        _old_tau_wvl, _old_tau_wvr,
        wheel_vel_l, wheel_vel_r,
        sag_pos_err,
        recenter_active=_apcr1nd_active,
    )
    # Adjust tau_sag: replace old wheel damping with new
    tau_sag = tau_sag.at[4].add(_new_tau_wvl - _old_tau_wvl)
    tau_sag = tau_sag.at[9].add(_new_tau_wvr - _old_tau_wvr)
    # Update diag for downstream diagnostics
    sag_diag["tau_wheel_vel_left"] = _new_tau_wvl
    sag_diag["tau_wheel_vel_right"] = _new_tau_wvr

    # === Step 5: Shape posture ===
    joint_pos_full = jnp.array([q_hr_l, q_hy_l, q_hp_l, q_kn_l, 0.0,
                                  q_hr_r, q_hy_r, q_hp_r, q_kn_r, 0.0], dtype=jnp.float64)
    joint_vel_full = jnp.array([qd_hr_l, qd_hy_l, qd_hp_l, qd_kn_l, 0.0,
                                  qd_hr_r, qd_hy_r, qd_hp_r, qd_kn_r, 0.0], dtype=jnp.float64)
    q_ref_full = jnp.array([qref_hr_l, qref_hy_l, qref_hp_l, qref_kn_l, 0.0,
                              qref_hr_r, qref_hy_r, qref_hp_r, qref_kn_r, 0.0], dtype=jnp.float64)

    tau_posture, _ = k2_jax_shape_posture_compute(q_ref_full, joint_pos_full, joint_vel_full)

    # === Step 6: Lateral roll ===
    # enable_stance_regularization=True matches Python behavior:
    # simulate_hierarchical_controller.py:6301-6307 always passes hip_roll_pos/vel/ref,
    # activating the stance term in lateral_roll_balance_controller.py:compute().
    tau_lateral, _ = k2_jax_lateral_roll_compute(roll_y, roll_rate,
        hip_roll_pos_left=q_hr_l, hip_roll_pos_right=q_hr_r,
        hip_roll_vel_left=qd_hr_l, hip_roll_vel_right=qd_hr_r,
        hip_roll_ref_left=qref_hr_l, hip_roll_ref_right=qref_hr_r,
        enable_stance_regularization=True)

    # === Step 7: Yaw ===
    tau_yaw = k2_jax_yaw_compute(yaw_err, yaw_rate)

    # === Step 8: Mode-div ===
    # D3 bugfix: validate ref_source — "zero_only_for_debug" is not supported by
    # JAX (requires different q_ref computation inside the controller).
    # The K2 runtime uses "target" which means hip_yaw_div_error is precomputed
    # by Python and packed into the input — JAX uses it directly.
    _ref_src_int = jnp.asarray(_mode_div_ref_source, dtype=jnp.int32)
    _disabled_ref = _ref_src_int >= 2  # 2 = disabled (matches Python --enable-mode-hip-yaw-divergence off)
    _unsupported_ref = _ref_src_int == 1  # 1 = zero_only_for_debug
    tau_mode_div = jnp.where(
        _disabled_ref | _unsupported_ref,
        jnp.zeros(10, dtype=jnp.float64),
        k2_jax_mode_div_compute(
            hy_div_err, hy_div_rate, com_z,
            soft_gain=_mode_div_soft_gain),
    )

    # === Step 9: Support feedforward ===
    tau_support_ff = k2_jax_support_feedforward_compute(
        support_pos_err, schedule_h)

    # === Step 10: Sum and compose (active K2 mechanism) ===
    # Yaw and mode_div are added to posture BEFORE composer,
    # matching the Python simulation order where all torque sources
    # pass through clip and rate-limit (simulate_hierarchical_controller.py:6332-6476).
    # This ensures hip-yaw [1,6] torque is clipped to torque_limit and rate-limited,
    # and prev_tau[1,6] stores the composer output for correct next-step rate-limiting.
    # Stage 7B: k2_jax_empirical_support_ff() included in tau_sum so composer
    # clips knee torque to max_position_tau, matching Python behavior.
    # NOTE: tau_support_ff (height-gated hip-yaw support) is EXCLUDED —
    # Python balance-core has no equivalent; inclusion causes divergence
    # during descending height transitions and push recovery.
    tau_posture_with_yaw = tau_posture.at[1].add(tau_yaw[1])
    tau_posture_with_yaw = tau_posture_with_yaw.at[6].add(tau_yaw[6])
    tau_posture_with_yaw = tau_posture_with_yaw.at[1].add(tau_mode_div[1])
    tau_posture_with_yaw = tau_posture_with_yaw.at[6].add(tau_mode_div[6])

    tau_sum = tau_sag + tau_posture_with_yaw + tau_lateral + k2_jax_empirical_support_ff()

    tau_final, tau_clipped, sat_mask, rate_mask = k2_jax_torque_composer_step(
        tau_sum, prev_tau, params_flat)

    # No post-composer additions needed — yaw and mode_div already in tau_sum

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

    # Adaptive bias state is already updated inside _k2_jax_adaptive_bias_trim
    # (state_flat was returned by the call). Copy ABS core fields from state_flat to new_state.
    new_state = new_state.at[_ABS_SLOW_SUM].set(state_flat[_ABS_SLOW_SUM])
    new_state = new_state.at[_ABS_FAST_SUM].set(state_flat[_ABS_FAST_SUM])
    new_state = new_state.at[_ABS_TRIM_TAU].set(state_flat[_ABS_TRIM_TAU])
    new_state = new_state.at[_ABS_HOLD_STEPS].set(state_flat[_ABS_HOLD_STEPS])
    new_state = new_state.at[_ABS_PREV_ERR_SIGN].set(state_flat[_ABS_PREV_ERR_SIGN])
    new_state = new_state.at[_ABS_ZC_COUNT].set(state_flat[_ABS_ZC_COUNT])
    new_state = new_state.at[_ABS_SLOW_COUNT].set(state_flat[_ABS_SLOW_COUNT])
    new_state = new_state.at[_ABS_SLOW_PTR].set(state_flat[_ABS_SLOW_PTR])
    new_state = new_state.at[_ABS_GUARD_TRIGGER].set(state_flat[_ABS_GUARD_TRIGGER])
    # Copy ring buffer
    new_state = new_state.at[_ABS_SLOW_BUF_START:_ABS_SLOW_BUF_END].set(
        state_flat[_ABS_SLOW_BUF_START:_ABS_SLOW_BUF_END])
    # Phase 6M: Copy ZC buffer state
    new_state = new_state.at[_ABS_ZC_BUF_COUNT].set(state_flat[_ABS_ZC_BUF_COUNT])
    new_state = new_state.at[_ABS_ZC_BUF_PTR].set(state_flat[_ABS_ZC_BUF_PTR])
    new_state = new_state.at[_ABS_ZC_BUF_START:_ABS_ZC_BUF_END].set(
        state_flat[_ABS_ZC_BUF_START:_ABS_ZC_BUF_END])

    # APCR1ND gating state (shifted by ZC buffer)
    new_state = new_state.at[_S_APCR1ND_STEP_COUNTER].set(_new_apcr1nd_step)
    new_state = new_state.at[_S_APCR1ND_PREV_ERROR].set(_new_apcr1nd_prev)
    new_state = new_state.at[_S_APCR1ND_CONVERGING_STEPS].set(_new_apcr1nd_conv)
    new_state = new_state.at[_S_APCR1ND_RECENTER_HELD].set(_new_apcr1nd_held)

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
    # Phase 2 push trace: tau_sag at wheel indices (post-APCR1ND override)
    diag = diag.at[_D_TAU_SAG_4].set(tau_sag[4])
    diag = diag.at[_D_TAU_SAG_9].set(tau_sag[9])
    diag = diag.at[_D_TAU_FINAL_START:_D_TAU_FINAL_START + 10].set(tau_final)
    diag = diag.at[_D_CLIP_COUNT].set(jnp.sum(sat_mask).astype(jnp.float64))
    diag = diag.at[_D_RATE_COUNT].set(jnp.sum(rate_mask).astype(jnp.float64))
    # Phase 0: Write ABS trim intermediates to diag
    diag = diag.at[_D_ABS_SLOW_MEAN].set(_abs_diag[0])
    diag = diag.at[_D_ABS_FAST_MEAN].set(_abs_diag[1])
    diag = diag.at[_D_ABS_SIGN_ERR].set(_abs_diag[2])
    diag = diag.at[_D_ABS_RAW_TARGET].set(_abs_diag[3])
    diag = diag.at[_D_ABS_CLIPPED].set(_abs_diag[4])
    diag = diag.at[_D_ABS_IS_DECAY].set(_abs_diag[5])
    diag = diag.at[_D_ABS_RATE].set(_abs_diag[6])
    diag = diag.at[_D_ABS_TRIM_DELTA].set(_abs_diag[7])
    diag = diag.at[_D_ABS_NEW_TRIM].set(_abs_diag[8])
    diag = diag.at[_D_ABS_SAFETY_PASS].set(_abs_diag[9])
    diag = diag.at[_D_EXTERNAL_POS_TRIM].set(_abs_diag[10])
    diag = diag.at[_D_ABS_HOLD_STEPS].set(_abs_diag[11])
    # Phase 3: tau_com_vy for wheel torque divergence investigation
    diag = diag.at[_D_TAU_COM_VY].set(sag_diag["tau_com_vy"])
    # Phase 0 APCR1ND push diagnostics: write JAX-computed APCR1ND state
    _apcr1nd_diag_safety = (com_z >= _apcr1nd_safe_com_z) & (jnp.abs(roll_y) <= _apcr1nd_safe_roll) & (jnp.abs(effective_pitch_x) <= _apcr1nd_safe_pitch)
    diag = diag.at[_D_APCR1ND_ACTIVE].set(jnp.where(_apcr1nd_active, 1.0, 0.0))
    diag = diag.at[_D_APCR1ND_NEW_STEP].set(_new_apcr1nd_step)
    diag = diag.at[_D_APCR1ND_NEW_PREV].set(_new_apcr1nd_prev)
    diag = diag.at[_D_APCR1ND_NEW_CONV].set(_new_apcr1nd_conv)
    diag = diag.at[_D_APCR1ND_NEW_HELD].set(_new_apcr1nd_held)
    diag = diag.at[_D_APCR1ND_SAFETY].set(jnp.where(_apcr1nd_diag_safety, 1.0, 0.0))
    # Wheel damping override applied: check if tau_wheel_vel was changed
    _apcr1nd_wd_applied = jnp.abs(sag_diag["tau_wheel_vel_left"] - _old_tau_wvl) > 1e-12
    _apcr1nd_wd_scale = jnp.where(
        jnp.abs(_old_tau_wvl) > 1e-12,
        sag_diag["tau_wheel_vel_left"] / _old_tau_wvl,
        1.0,
    )
    diag = diag.at[_D_APCR1ND_WD_APPLY].set(jnp.where(_apcr1nd_wd_applied, 1.0, 0.0))
    diag = diag.at[_D_APCR1ND_WD_SCALE].set(_apcr1nd_wd_scale)

    return tau_final, new_state, diag


# ===========================================================================
# Stage 4H: Adaptive bias trim (JAX port) — Stage 6L: sliding window fix
# ===========================================================================
# Replaces EMA-based mean with true sliding window ring buffer to match Python.
#
# New ABS state layout (ring buffer for slow window):
#   [0..18]  unchanged (notch, prev_tau, filtered_com_z, support, outer_loop)
#   [19]     abs_slow_sum   — running sum of slow window entries
#   [20]     abs_fast_sum   — running sum of fast window entries
#   [21]     abs_trim_tau
#   [22]     abs_hold_steps
#   [23]     abs_prev_err_sign
#   [24]     abs_zc_count
#   [25]     abs_slow_count — number of valid entries in slow ring buffer
#   [26]     abs_slow_ptr   — write pointer (0..SW-1) for slow ring buffer
#   [27]     abs_guard_trigger_count
#   [28..327] abs_slow_buffer — ring buffer, SW=300 entries

# Sliding window sizes from K2 profile (defined at module level)
_ABS_SLOW_WINDOW = _ABS_SLOW_WINDOW_MODULE    # 300
_ABS_FAST_WINDOW = _ABS_FAST_WINDOW_MODULE     # 100
_ABS_ZC_WINDOW = _ABS_ZC_WINDOW_MODULE          # 500

_ABS_SLOW_BUF_START = 28  # after 19 base + 9 ABS core fields
_ABS_SLOW_BUF_END = _ABS_SLOW_BUF_START + _ABS_SLOW_WINDOW  # 328
_ABS_SLOW_COUNT = 25
_ABS_SLOW_PTR = 26
_ABS_GUARD_TRIGGER = 27
# Remap indices for the original ABS fields
_ABS_SLOW_SUM = 19   # was _ABS_SLOW_EMA
_ABS_FAST_SUM = 20   # was _ABS_FAST_EMA
_ABS_TRIM_TAU = 21
_ABS_HOLD_STEPS = 22
_ABS_PREV_ERR_SIGN = 23
_ABS_ZC_COUNT = 24


def _abs_sliding_mean_slow(state_flat):
    """Compute sliding window mean from ring buffer (slow window, 300 entries)."""
    count = state_flat[_ABS_SLOW_COUNT]
    total = state_flat[_ABS_SLOW_SUM]
    return jnp.where(count > 0, total / count, 0.0)


def _abs_sliding_mean_fast(state_flat):
    """Compute sliding window mean from most recent FAST_WINDOW entries of slow buffer.

    JIT-compatible: uses jnp.where and arithmetic instead of Python loops.
    """
    count = state_flat[_ABS_SLOW_COUNT]
    ptr = state_flat[_ABS_SLOW_PTR].astype(jnp.int32)
    n_fast = jnp.minimum(count, _ABS_FAST_WINDOW).astype(jnp.float64)
    buf = state_flat[_ABS_SLOW_BUF_START:_ABS_SLOW_BUF_END]

    # Build mask of which entries are in the most recent n_fast (circular)
    indices = (jnp.arange(_ABS_SLOW_WINDOW) - ptr + _ABS_SLOW_WINDOW) % _ABS_SLOW_WINDOW
    # "most recent" = indices [0, n_fast) in circular order from ptr-1 going backward
    # Position k in the buffer is "recent" if (ptr - 1 - k) mod SW < n_fast (going backward)
    backward_pos = (ptr - 1 - jnp.arange(_ABS_SLOW_WINDOW) + _ABS_SLOW_WINDOW) % _ABS_SLOW_WINDOW
    mask = jnp.where(count > 0, backward_pos < n_fast, False)

    fast_sum = jnp.sum(jnp.where(mask, buf, 0.0))
    mean_fast = jnp.where(n_fast > 0, fast_sum / n_fast, 0.0)
    return mean_fast


def _abs_count_zero_crossings(state_flat):
    """Count zero crossings in the slow ring buffer (JIT-compatible).

    Walks backward from ptr-1 through the buffer, counting sign changes
    using JAX array operations only (no Python for-loops).
    """
    count = state_flat[_ABS_SLOW_COUNT]
    ptr = state_flat[_ABS_SLOW_PTR].astype(jnp.int32)
    buf = state_flat[_ABS_SLOW_BUF_START:_ABS_SLOW_BUF_END]

    # Build array of values in reverse chronological order (most recent first)
    i_range = jnp.arange(_ABS_SLOW_WINDOW)
    reverse_indices = (ptr - 1 - i_range + _ABS_SLOW_WINDOW) % _ABS_SLOW_WINDOW
    vals = buf[reverse_indices]

    # Shift by 1 to get previous value in the reverse-chronological sequence
    vals_prev = jnp.roll(vals, shift=1)
    # First entry (most recent) has no predecessor — set prev to same as current
    vals_prev = vals_prev.at[0].set(vals[0])

    # Valid: entry i is valid if i < count AND entry i-1 is valid
    valid_curr = i_range < count
    valid_prev = jnp.roll(valid_curr, shift=1)
    # Both current and previous must be within count
    both_valid = valid_curr & valid_prev

    # Sign change when signs differ and both are valid
    sign_change = (vals < 0) != (vals_prev < 0)
    sign_change_valid = sign_change & both_valid

    zc = jnp.sum(jnp.where(sign_change_valid, 1, 0))
    return zc


def _abs_update_zc_buffer(state_flat, error_signed):
    """Push error into ZC ring buffer (500 entries, separate from slow/fast).

    Phase 6M: Matches Python's separate _adaptive_bias_zero_crossing_history.
    Returns updated state_flat.
    """
    ptr = state_flat[_ABS_ZC_BUF_PTR].astype(jnp.int32)
    count = state_flat[_ABS_ZC_BUF_COUNT]

    new_state = state_flat.at[_ABS_ZC_BUF_START + ptr].set(error_signed)
    new_state = new_state.at[_ABS_ZC_BUF_PTR].set((ptr + 1) % _ABS_ZC_WINDOW)
    new_count = jnp.where(count >= _ABS_ZC_WINDOW, count, count + 1.0)
    new_state = new_state.at[_ABS_ZC_BUF_COUNT].set(new_count)

    return new_state


def _abs_count_zero_crossings_from_zc(state_flat):
    """Count zero crossings in ZC ring buffer (JIT-compatible, 500 entries).

    Phase 6M: Uses dedicated ZC buffer instead of slow ring buffer,
    matching Python's adaptive_bias_zero_crossing_window_steps = 500.
    """
    count = state_flat[_ABS_ZC_BUF_COUNT]
    ptr = state_flat[_ABS_ZC_BUF_PTR].astype(jnp.int32)
    buf = state_flat[_ABS_ZC_BUF_START:_ABS_ZC_BUF_END]

    # Build array of values in reverse chronological order (most recent first)
    i_range = jnp.arange(_ABS_ZC_WINDOW)
    reverse_indices = (ptr - 1 - i_range + _ABS_ZC_WINDOW) % _ABS_ZC_WINDOW
    vals = buf[reverse_indices]

    # Shift by 1 to get previous value in the reverse-chronological sequence
    vals_prev = jnp.roll(vals, shift=1)
    vals_prev = vals_prev.at[0].set(vals[0])

    # Valid: entry i is valid if i < count AND entry i-1 is valid
    valid_curr = i_range < count
    valid_prev = jnp.roll(valid_curr, shift=1)
    both_valid = valid_curr & valid_prev

    # Sign change when signs differ and both are valid
    sign_change = (vals < 0) != (vals_prev < 0)
    zc = jnp.sum(jnp.where(sign_change & both_valid, 1, 0))
    return zc


def _abs_update_ring_buffer(state_flat, error_signed):
    """Push new error into ring buffer, update running sum, count, and pointer.

    Returns updated state_flat with new buffer contents, sum, count, and ptr.
    """
    ptr = state_flat[_ABS_SLOW_PTR].astype(jnp.int32)
    count = state_flat[_ABS_SLOW_COUNT]
    total = state_flat[_ABS_SLOW_SUM]

    # Read oldest value from buffer before overwriting
    oldest = state_flat[_ABS_SLOW_BUF_START + ptr]

    # Write new value, update sum and count
    new_state = state_flat.at[_ABS_SLOW_BUF_START + ptr].set(error_signed)
    new_state = new_state.at[_ABS_SLOW_PTR].set((ptr + 1) % _ABS_SLOW_WINDOW)

    # Running sum: add new, subtract oldest (only if buffer is full)
    new_total = jnp.where(count >= _ABS_SLOW_WINDOW, total + error_signed - oldest,
                          total + error_signed)
    new_count = jnp.where(count >= _ABS_SLOW_WINDOW, count, count + 1.0)

    new_state = new_state.at[_ABS_SLOW_SUM].set(new_total)
    new_state = new_state.at[_ABS_SLOW_COUNT].set(new_count)

    return new_state


def pack_state_k2_final(
    notch_x1=0.0, notch_x2=0.0, notch_y1=0.0, notch_y2=0.0,
    prev_tau=None, filtered_com_z=0.4, prev_support_error=0.0,
    ol_pitch_ref_smoothed=0.0, ol_prev_support_error=0.0, ol_support_error_rate=0.0,
    abs_slow_ema=0.0, abs_fast_ema=0.0, abs_trim_tau=0.0,
    abs_hold_steps=0.0, abs_prev_err_sign=0.0, abs_zc_count=0.0,
    abs_zc_error_history=None,  # Phase 6M: ZC error history for parity
):
    """Pack full K2 state with ring buffer ABS + ZC buffer (backward-compatible signature)."""
    s = pack_state_k2(notch_x1, notch_x2, notch_y1, notch_y2, prev_tau,
                      filtered_com_z, prev_support_error,
                      ol_pitch_ref_smoothed, ol_prev_support_error, ol_support_error_rate)
    # pack_state_k2 already allocates full K2_JAX_STATE_SIZE (834) with zeros
    s = s.at[_ABS_SLOW_SUM].set(0.0)  # initialize sum to zero
    s = s.at[_ABS_FAST_SUM].set(0.0)
    s = s.at[_ABS_TRIM_TAU].set(abs_trim_tau)
    s = s.at[_ABS_HOLD_STEPS].set(abs_hold_steps)
    s = s.at[_ABS_PREV_ERR_SIGN].set(abs_prev_err_sign)
    s = s.at[_ABS_ZC_COUNT].set(abs_zc_count)
    s = s.at[_ABS_SLOW_COUNT].set(0.0)
    s = s.at[_ABS_SLOW_PTR].set(0.0)
    s = s.at[_ABS_GUARD_TRIGGER].set(0.0)
    # Phase 6M: Initialize ZC buffer fields (pack starting at position 0)
    s = s.at[_ABS_ZC_BUF_COUNT].set(0.0)
    s = s.at[_ABS_ZC_BUF_PTR].set(0.0)
    if abs_zc_error_history is not None and len(abs_zc_error_history) > 0:
        n_entries = min(len(abs_zc_error_history), _ABS_ZC_WINDOW)
        for i, val in enumerate(abs_zc_error_history[-n_entries:]):
            s = s.at[_ABS_ZC_BUF_START + i].set(float(val))
        s = s.at[_ABS_ZC_BUF_COUNT].set(float(n_entries))
        s = s.at[_ABS_ZC_BUF_PTR].set(float(n_entries % _ABS_ZC_WINDOW))
    return s


def _k2_jax_adaptive_bias_trim(
    signed_error, state_flat,
    schedule_h, pitch_x, safety_pass_in,
    contact_valid,  # Phase 6M: contact state for safety gate parity
):
    """Adaptive bias trim core logic — Stage 6L sliding window ring buffer + Phase 6M ZC buffer.

    Uses true sliding window mean (matching Python exactly) instead of EMA.
    State maintains a 300-element ring buffer for slow/fast + 500-element ZC buffer.

    Args:
        signed_error: Current signed sagittal position error [m]
        state_flat: Full JAX state array (with ring buffer + ZC buffer)
        schedule_h: Height for max_tau scheduling
        pitch_x: Pitch angle for safety gate
        safety_pass_in: Whether safety gate passes (upright, hip_yaw, abs_error)
        contact_valid: Contact state (0.0 or 1.0) for contact_ok gate

    Returns:
        (new_trim_tau, new_hold_steps, new_prev_err_sign, new_zc_count, trim_to_apply,
         slow_mean, fast_mean, new_state_flat)
    """
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import K2_NOTCH_LOW_Q_V1 as _sch

    # --- Update ring buffer with new signed error ---
    state_flat = _abs_update_ring_buffer(state_flat, signed_error)
    # Phase 6M: Update ZC buffer (separate 500-entry buffer, matching Python)
    state_flat = _abs_update_zc_buffer(state_flat, signed_error)

    # --- Compute sliding window means ---
    slow_mean = _abs_sliding_mean_slow(state_flat)
    fast_mean = _abs_sliding_mean_fast(state_flat)
    mean_err = slow_mean  # Use slow mean for trim (matching Python)

    # --- Height-scheduled max trim ---
    z_low = float(_sch.adaptive_bias_height_low_m)
    z_high = float(_sch.adaptive_bias_height_high_m)
    z_extreme = float(_sch.adaptive_bias_height_extreme_m)
    max_low = float(_sch.adaptive_bias_max_tau_low_nm)
    max_high = float(_sch.adaptive_bias_max_tau_high_nm)
    max_extreme = float(_sch.adaptive_bias_max_tau_extreme_nm)

    t_h = jnp.where(schedule_h <= z_low, 0.0,
           jnp.where(schedule_h >= z_extreme, 2.0,
           jnp.where(schedule_h <= z_high, (schedule_h - z_low) / jnp.maximum(z_high - z_low, 1e-9),
           1.0 + (schedule_h - z_high) / jnp.maximum(z_extreme - z_high, 1e-9))))
    max_tau_current = jnp.where(t_h <= 1.0, max_low + (max_high - max_low) * t_h,
                                max_high + (max_extreme - max_high) * (t_h - 1.0))

    # --- Zero-crossing guard ---
    # Phase 6M: Use dedicated ZC buffer (500 entries) instead of slow ring buffer (300).
    # Matches Python's separate _adaptive_bias_zero_crossing_history.
    zc_count = _abs_count_zero_crossings_from_zc(state_flat)
    zc_guard = (zc_count > float(_sch.adaptive_bias_zero_crossing_limit))
    guard_trigger = state_flat[_ABS_GUARD_TRIGGER]
    # Phase 6M fix: guard_trigger >= 3 → reset to 0 (matching Python svdbc.py:5633-5641)
    guard_trigger = jnp.where(
        zc_guard,
        jnp.where(guard_trigger + 1.0 >= 3.0, 0.0, guard_trigger + 1.0),
        0.0,
    )
    zc_guard_active = zc_guard
    guard_scale = jnp.where(zc_guard_active, float(_sch.adaptive_bias_zero_crossing_max_scale), 1.0)
    max_tau_g = max_tau_current * guard_scale

    # --- Sign-reversal guard ---
    sign_err = jnp.sign(mean_err)
    trim_tau = state_flat[_ABS_TRIM_TAU]
    hold_steps = state_flat[_ABS_HOLD_STEPS]
    prev_err_sign = state_flat[_ABS_PREV_ERR_SIGN]

    err_sign_changed = (sign_err != 0.0) & (sign_err != prev_err_sign)
    new_hold = jnp.where(err_sign_changed, float(_sch.adaptive_bias_sign_reversal_hold_steps),
                jnp.where(hold_steps > 0.0, hold_steps - 1.0, 0.0))
    sign_rev_blocked = (new_hold > 0.0) & err_sign_changed
    # Phase 6M fix: update prev_err_sign on hold>0 (matching Python elif branch)
    new_prev_sign = jnp.where(err_sign_changed | (new_hold > 0.0), sign_err, prev_err_sign)

    # --- Proportional target with hysteresis ---
    exit_th = float(_sch.adaptive_bias_exit_threshold_m)
    relief_th = float(_sch.adaptive_bias_relief_hysteresis_m)
    k_tau = float(_sch.adaptive_bias_k_tau_per_m)

    near_zero = jnp.abs(mean_err) <= exit_th
    in_hyst = jnp.abs(mean_err) <= exit_th + relief_th

    raw_target = jnp.where(near_zero, 0.0,
                  jnp.where(sign_rev_blocked, 0.0,
                  jnp.where(in_hyst, trim_tau,
                  -k_tau * (mean_err - sign_err * exit_th))))

    clipped = jnp.clip(raw_target, -max_tau_g, max_tau_g)

    # --- Asymmetric rate limiting ---
    is_decay = jnp.abs(clipped) < jnp.abs(trim_tau)
    rate = jnp.where(is_decay, float(_sch.adaptive_bias_decay_rate_nm_per_step),
                            float(_sch.adaptive_bias_rate_nm_per_step))
    delta = jnp.clip(clipped - trim_tau, -rate, rate)
    new_trim = jnp.clip(trim_tau + delta, -max_tau_g, max_tau_g)

    trim_to_apply = jnp.where(safety_pass_in, new_trim, 0.0)

    # --- Pack updated ABS state back ---
    state_flat = state_flat.at[_ABS_SLOW_SUM].set(state_flat[_ABS_SLOW_SUM])  # already updated
    state_flat = state_flat.at[_ABS_FAST_SUM].set(fast_mean)
    state_flat = state_flat.at[_ABS_TRIM_TAU].set(new_trim)
    state_flat = state_flat.at[_ABS_HOLD_STEPS].set(new_hold)
    state_flat = state_flat.at[_ABS_PREV_ERR_SIGN].set(new_prev_sign)
    state_flat = state_flat.at[_ABS_ZC_COUNT].set(zc_count.astype(jnp.float64))
    state_flat = state_flat.at[_ABS_GUARD_TRIGGER].set(guard_trigger)

    # Phase 0: Pack ABS diag intermediates (12 floats)
    abs_diag = jnp.array([
        slow_mean, fast_mean, sign_err, raw_target, clipped,
        jnp.where(is_decay, 1.0, 0.0), rate, delta, new_trim,
        jnp.where(safety_pass_in, 1.0, 0.0), trim_to_apply, new_hold,
    ], dtype=jnp.float64)

    return new_trim, new_hold, new_prev_sign, zc_count, trim_to_apply, slow_mean, fast_mean, state_flat, abs_diag
