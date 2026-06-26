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
