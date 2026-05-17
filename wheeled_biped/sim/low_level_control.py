"""
Low-level PID/PD control — reusable across tasks.

Provides ``pid_control``, a pure JAX function that converts a normalized
policy target action (in ``[-1, 1]``) into actuator control signals.

Two joint types are handled with appropriate control semantics:

- **Position-controlled joints** (legs) — PD+I control:
    error   = desired_position - current_position
    d_error = -joint_velocity   (= d(error)/dt for a fixed target)

- **Velocity-controlled joints** (wheels) — PI control (no derivative):
    error   = desired_velocity - current_velocity
    d_error = 0   (see note below)

**Why wheels use only PI (no derivative term)**

For position joints the derivative of position error is ``-joint_vel``,
so the ``kd`` term is mathematically correct as an anti-kickback damp.

For wheel joints the error is already in *velocity* space.  The true
derivative of velocity error would be ``-joint_acceleration``, which
requires either numerical differentiation (noisy) or an additional state
variable (complexity).  Using ``-joint_vel`` as a pseudo-derivative for
wheels would be a **unit mismatch**: it adds a ``kd × velocity`` term
(rad/s → wrong units for a velocity-error derivative) and acts as extra
viscous damping whose magnitude depends on the current speed rather than
the error rate.  Setting ``kd_wheel = 0`` avoids this mismatch while
preserving standard PI velocity control semantics.

The ``kd`` values in the config for wheel joints (indices 4 and 9) are
intentionally zero (or may be set to zero in the call-site; the controller
enforces this by masking them out internally).

Typical usage inside an env's ``step()``:

    from wheeled_biped.sim.low_level_control import pid_control

    ctrl, new_integral = pid_control(
        mjx_data,
        normalized_target,
        pid_integral,
        kp=self._pid_kp,
        ki=self._pid_ki,
        kd=self._pid_kd,
        joint_mins=self._joint_mins,
        joint_maxs=self._joint_maxs,
        wheel_mask=self._wheel_mask,
        wheel_vel_limit=self._wheel_vel_limit,
        i_limit=self._pid_i_limit,
        ctrl_min=self._ctrl_min,
        ctrl_max=self._ctrl_max,
        control_dt=self.CONTROL_DT,
    )
"""

from __future__ import annotations

import jax.numpy as jnp
from mujoco import mjx


def pid_control(
    mjx_data: mjx.Data,
    normalized_target: jnp.ndarray,
    pid_integral: jnp.ndarray,
    *,
    kp: jnp.ndarray,
    ki: jnp.ndarray,
    kd: jnp.ndarray,
    joint_mins: jnp.ndarray,
    joint_maxs: jnp.ndarray,
    wheel_mask: jnp.ndarray,
    wheel_vel_limit: float = 20.0,
    i_limit: float = 0.3,
    ctrl_min: jnp.ndarray,
    ctrl_max: jnp.ndarray,
    control_dt: float = 0.02,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """PD+I (legs) / PI (wheels) low-level control: normalized target → actuator ctrl.

    **Leg joints** (``wheel_mask == 0``): PD+I position control.
        desired = joint_mins + (target + 1) / 2 * (joint_maxs - joint_mins)
        error   = desired - current_position
        d_error = -joint_velocity              # correct: d(position_error)/dt

    **Wheel joints** (``wheel_mask == 1``): PI velocity control (no derivative).
        desired_vel = target * wheel_vel_limit
        error       = desired_vel - current_velocity
        d_error     = 0                        # kd masked to zero for wheels
                                               # see module docstring for rationale

    Integral state is updated per-call and anti-windup clamped by ``i_limit``.

    This is a pure function with no side-effects — safe inside ``jax.jit``
    and ``jax.vmap``.

    Args:
        mjx_data: Current MJX simulation data.
        normalized_target: Policy output in ``[-1, 1]``, shape ``(num_joints,)``.
        pid_integral: Running integral state, shape ``(num_joints,)``.
        kp: Proportional gains, shape ``(num_joints,)``.
        ki: Integral gains, shape ``(num_joints,)``.
        kd: Derivative gains, shape ``(num_joints,)``.
             For wheel joints this is internally masked to zero regardless of
             the value passed; see module docstring.
        joint_mins: Lower joint limits in radians, shape ``(num_joints,)``.
        joint_maxs: Upper joint limits in radians, shape ``(num_joints,)``.
        wheel_mask: Float mask; 1.0 for wheel joints, 0.0 otherwise.
        wheel_vel_limit: Max wheel angular velocity (rad/s).
        i_limit: Anti-windup clamp magnitude.
        ctrl_min: Actuator control range lower bound, shape ``(num_joints,)``.
        ctrl_max: Actuator control range upper bound, shape ``(num_joints,)``.
        control_dt: Control timestep in seconds (typically 0.02 s at 50 Hz).

    Returns:
        Tuple ``(ctrl, new_integral)`` where both have shape ``(num_joints,)``.
    """
    joint_pos = mjx_data.qpos[7:17]  # (num_joints,)
    joint_vel = mjx_data.qvel[6:16]  # (num_joints,)

    # ── Position target for leg joints, velocity target for wheels ──────────
    pos_target = joint_mins + (normalized_target + 1.0) * 0.5 * (joint_maxs - joint_mins)
    vel_target_wheel = normalized_target * wheel_vel_limit

    # ── Error and derivative ─────────────────────────────────────────────────
    pos_err = pos_target - joint_pos  # position error (legs)
    vel_err = vel_target_wheel - joint_vel  # velocity error (wheels)

    # Blend: position error for legs, velocity error for wheels
    error = (1.0 - wheel_mask) * pos_err + wheel_mask * vel_err

    # Derivative term:
    #   Legs:   d(pos_error)/dt = -joint_vel  (correct PD damping)
    #   Wheels: 0               (velocity-error derivative requires joint_accel;
    #                            -joint_vel is a unit mismatch — see module docstring)
    leg_d_error = -joint_vel  # anti-kickback for position joints
    d_error = (1.0 - wheel_mask) * leg_d_error  # zero for wheels

    # ── Integral with anti-windup ─────────────────────────────────────────────
    integral_new = jnp.clip(
        pid_integral + error * control_dt,
        -i_limit,
        i_limit,
    )

    ctrl = kp * error + kd * d_error + ki * integral_new
    ctrl = jnp.clip(ctrl, ctrl_min, ctrl_max)

    return ctrl, integral_new


def normalized_motor_torque_control(
    normalized_torque: jnp.ndarray,
    ctrl_min: jnp.ndarray,
    ctrl_max: jnp.ndarray,
    max_ctrl_fraction: float = 1.0,
    allow_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    normalized_torque = jnp.clip(normalized_torque, -1.0, 1.0)
    fraction = jnp.clip(jnp.asarray(max_ctrl_fraction, dtype=ctrl_min.dtype), 0.0, 1.0)
    ctrl_limit = jnp.minimum(jnp.abs(ctrl_min), jnp.abs(ctrl_max)) * fraction
    ctrl = normalized_torque * ctrl_limit
    if allow_mask is not None:
        ctrl = ctrl * allow_mask
    return jnp.clip(ctrl, ctrl_min, ctrl_max)


def hybrid_pid_plus_torque_control(
    pid_ctrl: jnp.ndarray,
    normalized_torque_residual: jnp.ndarray,
    ctrl_min: jnp.ndarray,
    ctrl_max: jnp.ndarray,
    max_ctrl_fraction: float = 1.0,
    allow_mask: jnp.ndarray | None = None,
    pid_authority_fraction: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Hybrid PID + torque control with optional authority reallocation.

    Args:
        pid_authority_fraction: Fraction of actuator range reserved for PID (0.0-1.0).
            1.0 = PID can use full range (default, backward compatible)
            0.7 = PID limited to 70% of range, reserving 30% for WBC residuals
            Prevents PID saturation from suppressing WBC corrections.
    """
    # Clamp PID output to reserved fraction before adding residual
    pid_fraction = jnp.clip(jnp.asarray(pid_authority_fraction, dtype=ctrl_min.dtype), 0.0, 1.0)
    pid_limit_min = ctrl_min * pid_fraction
    pid_limit_max = ctrl_max * pid_fraction
    pid_clamped = jnp.clip(pid_ctrl, pid_limit_min, pid_limit_max)

    # Compute WBC residual
    residual = normalized_motor_torque_control(
        normalized_torque_residual,
        ctrl_min,
        ctrl_max,
        max_ctrl_fraction,
        allow_mask,
    )

    # Blend and clip to full range
    final = jnp.clip(pid_clamped + residual, ctrl_min, ctrl_max)
    return final, residual


def actuator_saturation_flags(
    ctrl: jnp.ndarray,
    ctrl_min: jnp.ndarray,
    ctrl_max: jnp.ndarray,
    eps: float = 1e-6,
) -> jnp.ndarray:
    return (ctrl <= ctrl_min + eps) | (ctrl >= ctrl_max - eps)


def torque_first_wbc_control(
    mjx_data: mjx.Data,
    normalized_wbc_torque: jnp.ndarray,
    ctrl_min: jnp.ndarray,
    ctrl_max: jnp.ndarray,
    wbc_authority_fraction: float = 1.0,
    damping_gain: float = 0.0,
    smoothing_alpha: float = 0.0,
    prev_ctrl: jnp.ndarray | None = None,
    impedance_kp: float = 0.0,
    impedance_target: jnp.ndarray | None = None,
    joint_mins: jnp.ndarray | None = None,
    joint_maxs: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Torque-first WBC control: WBC commands primary, light stabilization only.

    Architecture:
        WBC torque (primary) -> light damping -> weak impedance -> smoothing -> actuators

    This mode gives WBC dominant authority (>70%) with only lightweight
    stabilization, eliminating PID position control that was suppressing
    WBC corrections in hybrid_pid_plus_torque mode.

    Args:
        mjx_data: Current MJX simulation data.
        normalized_wbc_torque: WBC torque commands in [-1, 1], shape (num_joints,).
        ctrl_min: Actuator control range lower bound.
        ctrl_max: Actuator control range upper bound.
        wbc_authority_fraction: Fraction of actuator range for WBC (0.0-1.0).
            Default 1.0 = WBC uses full range (100% authority).
        damping_gain: Optional light damping coefficient (default 0.0 = no damping).
            Small values (0.5-2.0) add velocity damping without position control.
        smoothing_alpha: Optional temporal smoothing (default 0.0 = no smoothing).
            Small values (0.1-0.3) reduce control chatter.
        prev_ctrl: Previous control output for smoothing (optional).
        impedance_kp: Optional weak impedance gain (default 0.0 = no impedance).
            Small values (1.0-5.0) add soft position feedback without dominance.
        impedance_target: Target joint positions for impedance (optional, radians).
            If None, uses current position (pure damping, no restoring force).
        joint_mins: Joint lower limits for impedance target normalization (optional).
        joint_maxs: Joint upper limits for impedance target normalization (optional).

    Returns:
        Final actuator control, shape (num_joints,).
    """
    # Scale WBC torque to actuator range
    normalized_wbc_torque = jnp.clip(normalized_wbc_torque, -1.0, 1.0)
    wbc_fraction = jnp.clip(jnp.asarray(wbc_authority_fraction, dtype=ctrl_min.dtype), 0.0, 1.0)
    ctrl_limit = jnp.minimum(jnp.abs(ctrl_min), jnp.abs(ctrl_max)) * wbc_fraction
    wbc_ctrl = normalized_wbc_torque * ctrl_limit

    # Optional light damping (velocity-based, not position-based)
    # Use jnp.where instead of if statement for JAX JIT compatibility
    joint_vel = mjx_data.qvel[6:16]
    damping_ctrl_raw = -jnp.asarray(damping_gain, dtype=ctrl_min.dtype) * joint_vel
    damping_ctrl_clipped = jnp.clip(damping_ctrl_raw, ctrl_min * 0.2, ctrl_max * 0.2)
    damping_ctrl = jnp.where(
        jnp.asarray(damping_gain, dtype=ctrl_min.dtype) > 0.0,
        damping_ctrl_clipped,
        jnp.zeros_like(wbc_ctrl)
    )

    # Optional weak impedance (soft position feedback)
    # Use jnp.where for JAX JIT compatibility
    joint_pos = mjx_data.qpos[7:17]
    impedance_target_safe = jnp.zeros_like(wbc_ctrl) if impedance_target is None else impedance_target
    pos_error = impedance_target_safe - joint_pos
    impedance_ctrl_raw = jnp.asarray(impedance_kp, dtype=ctrl_min.dtype) * pos_error
    impedance_ctrl_clipped = jnp.clip(impedance_ctrl_raw, ctrl_min * 0.15, ctrl_max * 0.15)
    impedance_ctrl = jnp.where(
        jnp.asarray(impedance_kp, dtype=ctrl_min.dtype) > 0.0,
        impedance_ctrl_clipped,
        jnp.zeros_like(wbc_ctrl)
    )

    # Combine: WBC (dominant) + damping + impedance
    ctrl = wbc_ctrl + damping_ctrl + impedance_ctrl
    ctrl = jnp.clip(ctrl, ctrl_min, ctrl_max)

    # Optional temporal smoothing
    # Use jnp.where for JAX JIT compatibility
    prev_ctrl_safe = jnp.zeros_like(ctrl) if prev_ctrl is None else prev_ctrl
    ctrl_smoothed = jnp.asarray(smoothing_alpha, dtype=ctrl.dtype) * prev_ctrl_safe + (1.0 - jnp.asarray(smoothing_alpha, dtype=ctrl.dtype)) * ctrl
    ctrl = jnp.where(
        jnp.asarray(smoothing_alpha, dtype=ctrl.dtype) > 0.0,
        jnp.clip(ctrl_smoothed, ctrl_min, ctrl_max),
        ctrl
    )

    return ctrl
