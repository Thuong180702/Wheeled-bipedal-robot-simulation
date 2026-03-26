"""
Low-level PID control — reusable across tasks.

Provides ``pid_control``, a pure JAX function that converts a normalized
policy target action (in ``[-1, 1]``) into actuator control signals through
a PD+I controller.

Two joint types are handled transparently:
- **Position-controlled joints** — error = desired_position - current_position
- **Velocity-controlled joints** (wheels) — error = desired_wheel_vel - current_vel

The caller selects which joints are wheels via ``wheel_mask``.

Typical usage inside an env's ``step()``::

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
    """PID low-level control: normalized target → actuator ctrl.

    For position joints:
        desired = joint_mins + (target + 1) / 2 * (joint_maxs - joint_mins)
        error   = desired - current_position

    For wheel joints (``wheel_mask == 1``):
        desired_vel = target * wheel_vel_limit
        error       = desired_vel - current_velocity

    The damping term ``d_error = -joint_vel`` is applied uniformly.
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

    # Position target for leg joints, velocity target for wheels
    pos_target = joint_mins + (normalized_target + 1.0) * 0.5 * (joint_maxs - joint_mins)
    vel_target_wheel = normalized_target * wheel_vel_limit

    pos_err = pos_target - joint_pos
    vel_err = -joint_vel  # damping

    # Blend: position error for legs, velocity error for wheels
    error = (1.0 - wheel_mask) * pos_err + wheel_mask * (vel_target_wheel - joint_vel)
    d_error = vel_err  # same damping for all joints

    integral_new = jnp.clip(
        pid_integral + error * control_dt,
        -i_limit,
        i_limit,
    )

    ctrl = kp * error + kd * d_error + ki * integral_new
    ctrl = jnp.clip(ctrl, ctrl_min, ctrl_max)

    return ctrl, integral_new
