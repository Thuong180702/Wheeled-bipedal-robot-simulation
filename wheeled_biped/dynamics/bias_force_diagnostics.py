"""Diagnostic decomposition helpers for bias force validation.

Identifies which terms fail in the bias force computation:
  1. Gravity-only error
  2. Base-only velocity error (pure base DOF velocities)
  3. Single actuated DOF velocity error
  4. Two-DOF cross velocity error (Coriolis coupling)
  5. All-actuated random velocity error
  6. Full mixed free-base + actuated velocity error
  7. Free-base force error (qfrc[0:3])
  8. Free-base torque error (qfrc[3:6])
  9. Actuated torque error (qfrc[6:16])

Also computes cross-term decomposition:
  bias(q, v_i + v_j) - bias(q, v_i) - bias(q, v_j) + bias(q, 0)

This isolates Coriolis coefficient errors in multi-joint interactions.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import mujoco
import numpy as np


def decompose_bias_errors(
    model: mujoco.MjModel,
    constants: dict[str, Any],
    qpos_np: np.ndarray,
    qvel_np: np.ndarray,
) -> dict[str, Any]:
    """Decompose bias force error into sub-components.

    Args:
        model: MuJoCo MjModel.
        constants: dict from ``build_bias_force_constants``.
        qpos_np: (nq,) numpy array.
        qvel_np: (nv,) numpy array.

    Returns:
        dict with per-component errors and verdicts.
    """
    from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces

    nv = model.nv
    qpos_jax = jnp.array(qpos_np, dtype=jnp.float32)
    qvel_jax = jnp.array(qvel_np, dtype=jnp.float32)

    # JAX bias
    jax_full = np.array(jax_bias_forces(qpos_jax, qvel_jax, constants), dtype=np.float64)

    # CPU bias
    d = mujoco.MjData(model)
    d.qpos[:] = qpos_np
    d.qvel[:] = qvel_np
    mujoco.mj_forward(model, d)
    cpu_full = np.array(d.qfrc_bias, dtype=np.float64)

    # Gravity-only
    zero_qvel_jax = jnp.zeros(nv, dtype=jnp.float32)
    jax_grav = np.array(jax_bias_forces(qpos_jax, zero_qvel_jax, constants), dtype=np.float64)
    d0 = mujoco.MjData(model)
    d0.qpos[:] = qpos_np
    mujoco.mj_forward(model, d0)
    cpu_grav = np.array(d0.qfrc_bias, dtype=np.float64)

    # Velocity-dependent (JAX vs CPU)
    jax_vel = jax_full - jax_grav
    cpu_vel = cpu_full - cpu_grav

    result = {
        "full_bias": _component(jax_full, cpu_full),
        "gravity": _component(jax_grav, cpu_grav),
        "velocity_dependent": _component(jax_vel, cpu_vel),
        "free_base_force": _component(jax_full[0:3], cpu_full[0:3]),
        "free_base_torque": _component(jax_full[3:6], cpu_full[3:6]),
        "actuated_torque": _component(jax_full[6:16], cpu_full[6:16]),
        "free_base_total": _component(jax_full[0:6], cpu_full[0:6]),
        "all_finite": bool(np.all(np.isfinite(jax_full))),
    }
    return result


def decompose_velocity_components(
    model: mujoco.MjModel,
    constants: dict[str, Any],
    qpos_np: np.ndarray,
) -> dict[str, Any]:
    """Test pure single-DOF and paired velocity cases to isolate error sources.

    Returns per-case errors for each individual and paired DOF excitation.
    """
    from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces

    nv = model.nv
    qpos_jax = jnp.array(qpos_np, dtype=jnp.float32)

    # Individual cases
    single_cases = {}
    # Base DOFs
    for idx, name in [(0, "base_vx"), (1, "base_vy"), (2, "base_vz"),
                       (3, "base_roll"), (4, "base_pitch"), (5, "base_yaw")]:
        qvel = np.zeros(nv, dtype=np.float64)
        qvel[idx] = 1.0
        single_cases[name] = _eval_case(model, constants, qpos_np, qpos_jax, qvel)

    # Actuated single DOFs
    actuated_names = {
        6: "l_hip_roll", 7: "l_hip_yaw", 8: "l_hip_pitch", 9: "l_knee", 10: "l_wheel",
        11: "r_hip_roll", 12: "r_hip_yaw", 13: "r_hip_pitch", 14: "r_knee", 15: "r_wheel",
    }
    for idx, name in actuated_names.items():
        qvel = np.zeros(nv, dtype=np.float64)
        qvel[idx] = 1.0
        single_cases[name] = _eval_case(model, constants, qpos_np, qpos_jax, qvel)

    # Paired velocity cases
    paired_cases = {}
    pairs = [
        ("base_yaw_l_hip_pitch", 5, 1.0, 8, 1.0),
        ("base_yaw_l_knee", 5, 1.0, 9, 1.0),
        ("l_hip_pitch_l_knee", 8, 1.0, 9, 1.0),
        ("l_wheel_r_wheel", 10, 5.0, 15, 5.0),
        ("l_hip_roll_r_hip_roll", 6, 1.0, 11, -1.0),
        ("base_yaw_l_wheel", 5, 1.0, 10, 5.0),
    ]
    for name, i1, v1, i2, v2 in pairs:
        qvel = np.zeros(nv, dtype=np.float64)
        qvel[i1] = v1
        qvel[i2] = v2
        paired_cases[name] = _eval_case(model, constants, qpos_np, qpos_jax, qvel)

    # Small random all-actuated
    rng = np.random.default_rng(123)
    qvel_act = np.zeros(nv, dtype=np.float64)
    qvel_act[6:16] = rng.uniform(-0.1, 0.1, 10)
    paired_cases["small_random_actuated"] = _eval_case(model, constants, qpos_np, qpos_jax, qvel_act)

    qvel_act2 = np.zeros(nv, dtype=np.float64)
    qvel_act2[6:16] = rng.uniform(-0.5, 0.5, 10)
    paired_cases["moderate_random_actuated"] = _eval_case(model, constants, qpos_np, qpos_jax, qvel_act2)

    return {
        "single_velocity": single_cases,
        "paired_velocity": paired_cases,
    }


def compute_cross_term_decomposition(
    model: mujoco.MjModel,
    constants: dict[str, Any],
    qpos_np: np.ndarray,
    vel_pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compute cross-term errors:
      bias(q, v_i + v_j) - bias(q, v_i) - bias(q, v_j) + bias(q, 0)

    For each velocity pair (v_i, v_j), compute both JAX and CPU versions of the
    cross-term expression and compare them.  This isolates Coriolis coupling.

    Args:
        model: MuJoCo MjModel.
        constants: dict from ``build_bias_force_constants``.
        qpos_np: (nq,) numpy array.
        vel_pairs: list of dicts with keys 'name', 'v_i', 'v_j'.

    Returns:
        list of dicts with cross-term errors.
    """
    from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces

    nv = model.nv
    qpos_jax = jnp.array(qpos_np, dtype=jnp.float32)
    results = []

    for pair in vel_pairs:
        name = pair["name"]
        v_i = np.array(pair["v_i"], dtype=np.float64)
        v_j = np.array(pair["v_j"], dtype=np.float64)
        v_sum = v_i + v_j

        # JAX cross-term
        def _jax_bias(v):
            return np.array(jax_bias_forces(qpos_jax, jnp.array(v, dtype=jnp.float32), constants),
                          dtype=np.float64)

        jax_sum = _jax_bias(v_sum)
        jax_i = _jax_bias(v_i)
        jax_j = _jax_bias(v_j)
        jax_zero = _jax_bias(np.zeros(nv, dtype=np.float64))

        jax_cross = jax_sum - jax_i - jax_j + jax_zero

        # CPU cross-term
        def _cpu_bias(v):
            d = mujoco.MjData(model)
            d.qpos[:] = qpos_np
            d.qvel[:] = v
            mujoco.mj_forward(model, d)
            return np.array(d.qfrc_bias, dtype=np.float64)

        cpu_sum = _cpu_bias(v_sum)
        cpu_i = _cpu_bias(v_i)
        cpu_j = _cpu_bias(v_j)
        cpu_zero = _cpu_bias(np.zeros(nv, dtype=np.float64))

        cpu_cross = cpu_sum - cpu_i - cpu_j + cpu_zero

        # Error
        cross_err = np.max(np.abs(jax_cross - cpu_cross))
        max_cpu_cross = np.max(np.abs(cpu_cross))
        cross_rel = cross_err / max_cpu_cross if max_cpu_cross > 1e-12 else cross_err

        results.append({
            "name": name,
            "cross_max_abs_error": float(cross_err),
            "cross_max_rel_error": float(cross_rel),
            "jax_cross_norm": float(np.max(np.abs(jax_cross))),
            "cpu_cross_norm": float(np.max(np.abs(cpu_cross))),
        })

    return results


# ── Internal helpers ────────────────────────────────────────────────────


def _eval_case(model, constants, qpos_np, qpos_jax, qvel_np):
    from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces

    nv = model.nv
    qvel_jax = jnp.array(qvel_np, dtype=jnp.float32)

    jax_full = np.array(jax_bias_forces(qpos_jax, qvel_jax, constants), dtype=np.float64)

    d = mujoco.MjData(model)
    d.qpos[:] = qpos_np
    d.qvel[:] = qvel_np
    mujoco.mj_forward(model, d)
    cpu_full = np.array(d.qfrc_bias, dtype=np.float64)

    full_err = float(np.max(np.abs(jax_full - cpu_full)))
    fb_err = float(np.max(np.abs(jax_full[0:6] - cpu_full[0:6])))
    act_err = float(np.max(np.abs(jax_full[6:16] - cpu_full[6:16])))

    return {
        "full_max_abs_error": full_err,
        "free_base_max_abs_error": fb_err,
        "actuated_max_abs_error": act_err,
    }


def _component(jax_arr, cpu_arr):
    abs_err = np.max(np.abs(jax_arr - cpu_arr))
    max_cpu = np.max(np.abs(cpu_arr))
    rel_err = abs_err / max_cpu if max_cpu > 1e-12 else abs_err
    pass_thresh, warn_thresh = 1e-3, 1e-2

    def _v(e):
        if e < pass_thresh:
            return "PASS"
        elif e < warn_thresh:
            return "WARN"
        return "FAIL"

    return {
        "max_abs_error": float(abs_err),
        "max_rel_error": float(rel_err),
        "verdict": _v(abs_err),
    }
