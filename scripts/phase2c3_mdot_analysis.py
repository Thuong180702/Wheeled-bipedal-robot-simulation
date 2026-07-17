#!/usr/bin/env python
"""Phase 2C.3 — M-dot analysis: identify exact discrepancy between RNEA and MuJoCo.

Method: compute qfrc_bias velocity-dependent part using three approaches:
  A. MuJoCo CPU (ground truth)
  B. Our JAX RNEA
  C. Mass-matrix trajectory derivative method (dM/dt @ qvel, numerical)

This isolates whether the error is in the RNEA or the mass matrix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wheeled_biped.dynamics.jax_bias_forces import (
    build_bias_force_constants,
    jax_bias_forces,
    jax_gravity_forces,
)
from wheeled_biped.dynamics.jax_mass_matrix import (
    jax_mass_matrix,
    build_mass_matrix_constants,
)
from wheeled_biped.utils.config import get_model_path


def _cpu_bias(model, qpos, qvel):
    d = mujoco.MjData(model)
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    d.qacc[:] = 0.0
    mujoco.mj_forward(model, d)
    mujoco.mj_inverse(model, d)
    return np.array(d.qfrc_inverse, dtype=np.float64)


def main() -> int:
    model_path = str(get_model_path())
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    nv = model.nv
    nq = model.nq

    constants = build_bias_force_constants(model)
    mm_constants = build_mass_matrix_constants(model)

    qpos_base = data.qpos.copy()
    qpos_base[3:7] = [1.0, 0.0, 0.0, 0.0]  # identity quaternion
    qpos_jax = jnp.array(qpos_base, dtype=jnp.float32)

    print("=" * 72)
    print("PHASE 2C.3 — M-DOT ANALYSIS")
    print("=" * 72)

    # ═══════════════════════════════════════════════════════════════════
    # Step 1: Compute M at identity, and verify it matches MuJoCo
    # ═══════════════════════════════════════════════════════════════════
    print("\n--- Step 1: Mass matrix verification ---")
    M_jax = np.array(jax_mass_matrix(qpos_jax, mm_constants))

    # Compute M_mjc column by column via finite difference of mj_inverse
    eps = 1e-4
    M_mjc = np.zeros((nv, nv))
    for i in range(nv):
        qacc = np.zeros(nv); qacc[i] = eps
        tau_eps = _cpu_bias(model, qpos_base, np.zeros(nv))
        # Actually need different approach: tau = M@qacc + bias
        # tau_eps = M@e_i*eps + bias_0
        # tau_0 = bias_0
        # (tau_eps - tau_0)/eps = M@e_i = column i of M
        # But cpu_bias is computed with qvel=0 and qacc=e_i*eps
        # Wait, _cpu_bias sets qacc=0. Let me fix this.

    # Better approach: use mj_forward + direct M computation
    d = mujoco.MjData(model)
    d.qpos[:] = qpos_base
    d.qvel[:] = 0.0
    mujoco.mj_forward(model, d)
    # qM is the mass matrix in sparse format, stored in d.qM
    # Actually, in newer MuJoCo versions, we can use mj_fullM
    M_mjc_full = np.zeros((nv, nv))
    mujoco.mj_fullM(model, M_mjc_full, d.qM)
    print(f"  JAX M vs MuJoCo M: max diff = {np.max(np.abs(M_jax - M_mjc_full)):.2e}")
    print(f"  Free-base block diff: {np.max(np.abs(M_jax[0:6, 0:6] - M_mjc_full[0:6, 0:6])):.2e}")
    print(f"  M[0:3, 3:6] JAX:\n{M_jax[0:3, 3:6]}")
    print(f"  M[0:3, 3:6] MJC:\n{M_mjc_full[0:3, 3:6]}")

    # ═══════════════════════════════════════════════════════════════════
    # Step 2: Compute dM/dt via trajectory method
    # ═══════════════════════════════════════════════════════════════════
    print("\n--- Step 2: M-dot via trajectory derivative ---")
    # For a rigid trajectory: q(t) = q0 + qvel*t (approximate, small t)
    qvel_test = np.zeros(nv)
    qvel_test[0] = 1.0   # vx
    qvel_test[5] = 1.0   # wz

    dt = 0.001
    # qpos change from qvel:
    dq = np.zeros(nq)
    dq[0:3] = qvel_test[0:3] * dt  # position update
    # Quaternion update from angular velocity (at identity):
    # dq/dt = 1/2 * [0, wx, wy, wz] (at identity quat)
    dq[3:7] = 0.5 * np.array([0.0, qvel_test[3], qvel_test[4], qvel_test[5]]) * dt
    q_dt = qpos_base + dq
    q_dt[3:7] /= np.linalg.norm(q_dt[3:7])
    M_dt_jax = np.array(jax_mass_matrix(jnp.array(q_dt, dtype=jnp.float32), mm_constants))

    # M_mjc at perturbed position
    d2 = mujoco.MjData(model)
    d2.qpos[:] = q_dt
    d2.qvel[:] = 0.0
    mujoco.mj_forward(model, d2)
    M_dt_mjc = np.zeros((nv, nv))
    mujoco.mj_fullM(model, M_dt_mjc, d2.qM)

    Mdot_jax = (M_dt_jax - M_jax) / dt
    Mdot_mjc = (M_dt_mjc - M_mjc_full) / dt

    print(f"  Mdot JAX vs MJC max diff: {np.max(np.abs(Mdot_jax - Mdot_mjc)):.2e}")

    # Compute dM/dt @ qvel
    Mdot_qvel_jax = Mdot_jax @ qvel_test
    Mdot_qvel_mjc = Mdot_mjc @ qvel_test

    print(f"  Mdot @ qvel (JAX) [0:6]: {Mdot_qvel_jax[0:6]}")
    print(f"  Mdot @ qvel (MJC) [0:6]: {Mdot_qvel_mjc[0:6]}")
    print(f"  diff [0:6]: {Mdot_qvel_jax[0:6] - Mdot_qvel_mjc[0:6]}")

    # ═══════════════════════════════════════════════════════════════════
    # Step 3: Compute dT/dq (partial derivative of kinetic energy)
    # ═══════════════════════════════════════════════════════════════════
    print("\n--- Step 3: dT/dq computation ---")
    # dT/dq_i = 1/2 qvel^T @ dM/dq_i @ qvel
    h = 1e-6
    dT_dq = np.zeros(nq)

    # Translational DOFs: dT/dx = 0 (M doesn't depend on position)
    # Rotational DOFs (q[3:7]): need finite differences of M
    for k in range(3, 7):
        qp = qpos_base.copy(); qp[k] += h
        qm = qpos_base.copy(); qm[k] -= h
        qp[3:7] /= np.linalg.norm(qp[3:7])
        qm[3:7] /= np.linalg.norm(qm[3:7])

        Mp = np.array(jax_mass_matrix(jnp.array(qp, dtype=jnp.float32), mm_constants))
        Mm = np.array(jax_mass_matrix(jnp.array(qm, dtype=jnp.float32), mm_constants))

        dM = (Mp - Mm) / (2 * h)
        dT_dq[k] = 0.5 * qvel_test @ dM @ qvel_test

    print(f"  dT/dq[quat components]: {dT_dq[3:7]}")

    # ═══════════════════════════════════════════════════════════════════
    # Step 4: Compute Coriolis force = d/dt(M qvel) - dT/dq
    # ═══════════════════════════════════════════════════════════════════
    print("\n--- Step 4: Coriolis force from Lagrangian ---")
    # For the "generalized force" from Lagrange:
    # tau_i = d/dt(dT/dqvel_i) - dT/dq_i
    # dT/dqvel = M @ qvel
    # d/dt(M qvel) = Mdot @ qvel + M @ qacc
    # For qacc=0: tau = Mdot @ qvel - dT/dq

    # But wait: the dT/dq we computed is in quaternion coordinates.
    # The generalized force tau = Mdot @ qvel - dT/dq is NOT in the same
    # coordinates as qfrc_bias. We need to map from quaternion tangent to
    # angular velocity.

    # The variational relationship: delta_quat -> delta_theta
    # At identity: delta_quat[1:4] = 1/2 * delta_theta
    # So the generalized force: tau_theta = 2 * tau_quat[1:4]

    # Actually, the relationship is more complex. Let me use a different approach.
    # I'll compute the Coriolis force directly in generalized velocity coordinates.

    # For the Lagrangian expressed in (x, y, z, theta_x, theta_y, theta_z):
    # T = 1/2 qvel^T M(body-frame) qvel  (M already in velocity coordinates)
    # dT/dtheta_i = 1/2 qvel^T dM/dtheta_i qvel

    # We need dM/dtheta, not dM/dquat.

    # At identity orientation, the quaternion derivative w.r.t. theta_i is:
    # dquat/dtheta_i = 1/2 * e_{i+1} (for i=0,1,2)
    # And by chain rule: dM/dtheta_i = dM/dquat @ dquat/dtheta_i

    # For theta_x (rotation about x-axis):
    # dquat/dtheta_x = [0, 1/2, 0, 0]
    # dM/dtheta_x = 1/2 * dM/dq_x (where q_x = quat[1])

    # Let's compute dM/dtheta more carefully using the rotation formula

    # Actually, let me use a MUCH simpler approach:
    # For small rotations, I can parameterize orientation as RPY and
    # compute M in RPY coordinates, then differentiate directly.

    # Simpler approach: use mj_differentiatePos
    # But that requires MuJoCo 3.0+

    # Let me just directly compare:
    # 1. MuJoCo qfrc_bias velocity part
    # 2. Our JAX RNEA velocity part
    # 3. The analytical formula from Mdot and dT/dq

    print("  Computing via M-quadratic form method...")

    # Alternative method: the Christoffel formula
    # Gamma_ijk = 1/2 (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i)
    # C_ij qvel_j = sum_k Gamma_ijk qvel_k qvel_j

    # I'll compute C @ qvel by computing dM/dq_k numerically and summing

    # For each DOF k in qpos that M depends on:
    # dM/dq_k contributes sum_k (dM/dq_k @ qvel) * qvel_k to d/dt(M qvel)
    # Wait, that's not right. dM/dt = sum_k dM/dq_k dq_k/dt = sum_k dM/dq_k qvel_k (where qvel_k drives q_k)

    # The issue: qvel[0:6] drives qpos[0:7] (position + quaternion).
    # d/dt qpos[0:3] = qvel[0:3] (direct)
    # d/dt qpos[3:7] = 1/2 Omega(qvel[3:6]) @ qpos[3:7] (quaternion derivative)

    # So dM/dt = sum_{k=0}^{2} dM/dx_k * qvel[k] + sum_{k=3}^{5} dM/dq_k * dq_k/dt
    # The quaternion derivatives make this messy.

    print("\n  Computing via direct rotation parameterization...")

    # Instead, I'll use RPY as local coordinates and compute M(RPY)
    # This is valid near identity where RPY ~ angular velocity integral

    def M_at_rpy(roll, pitch, yaw):
        """Compute M at orientation given by RPY angles."""
        from scipy.spatial.transform import Rotation
        R = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
        quat = Rotation.from_matrix(R).as_quat()  # [x,y,z,w]
        quat_mjc = np.array([quat[3], quat[0], quat[1], quat[2]])  # [w,x,y,z]
        qpos = qpos_base.copy()
        qpos[3:7] = quat_mjc / np.linalg.norm(quat_mjc)
        return np.array(jax_mass_matrix(jnp.array(qpos, dtype=jnp.float32), mm_constants))

    M0 = M_at_rpy(0, 0, 0)
    h_rpy = 1e-4
    dM_droll = (M_at_rpy(h_rpy, 0, 0) - M_at_rpy(-h_rpy, 0, 0)) / (2 * h_rpy)
    dM_dpitch = (M_at_rpy(0, h_rpy, 0) - M_at_rpy(0, -h_rpy, 0)) / (2 * h_rpy)
    dM_dyaw = (M_at_rpy(0, 0, h_rpy) - M_at_rpy(0, 0, -h_rpy)) / (2 * h_rpy)

    dM_dtheta = [dM_droll, dM_dpitch, dM_dyaw]
    theta_labels = ['roll', 'pitch', 'yaw']

    # dM/dt @ qvel = sum_i dM/dtheta_i * omega_i (since orientation drives all M dependence)
    # + sum_i dM/dx_i * v_i (but M doesn't depend on position)
    Mdot_qvel_from_theta = (dM_droll * qvel_test[3] + dM_dpitch * qvel_test[4] +
                             dM_dyaw * qvel_test[5]) @ qvel_test

    # dT/dtheta_i = 1/2 qvel^T @ dM/dtheta_i @ qvel
    dT_dtheta = np.zeros(3)
    for i in range(3):
        dT_dtheta[i] = 0.5 * qvel_test @ dM_dtheta[i] @ qvel_test

    print(f"  Mdot @ qvel (from RPY diff) [0:6]: {Mdot_qvel_from_theta[0:6]}")
    print(f"  dT/droll:  {dT_dtheta[0]:.6f}")
    print(f"  dT/dpitch: {dT_dtheta[1]:.6f}")
    print(f"  dT/dyaw:   {dT_dtheta[2]:.6f}")

    # ═══════════════════════════════════════════════════════════════════
    # Step 5: Coriolis force in generalized velocity coordinates
    # ═══════════════════════════════════════════════════════════════════
    print("\n--- Step 5: Coriolis force (C @ qvel) ---")
    # C_i = Mdot_qvel_i - dT/dq_i (for i in position coordinates)
    # For translational DOFs: dT/dx = dT/dy = dT/dz = 0
    # C[0:3] = Mdot_qvel[0:3] - 0 = Mdot_qvel[0:3]
    # For rotational DOFs: C[3:6] = Mdot_qvel[3:6] - dT/dtheta

    C_lagrange = np.zeros(nv)
    C_lagrange[0:3] = Mdot_qvel_from_theta[0:3]  # translational
    C_lagrange[3:6] = Mdot_qvel_from_theta[3:6] - dT_dtheta  # rotational
    C_lagrange[6:16] = Mdot_qvel_from_theta[6:16]  # actuated (dT/dq_actuated = 0 at identity)

    print(f"  Lagrange C@qvel [0:3] (translational): {C_lagrange[0:3]}")
    print(f"  Lagrange C@qvel [3:6] (rotational):    {C_lagrange[3:6]}")
    print(f"  Lagrange C@qvel [6:16] (actuated):     {C_lagrange[6:16]}")

    # ═══════════════════════════════════════════════════════════════════
    # Step 6: Compare all three methods
    # ═══════════════════════════════════════════════════════════════════
    print("\n--- Step 6: Three-way comparison ---")

    # MuJoCo velocity-dependent bias
    cpu_full = _cpu_bias(model, qpos_base, qvel_test)
    cpu_grav = _cpu_bias(model, qpos_base, np.zeros(nv))
    cpu_vel = cpu_full - cpu_grav

    # JAX RNEA
    jax_full = np.array(jax_bias_forces(qpos_jax, jnp.array(qvel_test, dtype=jnp.float32), constants), dtype=np.float64)
    jax_grav = np.array(jax_gravity_forces(qpos_jax, constants), dtype=np.float64)
    jax_vel = jax_full - jax_grav

    print(f"  {'Component':<20} {'MuJoCo':>12} {'JAX RNEA':>12} {'Lagrange':>12} {'JAX-MJC':>10} {'Lag-MJC':>10}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

    names = ["qfrc[0] (vx)", "qfrc[1] (vy)", "qfrc[2] (vz)",
             "qfrc[3] (wx)", "qfrc[4] (wy)", "qfrc[5] (wz)"]
    for i in range(6):
        diff_jax = jax_vel[i] - cpu_vel[i]
        diff_lag = C_lagrange[i] - cpu_vel[i]
        print(f"  {names[i]:<20} {cpu_vel[i]:>12.6f} {jax_vel[i]:>12.6f} {C_lagrange[i]:>12.6f} {diff_jax:>10.2e} {diff_lag:>10.2e}")

    print()
    for i in range(6, 16):
        diff_jax = jax_vel[i] - cpu_vel[i]
        if i == 6:
            print(f"  {'actuated[6:16]':<20} {cpu_vel[i]:>12.6f} {jax_vel[i]:>12.6f} {C_lagrange[i]:>12.6f} {diff_jax:>10.2e}")
        else:
            print(f"  {'':<20} {cpu_vel[i]:>12.6f} {jax_vel[i]:>12.6f} {C_lagrange[i]:>12.6f} {diff_jax:>10.2e}")

    # ═══════════════════════════════════════════════════════════════════
    # Step 7: KEY INSIGHT
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("KEY QUESTION: Does the Lagrange method match MuJoCo?")
    print("=" * 72)
    lag_mjc_err = np.max(np.abs(C_lagrange - cpu_vel))
    print(f"  Max Lagrange vs MuJoCo error: {lag_mjc_err:.2e}")
    if lag_mjc_err < 1e-3:
        print("  --> Lagrange method MATCHES MuJoCo (to 1e-3)")
        print("  --> The mass matrix dynamics are correctly computed by Lagrange")
        print("  --> Our RNEA must be producing a DIFFERENT spatial force than")
        print("      what the mass matrix dynamics predict for the free joint")
    else:
        print("  --> Lagrange method does NOT match MuJoCo")
        print(f"  --> Either the mass matrix is wrong, or the dM/dtheta computation is")
        print(f"      inaccurate (finite difference error at h={h_rpy})")

    # Check: is the JAX RNEA error in the free-base force equal to what we predict?
    print(f"\n  JAX RNEA vel bias [1] (vy error): {jax_vel[1]:.6f}")
    print(f"  Expected zero from MuJoCo: {cpu_vel[1]:.6f}")
    print(f"  Lagrange prediction: {C_lagrange[1]:.6f}")

    # ═══════════════════════════════════════════════════════════════════
    # Step 8: Check if mass matrix time-derivative matches RNEA expectation
    # ═══════════════════════════════════════════════════════════════════
    print("\n--- Step 8: M-dot consistency check ---")
    # If M is correct and Lagrange is correct, C@qvel from Lagrange should
    # equal MuJoCo. Let's verify M-dot against MuJoCo's internal computation.

    # Use mj_inverse with qacc != 0 to extract Mdot
    # mj_inverse(qvel, qacc) = M@qacc + bias(qvel) + passive
    # For qvel=0: mj_inverse(0, qacc) = M(q0)@qacc + bias(0) = M(q0)@qacc + grav
    # For qvel=v, qacc=0: mj_inverse(v, 0) = bias(v) = C(v)v + grav + passive

    # So velocity-dependent bias = mj_inverse(v, 0) - mj_inverse(0, 0)
    # And Mdot_qvel = d/dt(M qvel) at t=0

    # Let's verify our Lagrange method with a smaller test
    print("  Verifying Lagrange with central finite difference of M along trajectory...")

    # Compute M at q(-dt) and q(+dt) along the trajectory
    dt2 = 0.001
    dq_fwd = np.zeros(nq)
    dq_fwd[0:3] = qvel_test[0:3] * dt2
    dq_fwd[3:7] = 0.5 * np.array([0.0, qvel_test[3], qvel_test[4], qvel_test[5]]) * dt2
    q_fwd = qpos_base + dq_fwd
    q_fwd[3:7] /= np.linalg.norm(q_fwd[3:7])

    dq_bwd = np.zeros(nq)
    dq_bwd[0:3] = -qvel_test[0:3] * dt2
    dq_bwd[3:7] = -0.5 * np.array([0.0, qvel_test[3], qvel_test[4], qvel_test[5]]) * dt2
    q_bwd = qpos_base + dq_bwd
    q_bwd[3:7] /= np.linalg.norm(q_bwd[3:7])

    M_fwd = np.array(jax_mass_matrix(jnp.array(q_fwd, dtype=jnp.float32), mm_constants))
    M_bwd = np.array(jax_mass_matrix(jnp.array(q_bwd, dtype=jnp.float32), mm_constants))

    # M_dot_qvel_traj = (M_fwd @ qvel_test - M_bwd @ qvel_test) / (2 * dt2)
    # But wait, qvel changes along the trajectory too (for non-zero qacc).
    # For qacc=0, qvel is constant.
    Mdot_qvel_traj = (M_fwd - M_bwd) @ qvel_test / (2 * dt2)

    print(f"  Mdot@qvel from trajectory:   {Mdot_qvel_traj[0:6]}")
    print(f"  Mdot@qvel from RPY diff:     {Mdot_qvel_from_theta[0:6]}")
    print(f"  Diff: {np.max(np.abs(Mdot_qvel_traj[0:6] - Mdot_qvel_from_theta[0:6])):.2e}")

    # The Lagrangian C at identity:
    # C_i = Mdot_qvel_i - dT/dq_i  (but dT/dq_i is in q-coordinates, not v-coordinates)
    # For translational DOFs (0,1,2): dT/dx_i = 0
    # For rotational DOFs (3,4,5): need dT/dtheta_i
    # The issue is the coordinate representation

    return 0


if __name__ == "__main__":
    sys.exit(main())
