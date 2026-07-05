#!/usr/bin/env python
"""Phase 2C.3 — Diagnostic Probe: MuJoCo Free-Joint Convention Investigation

Probes:
  A. Floating body only, no child joint velocities
  B. Base linear velocity only: vx, vy, vz
  C. Base angular velocity only: wx, wy, wz
  D. Base angular + base linear pairs (9 combos)
  E. Non-identity base orientation with same velocity pairs
  F. Cross-term computation with detailed component breakdown
  G. Spatial force origin investigation
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wheeled_biped.dynamics.jax_bias_forces import (
    build_bias_force_constants,
    extract_jax_bias_arrays,
    extract_jax_fk_arrays,
    jax_bias_forces,
    jax_bias_forces_fk_arrays,
    jax_gravity_forces,
    jax_velocity_bias_forces,
    _crm, _crf, _skew3, _body_local_spatial_inertia,
    _motion_xup, _quat_to_rotmat, _axis_angle_to_rotmat,
    _jax_rnea_bias_body_local,
    rnea_body_local,
)
from wheeled_biped.utils.config import get_model_path


def _v(idx, val):
    arr = np.zeros(16)
    arr[idx] = val
    return arr


def _cpu_bias(model, qpos_np, qvel_np):
    d = mujoco.MjData(model)
    d.qpos[:] = qpos_np
    d.qvel[:] = qvel_np
    mujoco.mj_forward(model, d)
    return np.array(d.qfrc_bias, dtype=np.float64)


def _jax_bias(qpos_jax, qvel_np, constants):
    qvel_jax = jnp.array(qvel_np, dtype=jnp.float32)
    return np.array(jax_bias_forces(qpos_jax, qvel_jax, constants), dtype=np.float64)


def _quat_from_rpy(roll, pitch, yaw):
    """Convert RPY to quaternion (w,x,y,z)."""
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return np.array([w, x, y, z])


def _set_base_orientation(qpos_np, roll, pitch, yaw):
    q = qpos_np.copy()
    q[3:7] = _quat_from_rpy(np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw))
    return q


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
    fk_arrays = extract_jax_fk_arrays(constants)
    bias_arrays_full = extract_jax_bias_arrays(constants)
    _, *bias_rest = bias_arrays_full
    bias_arrays = tuple(bias_rest)

    # Base pose (identity orientation, zero height offset)
    qpos_base = data.qpos.copy()
    # Set identity quaternion
    qpos_base[3:7] = [1.0, 0.0, 0.0, 0.0]
    qpos_jax = jnp.array(qpos_base, dtype=jnp.float32)

    print("=" * 80)
    print("PHASE 2C.3 — MUJOCO FREE-JOINT CONVENTION DIAGNOSTIC")
    print("=" * 80)
    print(f"Model: nq={nq}, nv={nv}")
    print(f"Torso position (world): {qpos_base[0:3]}")
    print(f"Torso quaternion: {qpos_base[3:7]}")

    # ── A. Floating body, qvel=0 ─────────────────────────────────────────
    print("\n--- A. Zero velocity at identity ---")
    cpu0 = _cpu_bias(model, qpos_base, np.zeros(nv))
    jax0 = _jax_bias(qpos_jax, np.zeros(nv), constants)
    print(f"  CPU gravity[0:3] (force):  {cpu0[0:3]}")
    print(f"  JAX gravity[0:3] (force):  {jax0[0:3]}")
    print(f"  CPU gravity[3:6] (torque): {cpu0[3:6]}")
    print(f"  JAX gravity[3:6] (torque): {jax0[3:6]}")
    print(f"  Max gravity error: {np.max(np.abs(jax0 - cpu0)):.2e}")

    # Verify gravity is about COM, not body origin
    total_mass = float(np.sum(np.array(constants["body_mass"])))
    expected_grav_force = np.array([0.0, 0.0, total_mass * 9.81])
    print(f"  Expected gravity force (total mass × g): {expected_grav_force}")
    print(f"  CPU z-force / expected: {cpu0[2] / expected_grav_force[2]:.6f}")

    # ── B. Pure base linear velocity ─────────────────────────────────────
    print("\n--- B. Pure base linear velocity ---")
    for name, idx in [("vx", 0), ("vy", 1), ("vz", 2)]:
        qvel = np.zeros(nv); qvel[idx] = 1.0
        cpu = _cpu_bias(model, qpos_base, qvel)
        jax = _jax_bias(qpos_jax, qvel, constants)
        err = float(np.max(np.abs(jax - cpu)))
        print(f"  {name}=1.0: max_err={err:.2e}  CPU free-base[0:3]={cpu[0:3]} [3:6]={cpu[3:6]}")

    # ── C. Pure base angular velocity ────────────────────────────────────
    print("\n--- C. Pure base angular velocity ---")
    for name, idx in [("wx (roll)", 3), ("wy (pitch)", 4), ("wz (yaw)", 5)]:
        qvel = np.zeros(nv); qvel[idx] = 1.0
        cpu = _cpu_bias(model, qpos_base, qvel)
        jax = _jax_bias(qpos_jax, qvel, constants)
        err = float(np.max(np.abs(jax - cpu)))
        print(f"  {name}=1.0: max_err={err:.2e}  CPU free-base[0:3]={cpu[0:3]} [3:6]={cpu[3:6]}")

    # ── D. Base angular + linear pairs ───────────────────────────────────
    print("\n--- D. Base angular + linear velocity pairs ---")
    ang_pairs = [(3, "wx"), (4, "wy"), (5, "wz")]
    lin_pairs = [(0, "vx"), (1, "vy"), (2, "vz")]
    for ai, an in ang_pairs:
        for li, ln in lin_pairs:
            # Combined
            qvel_c = np.zeros(nv); qvel_c[ai] = 1.0; qvel_c[li] = 1.0
            cpu_c = _cpu_bias(model, qpos_base, qvel_c)
            jax_c = _jax_bias(qpos_jax, qvel_c, constants)

            # Individual
            qvel_a = np.zeros(nv); qvel_a[ai] = 1.0
            qvel_l = np.zeros(nv); qvel_l[li] = 1.0
            cpu_a = _cpu_bias(model, qpos_base, qvel_a)
            cpu_l = _cpu_bias(model, qpos_base, qvel_l)
            jax_a = _jax_bias(qpos_jax, qvel_a, constants)
            jax_l = _jax_bias(qpos_jax, qvel_l, constants)

            # Cross-term
            cpu_cross = cpu_c - cpu_a - cpu_l + cpu0
            jax_cross = jax_c - jax_a - jax_l + jax0
            cross_err = float(np.max(np.abs(jax_cross - cpu_cross)))
            full_err = float(np.max(np.abs(jax_c - cpu_c)))

            print(f"  {an}+{ln}: full_err={full_err:.2e} "
                  f"cpu_cross[0:3]={np.max(np.abs(cpu_cross[0:3])):.2e} "
                  f"cpu_cross[3:6]={np.max(np.abs(cpu_cross[3:6])):.2e} "
                  f"jax_cross[0:3]={np.max(np.abs(jax_cross[0:3])):.2e} "
                  f"jax_cross[3:6]={np.max(np.abs(jax_cross[3:6])):.2e} "
                  f"cross_err={cross_err:.2e}")

    # ── E. Non-identity orientation ──────────────────────────────────────
    print("\n--- E. Non-identity base orientation ---")
    orientations = [
        ("identity", 0, 0, 0),
        ("roll_10deg", 10, 0, 0),
        ("pitch_10deg", 0, 10, 0),
        ("yaw_15deg", 0, 0, 15),
        ("combined_small", 5, 8, 12),
    ]

    for oname, roll, pitch, yaw in orientations:
        qpos_rot = _set_base_orientation(qpos_base, roll, pitch, yaw)
        qpos_rot_jax = jnp.array(qpos_rot, dtype=jnp.float32)

        # Zero velocity (gravity)
        cpu_g = _cpu_bias(model, qpos_rot, np.zeros(nv))
        jax_g = _jax_bias(qpos_rot_jax, np.zeros(nv), constants)
        grav_err = float(np.max(np.abs(jax_g - cpu_g)))

        # wz + vx combined
        qvel_c = np.zeros(nv); qvel_c[5] = 1.0; qvel_c[0] = 1.0
        cpu_c = _cpu_bias(model, qpos_rot, qvel_c)
        jax_c = _jax_bias(qpos_rot_jax, qvel_c, constants)
        full_err = float(np.max(np.abs(jax_c - cpu_c)))

        # Cross term for wz + vx
        cpu_a = _cpu_bias(model, qpos_rot, _v(5, 1.0))
        cpu_l = _cpu_bias(model, qpos_rot, _v(0, 1.0))
        jax_a = _jax_bias(qpos_rot_jax, _v(5, 1.0), constants)
        jax_l = _jax_bias(qpos_rot_jax, _v(0, 1.0), constants)
        cpu_cross = cpu_c - cpu_a - cpu_l + cpu_g
        jax_cross = jax_c - jax_a - jax_l + jax_g
        cross_err = float(np.max(np.abs(jax_cross - cpu_cross)))

        # Pure wz
        cpu_wz = _cpu_bias(model, qpos_rot, _v(5, 1.0))
        jax_wz = _jax_bias(qpos_rot_jax, _v(5, 1.0), constants)
        wz_err = float(np.max(np.abs(jax_wz - cpu_wz)))

        # Pure vx
        cpu_vx = _cpu_bias(model, qpos_rot, _v(0, 1.0))
        jax_vx = _jax_bias(qpos_rot_jax, _v(0, 1.0), constants)
        vx_err = float(np.max(np.abs(jax_vx - cpu_vx)))

        print(f"  {oname}: grav={grav_err:.2e} wz={wz_err:.2e} vx={vx_err:.2e} "
              f"cross={cross_err:.2e} full={full_err:.2e}")

    # ── F. Detailed cross-term with per-body force breakdown ─────────────
    print("\n--- F. Detailed cross-term investigation ---")
    print("Comparing spatial force at each body for wz=1, vx=1, and wz+vx=1 at identity")

    # Run RNEA and extract per-body forces for diagnostic
    qvel_wz = _v(5, 1.0)
    qvel_vx = _v(0, 1.0)
    qvel_both = np.zeros(16); qvel_both[5] = 1.0; qvel_both[0] = 1.0

    # Full system (all bodies)
    cpu_both = _cpu_bias(model, qpos_base, qvel_both)
    cpu_wz_b = _cpu_bias(model, qpos_base, qvel_wz)
    cpu_vx_b = _cpu_bias(model, qpos_base, qvel_vx)
    cpu_zero = _cpu_bias(model, qpos_base, np.zeros(16))
    cpu_cross = cpu_both - cpu_wz_b - cpu_vx_b + cpu_zero

    print(f"  CPU cross-term free-base[0:3] (force):  {cpu_cross[0:3]}")
    print(f"  CPU cross-term free-base[3:6] (torque): {cpu_cross[3:6]}")
    print(f"  CPU cross-term ||free-base||: {np.linalg.norm(cpu_cross[0:6]):.6f}")

    # JAX side
    jax_both = _jax_bias(qpos_jax, qvel_both, constants)
    jax_wz_b = _jax_bias(qpos_jax, qvel_wz, constants)
    jax_vx_b = _jax_bias(qpos_jax, qvel_vx, constants)
    jax_zero = _jax_bias(qpos_jax, np.zeros(16), constants)
    jax_cross = jax_both - jax_wz_b - jax_vx_b + jax_zero

    print(f"  JAX cross-term free-base[0:3] (force):  {jax_cross[0:3]}")
    print(f"  JAX cross-term free-base[3:6] (torque): {jax_cross[3:6]}")
    print(f"  JAX cross-term ||free-base||: {np.linalg.norm(jax_cross[0:6]):.6f}")
    print(f"  Cross-term error (JAX-CPU): {np.max(np.abs(jax_cross - cpu_cross)):.2e}")

    # ── G. Test: is the free-base cross-term error from torso or children? ──
    print("\n--- G. Torso-only vs full system ---")
    # Disable all child joints by setting qvel[6:16] = 0 and also
    # checking the torso body's contribution in isolation

    # Get torso body local spatial inertia
    I_body_local = np.array(constants["I_body_local"])
    print(f"  Torso (body 1) spatial inertia shape: {I_body_local[1].shape}")
    print(f"  Torso mass: {float(np.array(constants['body_mass'])[1]):.4f} kg")
    print(f"  Torso COM in body frame (body_ipos): {np.array(constants['body_ipos'])[1]}")
    print(f"  Torso body inertia diag: {np.array(constants['body_inertia'])[1]}")

    # Check: for a single rigid body, qfrc_bias cross-term
    # The spatial Coriolis force is: crf(v) @ I @ v
    # For v = [R^T@ω; R^T@v_lin] at the body origin
    v_wz_local = np.concatenate([np.zeros(3), np.array([1.0, 0.0, 0.0])])  # R^T@[0,0,1] = [0,0,1], R^T@[0,0,0] = [0,0,0] → v_x
    # Actually let me be precise. At identity orientation, R=I, so:
    # qvel_wz = [0,0,0, 0,0,1, ...] → ω=[0,0,1], v_lin=[0,0,0]
    # qvel_vx = [1,0,0, 0,0,0, ...] → v_lin=[1,0,0], ω=[0,0,0]
    # v_wz_local = [R^T@ω; R^T@v_lin] = [0,0,1; 0,0,0]
    # v_vx_local = [R^T@ω; R^T@v_lin] = [0,0,0; 1,0,0]
    # v_both_local = [0,0,1; 1,0,0]

    I_torso = I_body_local[1]
    v_wz = np.array([0, 0, 1, 0, 0, 0], dtype=np.float64)
    v_vx = np.array([0, 0, 0, 1, 0, 0], dtype=np.float64)
    v_both = v_wz + v_vx

    # Coriolis force for single body at zero acceleration:
    # F = crf(v) @ I @ v
    # For bias: a = [0; -R^T@g] = [0,0,0; 0,0,g] since g = [0,0,-9.81], -R^T@g = [0,0,9.81]
    a_grav = np.array([0, 0, 0, 0, 0, 9.81], dtype=np.float64)

    def single_body_bias(I, v, a):
        """Bias force for single rigid body: I@a + crf(v)@I@v"""
        crf_v = np.zeros((6, 6))
        w = v[0:3]; vl = v[3:6]
        crf_v[0:3, 0:3] = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        crf_v[0:3, 3:6] = np.array([[0, -vl[2], vl[1]], [vl[2], 0, -vl[0]], [-vl[1], vl[0], 0]])
        crf_v[3:6, 3:6] = crf_v[0:3, 0:3]
        return I @ a + crf_v @ I @ v

    F_wz_single = single_body_bias(I_torso, v_wz, a_grav)
    F_vx_single = single_body_bias(I_torso, v_vx, a_grav)
    F_both_single = single_body_bias(I_torso, v_both, a_grav)
    F_zero_single = single_body_bias(I_torso, np.zeros(6), a_grav)
    cross_single = F_both_single - F_wz_single - F_vx_single + F_zero_single

    print(f"  Single-body cross-term [torque; force]:")
    print(f"    torque part: {cross_single[0:3]}")
    print(f"    force part:  {cross_single[3:6]}")
    print(f"    norm: {np.linalg.norm(cross_single):.6f}")

    # The single-body cross term in spatial force [tau; f]:
    # tau_cross = ω × (I_origin @ v_lin) + ω × (m*c × ...)  + ...
    # f_cross = ω × (m * v_lin)
    # So f_cross should have magnitude ~ m * |ω| * |v_lin| = mass * 1 * 1
    mass_torso = float(np.array(constants['body_mass'])[1])
    print(f"  Expected f_cross magnitude (m * ω * v): ~{mass_torso:.4f}")
    print(f"  Actual f_cross: {cross_single[3:6]}")

    # Now compare with full-system RNEA cross-term
    # The question: does the full-system cross-term match the single-body cross-term?
    print(f"\n  Full-system CPU cross-term force:  {cpu_cross[0:3]}")
    print(f"  Full-system CPU cross-term torque: {cpu_cross[3:6]}")
    print(f"  Single-body cross-term force (world frame = local since identity): {cross_single[3:6]}")
    print(f"  Single-body cross-term torque (world frame = local since identity): {cross_single[0:3]}")

    # ── H. Key insight: check if qvel[0:3] is velocity of body origin or COM ──
    print("\n--- H. Body origin vs COM velocity diagnostic ---")
    # If qvel[0:3] is body origin velocity:
    #   v_COM = v_origin + ω × com_offset
    # If qvel[0:3] is COM velocity:
    #   v_origin = v_COM - ω × com_offset

    com_offset = np.array(constants["body_ipos"])[1]  # torso COM in body frame
    print(f"  Torso COM offset (body frame): {com_offset}")

    # At identity orientation: body frame = world frame
    # qvel_wz gives ω = [0,0,1], v_lin = [0,0,0]
    # If qvel is body origin velocity, COM velocity = [0,0,0] + ω × com
    # ω × com_offset = [0,0,1] × [com_x, com_y, com_z]
    #                = [-com_y, com_x, 0]
    com_vel_if_origin = np.cross([0, 0, 1], com_offset)
    print(f"  If qvel=body origin vel: COM lateral vel for wz=1 = {com_vel_if_origin}")

    # Test: check CPU qfrc_bias for pure base angular velocity
    # If the kinetic energy is T = 1/2 m |v_COM|^2 + 1/2 ω^T I_cm ω
    # and v_COM depends on v_origin and ω, then the bias force should reflect this.
    # Let's check if the cross-term vanishes when we use COM velocity.
    print("\n  --- Testing COM-velocity hypothesis ---")
    # Hypothesis: MuJoCo qvel[0:3] is body ORIGIN velocity
    # Test: compute kinetic energy components and verify

    # For pure wz at identity:
    # If qvel[0:3] = v_origin:
    #   v_COM_world = [0,0,0] + [0,0,1] × com_offset  (since R=I)
    #   T_lin = 1/2 m |v_COM|^2 = 1/2 m |com_x^2 + com_y^2| * 1^2
    #   T_rot = 1/2 ω^T I_cm ω

    # For pure vx at identity:
    # If qvel[0:3] = v_origin:
    #   v_COM_world = [1,0,0]
    #   T_lin = 1/2 m * 1^2

    # The cross-term in the EOM comes from d/dt(∂T/∂q̇) - ∂T/∂q
    # For a free joint: M @ q̈ + C @ q̇ = τ
    # C_ij = Σ Γ_ijk q̇_k

    # The key question: does MuJoCo's free joint motion subspace
    # map qvel to spatial velocity of body origin or COM?

    # Let's use a different test: force application
    # Apply xfrc_applied at the COM and check qfrc_actuator
    print("  Force-at-origin test:")
    d_test = mujoco.MjData(model)
    d_test.qpos[:] = qpos_base
    mujoco.mj_forward(model, d_test)
    # Apply unit force at body 1 (torso) COM via xfrc_applied
    d_test.xfrc_applied[1, 0:3] = [1.0, 0.0, 0.0]  # force at COM
    mujoco.mj_inverse(model, d_test)
    qfrc_from_force = np.array(d_test.qfrc_inverse)
    print(f"  Unit force [1,0,0] at torso COM → qfrc_inverse[0:6]: {qfrc_from_force[0:6]}")

    # Now test force at body origin
    d_test2 = mujoco.MjData(model)
    d_test2.qpos[:] = qpos_base
    mujoco.mj_forward(model, d_test2)
    # Apply force at body origin by computing offset
    com_world = d_test2.xpos[1] + d_test2.xmat[1].reshape(3,3) @ com_offset  # COM in world
    origin_world = d_test2.xpos[1]
    d_test2.xfrc_applied[1, 0:3] = [1.0, 0.0, 0.0]  # at COM
    # The equivalent force at origin produces additional torque: r × f
    r_com_to_origin = origin_world - com_world
    # Actually, xfrc_applied is at COM. For unit force at COM:
    # qfrc_translational = force (since xfrc is at COM, the generalized force is: J^T @ F)
    # The translational Jacobian for COM is different from body origin.
    print(f"  Body origin in world: {origin_world}")
    print(f"  COM in world: {com_world}")
    print(f"  COM→origin vector: {r_com_to_origin}")

    # Let's try a different approach: apply force at body origin by setting it at COM
    # and accounting for the torque
    # Actually, the simplest test is to check qfrc_bias for pure vx.
    # If the kinetic energy uses COM velocity:
    #   T = 1/2 m (v_origin + ω × com)^2 + 1/2 ω^T I_cm ω
    #   = 1/2 m v_origin^2 + 1/2 ω^T (I_cm + m skew(com) skew(com)^T) ω + m v_origin^T (ω × com)
    # The last term is the coupling: L = m v_origin × com · ω

    # For the EOM: d/dt(∂T/∂v_origin) - ∂T/∂x = m a_origin + m ω̇ × com  (Coriolis for origin→COM)
    # This will produce coupling in the acceleration terms (mass matrix), not in the bias forces.
    # The bias force coupling through C(q̇)q̇ comes from the Christoffel symbols,
    # which for the kinetic energy above would involve terms like m ω × (v_origin × com)

    # Actually, I think the answer is simpler. Let me just compute the kinetic energy
    # decomposition numerically by querying the mass matrix.

    print("\n  --- Mass matrix verification ---")
    from wheeled_biped.dynamics.jax_mass_matrix import jax_mass_matrix
    M = np.array(jax_mass_matrix(qpos_jax, constants))
    print(f"  M[0:6, 0:6] =\n{M[0:6, 0:6]}")
    print(f"  M[0:3, 3:6] (linear×angular coupling):\n{M[0:3, 3:6]}")

    # The mass matrix block M[0:3, 3:6] is m * skew(com)
    # This is the coupling between linear acceleration and angular acceleration
    # at the BODY ORIGIN level (not COM level).

    # For a single rigid body with spatial velocity at body origin:
    # M_spatial = [[I_origin, m*skew(com)], [m*skew(com)^T, m*I]]
    # In generalized coordinates with R=I:
    # M[3:6, 3:6] = I_origin (inertia about body origin)
    # M[0:3, 3:6] = m * skew(com)  (coupling)
    # M[0:3, 0:3] = m * I
    # M[3:6, 0:3] = -m * skew(com)

    # Check if this matches
    print(f"  Expected M[0:3, 3:6] (= m*skew(com)):\n{mass_torso * np.array([[0, -com_offset[2], com_offset[1]], [com_offset[2], 0, -com_offset[0]], [-com_offset[1], com_offset[0], 0]])}")
    print(f"  Actual top-left 6×6 norm: {np.linalg.norm(M[0:6, 0:6]):.4f}")

    # ── I. Test torso-only (remove all children) ──────────────────────────
    print("\n--- I. Torso-only floating body test ---")
    # The key insight: if we isolate the torso from all children,
    # does the single-body bias force cross-term match the full system?

    # We can't easily remove children in MuJoCo, but we can compare:
    # 1. Full system bias with qvel only on free-base DOFs (children stationary)
    # 2. Theoretical single-body bias

    # For a single floating body at identity orientation:
    # qfrc_bias = [f_world; τ_world]
    # where [τ_world; f_world] = I @ [0; R^T @ g] + v ×* I @ v
    # (with spatial convention [angular; linear])

    # Since R=I at identity:
    # a_grav_body_local = [0, 0, 0; 0, 0, 9.81] = [0,0,0, 0,0,9.81]

    # For pure wz: v_body_local = [0,0,1; 0,0,0]
    # Coriolis: v ×* I @ v
    # = [[skew(ω), skew(v_lin)], [0, skew(ω)]] @ I @ [ω; v_lin]
    # For purely angular velocity:
    # I @ v = I @ [ω; 0] = [I_origin @ ω; m * skew(com)^T @ ω]
    # v ×* I @ v = [[skew(ω), 0], [0, skew(ω)]] @ [I_origin @ ω; m * skew(com)^T @ ω]
    # = [skew(ω) @ I_origin @ ω; skew(ω) @ m * skew(com)^T @ ω]

    # Total spatial force = I @ a + v ×* I @ v
    # = [I_origin @ 0 + m*skew(com) @ [0,0,9.81] + skew(ω) @ I_origin @ ω;
    #    m*skew(com)^T @ 0 + m*[0,0,9.81] + skew(ω) @ m*skew(com)^T @ ω]

    # The pure wz bias should NOT have a force term [f_world] from gravity (since a_z in body = g_z).
    # Wait: a_grav[3:6] = -R^T @ gravity = -[0,0,-9.81] = [0,0,9.81] for identity
    # I @ a_grav = [I_origin @ 0 + m*skew(com) @ [0,0,9.81];
    #               m*skew(com)^T @ 0 + m @ [0,0,9.81]]
    # = [m * [com_y*g, -com_x*g, 0];  [0, 0, m*g]]

    # So gravity produces: torque = m*g*[com_y, -com_x, 0], force = [0, 0, m*g]
    # This is the gravity wrench about the body origin. Correct.

    # For the velocity part (pure wz):
    # v ×* I @ v = [ω × I_origin @ ω; ω × (m * skew(com)^T @ ω)]
    # ω × (I_origin @ ω): centrifugal torque (nonzero for asymmetric bodies)
    # ω × (m * skew(com)^T @ ω):
    #   skew(com)^T @ ω = [-com × ω]
    #   ω × (m * (-com × ω)) = -m * ω × (com × ω) = -m * (com*(ω·ω) - ω*(com·ω))
    #   For ω = [0,0,wz]: ω × (com × ω) = -wz^2 * [com_x, com_y, 0]
    #   So ω × (skew(com)^T @ ω) = m * wz^2 * [com_x, com_y, 0]

    # This gives a nonzero force at the body origin (centrifugal effect).

    # Now for pure vx: v_body_local = [0,0,0; 1,0,0]
    # v ×* I @ v = [[0, skew(v_lin); 0, 0]] @ I @ [0; v_lin]
    #            = [skew(v_lin) @ m @ v_lin; 0]
    #            = [0; 0]  (since v_lin × v_lin = 0 for any v_lin aligned with itself)

    # Wait that's not right. Let me recompute.
    # I @ v = [I_origin @ 0 + m*skew(com) @ v_lin; m*skew(com)^T @ 0 + m @ v_lin]
    # = [m*skew(com) @ v_lin; m @ v_lin]
    # v ×* I @ v = [[0, skew(v_lin)], [0, 0]] @ [m*skew(com) @ v_lin; m @ v_lin]
    # = [skew(v_lin) @ m @ v_lin; 0]

    # skew(v_lin) @ v_lin = v_lin × v_lin = 0!
    # So the pure linear velocity bias force is ZERO (only gravity remains).

    # For the combined case (wz + vx):
    # v = [0,0,1; 1,0,0]
    # I @ v = [I_origin @ [0,0,1] + m*skew(com) @ [1,0,0]; m*skew(com)^T @ [0,0,1] + m @ [1,0,0]]
    # Let's just compute this numerically...

    print("  Computing single-body bias forces numerically...")
    # Use the same I_torso, but as a complete spatial inertia
    I_t = I_torso  # 6×6 about torso body origin, torso-local frame

    # Gravity acceleration
    grav = np.array(constants["gravity"], dtype=np.float64)  # [0, 0, -9.81]
    R = np.eye(3)  # identity
    a_torso = np.concatenate([np.zeros(3), -R.T @ grav])  # [0,0,0, 0,0,9.81]

    def torso_single_body_bias(qvel_06):
        """Compute single-body bias force at torso body origin."""
        # qvel_06: [vx,vy,vz, wx,wy,wz] — MuJoCo ordering
        v_lin_world = qvel_06[0:3]
        w_world = qvel_06[3:6]
        # Transform to body-local
        v_lin_local = R.T @ v_lin_world
        w_local = R.T @ w_world
        v_local = np.concatenate([w_local, v_lin_local])

        # Bias spatial force: I @ a + crf(v) @ I @ v
        Ia = I_t @ a_torso
        Iv = I_t @ v_local
        crf_v = np.zeros((6, 6))
        w = v_local[0:3]; vl = v_local[3:6]
        w_skew = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        vl_skew = np.array([[0, -vl[2], vl[1]], [vl[2], 0, -vl[0]], [-vl[1], vl[0], 0]])
        crf_v[0:3, 0:3] = w_skew
        crf_v[0:3, 3:6] = vl_skew
        crf_v[3:6, 3:6] = w_skew
        crf_Iv = crf_v @ Iv

        F_local = Ia + crf_Iv  # spatial force [torque; force] at body origin, body-local

        # Transform to world frame, then to MuJoCo qfrc ordering
        F_world = np.concatenate([R @ F_local[0:3], R @ F_local[3:6]])
        qfrc = np.zeros(16)
        qfrc[0:3] = F_world[3:6]  # force
        qfrc[3:6] = F_world[0:3]  # torque
        return qfrc

    s_wz = torso_single_body_bias(np.array([0, 0, 0, 0, 0, 1.0]))
    s_vx = torso_single_body_bias(np.array([1, 0, 0, 0, 0, 0.0]))
    s_both = torso_single_body_bias(np.array([1, 0, 0, 0, 0, 1.0]))
    s_zero = torso_single_body_bias(np.zeros(6))
    s_cross = s_both - s_wz - s_vx + s_zero

    print(f"  Single-body theory: cross[0:3] (force) = {s_cross[0:3]}")
    print(f"  Single-body theory: cross[3:6] (torque) = {s_cross[3:6]}")
    print(f"  CPU full system: cross[0:3] (force) = {cpu_cross[0:3]}")
    print(f"  CPU full system: cross[3:6] (torque) = {cpu_cross[3:6]}")
    print(f"  Difference: {np.max(np.abs(s_cross - cpu_cross)):.2e}")

    # ── J. The critical question ─────────────────────────────────────────
    print("\n--- J. CRITICAL FINDING ---")
    print("Compare: single-body theoretical cross-term vs CPU MuJoCo full-system cross-term")
    s_cross_norm = np.max(np.abs(s_cross[0:6]))
    cpu_cross_norm = np.max(np.abs(cpu_cross[0:6]))
    print(f"  Single-body cross norm (free-base): {s_cross_norm:.6f}")
    print(f"  CPU full-system cross norm (free-base): {cpu_cross_norm:.6f}")

    if s_cross_norm < 1e-6 and cpu_cross_norm < 1e-6:
        print("  → Both are zero: cross-term is structurally zero for free-base ω×v")
        print("  → This is a property of the free-joint motion subspace")
    elif s_cross_norm > 1e-6 and cpu_cross_norm < 1e-6:
        print("  → Single-body theory is nonzero but CPU is zero")
        print("  → FREE-JOINT PROJECTION MUST BE DIFFERENT from direct spatial force mapping!")
    elif s_cross_norm > 1e-6 and cpu_cross_norm > 1e-6:
        if np.max(np.abs(s_cross[0:6] - cpu_cross[0:6])) < 1e-6:
            print("  → Both match: single-body theory = CPU = nonzero")
            print("  → Cross-term IS physically nonzero for free-base")
            print("  → The bug is in how children's forces cancel/propagate to the free-base")
        else:
            print(f"  → Both nonzero but differ: diff={np.max(np.abs(s_cross[0:6]-cpu_cross[0:6])):.2e}")
            print("  → Need to investigate origin of difference")

    # ── K. Test with all other bodies massless ────────────────────────────
    print("\n--- K. Force decomposition along kinematic chain ---")
    # The full-system spatial force at torso includes contributions from children
    # We need to understand if children's forces create the cancellation

    # For each body, compute its own contribution (I_i @ a_i + crf(v_i) @ I_i @ v_i)
    # and the propagated force to parent, to see what adds to F_torso

    # Actually, we can test if the error is in the spatial force propagation
    # by checking: does F_torso = I_torso @ a_torso + crf(v_torso) @ I_torso @ v_torso + Σ X_child→torso^T @ F_child?

    # Alternative: use rnea_body_local to inspect intermediate values
    # Actually, let's just run a clean single-body test in a stripped model
    print("  To isolate: we need to test if children's forces create the cancellation.")
    print("  The key diagnostic: run RNEA, inspect F_spatial at each body,")
    print("  and verify the backward pass propagation.")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
