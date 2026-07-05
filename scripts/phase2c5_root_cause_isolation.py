#!/usr/bin/env python
"""Phase 2C.5 — Root cause isolation: body-local vs world-frame RNEA comparison."""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco, numpy as np, jax.numpy as jnp
from wheeled_biped.dynamics.jax_bias_forces import (
    build_bias_force_constants, jax_bias_forces, jax_gravity_forces,
    extract_jax_fk_arrays, extract_jax_bias_arrays,
    _quat_to_rotmat, _skew3, _crm, _crf, _motion_xup, _axis_angle_to_rotmat,
)
from wheeled_biped.dynamics.jax_kinematics import jax_forward_kinematics_fk_arrays

model_path = str(PROJECT_ROOT / "assets" / "robot" / "wheeled_biped_real.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

constants = build_bias_force_constants(model)
nv, nq = model.nv, model.nq
nbody = model.nbody

# ── Extract arrays ─────────────────────────────────────────────────────
fk_arrays = extract_jax_fk_arrays(constants)
bias_arrays_full = extract_jax_bias_arrays(constants)
_, *bias_rest = bias_arrays_full
bias_arrays = tuple(bias_rest)

(bm, bipos, biquat, binertia, binertia3x3, jdofadr, border, children, grav,
 I_body_local, R_tree, body_pos_local_origin, S_body_local,
 body_dof_adr, joint_type_from_body, num_children,
 total_mass, total_com_body, M_cross_world_identity,
 body_mass_mm, body_ipos_mm, body_iquat_mm, body_inertia_mm, dof_armature,
) = bias_arrays

(parent_ids, body_jntadr, body_pos_local_fk, body_quat_local,
 _joint_type, joint_axis, joint_qpos_adr, body_categories) = fk_arrays

# ── Test case: wz + vx ─────────────────────────────────────────────────
qpos = data.qpos.copy()
qvel = np.zeros(nv); qvel[5] = 1.0; qvel[0] = 1.0
qpos_jax = jnp.array(qpos, dtype=jnp.float32)
qvel_jax = jnp.array(qvel, dtype=jnp.float32)

# CPU
d = mujoco.MjData(model); d.qpos[:] = qpos; d.qvel[:] = qvel
mujoco.mj_forward(model, d)
cpu_bias = np.array(d.qfrc_bias, dtype=np.float64)

# JAX standard
jax_std = np.array(jax_bias_forces(qpos_jax, qvel_jax, constants), dtype=np.float64)

print("=" * 60)
print("ROOT CAUSE ISOLATION: wz + vx")
print("=" * 60)
print(f"\nStandard JAX actuated max error: {np.max(np.abs(jax_std[6:16] - cpu_bias[6:16])):.6f}")

# ══════════════════════════════════════════════════════════════════════════
# Body-local RNEA trace (replicating the internal logic)
# ══════════════════════════════════════════════════════════════════════════

n_active = int(border.shape[0])
torso_id = 1

fk = jax_forward_kinematics_fk_arrays(qpos_jax, fk_arrays)
body_quat_world_np = np.array(fk["body_quat_world"])
body_pos_world_np = np.array(fk["body_pos_world"])

R_torso_np = np.array(_quat_to_rotmat(jnp.array(body_quat_world_np[torso_id], dtype=jnp.float32)))
R_torso_T = R_torso_np.T

# Body-local torso init
v_torso_body = np.concatenate([qvel[3:6], R_torso_T @ qvel[0:3]])
a_torso_body = np.concatenate([np.zeros(3), -R_torso_T @ np.array(grav)])

v_body = np.zeros((nbody, 6), dtype=np.float64)
a_body = np.zeros((nbody, 6), dtype=np.float64)
X_up_cache = np.zeros((nbody, 6, 6), dtype=np.float64)
v_body[torso_id] = v_torso_body
a_body[torso_id] = a_torso_body
X_up_cache[torso_id] = np.eye(6)

# Forward pass (body-local)
for k in range(1, n_active):
    body_id = int(border[k])
    parent = int(parent_ids[body_id])
    jid = int(body_jntadr[body_id])
    R_tr = np.array(R_tree[body_id])
    p_parent = np.array(body_pos_local_origin[body_id])
    axis_local = np.array(joint_axis[max(jid, 0)])
    q_adr = int(joint_qpos_adr[max(jid, 0)])
    q_j = qpos[q_adr]
    R_joint_np = np.array(_axis_angle_to_rotmat(
        jnp.array(axis_local, dtype=jnp.float32),
        jnp.array(q_j, dtype=jnp.float32)))
    if jid < 0: R_joint_np = np.eye(3)
    R_pc = R_tr @ R_joint_np
    R_pc_T = R_pc.T
    X_up = np.array(_motion_xup(
        jnp.array(R_pc_T, dtype=jnp.float32),
        jnp.array(p_parent, dtype=jnp.float32)))
    S_i = np.array(S_body_local[body_id])
    dof_idx = int(body_dof_adr[body_id])
    qdot = qvel[dof_idx] if dof_idx >= 0 else 0.0
    S_qdot = S_i * qdot
    v_i = X_up @ v_body[parent] + S_qdot
    a_i = X_up @ a_body[parent] + np.array(_crm(jnp.array(v_i, dtype=jnp.float32))) @ S_qdot
    v_body[body_id] = v_i
    a_body[body_id] = a_i
    X_up_cache[body_id] = X_up

# Backward pass (body-local)
F_body = np.zeros((nbody, 6), dtype=np.float64)
for k in range(n_active - 1, -1, -1):
    body_id = int(border[k])
    I_b = np.array(I_body_local[body_id])
    v_b = v_body[body_id]
    a_b = a_body[body_id]
    Ia = I_b @ a_b
    Iv = I_b @ v_b
    crf_v = np.array(_crf(jnp.array(v_b, dtype=jnp.float32)))
    F_body_i = Ia + crf_v @ Iv
    F_body[body_id] += F_body_i
    parent = int(parent_ids[body_id])
    X_up = X_up_cache[body_id]
    R_T_np = X_up[0:3, 0:3]   # = child→parent
    R_np = R_T_np.T             # = parent→child
    tau_c = F_body[body_id, 0:3]
    f_c = F_body[body_id, 3:6]
    p_np = np.array(body_pos_local_origin[body_id])
    tau_parent = R_np @ tau_c + np.array(_skew3(jnp.array(p_np, dtype=jnp.float32))) @ (R_np @ f_c)
    f_parent = R_np @ f_c
    F_from_child = np.concatenate([tau_parent, f_parent])
    F_body[parent] += F_from_child

print("\n--- Body-Local RNEA actuated torques ---")
for j in range(6, 16):
    jname = ['l_hr','l_hy','l_hp','l_kn','l_wh','r_hr','r_hy','r_hp','r_kn','r_wh'][j-6]
    dof_idx = j
    body_id_for_dof = None
    for b in range(1, nbody):
        if int(body_dof_adr[b]) == dof_idx:
            body_id_for_dof = b; break
    S_i = np.array(S_body_local[body_id_for_dof]) if body_id_for_dof else np.zeros(6)
    tau_bl = float(np.dot(S_i, F_body[body_id_for_dof])) if body_id_for_dof else 0.0
    print(f"  {jname}: BL={tau_bl:.6f} CPU={cpu_bias[j]:.6f} err={abs(tau_bl-cpu_bias[j]):.2e}")

# ══════════════════════════════════════════════════════════════════════════
# World-frame RNEA
# ══════════════════════════════════════════════════════════════════════════

print("\n--- World-Frame RNEA ---")

# World-frame spatial inertias
I_world = np.zeros((nbody, 6, 6), dtype=np.float64)
for b in range(1, nbody):
    I_b_body = np.array(I_body_local[b])
    quat_w = body_quat_world_np[b]
    R_w = np.array(_quat_to_rotmat(jnp.array(quat_w, dtype=jnp.float32)))
    # X_wb: body→world spatial transform (pure rotation at body origin)
    X_wb = np.zeros((6, 6))
    X_wb[0:3, 0:3] = R_w; X_wb[3:6, 3:6] = R_w
    I_world[b] = X_wb @ I_b_body @ X_wb.T

# World-frame torso init: [omega_world; v_world]
omega_world = R_torso_np @ qvel[3:6]
v_torso_world = np.concatenate([omega_world, qvel[0:3]])

# Gravity in world frame: fictitious acceleration = [0; -g]
a_grav_world = np.concatenate([np.zeros(3), -np.array(grav)])

v_world = np.zeros((nbody, 6), dtype=np.float64)
a_world = np.zeros((nbody, 6), dtype=np.float64)
v_world[torso_id] = v_torso_world
a_world[torso_id] = a_grav_world

# Forward pass (world frame)
for k in range(1, n_active):
    body_id = int(border[k])
    parent = int(parent_ids[body_id])
    p_parent_body = np.array(body_pos_local_origin[body_id])
    quat_parent_w = body_quat_world_np[parent]
    R_parent_w = np.array(_quat_to_rotmat(jnp.array(quat_parent_w, dtype=jnp.float32)))
    r_world = R_parent_w @ p_parent_body

    # World-frame X_up: no rotation (orientation in I_world), only translation
    X_up_w = np.eye(6)
    X_up_w[3:6, 0:3] = -np.array(_skew3(jnp.array(r_world, dtype=jnp.float32)))

    # Joint contribution in world frame
    jid = int(body_jntadr[body_id])
    S_i_body = np.array(S_body_local[body_id])
    dof_idx = int(body_dof_adr[body_id])
    qdot = qvel[dof_idx] if dof_idx >= 0 else 0.0

    quat_child_w = body_quat_world_np[body_id]
    R_child_w = np.array(_quat_to_rotmat(jnp.array(quat_child_w, dtype=jnp.float32)))
    S_i_w = np.zeros(6)
    S_i_w[0:3] = R_child_w @ S_i_body[0:3]  # axis in world frame
    S_qdot_w = S_i_w * qdot

    v_i_w = X_up_w @ v_world[parent] + S_qdot_w
    a_i_w = X_up_w @ a_world[parent] + np.array(_crm(jnp.array(v_i_w, dtype=jnp.float32))) @ S_qdot_w

    v_world[body_id] = v_i_w
    a_world[body_id] = a_i_w

# Backward pass (world frame)
F_world = np.zeros((nbody, 6), dtype=np.float64)
for k in range(n_active - 1, -1, -1):
    body_id = int(border[k])
    I_b_w = I_world[body_id]
    v_b_w = v_world[body_id]
    a_b_w = a_world[body_id]
    Ia_w = I_b_w @ a_b_w
    Iv_w = I_b_w @ v_b_w
    crf_v_w = np.array(_crf(jnp.array(v_b_w, dtype=jnp.float32)))
    F_body_i_w = Ia_w + crf_v_w @ Iv_w
    F_world[body_id] += F_body_i_w

    parent = int(parent_ids[body_id])
    p_parent_body = np.array(body_pos_local_origin[body_id])
    quat_parent_w = body_quat_world_np[parent]
    R_parent_w = np.array(_quat_to_rotmat(jnp.array(quat_parent_w, dtype=jnp.float32)))
    r_world = R_parent_w @ p_parent_body

    tau_c_w = F_world[body_id, 0:3]
    f_c_w = F_world[body_id, 3:6]
    # X_up_world^T = [[I, skew(r)], [0, I]]
    tau_parent_w = tau_c_w + np.array(_skew3(jnp.array(r_world, dtype=jnp.float32))) @ f_c_w
    f_parent_w = f_c_w
    F_world[parent] += np.concatenate([tau_parent_w, f_parent_w])

# Project to MuJoCo qfrc
qfrc_wf = np.zeros(nv, dtype=np.float64)
# Free-base: qfrc[0:3]=force_world, qfrc[3:6]=torque_body
qfrc_wf[0:3] = F_world[torso_id, 3:6]   # force in world
qfrc_wf[3:6] = R_torso_T @ F_world[torso_id, 0:3]  # torque world→body

# Actuated joints
for k in range(1, n_active):
    body_id = int(border[k])
    dof_idx = int(body_dof_adr[body_id])
    S_i_body = np.array(S_body_local[body_id])
    quat_child_w = body_quat_world_np[body_id]
    R_child_w = np.array(_quat_to_rotmat(jnp.array(quat_child_w, dtype=jnp.float32)))
    S_i_w = np.zeros(6)
    S_i_w[0:3] = R_child_w @ S_i_body[0:3]
    tau_j_w = float(np.dot(S_i_w, F_world[body_id]))
    qfrc_wf[dof_idx] = tau_j_w

print("World-Frame RNEA actuated torques:")
for j in range(6, 16):
    jname = ['l_hr','l_hy','l_hp','l_kn','l_wh','r_hr','r_hy','r_hp','r_kn','r_wh'][j-6]
    print(f"  {jname}: WF={qfrc_wf[j]:.6f} CPU={cpu_bias[j]:.6f} err={abs(qfrc_wf[j]-cpu_bias[j]):.2e}")

act_err_wf = np.max(np.abs(qfrc_wf[6:16] - cpu_bias[6:16]))
fb_err_wf = np.max(np.abs(qfrc_wf[0:6] - cpu_bias[0:6]))
print(f"\nWorld-frame max actuated error: {act_err_wf:.2e}")
print(f"World-frame max free-base error: {fb_err_wf:.2e}")

# ══════════════════════════════════════════════════════════════════════════
# KEY DIAGNOSTIC: World-frame RNEA WITHOUT gravity in forward pass
# (gravity treated as external force, not fictitious acceleration)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("KEY TEST: World-frame RNEA, gravity via backward pass only")
print("=" * 60)

# In MuJoCo's RNEA for bias forces, gravity is added as a uniform
# field. The treatment can be either:
#   a) Fictitious acceleration in forward pass: a_base = [0; -g_world]
#   b) Body force in backward pass: add m_i * g to each body

# Let's try both approaches

# Approach (b): Gravity as body force, forward pass has zero acceleration
v_w2 = np.zeros((nbody, 6), dtype=np.float64)
a_w2 = np.zeros((nbody, 6), dtype=np.float64)
v_w2[torso_id] = v_torso_world
a_w2[torso_id] = np.zeros(6)  # NO gravity in forward pass

for k in range(1, n_active):
    body_id = int(border[k])
    parent = int(parent_ids[body_id])
    p_parent_body = np.array(body_pos_local_origin[body_id])
    quat_parent_w = body_quat_world_np[parent]
    R_parent_w = np.array(_quat_to_rotmat(jnp.array(quat_parent_w, dtype=jnp.float32)))
    r_world = R_parent_w @ p_parent_body
    X_up_w = np.eye(6)
    X_up_w[3:6, 0:3] = -np.array(_skew3(jnp.array(r_world, dtype=jnp.float32)))
    jid = int(body_jntadr[body_id])
    S_i_body = np.array(S_body_local[body_id])
    dof_idx = int(body_dof_adr[body_id])
    qdot = qvel[dof_idx] if dof_idx >= 0 else 0.0
    quat_child_w = body_quat_world_np[body_id]
    R_child_w = np.array(_quat_to_rotmat(jnp.array(quat_child_w, dtype=jnp.float32)))
    S_i_w = np.zeros(6)
    S_i_w[0:3] = R_child_w @ S_i_body[0:3]
    S_qdot_w = S_i_w * qdot
    v_i_w = X_up_w @ v_w2[parent] + S_qdot_w
    a_i_w = X_up_w @ a_w2[parent] + np.array(_crm(jnp.array(v_i_w, dtype=jnp.float32))) @ S_qdot_w
    v_w2[body_id] = v_i_w
    a_w2[body_id] = a_i_w

F_w2 = np.zeros((nbody, 6), dtype=np.float64)
for k in range(n_active - 1, -1, -1):
    body_id = int(border[k])
    I_b_w = I_world[body_id]
    v_b_w = v_w2[body_id]
    a_b_w = a_w2[body_id]
    Ia_w = I_b_w @ a_b_w
    Iv_w = I_b_w @ v_b_w
    crf_v_w = np.array(_crf(jnp.array(v_b_w, dtype=jnp.float32)))
    F_body_i_w = Ia_w + crf_v_w @ Iv_w

    # Add gravity body force: F_grav = [0; m_i * g] in world frame
    mass_i = float(constants["body_mass"][body_id])
    g_world = np.array(grav)
    F_grav_i = np.concatenate([np.zeros(3), mass_i * g_world])
    F_body_i_w = F_body_i_w + F_grav_i

    F_w2[body_id] += F_body_i_w

    parent = int(parent_ids[body_id])
    p_parent_body = np.array(body_pos_local_origin[body_id])
    quat_parent_w = body_quat_world_np[parent]
    R_parent_w = np.array(_quat_to_rotmat(jnp.array(quat_parent_w, dtype=jnp.float32)))
    r_world = R_parent_w @ p_parent_body
    tau_c_w = F_w2[body_id, 0:3]
    f_c_w = F_w2[body_id, 3:6]
    tau_parent_w = tau_c_w + np.array(_skew3(jnp.array(r_world, dtype=jnp.float32))) @ f_c_w
    f_parent_w = f_c_w
    F_w2[parent] += np.concatenate([tau_parent_w, f_parent_w])

qfrc_w2 = np.zeros(nv, dtype=np.float64)
qfrc_w2[0:3] = F_w2[torso_id, 3:6]
qfrc_w2[3:6] = R_torso_T @ F_w2[torso_id, 0:3]
for k in range(1, n_active):
    body_id = int(border[k])
    dof_idx = int(body_dof_adr[body_id])
    S_i_body = np.array(S_body_local[body_id])
    quat_child_w = body_quat_world_np[body_id]
    R_child_w = np.array(_quat_to_rotmat(jnp.array(quat_child_w, dtype=jnp.float32)))
    S_i_w = np.zeros(6)
    S_i_w[0:3] = R_child_w @ S_i_body[0:3]
    tau_j_w = float(np.dot(S_i_w, F_w2[body_id]))
    qfrc_w2[dof_idx] = tau_j_w

print("Gravity-as-body-force actuated torques:")
for j in range(6, 16):
    jname = ['l_hr','l_hy','l_hp','l_kn','l_wh','r_hr','r_hy','r_hp','r_kn','r_wh'][j-6]
    print(f"  {jname}: GF={qfrc_w2[j]:.6f} CPU={cpu_bias[j]:.6f} err={abs(qfrc_w2[j]-cpu_bias[j]):.2e}")

act_err_gf = np.max(np.abs(qfrc_w2[6:16] - cpu_bias[6:16]))
fb_err_gf = np.max(np.abs(qfrc_w2[0:6] - cpu_bias[0:6]))
print(f"\nGravity-as-body-force max actuated error: {act_err_gf:.2e}")
print(f"Gravity-as-body-force max free-base error: {fb_err_gf:.2e}")

# ══════════════════════════════════════════════════════════════════════════
# KEY TEST: What if the body-local RNEA uses R_pc (not R_pc^T) in X_up?
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("KEY TEST: Body-local RNEA with FLIPPED X_up rotation convention")
print("=" * 60)

# Current: X_up = [[R_pc^T, 0], [-R_pc^T @ skew(p), R_pc^T]]
# Flipped:  X_up' = [[R_pc, 0], [-R_pc @ skew(p), R_pc]]

def _motion_xup_flipped(R_pc, p_parent):
    Z33 = np.zeros((3, 3))
    top = np.concatenate([R_pc, Z33], axis=1)
    bot = np.concatenate([-R_pc @ np.array(_skew3(jnp.array(p_parent, dtype=jnp.float32))), R_pc], axis=1)
    return np.concatenate([top, bot], axis=0)

v_flip = np.zeros((nbody, 6), dtype=np.float64)
a_flip = np.zeros((nbody, 6), dtype=np.float64)
X_flip_cache = np.zeros((nbody, 6, 6), dtype=np.float64)
v_flip[torso_id] = v_torso_body
a_flip[torso_id] = a_torso_body
X_flip_cache[torso_id] = np.eye(6)

for k in range(1, n_active):
    body_id = int(border[k])
    parent = int(parent_ids[body_id])
    jid = int(body_jntadr[body_id])
    R_tr = np.array(R_tree[body_id])
    p_parent = np.array(body_pos_local_origin[body_id])
    axis_local = np.array(joint_axis[max(jid, 0)])
    q_adr = int(joint_qpos_adr[max(jid, 0)])
    q_j = qpos[q_adr]
    R_joint_np = np.array(_axis_angle_to_rotmat(
        jnp.array(axis_local, dtype=jnp.float32), jnp.array(q_j, dtype=jnp.float32)))
    if jid < 0: R_joint_np = np.eye(3)
    R_pc = R_tr @ R_joint_np
    X_up_f = _motion_xup_flipped(R_pc, p_parent)  # Use FLIPPED convention
    S_i = np.array(S_body_local[body_id])
    dof_idx = int(body_dof_adr[body_id])
    qdot = qvel[dof_idx] if dof_idx >= 0 else 0.0
    S_qdot = S_i * qdot
    v_i_f = X_up_f @ v_flip[parent] + S_qdot
    a_i_f = X_up_f @ a_flip[parent] + np.array(_crm(jnp.array(v_i_f, dtype=jnp.float32))) @ S_qdot
    v_flip[body_id] = v_i_f
    a_flip[body_id] = a_i_f
    X_flip_cache[body_id] = X_up_f

F_flip = np.zeros((nbody, 6), dtype=np.float64)
for k in range(n_active - 1, -1, -1):
    body_id = int(border[k])
    I_b = np.array(I_body_local[body_id])
    v_b = v_flip[body_id]
    a_b = a_flip[body_id]
    Ia = I_b @ a_b
    Iv = I_b @ v_b
    crf_v = np.array(_crf(jnp.array(v_b, dtype=jnp.float32)))
    F_body_i = Ia + crf_v @ Iv
    F_flip[body_id] += F_body_i
    parent = int(parent_ids[body_id])
    X_up_f = X_flip_cache[body_id]
    # With flipped X_up, X_up^T = [[R_pc^T, skew(p) @ R_pc^T], [0, R_pc^T]]
    R_f = X_up_f[0:3, 0:3]  # = R_pc
    R_f_T = R_f.T            # = R_pc^T
    tau_c = F_flip[body_id, 0:3]
    f_c = F_flip[body_id, 3:6]
    p_np = np.array(body_pos_local_origin[body_id])
    # X_up^T = [[R^T, skew(p) @ R^T], [0, R^T]] where R = R_pc
    tau_parent_f = R_f_T @ tau_c + np.array(_skew3(jnp.array(p_np, dtype=jnp.float32))) @ (R_f_T @ f_c)
    f_parent_f = R_f_T @ f_c
    F_flip[parent] += np.concatenate([tau_parent_f, f_parent_f])

print("FLIPPED-X_up actuated torques:")
for j in range(6, 16):
    jname = ['l_hr','l_hy','l_hp','l_kn','l_wh','r_hr','r_hy','r_hp','r_kn','r_wh'][j-6]
    dof_idx = j
    body_id_for_dof = None
    for b in range(1, nbody):
        if int(body_dof_adr[b]) == dof_idx:
            body_id_for_dof = b; break
    S_i = np.array(S_body_local[body_id_for_dof]) if body_id_for_dof else np.zeros(6)
    tau_flip = float(np.dot(S_i, F_flip[body_id_for_dof])) if body_id_for_dof else 0.0
    print(f"  {jname}: FLIP={tau_flip:.6f} CPU={cpu_bias[j]:.6f} err={abs(tau_flip-cpu_bias[j]):.2e}")

act_err_flip = np.max([abs(float(np.dot(np.array(S_body_local[b]), F_flip[b])) - cpu_bias[int(body_dof_adr[b])])
                       for b in range(2, nbody) if int(body_dof_adr[b]) >= 6])
# More careful computation
max_act_flip = 0.0
for j in range(6, 16):
    for b in range(1, nbody):
        if int(body_dof_adr[b]) == j:
            tau_f = float(np.dot(np.array(S_body_local[b]), F_flip[b]))
            err_f = abs(tau_f - cpu_bias[j])
            max_act_flip = max(max_act_flip, err_f)
print(f"\nFlipped-X_up max actuated error: {max_act_flip:.2e}")

# Also compute free-base
R_t = R_torso_np
f_w_flip = R_t @ F_flip[torso_id, 3:6]
tau_b_flip = F_flip[torso_id, 0:3]
print(f"Flipped FB force:  {f_w_flip} CPU: {cpu_bias[0:3]} err: {np.max(np.abs(f_w_flip - cpu_bias[0:3])):.2e}")
print(f"Flipped FB torque: {tau_b_flip} CPU: {cpu_bias[3:6]} err: {np.max(np.abs(tau_b_flip - cpu_bias[3:6])):.2e}")
