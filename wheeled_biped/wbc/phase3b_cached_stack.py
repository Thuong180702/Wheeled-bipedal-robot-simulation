"""Phase 3B.1 — Shape-Stable Contact Stack and Snapshot Caching.

Provides:
  - ``PaddedContactStack``: fixed-shape contact data for JIT stability
  - ``prepare_phase3b_snapshot()``: precompute all mode-independent data once
  - ``build_phase3b_qp_from_snapshot()``: mode-specific QP build from cached data

All functions are offline only. No realtime integration. No controller coupling.
No torque injection.

Design principle:
  Scenario data (contacts, Jacobians, dynamics) is computed ONCE per scenario
  and reused across all 5 task modes. Task modes only change soft cost weights.
"""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

# ── Maximum contacts for shape-stable padding ──────────────────────────

MAX_CONTACTS = 4  # Phase 2D.1 scenarios have 2 or 4 contacts


# ═══════════════════════════════════════════════════════════════════════════
# PaddedContactStack
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PaddedContactStack:
    """Fixed-shape contact stack for JIT stability.

    All arrays have first dimension = MAX_CONTACTS. Inactive padded entries
    are masked out via ``active_mask``.
    """
    Jp: np.ndarray          # shape (MAX_CONTACTS, 3, 16)
    Jr: np.ndarray          # shape (MAX_CONTACTS, 3, 16) — rotational, unused currently
    JcT: np.ndarray         # shape (16, 3*MAX_CONTACTS)
    frame: np.ndarray       # shape (MAX_CONTACTS, 3, 3)
    local_point: np.ndarray # shape (MAX_CONTACTS, 3)
    body_id: np.ndarray     # shape (MAX_CONTACTS,)
    normal: np.ndarray      # shape (MAX_CONTACTS, 3)
    position_world: np.ndarray  # shape (MAX_CONTACTS, 3)
    active_mask: np.ndarray # shape (MAX_CONTACTS,), bool
    num_contacts: int

    @property
    def m(self) -> int:
        return self.num_contacts

    @property
    def n_lambda(self) -> int:
        return 3 * self.num_contacts

    @property
    def n_lambda_padded(self) -> int:
        return 3 * MAX_CONTACTS

    def get_active_JcT(self) -> np.ndarray:
        """Return JcT for only active contacts: shape (16, 3*num_contacts)."""
        if self.num_contacts == 0:
            return np.zeros((16, 0), dtype=np.float64)
        return self.JcT[:, :3 * self.num_contacts]

    def get_active_Jp_stack(self) -> np.ndarray:
        """Return stacked Jp for only active contacts: shape (3*num_contacts, 16)."""
        if self.num_contacts == 0:
            return np.zeros((0, 16), dtype=np.float64)
        return self.Jp[:self.num_contacts, :, :].reshape(3 * self.num_contacts, 16)

    def get_active_normals(self) -> np.ndarray:
        """Return normals for only active contacts: shape (num_contacts, 3)."""
        if self.num_contacts == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return self.normal[:self.num_contacts, :]

    def get_active_frames(self) -> np.ndarray:
        """Return frames for only active contacts: shape (num_contacts, 3, 3)."""
        if self.num_contacts == 0:
            return np.zeros((0, 3, 3), dtype=np.float64)
        return self.frame[:self.num_contacts, :, :]

    def get_active_body_ids(self) -> np.ndarray:
        """Return body IDs for only active contacts: shape (num_contacts,)."""
        if self.num_contacts == 0:
            return np.zeros(0, dtype=np.int32)
        return self.body_id[:self.num_contacts]

    def get_active_local_points(self) -> np.ndarray:
        """Return local points for only active contacts: shape (num_contacts, 3)."""
        if self.num_contacts == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return self.local_point[:self.num_contacts, :]


def build_padded_contact_stack(
    qpos: np.ndarray,
    contacts: list[dict[str, Any]],
    contact_constants: dict[str, Any],
    max_contacts: int = MAX_CONTACTS,
) -> PaddedContactStack:
    """Build a shape-stable padded contact stack from active contacts.

    Active contacts fill the first m slots; remaining slots are zero-padded.
    The ``active_mask`` array marks which entries are real.

    Args:
        qpos: (nq,) generalized positions.
        contacts: list of active contact dicts (from ``extract_active_contacts``).
        contact_constants: dict from ``build_contact_dynamics_constants``.
        max_contacts: maximum number of contacts to pad to.

    Returns:
        ``PaddedContactStack`` with all arrays having first dim = max_contacts.
    """
    from wheeled_biped.dynamics.jax_contact_dynamics import (
        contact_point_translational_jacobian,
    )

    m = len(contacts)
    if m > max_contacts:
        raise ValueError(
            f"Number of contacts ({m}) exceeds max_contacts ({max_contacts}). "
            f"Increase max_contacts or fix scenario generation."
        )

    qpos_jax = jnp.array(qpos, dtype=jnp.float32)

    # Allocate padded arrays
    Jp = np.zeros((max_contacts, 3, 16), dtype=np.float64)
    Jr = np.zeros((max_contacts, 3, 16), dtype=np.float64)  # reserved
    JcT = np.zeros((16, 3 * max_contacts), dtype=np.float64)
    frame = np.zeros((max_contacts, 3, 3), dtype=np.float64)
    local_point = np.zeros((max_contacts, 3), dtype=np.float64)
    body_id = np.zeros(max_contacts, dtype=np.int32)
    normal = np.zeros((max_contacts, 3), dtype=np.float64)
    position_world = np.zeros((max_contacts, 3), dtype=np.float64)
    active_mask = np.zeros(max_contacts, dtype=bool)

    for i in range(m):
        c = contacts[i]
        bid = int(c["body_id"])
        lp = np.array(c["local_point"], dtype=np.float32)
        fr = np.array(c["frame"], dtype=np.float32)
        pos = np.array(c["position"], dtype=np.float64)

        # Translational Jacobian
        Jp_i = contact_point_translational_jacobian(
            qpos_jax, bid, jnp.array(lp, dtype=jnp.float32), contact_constants,
        )  # (3, 16)
        Jp_i_np = np.array(Jp_i, dtype=np.float64)

        # Normal in world frame
        n_world = fr[:, 0].copy()

        # JcT contribution
        JcT_i = Jp_i_np.T @ fr  # (16, 3)

        Jp[i, :, :] = Jp_i_np
        JcT[:, 3*i:3*i+3] = JcT_i
        frame[i, :, :] = fr
        local_point[i, :] = lp
        body_id[i] = bid
        normal[i, :] = n_world
        position_world[i, :] = pos
        active_mask[i] = True

    return PaddedContactStack(
        Jp=Jp,
        Jr=Jr,
        JcT=JcT,
        frame=frame,
        local_point=local_point,
        body_id=body_id,
        normal=normal,
        position_world=position_world,
        active_mask=active_mask,
        num_contacts=m,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Per-scenario snapshot (precomputed once, reused across modes)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Phase3BSnapshot:
    """Precomputed mode-independent data for one scenario.

    All Jacobians and dynamics are computed ONCE and reused across all 5 modes.
    """
    # Scenario identity
    scenario_name: str

    # State
    qpos: np.ndarray          # (17,)
    qvel: np.ndarray          # (16,)

    # Dynamics
    M: np.ndarray             # (16, 16) mass matrix
    h: np.ndarray             # (16,) bias forces
    S: np.ndarray             # (16, 10) actuator selection

    # Contact
    contact_stack: PaddedContactStack
    jdot_qdot: np.ndarray     # (3*max_contacts,) Jdot @ qvel
    mu: float

    # COM
    Jcom: np.ndarray          # (3, 16) COM translational Jacobian
    jdq_com: np.ndarray       # (3,) Jdot_com @ qvel
    com_position: np.ndarray  # (3,) current COM position

    # Torso orientation
    Jr: np.ndarray            # (3, 16) torso angular velocity Jacobian
    jdw_torso: np.ndarray     # (3,) Jdot_w_torso @ qvel
    e_R: np.ndarray           # (3,) orientation error (log_SO3)
    omega_current: np.ndarray # (3,) current angular velocity
    current_rpy: np.ndarray   # (3,) current roll/pitch/yaw

    # Torque limits
    tau_min: np.ndarray       # (10,)
    tau_max: np.ndarray       # (10,)

    # Robot mass info
    total_mass: float
    robot_weight: float

    # Timing
    snapshot_time_s: float

    @property
    def nv(self) -> int:
        return 16

    @property
    def nu(self) -> int:
        return 10

    @property
    def m(self) -> int:
        return self.contact_stack.num_contacts

    @property
    def n_lambda(self) -> int:
        return 3 * self.m

    @property
    def n_lambda_padded(self) -> int:
        return 3 * MAX_CONTACTS


def prepare_phase3b_snapshot(
    scenario_name: str,
    qpos: np.ndarray,
    qvel: np.ndarray,
    contacts: list[dict[str, Any]],
    constants: dict[str, Any],
    max_contacts: int = MAX_CONTACTS,
) -> Phase3BSnapshot:
    """Precompute all mode-independent data for one scenario.

    This function is called ONCE per scenario. The resulting snapshot is
    reused across all 5 task modes via ``build_phase3b_qp_from_snapshot``.

    Computes:
      - M(q), h(q, qvel), S
      - Padded contact stack, Jdot_qdot
      - COM Jacobian and Jdot_com_qdot
      - Torso angular velocity Jacobian and Jdot_w_qdot
      - Orientation error
      - Torque limits
      - Robot mass info

    Args:
        scenario_name: human-readable scenario identifier.
        qpos: (nq,) generalized positions.
        qvel: (nv,) generalized velocities.
        contacts: list of active contact dicts.
        constants: dict from ``build_qp_wbc_constants`` (must include
                   dynamics, contact, and kinematics constants).
        max_contacts: max contacts for padded stack.

    Returns:
        ``Phase3BSnapshot`` with all precomputed data.
    """
    import time
    t0 = time.perf_counter()

    nv = 16
    nu = 10
    m = len(contacts)

    # Ensure required constants are available
    from wheeled_biped.wbc.offline_qp_wbc import (
        _ensure_dynamics_constants,
        _ensure_contact_constants,
    )
    _ensure_dynamics_constants(constants)
    _ensure_contact_constants(constants)

    mass_constants = constants["_mass_matrix_constants"]
    bias_constants = constants["_dynamics_constants"]
    contact_constants = constants["_contact_constants"]

    # Ensure kinematics constants
    from wheeled_biped.wbc.offline_task_stack import _ensure_kinematics_constants_for_tasks
    _ensure_kinematics_constants_for_tasks(constants)
    kc = constants["_kinematics_constants"]

    from wheeled_biped.dynamics.jax_mass_matrix import jax_mass_matrix
    from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces

    qpos_jax = jnp.array(qpos, dtype=jnp.float32)
    qvel_jax = jnp.array(qvel, dtype=jnp.float32)

    # ── Mass matrix and bias forces ──────────────────────────────────
    M_jax = jax_mass_matrix(qpos_jax, mass_constants)
    h_jax = jax_bias_forces(qpos_jax, qvel_jax, bias_constants)
    M = np.array(M_jax, dtype=np.float64)
    h = np.array(h_jax, dtype=np.float64)

    # ── Actuator selection matrix ────────────────────────────────────
    S_np = np.array(constants["S"], dtype=np.float64)

    # ── Padded contact stack ─────────────────────────────────────────
    if m > 0:
        contact_stack = build_padded_contact_stack(
            qpos, contacts, contact_constants, max_contacts=max_contacts,
        )
    else:
        # Empty padded stack
        contact_stack = PaddedContactStack(
            Jp=np.zeros((max_contacts, 3, nv), dtype=np.float64),
            Jr=np.zeros((max_contacts, 3, nv), dtype=np.float64),
            JcT=np.zeros((nv, 3 * max_contacts), dtype=np.float64),
            frame=np.zeros((max_contacts, 3, 3), dtype=np.float64),
            local_point=np.zeros((max_contacts, 3), dtype=np.float64),
            body_id=np.zeros(max_contacts, dtype=np.int32),
            normal=np.zeros((max_contacts, 3), dtype=np.float64),
            position_world=np.zeros((max_contacts, 3), dtype=np.float64),
            active_mask=np.zeros(max_contacts, dtype=bool),
            num_contacts=0,
        )

    # ── Jdot_qdot for contact normal acceleration ────────────────────
    from wheeled_biped.wbc.offline_qp_wbc import compute_contact_jdot_qdot
    if m > 0:
        jdot_qdot_raw = compute_contact_jdot_qdot(qpos, qvel, contacts, contact_constants)
        # Pad to max_contacts
        jdot_qdot = np.zeros(3 * max_contacts, dtype=np.float64)
        jdot_qdot[:3 * m] = jdot_qdot_raw
    else:
        jdot_qdot = np.zeros(3 * max_contacts, dtype=np.float64)

    # ── COM Jacobian and Jdot_qdot ───────────────────────────────────
    from wheeled_biped.wbc.offline_task_stack import (
        compute_com_jacobian,
        compute_com_jdot_qdot,
        compute_torso_angular_velocity_jacobian,
        compute_torso_jdotw_qdot,
        compute_torso_orientation_error,
    )

    Jcom = compute_com_jacobian(qpos, kc)
    jdq_com = compute_com_jdot_qdot(qpos, qvel, kc)

    # ── COM current position ────────────────────────────────────────
    from wheeled_biped.wbc.offline_task_stack import _compute_com
    com_pos = np.array(_compute_com(qpos_jax, kc), dtype=np.float64)

    # ── Torso orientation ───────────────────────────────────────────
    Jr = compute_torso_angular_velocity_jacobian(qpos, kc)
    jdw_torso = compute_torso_jdotw_qdot(qpos, qvel, kc)

    orient_result = compute_torso_orientation_error(qpos, kc)
    e_R = orient_result["e_R"]
    current_rpy = orient_result["current_rpy"]

    qvel_np = np.array(qvel, dtype=np.float64)
    omega_current = Jr @ qvel_np

    # ── Torque limits ────────────────────────────────────────────────
    tau_min = np.array(constants["tau_min"], dtype=np.float64)
    tau_max = np.array(constants["tau_max"], dtype=np.float64)

    # ── Robot mass info ─────────────────────────────────────────────
    body_mass_arr = constants.get("body_mass", np.ones(1, dtype=np.float32))
    total_mass = float(np.sum(np.array(body_mass_arr)))
    g_val = float(np.array(constants.get("gravity", jnp.array([0, 0, -9.81]))[2]))
    robot_weight = total_mass * abs(g_val)

    # ── Friction coefficient ────────────────────────────────────────
    mu = float(constants.get("mu", 0.8))

    snapshot_time = time.perf_counter() - t0

    return Phase3BSnapshot(
        scenario_name=scenario_name,
        qpos=qpos.copy(),
        qvel=qvel.copy(),
        M=M,
        h=h,
        S=S_np,
        contact_stack=contact_stack,
        jdot_qdot=jdot_qdot,
        mu=mu,
        Jcom=Jcom,
        jdq_com=jdq_com,
        com_position=com_pos,
        Jr=Jr,
        jdw_torso=jdw_torso,
        e_R=e_R,
        omega_current=omega_current,
        current_rpy=current_rpy,
        tau_min=tau_min,
        tau_max=tau_max,
        total_mass=total_mass,
        robot_weight=robot_weight,
        snapshot_time_s=snapshot_time,
    )


# ═══════════════════════════════════════════════════════════════════════════
# QP building from snapshot (only adds mode-specific weights)
# ═══════════════════════════════════════════════════════════════════════════

def build_phase3b_qp_from_snapshot(
    snapshot: Phase3BSnapshot,
    task_mode: str,
    constants: dict[str, Any],
) -> dict[str, Any]:
    """Build QP matrices from a precomputed snapshot for a specific task mode.

    This function does NOT recompute any Jacobians or dynamics. It only:
      1. Builds the base Phase 3 hard-constraint QP from cached data.
      2. Builds mode-specific task cost matrices from cached Jacobians.
      3. Adds them together.

    Args:
        snapshot: precomputed ``Phase3BSnapshot``.
        task_mode: one of "feasibility_only", "balanced_default",
                   "posture_priority", "torso_priority", "com_priority".
        constants: dict from ``build_qp_wbc_constants``.

    Returns:
        dict with all QP matrices (base + task costs).
    """
    import time
    t0 = time.perf_counter()

    from wheeled_biped.wbc.offline_task_stack import TASK_WEIGHT_MODES

    if task_mode not in TASK_WEIGHT_MODES:
        raise ValueError(f"Unknown task mode: {task_mode}")

    weights = TASK_WEIGHT_MODES[task_mode]

    m = snapshot.m
    nv = snapshot.nv
    nu = snapshot.nu
    k = 0  # no slack variables in current design
    n_lambda = 3 * m
    nz = nv + nu + n_lambda + k

    M = snapshot.M
    h_vec = snapshot.h
    S = snapshot.S
    contact_stack = snapshot.contact_stack
    jdot_qdot = snapshot.jdot_qdot

    # ── Variable slices ──────────────────────────────────────────────
    slices = {
        "qdd": (0, 16),
        "tau": (16, 26),
        "lambda": (26, 26 + n_lambda),
        "slack": (26 + n_lambda, nz),
    }

    # ── Base QP matrices (hard constraints from Phase 3) ─────────────
    # Quadratic cost: H (base regularization)
    w_qdd_base = weights["w_qdd"]
    w_tau_base = weights["w_tau"]
    w_lambda_base = weights["w_lambda"]
    w_slack_base = weights["w_slack"]

    H_diag = np.concatenate([
        np.full(nv, w_qdd_base, dtype=np.float64),
        np.full(nu, w_tau_base, dtype=np.float64),
        np.full(n_lambda, w_lambda_base, dtype=np.float64),
        np.full(k, w_slack_base, dtype=np.float64),
    ])
    H = np.diag(H_diag)

    # Linear cost: g
    g = np.zeros(nz, dtype=np.float64)

    # ── Equality: dynamics M qdd + h = S tau + JcT lambda ────────────
    JcT_active = contact_stack.get_active_JcT()  # (16, 3m)

    A_dyn = np.zeros((nv, nz), dtype=np.float64)
    A_dyn[:, 0:16] = M
    A_dyn[:, 16:26] = -S
    if m > 0:
        A_dyn[:, 26:26 + n_lambda] = -JcT_active
    b_dyn = -h_vec

    # ── Equality: contact normal acceleration ────────────────────────
    n_eq_contact = m
    A_contact = np.zeros((n_eq_contact, nz), dtype=np.float64)
    b_contact = np.zeros(n_eq_contact, dtype=np.float64)

    if m > 0:
        Jp_active = contact_stack.get_active_Jp_stack()  # (3m, 16)
        normals_active = contact_stack.get_active_normals()  # (m, 3)

        for i in range(m):
            n_i = normals_active[i]
            Jp_i = Jp_active[3*i:3*i+3, :]
            row_i = n_i @ Jp_i
            A_contact[i, 0:16] = row_i
            b_contact[i] = -np.dot(n_i, jdot_qdot[3*i:3*i+3])

    # ── Stack equalities ────────────────────────────────────────────
    A_eq_parts = [A_dyn]
    b_eq_parts = [b_dyn]
    if m > 0:
        A_eq_parts.append(A_contact)
        b_eq_parts.append(b_contact)

    A_eq = np.concatenate(A_eq_parts, axis=0)
    b_eq = np.concatenate(b_eq_parts)

    # ── Inequality: friction cone (linearized pyramid) ──────────────
    mu = snapshot.mu
    n_friction = 5 * m
    A_friction = np.zeros((n_friction, nz), dtype=np.float64)
    b_friction = np.zeros(n_friction, dtype=np.float64)

    if m > 0:
        for i in range(m):
            row_start = 5 * i
            col_start = 26 + 3 * i

            A_friction[row_start + 0, col_start + 0] = 1.0  # fn >= 0

            A_friction[row_start + 1, col_start + 0] = mu
            A_friction[row_start + 1, col_start + 1] = -1.0  # mu*fn - ft1 >= 0

            A_friction[row_start + 2, col_start + 0] = mu
            A_friction[row_start + 2, col_start + 1] = 1.0   # mu*fn + ft1 >= 0

            A_friction[row_start + 3, col_start + 0] = mu
            A_friction[row_start + 3, col_start + 2] = -1.0  # mu*fn - ft2 >= 0

            A_friction[row_start + 4, col_start + 0] = mu
            A_friction[row_start + 4, col_start + 2] = 1.0   # mu*fn + ft2 >= 0

    # ── Bounds ──────────────────────────────────────────────────────
    bounds_list = []
    for _ in range(nv):
        bounds_list.append((-1e6, 1e6))
    for i in range(nu):
        bounds_list.append((float(snapshot.tau_min[i]), float(snapshot.tau_max[i])))
    for _ in range(n_lambda):
        bounds_list.append((-1e6, 1e6))
    for _ in range(k):
        bounds_list.append((-1e6, 1e6))

    # ── Build task cost matrices from snapshot (cached Jacobians) ────
    H_task, g_task, per_task_meta = _build_task_costs_from_snapshot(
        snapshot, weights,
    )

    H += H_task
    g += g_task

    # ── Assemble ────────────────────────────────────────────────────
    build_time = time.perf_counter() - t0

    return {
        "H": H,
        "g": g,
        "A_eq": A_eq,
        "b_eq": b_eq,
        "A_friction": A_friction,
        "b_friction": b_friction,
        "bounds": bounds_list,
        "slices": slices,
        "nz": nz,
        "nv": nv,
        "nu": nu,
        "m": m,
        "k": k,
        "M": M,
        "h": h_vec,
        "S": S,
        "JcT": JcT_active,
        "contact_stack_raw": contact_stack,
        "jdot_qdot": jdot_qdot[:3*m] if m > 0 else np.zeros(0),
        "n_eq_dyn": nv,
        "n_eq_contact": m,
        "n_ineq_friction": n_friction,
        "task_version": "phase3b1_cached_snapshot",
        "task_mode": task_mode,
        "per_task_metadata": per_task_meta,
        "qp_build_time_s": build_time,
        "snapshot": snapshot,
    }


def _build_task_costs_from_snapshot(
    snapshot: Phase3BSnapshot,
    weights: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build task cost matrices H_task, g_task using CACHED Jacobians.

    Does NOT call any JAX functions — uses only precomputed data from snapshot.

    Returns:
        (H_task, g_task, per_task_metadata)
    """
    from wheeled_biped.wbc.offline_task_stack import (
        DEFAULT_KP_COM, DEFAULT_KD_COM,
        DEFAULT_KP_R, DEFAULT_KD_R,
        DEFAULT_KP_POSTURE, DEFAULT_KD_POSTURE,
    )

    m = snapshot.m
    nv = snapshot.nv
    nu = snapshot.nu
    n_lambda = 3 * m
    k = 0
    nz = nv + nu + n_lambda + k

    H_task = np.zeros((nz, nz), dtype=np.float64)
    g_task = np.zeros(nz, dtype=np.float64)
    per_task = {}

    qdd_slice = slice(0, 16)
    tau_slice = slice(16, 26)
    lambda_slice = slice(26, 26 + n_lambda)

    # ── COM height task ─────────────────────────────────────────────
    w_com = weights.get("w_com", 0.0)
    if w_com > 0.0:
        Jcom = snapshot.Jcom
        Jcom_z = Jcom[2:3, :]
        jdq_com_z = snapshot.jdq_com[2]

        z_com_current = float(snapshot.com_position[2])
        qvel_np = snapshot.qvel
        vz_com = float(np.dot(Jcom_z, qvel_np)[0])

        kp_z = DEFAULT_KP_COM
        kd_z = DEFAULT_KD_COM
        z_ref = z_com_current  # hold current height
        vz_ref = 0.0

        a_com_z_des = kp_z * (z_ref - z_com_current) + kd_z * (vz_ref - vz_com)

        A_com = np.zeros((1, nz), dtype=np.float64)
        A_com[0, qdd_slice] = Jcom_z[0, :]
        b_com = np.array([a_com_z_des - jdq_com_z], dtype=np.float64)

        H_task += w_com * (A_com.T @ A_com)
        g_task += -w_com * (A_com.T @ b_com).flatten()
        # Check for NaN after COM task
        if not np.all(np.isfinite(H_task)):
            import sys as _sc
            print(f"[WARN] COM task produced NaN! Jcom_z finite={np.all(np.isfinite(Jcom_z))} "
                  f"jdq_com_z={jdq_com_z:.6f} z_com={z_com_current:.4f}",
                  file=_sc.stderr, flush=True)
            H_task = np.nan_to_num(H_task, nan=0.0)
            g_task = np.nan_to_num(g_task, nan=0.0)

        per_task["com_height"] = {
            "A": A_com, "b": b_com, "weight": w_com,
            "Jcom_z": Jcom_z, "jdq_com_z": jdq_com_z,
            "a_des": a_com_z_des, "z_current": z_com_current, "z_ref": z_ref,
        }

    # ── Torso orientation task ──────────────────────────────────────
    w_torso = weights.get("w_torso", 0.0)
    if w_torso > 0.0:
        Jr = snapshot.Jr
        jdw_torso = snapshot.jdw_torso
        e_R = snapshot.e_R
        omega_current = snapshot.omega_current

        # NaN guard: clamp degenerate Jacobian values before matmul
        Jr = np.nan_to_num(Jr, nan=0.0, posinf=1e3, neginf=-1e3)
        Jr = np.clip(Jr, -1e4, 1e4)
        jdw_torso = np.nan_to_num(jdw_torso, nan=0.0, posinf=1e3, neginf=-1e3)
        e_R = np.nan_to_num(e_R, nan=0.0, posinf=1.0, neginf=-1.0)
        omega_current = np.nan_to_num(omega_current, nan=0.0, posinf=1e2, neginf=-1e2)

        kp_R = DEFAULT_KP_R
        kd_R = DEFAULT_KD_R
        omega_target = np.zeros(3, dtype=np.float64)

        alpha_des = kp_R * e_R + kd_R * (omega_target - omega_current)

        A_torso = np.zeros((3, nz), dtype=np.float64)
        A_torso[:, qdd_slice] = Jr
        b_torso = alpha_des - jdw_torso

        W_torso = np.diag([w_torso, w_torso, w_torso])
        # Skip torso task if Jr is degenerate (produces NaN Hessian)
        _jr_ok = np.all(np.isfinite(Jr)) and np.all(np.isfinite(jdw_torso)) and np.all(np.isfinite(e_R))
        if not _jr_ok:
            import sys as _sys2
            _jr_max = float(np.max(np.abs(Jr)))
            print(f"[WARN] Torso task SKIPPED: Jr max_abs={_jr_max:.1f} ok={_jr_ok}",
                  file=_sys2.stderr, flush=True)
        else:
            H_task += A_torso.T @ W_torso @ A_torso
            g_task += -(A_torso.T @ W_torso @ b_torso).flatten()
        # Clean up NaN after torso task (regardless of _jr_ok)
        if not np.all(np.isfinite(H_task)):
            import sys as _sc2
            _jr_max_val = float(np.max(np.abs(Jr)))
            print(f"[WARN] Torso task produced NaN! Jr max={_jr_max_val:.1f} finite={np.all(np.isfinite(Jr))} "
                  f"jdw_torso finite={np.all(np.isfinite(jdw_torso))}",
                  file=_sc2.stderr, flush=True)
            H_task = np.nan_to_num(H_task, nan=0.0)
            g_task = np.nan_to_num(g_task, nan=0.0)

        per_task["torso_orientation"] = {
            "A": A_torso, "b": b_torso, "weight": w_torso,
            "Jr": Jr, "jdw_torso": jdw_torso,
            "e_R": e_R, "alpha_des": alpha_des, "omega_current": omega_current,
        }

    # ── Posture task ────────────────────────────────────────────────
    w_posture = weights.get("w_posture", 0.0)
    if w_posture > 0.0:
        q_act_current = snapshot.qpos[7:17].copy()
        qd_act_current = snapshot.qvel[6:16].copy()

        kp_p = DEFAULT_KP_POSTURE
        kd_p = DEFAULT_KD_POSTURE
        q_act_ref = q_act_current
        qd_act_ref = np.zeros(10, dtype=np.float64)

        qdd_act_des = kp_p * (q_act_ref - q_act_current) + kd_p * (qd_act_ref - qd_act_current)

        A_posture = np.zeros((10, nz), dtype=np.float64)
        A_posture[:, 6:16] = np.eye(10)
        b_posture = qdd_act_des

        H_task += w_posture * (A_posture.T @ A_posture)
        g_task += -w_posture * (A_posture.T @ b_posture).flatten()
        if not np.all(np.isfinite(H_task)):
            H_task = np.nan_to_num(H_task, nan=0.0)
            g_task = np.nan_to_num(g_task, nan=0.0)

        per_task["posture"] = {
            "A": A_posture, "b": b_posture, "weight": w_posture,
            "q_act_current": q_act_current, "q_act_ref": q_act_ref,
            "qdd_act_des": qdd_act_des,
        }

    # ── Wheel acceleration regularization ───────────────────────────
    w_wheel = weights.get("w_wheel", 0.0)
    if w_wheel > 0.0:
        wheel_indices = [10, 15]  # l_wheel, r_wheel in qvel

        A_wheel = np.zeros((len(wheel_indices), nz), dtype=np.float64)
        for wi, idx in enumerate(wheel_indices):
            if 0 <= idx < nv:
                A_wheel[wi, idx] = 1.0
        b_wheel = np.zeros(len(wheel_indices), dtype=np.float64)

        H_task += w_wheel * (A_wheel.T @ A_wheel)

        per_task["wheel_accel"] = {
            "A": A_wheel, "b": b_wheel, "weight": w_wheel,
            "wheel_indices": wheel_indices,
        }

    # ── Contact force distribution regularization ───────────────────
    w_force = weights.get("w_force_distribution", 0.0)
    if w_force > 0.0 and m > 0:
        fn_ref = snapshot.robot_weight / max(m, 1)

        lambda_ref = np.zeros(n_lambda, dtype=np.float64)
        for i in range(m):
            lambda_ref[3*i + 0] = fn_ref * 0.1
            lambda_ref[3*i + 1] = 0.0
            lambda_ref[3*i + 2] = 0.0

        A_force = np.zeros((n_lambda, nz), dtype=np.float64)
        A_force[:, lambda_slice] = np.eye(n_lambda)
        b_force = lambda_ref

        H_task += w_force * (A_force.T @ A_force)
        g_task += -w_force * (A_force.T @ b_force).flatten()

        per_task["contact_force"] = {
            "A": A_force, "b": b_force, "weight": w_force,
            "fn_ref": fn_ref, "lambda_ref": lambda_ref,
        }

    # ── Hessian regularization (prevents singularity from degenerate Jacobians) ──
    REG_EPS = 1e-4
    H_task += REG_EPS * np.eye(nz, dtype=np.float64)

    return H_task, g_task, per_task


# ═══════════════════════════════════════════════════════════════════════════
# Task residual evaluation from snapshot (cached Jacobians)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_task_residuals_from_snapshot(
    snapshot: Phase3BSnapshot,
    solution: dict[str, Any],
    task_mode: str,
) -> dict[str, Any]:
    """Evaluate task residuals using cached snapshot data.

    Does NOT call any JAX functions — uses only precomputed data.

    Args:
        snapshot: precomputed ``Phase3BSnapshot``.
        solution: dict from ``solve_offline_qp``.
        task_mode: task mode name.

    Returns:
        dict with per-task residuals and metadata.
    """
    from wheeled_biped.wbc.offline_task_stack import (
        TASK_WEIGHT_MODES,
        DEFAULT_KP_COM, DEFAULT_KD_COM,
        DEFAULT_KP_R, DEFAULT_KD_R,
        DEFAULT_KP_POSTURE, DEFAULT_KD_POSTURE,
    )

    weights = TASK_WEIGHT_MODES.get(task_mode, TASK_WEIGHT_MODES["balanced_default"])

    m = snapshot.m
    nv = snapshot.nv
    n_lambda = 3 * m

    qdd = solution.get("qdd", np.zeros(nv))
    tau = solution.get("tau", np.zeros(10))
    lam = solution.get("lambda", np.zeros(n_lambda))

    residuals = {}

    # ── COM task residual ───────────────────────────────────────────
    w_com = weights.get("w_com", 0.0)
    if w_com > 0.0:
        Jcom_z = snapshot.Jcom[2:3, :]
        jdq_com_z = snapshot.jdq_com[2]
        a_com_z = float(np.dot(Jcom_z, qdd)[0]) + jdq_com_z

        z_com_current = float(snapshot.com_position[2])
        vz_com = float(np.dot(Jcom_z, snapshot.qvel)[0])

        a_des = DEFAULT_KP_COM * (z_com_current - z_com_current) + DEFAULT_KD_COM * (0.0 - vz_com)

        residuals["com"] = {
            "residual": abs(a_com_z - a_des),
            "a_achieved": a_com_z,
            "a_desired": a_des,
            "z_current": z_com_current,
            "z_ref": z_com_current,
            "vz_current": vz_com,
        }

    # ── Torso orientation task residual ─────────────────────────────
    w_torso = weights.get("w_torso", 0.0)
    if w_torso > 0.0:
        Jr = snapshot.Jr
        jdw_torso = snapshot.jdw_torso
        e_R = snapshot.e_R
        omega_current = snapshot.omega_current

        kp_R = DEFAULT_KP_R
        kd_R = DEFAULT_KD_R
        omega_target = np.zeros(3, dtype=np.float64)
        alpha_des = kp_R * e_R + kd_R * (omega_target - omega_current)

        alpha_achieved = Jr @ qdd + jdw_torso
        residual_vec = alpha_achieved - alpha_des

        residuals["torso"] = {
            "residual": float(np.linalg.norm(residual_vec)),
            "residual_vec": residual_vec,
            "alpha_achieved": alpha_achieved,
            "alpha_desired": alpha_des,
            "e_R": e_R,
            "omega_current": omega_current,
        }

    # ── Posture task residual ───────────────────────────────────────
    w_posture = weights.get("w_posture", 0.0)
    if w_posture > 0.0:
        q_act_current = snapshot.qpos[7:17].copy()
        qd_act_current = snapshot.qvel[6:16].copy()

        qdd_act_des = (DEFAULT_KP_POSTURE * (q_act_current - q_act_current) +
                       DEFAULT_KD_POSTURE * (np.zeros(10) - qd_act_current))
        qdd_act = qdd[6:16]
        residual_vec = qdd_act - qdd_act_des

        residuals["posture"] = {
            "residual": float(np.linalg.norm(residual_vec)),
            "residual_vec": residual_vec,
            "qdd_act": qdd_act,
            "qdd_act_des": qdd_act_des,
            "max_qdd_act_des": float(np.max(np.abs(qdd_act_des))),
            "max_qdd_act_solved": float(np.max(np.abs(qdd_act))),
        }

    # ── Wheel acceleration residual ─────────────────────────────────
    w_wheel = weights.get("w_wheel", 0.0)
    if w_wheel > 0.0:
        wheel_indices = [10, 15]
        wheel_qdd = np.array([qdd[i] for i in wheel_indices if 0 <= i < nv])
        residuals["wheel"] = {
            "residual": float(np.linalg.norm(wheel_qdd)),
            "qdd_wheel": wheel_qdd,
            "max_wheel_qdd": float(np.max(np.abs(wheel_qdd))) if len(wheel_qdd) > 0 else 0.0,
            "wheel_indices": wheel_indices,
        }

    # ── Contact force regularization residual ───────────────────────
    w_force = weights.get("w_force_distribution", 0.0)
    if w_force > 0.0 and m > 0:
        fn_ref = snapshot.robot_weight / max(m, 1)
        normal_forces = [float(lam[3*i + 0]) for i in range(m)]

        fn_ref_weak = fn_ref * 0.1
        fn_residual = np.array([abs(fn - fn_ref_weak) for fn in normal_forces])

        residuals["force_distribution"] = {
            "residual": float(np.linalg.norm(fn_residual)),
            "normal_forces": normal_forces,
            "min_normal_force": float(min(normal_forces)) if normal_forces else 0.0,
            "max_normal_force": float(max(normal_forces)) if normal_forces else 0.0,
            "fn_ref_weak": fn_ref_weak,
            "tangent_forces": [float(abs(lam[3*i+1])) + float(abs(lam[3*i+2])) for i in range(m)],
        }

    # ── qdd/tau/lambda magnitudes ──────────────────────────────────
    residuals["qdd_magnitude"] = {
        "max_abs_qdd": float(np.max(np.abs(qdd))),
        "rms_qdd": float(np.sqrt(np.mean(qdd**2))),
    }
    residuals["tau_magnitude"] = {
        "max_abs_tau": float(np.max(np.abs(tau))),
        "rms_tau": float(np.sqrt(np.mean(tau**2))),
    }
    if m > 0:
        residuals["lambda_magnitude"] = {
            "max_abs_lambda": float(np.max(np.abs(lam))),
            "rms_lambda": float(np.sqrt(np.mean(lam**2))),
        }
    residuals["slack"] = {"max_abs_slack": 0.0, "rms_slack": 0.0}

    return residuals


# ═══════════════════════════════════════════════════════════════════════════
# Solution validation from snapshot (cached data)
# ═══════════════════════════════════════════════════════════════════════════

def validate_solution_from_snapshot(
    snapshot: Phase3BSnapshot,
    solution: dict[str, Any],
    _constants: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate QP solution against hard constraints using snapshot data.

    Uses ONLY cached data from the snapshot — no JAX calls.

    Args:
        snapshot: precomputed ``Phase3BSnapshot``.
        solution: dict from ``solve_offline_qp``.
        _constants: unused, kept for API compatibility.

    Returns:
        dict with per-check verdicts and metrics.
    """
    m = snapshot.m
    tau = solution.get("tau", np.zeros(10))
    lam = solution.get("lambda", np.zeros(3 * m))
    qdd = solution.get("qdd", np.zeros(16))

    tau_min = snapshot.tau_min
    tau_max = snapshot.tau_max
    mu = snapshot.mu

    # ── Dynamics residual ───────────────────────────────────────────
    M = snapshot.M
    h_vec = snapshot.h
    S = snapshot.S
    JcT_active = snapshot.contact_stack.get_active_JcT()

    if m > 0:
        dyn_residual = M @ qdd + h_vec - S @ tau - JcT_active @ lam
    else:
        dyn_residual = M @ qdd + h_vec - S @ tau

    max_dyn = float(np.max(np.abs(dyn_residual)))
    max_dyn_fb = float(np.max(np.abs(dyn_residual[0:6])))
    max_dyn_act = float(np.max(np.abs(dyn_residual[6:16])))

    if max_dyn < 1e-5:
        dyn_verdict = "PASS"
    elif max_dyn < 1e-4:
        dyn_verdict = "WARN"
    else:
        dyn_verdict = "FAIL"

    # ── Contact normal acceleration residual ────────────────────────
    max_contact_accel_res = 0.0
    contact_accel_verdict = "PASS"
    if m > 0:
        Jp_active = snapshot.contact_stack.get_active_Jp_stack()
        normals_active = snapshot.contact_stack.get_active_normals()
        jdq = snapshot.jdot_qdot

        accel_residuals = []
        for i in range(m):
            n_i = normals_active[i]
            Jp_i = Jp_active[3*i:3*i+3, :]
            a_p = Jp_i @ qdd + jdq[3*i:3*i+3]
            a_n = np.dot(n_i, a_p)
            accel_residuals.append(abs(a_n))

        max_contact_accel_res = float(max(accel_residuals)) if accel_residuals else 0.0

        if max_contact_accel_res < 1e-4:
            contact_accel_verdict = "PASS"
        elif max_contact_accel_res < 1e-3:
            contact_accel_verdict = "WARN"
        else:
            contact_accel_verdict = "FAIL"

    # ── Friction cone ───────────────────────────────────────────────
    max_friction_violation = 0.0
    friction_verdict = "PASS"
    if m > 0:
        friction_violations = []
        for i in range(m):
            fn = lam[3*i + 0]
            ft1 = lam[3*i + 1]
            ft2 = lam[3*i + 2]
            v_fn = max(0.0, -fn)
            v_ft1 = max(0.0, abs(ft1) - mu * fn)
            v_ft2 = max(0.0, abs(ft2) - mu * fn)
            friction_violations.extend([v_fn, v_ft1, v_ft2])
        max_friction_violation = float(max(friction_violations))

        if max_friction_violation < 1e-6:
            friction_verdict = "PASS"
        elif max_friction_violation < 1e-4:
            friction_verdict = "WARN"
        else:
            friction_verdict = "FAIL"

    # ── Torque limits ───────────────────────────────────────────────
    tau_violations = []
    for i in range(len(tau)):
        v_lo = max(0.0, tau_min[i] - tau[i])
        v_hi = max(0.0, tau[i] - tau_max[i])
        tau_violations.extend([v_lo, v_hi])
    max_torque_violation = float(max(tau_violations))

    if max_torque_violation < 1e-6:
        torque_verdict = "PASS"
    elif max_torque_violation < 1e-4:
        torque_verdict = "WARN"
    else:
        torque_verdict = "FAIL"

    # ── Solution magnitude ──────────────────────────────────────────
    max_abs_qdd = float(np.max(np.abs(qdd)))
    max_abs_tau = float(np.max(np.abs(tau)))
    max_abs_lambda = float(np.max(np.abs(lam))) if m > 0 else 0.0

    return {
        "dynamics": {
            "max_residual": max_dyn,
            "max_free_base_residual": max_dyn_fb,
            "max_actuated_residual": max_dyn_act,
            "verdict": dyn_verdict,
        },
        "contact_normal_acceleration": {
            "max_residual": max_contact_accel_res,
            "verdict": contact_accel_verdict,
        },
        "friction_cone": {
            "max_violation": max_friction_violation,
            "verdict": friction_verdict,
            "mu": mu,
        },
        "torque_limits": {
            "max_violation": max_torque_violation,
            "verdict": torque_verdict,
        },
        "solution_magnitude": {
            "max_abs_qdd": max_abs_qdd,
            "max_abs_tau": max_abs_tau,
            "max_abs_lambda": max_abs_lambda,
        },
        "finite_solution": solution.get("finite_solution", False),
        "solver_success": solution.get("success", False),
    }
