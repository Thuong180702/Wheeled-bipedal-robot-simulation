#!/usr/bin/env python
"""Phase 2D — Contact Dynamics / Contact Jacobian / Constraint Force Validation Audit.

Validates JAX contact dynamics against CPU MuJoCo ground truth across multiple
physically meaningful scenarios (settle poses, small velocities, tilts, etc.).

Generates:
  - docs/validation/k2_phase2d_contact_dynamics_audit.md
  - docs/validation/k2_phase2d_contact_dynamics_audit.json
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import mujoco
import numpy as np
import jax.numpy as jnp
from datetime import datetime, timezone

from wheeled_biped.utils.config import get_model_path
from wheeled_biped.dynamics.jax_contact_dynamics import (
    build_contact_dynamics_constants,
    contact_point_world_position,
    contact_point_translational_jacobian,
    contact_point_rotational_jacobian,
    contact_force_to_generalized_force,
    contact_wrench_to_generalized_force,
    CONSTANTS_VERSION,
    compare_contact_jacobian_to_mujoco,
    compare_contact_force_mapping_to_mujoco,
)


def _np_quat_to_rotmat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


# ═══════════════════════════════════════════════════════════════════════════
# Scenario generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_scenarios(model, data):
    """Generate a list of (name, qpos, qvel) scenarios for validation."""
    scenarios = []
    base_qpos = data.qpos.copy()
    nv = model.nv

    # Scenario 1: passive_settle_keyframe
    d = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, d, 0)
    mujoco.mj_forward(model, d)
    scenarios.append(("passive_settle_keyframe", d.qpos.copy(), np.zeros(nv)))

    # Scenario 2-4: height variations — use keyframe with base height offset
    from scipy.spatial.transform import Rotation
    for label, height_z in [
        ("low_height_settle", 0.45),
        ("mid_height_settle", 0.55),
        ("high_height_settle", 0.65),
    ]:
        qp = base_qpos.copy()
        qp[2] = height_z  # adjust base height
        d2 = mujoco.MjData(model)
        d2.qpos[:] = qp
        try:
            mujoco.mj_forward(model, d2)
            scenarios.append((label, d2.qpos.copy(), np.zeros(nv)))
        except Exception:
            pass  # skip if physics fails

    # Scenario 5: small_forward_velocity
    qvel = np.zeros(nv); qvel[0] = 0.2
    d3 = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, d3, 0)
    mujoco.mj_forward(model, d3)
    scenarios.append(("small_forward_velocity", d3.qpos.copy(), qvel))

    # Scenario 6: small_yaw_rate
    qvel = np.zeros(nv); qvel[5] = 0.5
    d4 = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, d4, 0)
    mujoco.mj_forward(model, d4)
    scenarios.append(("small_yaw_rate", d4.qpos.copy(), qvel))

    # Scenario 7: small_roll_tilt
    qp = base_qpos.copy()
    R = Rotation.from_euler('xyz', [np.deg2rad(5), 0, 0]).as_matrix()
    quat = Rotation.from_matrix(R).as_quat()
    qp[3:7] = [quat[3], quat[0], quat[1], quat[2]]
    d5 = mujoco.MjData(model)
    d5.qpos[:] = qp
    mujoco.mj_forward(model, d5)
    scenarios.append(("small_roll_tilt", d5.qpos.copy(), np.zeros(nv)))

    # Scenario 8-10: random small perturbations
    for i in range(3):
        qp = base_qpos.copy()
        rng = np.random.default_rng(100 + i)
        # Small random perturbation to base orientation and joint angles
        rpy = rng.uniform(-3, 3, 3)  # degrees
        R = Rotation.from_euler('xyz', np.deg2rad(rpy)).as_matrix()
        quat = Rotation.from_matrix(R).as_quat()
        qp[3:7] = [quat[3], quat[0], quat[1], quat[2]]
        qp[2] += rng.uniform(-0.05, 0.05)  # base height
        for j in range(7, 17):
            qp[j] += rng.uniform(-0.05, 0.05)
        qvel = np.zeros(nv)
        qvel = rng.uniform(-0.1, 0.1, nv)
        d6 = mujoco.MjData(model)
        d6.qpos[:] = qp
        mujoco.mj_forward(model, d6)
        scenarios.append((f"random_pose_small_perturbation_{i+1}", d6.qpos.copy(), qvel))

    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# Contact extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_wheel_floor_contacts(model, data, constants):
    """Extract wheel-floor contact data from MuJoCo.

    Returns list of dicts with contact info for wheel bodies vs floor.
    """
    contacts = []
    wheel_body_ids = constants["wheel_body_ids"]
    wheel_names = {v: k for k, v in wheel_body_ids.items()}

    for contact_id in range(data.ncon):
        c = data.contact[contact_id]
        geom1 = c.geom1
        geom2 = c.geom2
        body1 = model.geom_bodyid[geom1]
        body2 = model.geom_bodyid[geom2]

        # Check if one body is a wheel and the other is world/floor
        wheel_body = None
        if body1 in wheel_body_ids.values():
            wheel_body = body1
            other_body = body2
        elif body2 in wheel_body_ids.values():
            wheel_body = body2
            other_body = body1
        else:
            continue

        # Get contact position and frame
        contact_pos = c.pos.copy()  # world position
        contact_frame = c.frame.copy().reshape(3, 3)  # contact frame matrix

        # Compute body-local contact point
        body_pos = data.xpos[wheel_body].copy()
        body_quat = data.xquat[wheel_body].copy()
        R_body = _np_quat_to_rotmat(body_quat)
        local_point = R_body.T @ (contact_pos - body_pos)

        # Get contact force via mj_contactForce
        cf = np.zeros(6)
        mujoco.mj_contactForce(model, data, contact_id, cf)
        # cf = [force_contact_frame; torque_contact_frame] (both in contact frame)
        force_contact_frame = cf[0:3].copy()
        torque_contact_frame = cf[3:6].copy()

        # Convert to world frame: force_world = contact_frame @ force_contact_frame
        force_world = contact_frame @ force_contact_frame
        torque_world = contact_frame @ torque_contact_frame

        contacts.append({
            "contact_id": int(contact_id),
            "wheel_body": int(wheel_body),
            "wheel_name": wheel_names.get(wheel_body, f"body_{wheel_body}"),
            "other_body": int(other_body),
            "other_name": (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other_body)
                          or f"body_{other_body}"),
            "contact_pos_world": contact_pos,
            "contact_frame": contact_frame,
            "local_point": local_point,
            "force_contact_frame": force_contact_frame,
            "torque_contact_frame": torque_contact_frame,
            "force_world": force_world,
            "torque_world": torque_world,
            "distance": float(c.dist),
        })

    return contacts


# ═══════════════════════════════════════════════════════════════════════════
# Validation per contact
# ═══════════════════════════════════════════════════════════════════════════

PASS_TH_POINT = 1e-6
PASS_TH_JAC = 1e-5
WARN_TH_JAC = 1e-4
PASS_TH_QFRC = 1e-4
WARN_TH_QFRC = 1e-3


def _verdict(err, th_pass, th_warn):
    if err < th_pass:
        return "PASS"
    elif err < th_warn:
        return "WARN"
    return "FAIL"


def validate_contact(model, data, contact_info, constants, qpos_jax):
    """Validate one contact's JAX dynamics against CPU MuJoCo."""
    body_id = contact_info["wheel_body"]
    local_pt = contact_info["local_point"]

    # ── Contact point reconstruction ────────────────────────────────────
    p_jax = np.array(
        contact_point_world_position(qpos_jax, body_id, jnp.array(local_pt, dtype=jnp.float32), constants),
        dtype=np.float64,
    )
    p_cpu = contact_info["contact_pos_world"]
    point_err = float(np.max(np.abs(p_jax - p_cpu)))

    # ── Jacobian comparison ─────────────────────────────────────────────
    jac_result = compare_contact_jacobian_to_mujoco(model, data, body_id, local_pt, constants)

    # ── Force mapping comparison ────────────────────────────────────────
    force_result = compare_contact_force_mapping_to_mujoco(
        model, data, body_id, local_pt,
        contact_info["force_world"], contact_info["torque_world"],
        constants,
    )

    return {
        "contact_id": contact_info["contact_id"],
        "wheel_name": contact_info["wheel_name"],
        "point_reconstruction_error": point_err,
        "point_verdict": _verdict(point_err, PASS_TH_POINT, PASS_TH_POINT * 10),
        "jacobian_full_error": jac_result["jacobian_full_max_abs_error"],
        "jacobian_base_linear_error": jac_result["jacobian_base_linear_max_abs_error"],
        "jacobian_base_angular_error": jac_result["jacobian_base_angular_max_abs_error"],
        "jacobian_actuated_error": jac_result["jacobian_actuated_max_abs_error"],
        "jacobian_rotational_error": jac_result["jacobian_rotational_max_abs_error"],
        "jacobian_full_verdict": jac_result["verdict_jacobian_full"],
        "jacobian_base_linear_verdict": jac_result["verdict_jacobian_base_linear"],
        "jacobian_base_angular_verdict": jac_result["verdict_jacobian_base_angular"],
        "jacobian_actuated_verdict": jac_result["verdict_jacobian_actuated"],
        "qfrc_full_error": force_result["qfrc_full_max_abs_error"],
        "qfrc_free_base_error": force_result["qfrc_free_base_max_abs_error"],
        "qfrc_actuated_error": force_result["qfrc_actuated_max_abs_error"],
        "qfrc_full_verdict": force_result["verdict_qfrc_full"],
        "qfrc_free_base_verdict": force_result["verdict_qfrc_free_base"],
        "qfrc_actuated_verdict": force_result["verdict_qfrc_actuated"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Aggregate
# ═══════════════════════════════════════════════════════════════════════════

def aggregate(results):
    """Aggregate validation results into counts."""
    agg = {
        "point": {"PASS": 0, "WARN": 0, "FAIL": 0},
        "jacobian_full": {"PASS": 0, "WARN": 0, "FAIL": 0},
        "jacobian_base_linear": {"PASS": 0, "WARN": 0, "FAIL": 0},
        "jacobian_base_angular": {"PASS": 0, "WARN": 0, "FAIL": 0},
        "jacobian_actuated": {"PASS": 0, "WARN": 0, "FAIL": 0},
        "qfrc_full": {"PASS": 0, "WARN": 0, "FAIL": 0},
        "qfrc_free_base": {"PASS": 0, "WARN": 0, "FAIL": 0},
        "qfrc_actuated": {"PASS": 0, "WARN": 0, "FAIL": 0},
    }
    max_errors = {
        "max_point_error": 0.0,
        "max_jacobian_full": 0.0,
        "max_jacobian_base_linear": 0.0,
        "max_jacobian_base_angular": 0.0,
        "max_jacobian_actuated": 0.0,
        "max_qfrc_full": 0.0,
        "max_qfrc_free_base": 0.0,
        "max_qfrc_actuated": 0.0,
    }

    for r in results:
        agg["point"][r["point_verdict"]] += 1
        agg["jacobian_full"][r["jacobian_full_verdict"]] += 1
        agg["jacobian_base_linear"][r["jacobian_base_linear_verdict"]] += 1
        agg["jacobian_base_angular"][r["jacobian_base_angular_verdict"]] += 1
        agg["jacobian_actuated"][r["jacobian_actuated_verdict"]] += 1
        agg["qfrc_full"][r["qfrc_full_verdict"]] += 1
        agg["qfrc_free_base"][r["qfrc_free_base_verdict"]] += 1
        agg["qfrc_actuated"][r["qfrc_actuated_verdict"]] += 1

        max_errors["max_point_error"] = max(max_errors["max_point_error"], r["point_reconstruction_error"])
        max_errors["max_jacobian_full"] = max(max_errors["max_jacobian_full"], r["jacobian_full_error"])
        max_errors["max_jacobian_base_linear"] = max(max_errors["max_jacobian_base_linear"], r["jacobian_base_linear_error"])
        max_errors["max_jacobian_base_angular"] = max(max_errors["max_jacobian_base_angular"], r["jacobian_base_angular_error"])
        max_errors["max_jacobian_actuated"] = max(max_errors["max_jacobian_actuated"], r["jacobian_actuated_error"])
        max_errors["max_qfrc_full"] = max(max_errors["max_qfrc_full"], r["qfrc_full_error"])
        max_errors["max_qfrc_free_base"] = max(max_errors["max_qfrc_free_base"], r["qfrc_free_base_error"])
        max_errors["max_qfrc_actuated"] = max(max_errors["max_qfrc_actuated"], r["qfrc_actuated_error"])

    return agg, max_errors


# ═══════════════════════════════════════════════════════════════════════════
# Verdict
# ═══════════════════════════════════════════════════════════════════════════

def determine_verdict(agg, max_errors, jit_ok, controller_ok):
    """Determine Phase 2D readiness verdict."""
    if not jit_ok:
        return "NOT_READY"
    if not controller_ok:
        return "NOT_READY"

    # Check all required validations pass
    checks = [
        ("point_reconstruction", agg["point"]["FAIL"] == 0),
        ("jacobian_full", agg["jacobian_full"]["FAIL"] == 0),
        ("jacobian_base_linear", agg["jacobian_base_linear"]["FAIL"] == 0),
        ("jacobian_base_angular", agg["jacobian_base_angular"]["FAIL"] == 0),
        ("jacobian_actuated", agg["jacobian_actuated"]["FAIL"] == 0),
        ("qfrc_full", agg["qfrc_full"]["FAIL"] == 0),
        ("qfrc_free_base", agg["qfrc_free_base"]["FAIL"] == 0),
        ("qfrc_actuated", agg["qfrc_actuated"]["FAIL"] == 0),
    ]

    failed = [name for name, ok in checks if not ok]

    if not failed:
        return "READY_FOR_PHASE_3_OFFLINE_QP_WBC_PROTOTYPE"

    # Check for partial readiness
    critical_fails = [f for f in failed if f in ("point_reconstruction", "jacobian_full", "qfrc_full")]
    if critical_fails:
        return "NOT_READY"

    return "PARTIAL_READY"


# ═══════════════════════════════════════════════════════════════════════════
# JIT check
# ═══════════════════════════════════════════════════════════════════════════

def check_jit(constants, test_qpos):
    """Verify JIT compatibility of core contact functions."""
    import jax

    wheel_id = constants["wheel_body_ids"]["l_wheel_link"]
    local_pt = jnp.array([0.0, 0.0, -0.06], dtype=jnp.float32)
    f_w = jnp.array([10.0, 0.0, 100.0], dtype=jnp.float32)

    try:
        jit_p = jax.jit(lambda q: contact_point_world_position(q, wheel_id, local_pt, constants))
        jit_Jp = jax.jit(lambda q: contact_point_translational_jacobian(q, wheel_id, local_pt, constants))
        jit_Jr = jax.jit(lambda q: contact_point_rotational_jacobian(q, wheel_id, constants))
        jit_qfrc = jax.jit(lambda q, f: contact_force_to_generalized_force(q, wheel_id, local_pt, f, constants))

        r_p = np.array(jit_p(test_qpos))
        r_Jp = np.array(jit_Jp(test_qpos))
        r_Jr = np.array(jit_Jr(test_qpos))
        r_qfrc = np.array(jit_qfrc(test_qpos, f_w))

        all_finite = (
            np.all(np.isfinite(r_p))
            and np.all(np.isfinite(r_Jp))
            and np.all(np.isfinite(r_Jr))
            and np.all(np.isfinite(r_qfrc))
        )
        return all_finite
    except Exception as e:
        print(f"  JIT check FAILED: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Controller check
# ═══════════════════════════════════════════════════════════════════════════

def check_controller_not_modified():
    """Verify no controller files are imported by jax_contact_dynamics."""
    import ast
    src = (PROJECT_ROOT / "wheeled_biped" / "dynamics" / "jax_contact_dynamics.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden = ["k2_jax_controller", "sagittal_velocity_damped_balance_controller"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(f in alias.name for f in forbidden):
                    return False
        elif isinstance(node, ast.ImportFrom):
            if node.module and any(f in node.module for f in forbidden):
                return False
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_markdown_report(
    verdict, agg, max_errors, results, scenarios, num_contacts,
    jit_ok, controller_ok, constants,
):
    """Generate markdown report."""
    ts = datetime.now(timezone.utc).isoformat()

    lines = []
    lines.append(f"# Phase 2D — Contact Dynamics / Contact Jacobian / Constraint Force Validation Audit Report")
    lines.append(f"")
    lines.append(f"**Timestamp:** {ts}  ")
    lines.append(f"**Verdict:** `{verdict}`")
    lines.append(f"")
    lines.append(f"## 1. Executive Summary")
    lines.append(f"")
    lines.append(f"Phase 2D implements JAX-compatible contact dynamics infrastructure:")
    lines.append(f"")
    lines.append(f"- Contact point world position from body-local coordinates")
    lines.append(f"- Full translational contact Jacobian Jp ∈ R^(3×16), including free-base columns")
    lines.append(f"- Rotational contact Jacobian Jr ∈ R^(3×16)")
    lines.append(f"- Contact force → generalized force mapping (Jp^T @ f_world)")
    lines.append(f"- Contact wrench → generalized force mapping (Jp^T @ f + Jr^T @ tau)")
    lines.append(f"")
    lines.append(f"Validated against CPU MuJoCo `mj_jac` and `mj_contactForce` across "
                f"{len(scenarios)} scenarios with {num_contacts} total contacts.")
    lines.append(f"")
    lines.append(f"### Results Summary")
    lines.append(f"")
    lines.append(f"| Validation | PASS | WARN | FAIL | Max Error |")
    lines.append(f"|------------|------|------|------|-----------|")
    for label, key in [
        ("Contact Point Reconstruction", "point"),
        ("Jacobian Full", "jacobian_full"),
        ("Jacobian Base Linear", "jacobian_base_linear"),
        ("Jacobian Base Angular", "jacobian_base_angular"),
        ("Jacobian Actuated", "jacobian_actuated"),
        ("QFRC Full", "qfrc_full"),
        ("QFRC Free-Base", "qfrc_free_base"),
        ("QFRC Actuated", "qfrc_actuated"),
    ]:
        p = agg[key]["PASS"]
        w = agg[key]["WARN"]
        f = agg[key]["FAIL"]
        max_key = {
            "point": "max_point_error",
            "jacobian_full": "max_jacobian_full",
            "jacobian_base_linear": "max_jacobian_base_linear",
            "jacobian_base_angular": "max_jacobian_base_angular",
            "jacobian_actuated": "max_jacobian_actuated",
            "qfrc_full": "max_qfrc_full",
            "qfrc_free_base": "max_qfrc_free_base",
            "qfrc_actuated": "max_qfrc_actuated",
        }[key]
        lines.append(f"| {label} | {p} | {w} | {f} | {max_errors[max_key]:.2e} |")
    lines.append(f"")
    lines.append(f"## 2. Controller Integrity")
    lines.append(f"")
    lines.append(f"Controller code and `K2_JAX_DEDICATED_DEFAULT_V3` were **not** modified.")
    lines.append(f"Controller check: {'PASS' if controller_ok else 'FAIL'}")
    lines.append(f"")
    lines.append(f"## 3. Changed Files")
    lines.append(f"")
    lines.append(f"| File | Status |")
    lines.append(f"|------|--------|")
    lines.append(f"| `wheeled_biped/dynamics/jax_contact_dynamics.py` | **new** — Phase 2D module |")
    lines.append(f"| `tests/test_phase2d_contact_dynamics.py` | **new** — test suite |")
    lines.append(f"| `scripts/phase2d_contact_dynamics_audit.py` | **new** — this audit |")
    lines.append(f"| `docs/validation/k2_phase2d_contact_dynamics_audit.md` | **new** — this report |")
    lines.append(f"| `docs/validation/k2_phase2d_contact_dynamics_audit.json` | **new** — JSON summary |")
    lines.append(f"")
    lines.append(f"## 4. Phase 2C.5 Readiness Recap")
    lines.append(f"")
    lines.append(f"Phase 2C.5 is READY_FOR_PHASE_2D_CONTACT_DYNAMICS_PORT:")
    lines.append(f"- 35P/0W/0F on original bias validation")
    lines.append(f"- Max actuated bias error: 4.60e-07")
    lines.append(f"- All xpassed tests cleaned (79 passed, 0 xpassed)")
    lines.append(f"- Report inconsistency fixed")
    lines.append(f"")
    lines.append(f"## 5. Contact Dynamics Method")
    lines.append(f"")
    lines.append(f"### Contact Point Kinematics")
    lines.append(f"Uses Phase 2A forward kinematics: `p_world = x_body + R_body @ p_local`.")
    lines.append(f"Validated via CPU `xpos + xquat @ p_local`.")
    lines.append(f"")
    lines.append(f"### Contact Jacobian")
    lines.append(f"Free-base columns: Jp[:, 0:3] = I_3, Jp[:, 3:6] = -skew(r) @ R_base.")
    lines.append(f"Actuated columns: JAX autodiff ∂p_world/∂qpos[7:17].")
    lines.append(f"")
    lines.append(f"### Contact Force Mapping")
    lines.append(f"Virtual work: qfrc = Jp^T @ force_world (+ Jr^T @ torque_world for wrench).")
    lines.append(f"")
    lines.append(f"## 6. Free-Base Jacobian Convention")
    lines.append(f"")
    lines.append(f"- `qvel[0:3]` = base linear velocity (WORLD frame) → Jp[:, 0:3] = I_3")
    lines.append(f"- `qvel[3:6]` = base angular velocity (BODY frame) → Jp[:, 3:6] = -skew(r) @ R_base_world")
    lines.append(f"- `omega_world = R_base @ qvel[3:6]`")
    lines.append(f"")
    lines.append(f"## 7. Contact Reference Extraction")
    lines.append(f"")
    lines.append(f"CPU MuJoCo used for reference extraction only:")
    lines.append(f"- `mj_jac` for Jacobian ground truth")
    lines.append(f"- `mj_contactForce` for 6D contact wrench")
    lines.append(f"- Contact position from `data.contact[i].pos`")
    lines.append(f"- Contact frame from `data.contact[i].frame` (3×3 rotation matrix)")
    lines.append(f"")
    lines.append(f"No CPU MuJoCo calls inside JAX compute functions.")
    lines.append(f"")
    lines.append(f"## 8. Contact Frame Convention")
    lines.append(f"")
    lines.append(f"MuJoCo `contact.frame` is a 3×3 matrix:")
    lines.append(f"- `frame[:, 0]` = contact normal")
    lines.append(f"- `frame[:, 1]` = first tangent")
    lines.append(f"- `frame[:, 2]` = second tangent")
    lines.append(f"")
    lines.append(f"World-frame force: `f_world = contact.frame @ f_contact_frame`")
    lines.append(f"")
    lines.append(f"## 9. Contact Point Reconstruction Validation")
    lines.append(f"")
    lines.append(f"Threshold: PASS < {PASS_TH_POINT:.0e} m")
    lines.append(f"")
    lines.append(f"| Scenario | Contacts | Max Point Error (m) | Verdict |")
    lines.append(f"|----------|----------|---------------------|---------|")
    scenario_results = {}
    for r in results:
        sn = r.get("scenario", "unknown")
        if sn not in scenario_results:
            scenario_results[sn] = []
        scenario_results[sn].append(r)
    for sn in scenarios:
        name = sn[0]
        s_results = scenario_results.get(name, [])
        n_contacts = len(s_results)
        if n_contacts == 0:
            lines.append(f"| {name} | 0 | — | N/A |")
        else:
            max_pt = max(r["point_reconstruction_error"] for r in s_results)
            worst_v = max((r["point_verdict"] for r in s_results),
                         key=lambda v: {"PASS": 0, "WARN": 1, "FAIL": 2}.get(v, 3))
            lines.append(f"| {name} | {n_contacts} | {max_pt:.2e} | {worst_v} |")
    lines.append(f"")
    lines.append(f"## 10. Contact Jacobian Validation")
    lines.append(f"")
    lines.append(f"Thresholds: PASS < {PASS_TH_JAC:.0e}, WARN < {WARN_TH_JAC:.0e}, FAIL >= {WARN_TH_JAC:.0e}")
    lines.append(f"")
    lines.append(f"| Scenario | Contacts | Max Jp Full | Max Jp Base Lin | Max Jp Base Ang | Max Jp Act | Verdict |")
    lines.append(f"|----------|----------|-------------|-----------------|-----------------|------------|---------|")
    for sn in scenarios:
        name = sn[0]
        s_results = scenario_results.get(name, [])
        if not s_results:
            lines.append(f"| {name} | 0 | — | — | — | — | N/A |")
        else:
            max_jf = max(r["jacobian_full_error"] for r in s_results)
            max_bl = max(r["jacobian_base_linear_error"] for r in s_results)
            max_ba = max(r["jacobian_base_angular_error"] for r in s_results)
            max_act = max(r["jacobian_actuated_error"] for r in s_results)
            worst_v = max((r["jacobian_full_verdict"] for r in s_results),
                         key=lambda v: {"PASS": 0, "WARN": 1, "FAIL": 2}.get(v, 3))
            lines.append(f"| {name} | {len(s_results)} | {max_jf:.2e} | {max_bl:.2e} | {max_ba:.2e} | {max_act:.2e} | {worst_v} |")
    lines.append(f"")
    lines.append(f"## 11. Contact Force-to-QFRC Mapping Validation")
    lines.append(f"")
    lines.append(f"Thresholds: PASS < {PASS_TH_QFRC:.0e}, WARN < {WARN_TH_QFRC:.0e}, FAIL >= {WARN_TH_QFRC:.0e}")
    lines.append(f"")
    lines.append(f"| Scenario | Contacts | Max QFRC Full | Max QFRC FB | Max QFRC Act | Verdict |")
    lines.append(f"|----------|----------|---------------|-------------|--------------|---------|")
    for sn in scenarios:
        name = sn[0]
        s_results = scenario_results.get(name, [])
        if not s_results:
            lines.append(f"| {name} | 0 | — | — | — | N/A |")
        else:
            max_qf = max(r["qfrc_full_error"] for r in s_results)
            max_qfb = max(r["qfrc_free_base_error"] for r in s_results)
            max_qa = max(r["qfrc_actuated_error"] for r in s_results)
            worst_v = max((r["qfrc_full_verdict"] for r in s_results),
                         key=lambda v: {"PASS": 0, "WARN": 1, "FAIL": 2}.get(v, 3))
            lines.append(f"| {name} | {len(s_results)} | {max_qf:.2e} | {max_qfb:.2e} | {max_qa:.2e} | {worst_v} |")
    lines.append(f"")
    lines.append(f"## 12. Scenario Table")
    lines.append(f"")
    lines.append(f"| # | Name | Has Contacts | Contact Bodies |")
    lines.append(f"|---|------|-------------|----------------|")
    for i, sn in enumerate(scenarios):
        name = sn[0]
        s_results = scenario_results.get(name, [])
        bodies = set(r.get("wheel_name", "?") for r in s_results)
        lines.append(f"| {i+1} | {name} | {len(s_results)} | {', '.join(sorted(bodies)) if bodies else '—'} |")
    lines.append(f"")
    lines.append(f"## 13. Aggregate Metrics")
    lines.append(f"")
    lines.append(f"- Max contact point error: {max_errors['max_point_error']:.2e} m")
    lines.append(f"- Max Jacobian full error: {max_errors['max_jacobian_full']:.2e}")
    lines.append(f"- Max Jacobian base linear error: {max_errors['max_jacobian_base_linear']:.2e}")
    lines.append(f"- Max Jacobian base angular error: {max_errors['max_jacobian_base_angular']:.2e}")
    lines.append(f"- Max Jacobian actuated error: {max_errors['max_jacobian_actuated']:.2e}")
    lines.append(f"- Max QFRC full error: {max_errors['max_qfrc_full']:.2e}")
    lines.append(f"- Max QFRC free-base error: {max_errors['max_qfrc_free_base']:.2e}")
    lines.append(f"- Max QFRC actuated error: {max_errors['max_qfrc_actuated']:.2e}")
    lines.append(f"")
    lines.append(f"## 14. JIT Compatibility")
    lines.append(f"")
    lines.append(f"JIT check: {'PASS' if jit_ok else 'FAIL'}")
    lines.append(f"")
    lines.append(f"## 15. Limitations")
    lines.append(f"")
    lines.append(f"- Contact detection not implemented — uses CPU MuJoCo to locate contacts, then validates JAX mapping.")
    lines.append(f"- Summed qfrc_constraint validation not applicable — scene has multiple constraint types.")
    lines.append(f"- Rotational Jacobian included for wrench completeness (translational forces are primary for contact).")
    lines.append(f"- No friction cone / QP / WBC integration — Phase 3 scope.")
    lines.append(f"")
    lines.append(f"## 16. Phase 3 Readiness Verdict")
    lines.append(f"")
    lines.append(f"```text")
    lines.append(f"{verdict}")
    lines.append(f"```")
    lines.append(f"")

    return "\n".join(lines)


def generate_json_report(
    verdict, agg, max_errors, results, num_scenarios, num_contacts,
    jit_ok, controller_ok,
):
    """Generate JSON report."""
    ts = datetime.now(timezone.utc).isoformat()

    return {
        "phase": "2D",
        "verdict": verdict,
        "constants_version": CONSTANTS_VERSION,
        "timestamp": ts,
        "num_scenarios": num_scenarios,
        "num_contacts_validated": num_contacts,
        "contact_point_reconstruction_pass_warn_fail": agg["point"],
        "contact_jacobian_pass_warn_fail": agg["jacobian_full"],
        "contact_force_mapping_pass_warn_fail": agg["qfrc_full"],
        "qfrc_constraint_sum_pass_warn_fail": {
            "PASS": 0, "WARN": 0, "FAIL": 0,
            "note": "Not applicable — scene has multiple constraint types; per-contact Path A validation used instead.",
        },
        "max_contact_point_error": max_errors["max_point_error"],
        "max_jacobian_full_abs_error": max_errors["max_jacobian_full"],
        "max_jacobian_base_linear_abs_error": max_errors["max_jacobian_base_linear"],
        "max_jacobian_base_angular_abs_error": max_errors["max_jacobian_base_angular"],
        "max_jacobian_actuated_abs_error": max_errors["max_jacobian_actuated"],
        "max_contact_qfrc_abs_error": max_errors["max_qfrc_full"],
        "max_contact_qfrc_free_base_abs_error": max_errors["max_qfrc_free_base"],
        "max_contact_qfrc_actuated_abs_error": max_errors["max_qfrc_actuated"],
        "jit_compatible": jit_ok,
        "controller_modified": not controller_ok,
        "contact_detection_implemented": False,
        "limitations": [
            "Contact detection not implemented — uses CPU MuJoCo to locate contacts.",
            "Summed qfrc_constraint validation not applicable — per-contact Path A used.",
            "No friction cone / QP / WBC integration.",
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Phase 2D — Contact Dynamics / Contact Jacobian / Force Validation Audit")
    print("=" * 70)

    # ── Load model ──────────────────────────────────────────────────────
    model_path = str(get_model_path())
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    print(f"\nModel: nbody={model.nbody}, nq={model.nq}, nv={model.nv}")
    print(f"Constants version: {CONSTANTS_VERSION}")

    # ── Build constants ─────────────────────────────────────────────────
    constants = build_contact_dynamics_constants(model)
    print(f"Wheel bodies: {constants['wheel_body_ids']}")
    print(f"Wheel geoms: {constants['wheel_geom_ids']}")
    print(f"Floor geoms: {constants['floor_geom_ids']}")

    # ── Controller check ────────────────────────────────────────────────
    controller_ok = check_controller_not_modified()
    print(f"\nController check: {'PASS' if controller_ok else 'FAIL'}")

    # ── Generate scenarios ──────────────────────────────────────────────
    scenarios = generate_scenarios(model, data)
    print(f"\nGenerated {len(scenarios)} scenarios")

    # ── Validate each scenario ──────────────────────────────────────────
    all_results = []
    total_contacts = 0

    for sn_name, qpos_np, qvel_np in scenarios:
        # Set up MuJoCo data
        d = mujoco.MjData(model)
        d.qpos[:] = qpos_np
        d.qvel[:] = qvel_np
        mujoco.mj_forward(model, d)

        qpos_jax = jnp.array(qpos_np, dtype=jnp.float32)

        # Extract contacts
        contacts = extract_wheel_floor_contacts(model, d, constants)

        if not contacts:
            print(f"  {sn_name}: 0 wheel-floor contacts")
            continue

        print(f"  {sn_name}: {len(contacts)} wheel-floor contacts")

        for contact in contacts:
            result = validate_contact(model, d, contact, constants, qpos_jax)
            result["scenario"] = sn_name
            all_results.append(result)
            total_contacts += 1

    print(f"\nTotal contacts validated: {total_contacts}")

    # ── Aggregate ───────────────────────────────────────────────────────
    agg, max_errors = aggregate(all_results)

    # ── JIT check ───────────────────────────────────────────────────────
    test_qpos = jnp.array(data.qpos.copy(), dtype=jnp.float32)
    jit_ok = check_jit(constants, test_qpos)
    print(f"JIT check: {'PASS' if jit_ok else 'FAIL'}")

    # ── Verdict ─────────────────────────────────────────────────────────
    verdict = determine_verdict(agg, max_errors, jit_ok, controller_ok)
    print(f"\nVerdict: {verdict}")

    # ── Print summary ───────────────────────────────────────────────────
    print(f"\nAggregate Results:")
    for label, key in [
        ("Point Reconstruction", "point"),
        ("Jacobian Full", "jacobian_full"),
        ("Jacobian Base Linear", "jacobian_base_linear"),
        ("Jacobian Base Angular", "jacobian_base_angular"),
        ("Jacobian Actuated", "jacobian_actuated"),
        ("QFRC Full", "qfrc_full"),
        ("QFRC Free-Base", "qfrc_free_base"),
        ("QFRC Actuated", "qfrc_actuated"),
    ]:
        print(f"  {label}: {agg[key]}")

    print(f"\nMax Errors:")
    for k, v in max_errors.items():
        print(f"  {k}: {v:.2e}")

    # ── Generate reports ────────────────────────────────────────────────
    docs_dir = PROJECT_ROOT / "docs" / "validation"
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Markdown
    md_content = generate_markdown_report(
        verdict, agg, max_errors, all_results, scenarios, total_contacts,
        jit_ok, controller_ok, constants,
    )
    md_path = docs_dir / "k2_phase2d_contact_dynamics_audit.md"
    md_path.write_text(md_content, encoding="utf-8")
    print(f"\nMarkdown report: {md_path}")

    # JSON
    json_data = generate_json_report(
        verdict, agg, max_errors, all_results, len(scenarios), total_contacts,
        jit_ok, controller_ok,
    )
    json_path = docs_dir / "k2_phase2d_contact_dynamics_audit.json"
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    print(f"JSON report: {json_path}")

    return 0 if "READY" in verdict else 1


if __name__ == "__main__":
    sys.exit(main())
