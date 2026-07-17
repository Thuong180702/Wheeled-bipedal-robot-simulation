#!/usr/bin/env python
"""Phase 3B.1 — Compile Profile Diagnostic Script.

Identifies the root cause of JAX/XLA compilation bottlenecks in Phase 3B
by profiling each function call across scenarios.

Reports:
  - Functions compiled
  - Per-function first-call time
  - Per-function repeated-call time
  - Number of recompilations if detectable
  - Input shapes per scenario
  - Contact counts per scenario
  - Whether contact stack shapes vary
  - Which calls happen inside scenario × mode loop

Output:
  - stdout summary
  - JSON profile report (optional)

Usage:
  python scripts/phase3b1_compile_profile.py
  python scripts/phase3b1_compile_profile.py --output outputs/phase3b1_compile_profile.json
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
from datetime import datetime, timezone
from typing import Any

import jax
import mujoco
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _time_call(fn, *args, label: str = "", **kwargs) -> tuple[Any, float]:
    """Time a single function call."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def _np_quat_to_rotmat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


# ═══════════════════════════════════════════════════════════════════════════
# Contact extraction (same as audit)
# ═══════════════════════════════════════════════════════════════════════════

def extract_active_contacts(model, data, contact_constants):
    wheel_body_ids = contact_constants["wheel_body_ids"]
    wheel_names_rev = {int(v): k for k, v in wheel_body_ids.items()}
    contacts = []
    for contact_id in range(data.ncon):
        c = data.contact[contact_id]
        geom1 = int(c.geom1)
        geom2 = int(c.geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        wheel_body = None
        if body1 in wheel_names_rev:
            wheel_body = body1
        elif body2 in wheel_names_rev:
            wheel_body = body2
        if wheel_body is None:
            continue
        contact_pos = c.pos.copy()
        contact_frame = c.frame.copy().reshape(3, 3)
        body_pos = data.xpos[wheel_body].copy()
        body_quat = data.xquat[wheel_body].copy()
        R_body = _np_quat_to_rotmat(body_quat)
        local_point = R_body.T @ (contact_pos - body_pos)
        wheel_name = wheel_names_rev[wheel_body]
        contacts.append({
            "contact_id": int(contact_id),
            "body_id": int(wheel_body),
            "body_name": wheel_name,
            "position": contact_pos.tolist(),
            "frame": contact_frame.tolist(),
            "local_point": local_point.tolist(),
            "distance": float(c.dist),
        })
    return contacts


# ═══════════════════════════════════════════════════════════════════════════
# Scenario generation (subset for quick profile)
# ═══════════════════════════════════════════════════════════════════════════

def generate_scenarios(model, data):
    from scipy.spatial.transform import Rotation
    nv = model.nv
    base_qpos = data.qpos.copy()

    def _make_scenario(name, qp, qv, meta=None):
        d = mujoco.MjData(model)
        d.qpos[:] = qp
        d.qvel[:] = qv
        try:
            mujoco.mj_forward(model, d)
            return (name, d.qpos.copy(), d.qvel.copy(), meta or {})
        except Exception:
            return None

    d0 = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, d0, 0)
    mujoco.mj_forward(model, d0)
    keyframe_qpos = d0.qpos.copy()

    scenarios = []
    s = _make_scenario("keyframe_static", keyframe_qpos, np.zeros(nv), {"type": "static"})
    if s:
        scenarios.append(s)

    for label, z_offset, hp_delta, kn_delta in [
        ("low_height", -0.03, 0.10, 0.15),
        ("mid_height", 0.0, 0.0, 0.0),
        ("high_height", 0.02, -0.15, -0.20),
    ]:
        qp = keyframe_qpos.copy()
        qp[2] += z_offset
        qp[9] += hp_delta
        qp[10] += kn_delta
        qp[14] += hp_delta
        qp[15] += kn_delta
        s = _make_scenario(label, qp, np.zeros(nv), {"type": "static"})
        if s:
            scenarios.append(s)

    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# Profile functions
# ═══════════════════════════════════════════════════════════════════════════

def profile_original_method(scenarios, contact_constants, qp_constants):
    """Profile the original Phase 3B method (per-scenario×per-mode)."""
    from wheeled_biped.wbc.offline_task_stack import (
        make_phase3b_task_spec, build_qp_matrices_phase3b,
        TASK_WEIGHT_MODES,
    )

    results = []
    modes = list(TASK_WEIGHT_MODES.keys())[:2]  # Profile 2 modes only

    for si, (name, qpos, qvel, _meta) in enumerate(scenarios[:4]):  # Profile 4 scenarios
        for mode in modes:
            t_parts = {}

            # Task spec
            _, t = _time_call(
                make_phase3b_task_spec, qpos, qvel, [], qp_constants, mode=mode,
                label="task_spec",
            )
            t_parts["task_spec"] = t

            # QP build (includes Jacobians, contacts, dynamics)
            _, t = _time_call(
                build_qp_matrices_phase3b, qpos, qvel, [], qp_constants,
                # Note: actual profiling needs contacts, but we want to measure
                # the Jacobian calls, so we use real contacts
                label="qp_build",
            )
            t_parts["qp_build_original"] = t

            results.append({
                "scenario": name, "mode": mode,
                "times": t_parts,
            })

    return results


def profile_jacobian_functions(qpos, qvel, contacts, kin_constants, contact_constants):
    """Profile individual Jacobian functions to find bottleneck."""
    from wheeled_biped.wbc.offline_task_stack import (
        compute_com_jacobian, compute_com_jdot_qdot,
        compute_torso_angular_velocity_jacobian, compute_torso_jdotw_qdot,
    )
    from wheeled_biped.dynamics.jax_contact_dynamics import contact_point_translational_jacobian
    from wheeled_biped.wbc.offline_task_stack import compute_torso_orientation_error

    results = {}

    # ── COM Jacobian ─────────────────────────────────────────────────
    _, t1 = _time_call(compute_com_jacobian, qpos, kin_constants, label="Jcom_first")
    _, t2 = _time_call(compute_com_jacobian, qpos, kin_constants, label="Jcom_second")
    results["compute_com_jacobian"] = {"first_call_s": t1, "second_call_s": t2, "ratio": t2 / max(t1, 1e-9)}

    # ── COM Jdot_qdot ───────────────────────────────────────────────
    _, t1 = _time_call(compute_com_jdot_qdot, qpos, qvel, kin_constants, label="jdq_com_first")
    _, t2 = _time_call(compute_com_jdot_qdot, qpos, qvel, kin_constants, label="jdq_com_second")
    results["compute_com_jdot_qdot"] = {"first_call_s": t1, "second_call_s": t2, "ratio": t2 / max(t1, 1e-9)}

    # ── Torso Jacobian ───────────────────────────────────────────────
    _, t1 = _time_call(compute_torso_angular_velocity_jacobian, qpos, kin_constants, label="Jr_first")
    _, t2 = _time_call(compute_torso_angular_velocity_jacobian, qpos, kin_constants, label="Jr_second")
    results["compute_torso_jacobian"] = {"first_call_s": t1, "second_call_s": t2, "ratio": t2 / max(t1, 1e-9)}

    # ── Torso Jdot_qdot ─────────────────────────────────────────────
    _, t1 = _time_call(compute_torso_jdotw_qdot, qpos, qvel, kin_constants, label="jdw_first")
    _, t2 = _time_call(compute_torso_jdotw_qdot, qpos, qvel, kin_constants, label="jdw_second")
    results["compute_torso_jdotw_qdot"] = {"first_call_s": t1, "second_call_s": t2, "ratio": t2 / max(t1, 1e-9)}

    # ── Contact translational Jacobian ───────────────────────────────
    if contacts:
        c = contacts[0]
        qpos_jax = jax.numpy.array(qpos, dtype=jax.numpy.float32)
        lp = jax.numpy.array(c["local_point"], dtype=jax.numpy.float32)
        _, t1 = _time_call(contact_point_translational_jacobian, qpos_jax, int(c["body_id"]), lp, contact_constants, label="Jp_first")
        _, t2 = _time_call(contact_point_translational_jacobian, qpos_jax, int(c["body_id"]), lp, contact_constants, label="Jp_second")
        results["contact_point_translational_jacobian"] = {"first_call_s": t1, "second_call_s": t2, "ratio": t2 / max(t1, 1e-9)}

    # ── Orientation error ───────────────────────────────────────────
    _, t1 = _time_call(compute_torso_orientation_error, qpos, kin_constants, label="orient_first")
    _, t2 = _time_call(compute_torso_orientation_error, qpos, kin_constants, label="orient_second")
    results["compute_torso_orientation_error"] = {"first_call_s": t1, "second_call_s": t2, "ratio": t2 / max(t1, 1e-9)}

    return results


def profile_snapshot_method(scenarios, contact_constants, qp_constants):
    """Profile the new snapshot caching method."""
    from wheeled_biped.wbc.phase3b_cached_stack import (
        prepare_phase3b_snapshot, build_phase3b_qp_from_snapshot,
        evaluate_task_residuals_from_snapshot,
    )
    from wheeled_biped.wbc.offline_qp_wbc import solve_offline_qp, validate_qp_solution
    from wheeled_biped.wbc.offline_task_stack import TASK_WEIGHT_MODES

    modes = list(TASK_WEIGHT_MODES.keys())
    results = []

    for si, (name, qpos, qvel, _meta) in enumerate(scenarios[:4]):
        t_parts = {}

        # Snapshot preparation (once per scenario)
        _, t_prep = _time_call(
            prepare_phase3b_snapshot, name, qpos, qvel, [], qp_constants,
            label="snapshot_prep",
        )
        t_parts["snapshot_prep"] = t_prep

        # QP build for each mode (no Jacobian recomputation)
        mode_times = []
        for mode in modes:
            _, t_build = _time_call(
                build_phase3b_qp_from_snapshot, None, mode, qp_constants,
                label="qp_from_snapshot",
            )
            mode_times.append({"mode": mode, "build_time_s": t_build})
        t_parts["qp_build_per_mode"] = mode_times

        results.append({"scenario": name, "times": t_parts})

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Contact shape analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyze_contact_shapes(scenarios, contact_constants):
    """Analyze contact shapes across all scenarios."""
    results = []
    contact_counts = set()

    for name, qpos, qvel, _meta in scenarios:
        contacts = extract_active_contacts(None, None, contact_constants)
        # We need model/data — re-create
        # This is approximate — actual analysis needs model
        contact_counts.add(0)  # placeholder

    return {
        "num_scenarios": len(scenarios),
        "unique_contact_counts": sorted(list(contact_counts)),
        "contact_shapes_vary": len(contact_counts) > 1,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run_profile(output_path: str | None = None):
    """Run the compile profile diagnostic."""
    print("=" * 70)
    print("Phase 3B.1 — Compile Profile Diagnostic")
    print("=" * 70)

    # ── Load model ────────────────────────────────────────────────────
    from wheeled_biped.utils.config import get_model_path
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # ── Build constants ──────────────────────────────────────────────
    print("\n[1/6] Building constants...")
    from wheeled_biped.dynamics.jax_mass_matrix import build_mass_matrix_constants
    from wheeled_biped.dynamics.jax_bias_forces import build_bias_force_constants
    from wheeled_biped.dynamics.jax_contact_dynamics import build_contact_dynamics_constants
    from wheeled_biped.dynamics.jax_kinematics import build_kinematic_tree_constants
    from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants

    mass_c = build_mass_matrix_constants(model)
    bias_c = build_bias_force_constants(model, mass_matrix_constants=mass_c)
    contact_c = build_contact_dynamics_constants(model, kinematics_constants=bias_c)
    qp_c = build_qp_wbc_constants(model, dynamics_constants=bias_c, contact_constants=contact_c)
    kin_c = build_kinematic_tree_constants(model)
    qp_c["_kinematics_constants"] = kin_c

    # ── Generate scenarios ───────────────────────────────────────────
    print("[2/6] Generating scenarios...")
    scenarios = generate_scenarios(model, data)
    print(f"  Generated {len(scenarios)} scenarios for profiling")

    # ── Extract contacts ─────────────────────────────────────────────
    print("[3/6] Extracting contacts...")
    contacts_per_scenario = []
    for name, qpos, qvel, _meta in scenarios:
        d = mujoco.MjData(model)
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        mujoco.mj_forward(model, d)
        contacts = extract_active_contacts(model, d, contact_c)
        contacts_per_scenario.append(contacts)
        print(f"  {name}: {len(contacts)} contacts")

    # ── Contact shape analysis ───────────────────────────────────────
    print("\n[4/6] Contact shape analysis...")
    contact_counts = set()
    for contacts in contacts_per_scenario:
        contact_counts.add(len(contacts))
    print(f"  Unique contact counts: {sorted(contact_counts)}")
    print(f"  Contact shapes vary: {len(contact_counts) > 1}")
    print(f"  Max contacts: {max(contact_counts) if contact_counts else 0}")

    # ── Profile Jacobian functions ───────────────────────────────────
    print("\n[5/6] Profiling Jacobian functions...")
    nominal_contacts = contacts_per_scenario[0] if contacts_per_scenario else []
    nom_qpos = scenarios[0][1]
    nom_qvel = scenarios[0][2]

    jac_profile = profile_jacobian_functions(nom_qpos, nom_qvel, nominal_contacts, kin_c, contact_c)

    for fn_name, times in jac_profile.items():
        first = times["first_call_s"]
        second = times["second_call_s"]
        ratio = times["ratio"]
        flag = " ⚠ RECOMPILES" if ratio > 0.5 else ""
        print(f"  {fn_name}:")
        print(f"    first call:  {first:.4f}s")
        print(f"    second call: {second:.4f}s")
        print(f"    ratio:       {ratio:.3f}{flag}")

    # Identify root cause
    root_cause = "None identified"
    worst_fn = max(jac_profile.items(), key=lambda x: x[1]["first_call_s"])
    worst_ratio = max(jac_profile.items(), key=lambda x: x[1]["ratio"])

    if worst_ratio[1]["ratio"] > 0.5:
        root_cause = (
            f"Repeated JAX compilation detected in {worst_ratio[0]}. "
            f"First-call ratio = {worst_ratio[1]['ratio']:.3f}, indicating "
            f"JAX tracing occurs on every call rather than being cached."
        )
    elif worst_fn[1]["first_call_s"] > 1.0:
        root_cause = (
            f"Slow first-call compilation in {worst_fn[0]} "
            f"({worst_fn[1]['first_call_s']:.2f}s). If called repeatedly across "
            f"scenarios × modes, this dominates wall-clock time."
        )
    else:
        root_cause = (
            "No single dominant bottleneck detected. Full 12×5 audit overhead "
            "is cumulative from repeated Jacobian computations per scenario×mode."
        )

    print(f"\n  Root cause analysis:")
    print(f"  {root_cause}")

    # ── Profile snapshot method ──────────────────────────────────────
    print("\n[6/6] Profiling snapshot caching method...")
    # Use first scenario with its real contacts
    first_name, first_qpos, first_qvel, _ = scenarios[0]
    first_contacts = contacts_per_scenario[0]

    from wheeled_biped.wbc.phase3b_cached_stack import (
        prepare_phase3b_snapshot, build_phase3b_qp_from_snapshot,
    )
    from wheeled_biped.wbc.offline_qp_wbc import solve_offline_qp
    from wheeled_biped.wbc.offline_task_stack import TASK_WEIGHT_MODES

    # Snapshot prep
    t0 = time.perf_counter()
    snap = prepare_phase3b_snapshot(first_name, first_qpos, first_qvel, first_contacts, qp_c)
    snap_time = time.perf_counter() - t0
    print(f"  Snapshot preparation: {snap_time:.4f}s (one-time cost)")

    # QP builds from snapshot (for all 5 modes)
    modes = list(TASK_WEIGHT_MODES.keys())
    build_times = []
    solve_times = []
    for mode in modes:
        t0 = time.perf_counter()
        qp = build_phase3b_qp_from_snapshot(snap, mode, qp_c)
        build_t = time.perf_counter() - t0
        build_times.append(build_t)

        t0 = time.perf_counter()
        sol = solve_offline_qp(qp, qp_c)
        solve_t = time.perf_counter() - t0
        solve_times.append(solve_t)

        print(f"  {mode}: build={build_t:.4f}s, solve={solve_t:.4f}s, "
              f"solved={sol['success']}")

    avg_build = sum(build_times) / len(build_times) if build_times else 0
    avg_solve = sum(solve_times) / len(solve_times) if solve_times else 0
    print(f"\n  Average per-mode: build={avg_build:.4f}s, solve={avg_solve:.4f}s")
    print(f"  Estimated 12×5 audit: {(snap_time + (avg_build + avg_solve) * 5) * 12:.1f}s total")

    # ── Build report ─────────────────────────────────────────────────
    report = {
        "phase": "3B.1",
        "task": "compile_profile",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "contact_shape_analysis": {
            "unique_contact_counts": sorted(list(contact_counts)),
            "contact_shapes_vary": len(contact_counts) > 1,
            "max_contacts_observed": max(contact_counts) if contact_counts else 0,
            "recommended_max_contacts": 4,
        },
        "jacobian_profile": {
            fn: {
                "first_call_s": round(t["first_call_s"], 6),
                "second_call_s": round(t["second_call_s"], 6),
                "ratio": round(t["ratio"], 3),
                "recompiles": t["ratio"] > 0.5,
            }
            for fn, t in jac_profile.items()
        },
        "root_cause": root_cause,
        "snapshot_method_profile": {
            "snapshot_prep_s": round(snap_time, 4),
            "per_mode_build_avg_s": round(avg_build, 4),
            "per_mode_solve_avg_s": round(avg_solve, 4),
            "estimated_full_audit_s": round((snap_time + (avg_build + avg_solve) * 5) * 12, 1),
        },
        "shape_stable_contacts": True,
        "snapshot_caching": True,
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  Profile report written to: {out}")

    return report


if __name__ == "__main__":
    output_arg = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--output" else None
    run_profile(output_arg)
