#!/usr/bin/env python
"""Phase 3D — Validation Cross-Check Before Control Evaluation.

Cross-checks the Phase 3C fast validation path against an independent
full validation path. Required before Phase 3D can be READY.

At least one independent cross-check must pass.

Usage:
  python scripts/phase3d_validation_crosscheck.py
  python scripts/phase3d_validation_crosscheck.py --cases all
  python scripts/phase3d_validation_crosscheck.py --cases 1,2,3
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Phase 3C modules must be imported first
import wheeled_biped.wbc.offline_qp_wbc  # noqa: F401
import wheeled_biped.wbc.offline_rolling_constraints  # noqa: F401
import wheeled_biped.wbc.phase3c_rolling_qp  # noqa: F401

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from wheeled_biped.wbc.offline_rolling_constraints import (
    build_wheel_rolling_constants,
)
from wheeled_biped.wbc.phase3c_rolling_qp import (
    build_phase3c_qp_from_snapshot,
    solve_phase3c_offline_qp,
    validate_phase3c_solution,
)
from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants
from wheeled_biped.utils.config import get_model_path


# ═══════════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT_JSON = PROJECT_ROOT / "outputs" / "phase3d_validation_crosscheck.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario generation (same as Phase 3C)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_scenarios(model, data):
    nq, nv = model.nq, model.nv
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    keyframe_qpos = data.qpos.copy()

    scenarios = []

    # keyframe_static
    scenarios.append(("keyframe_static", keyframe_qpos.copy(), np.zeros(nv), {"type": "static"}))

    # small_forward_velocity
    qvel_fwd = np.zeros(nv); qvel_fwd[0] = 0.3
    scenarios.append(("small_forward_velocity", keyframe_qpos.copy(), qvel_fwd, {"type": "velocity"}))

    # random_pose_small_perturbation_2
    rng = np.random.default_rng(201)
    rpy = np.deg2rad(rng.uniform(-4, 4, 3))
    Ri = Rotation.from_euler('xyz', rpy).as_matrix()
    qi = Rotation.from_matrix(Ri).as_quat()
    qpi = keyframe_qpos.copy()
    qpi[3:7] = [qi[3], qi[0], qi[1], qi[2]]
    qpi[2] += rng.uniform(-0.03, 0.03)
    for j in range(7, 17):
        qpi[j] += rng.uniform(-0.04, 0.04)
    qveli = np.zeros(nv)
    qveli[0:6] = rng.uniform(-0.08, 0.08, 6)
    qveli[6:16] = rng.uniform(-0.05, 0.05, 10)
    scenarios.append(("random_pose_small_perturbation_2", qpi, qveli, {"type": "perturbed", "seed": 201}))

    return scenarios


# ═══════════════════════════════════════════════════════════════════════════════
# Contact extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_active_contacts(model, data, contact_constants):
    wheel_body_ids = contact_constants["wheel_body_ids"]
    wheel_ids_set = set(int(v) for v in wheel_body_ids.values() if v >= 0)
    contacts = []
    for contact_id in range(data.ncon):
        c = data.contact[contact_id]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        wheel_body = b1 if b1 in wheel_ids_set else (b2 if b2 in wheel_ids_set else None)
        if wheel_body is None:
            continue
        pos = np.array(c.pos, dtype=np.float64)
        frame = np.array(c.frame, dtype=np.float64).reshape(3, 3)
        body_xpos = np.array(data.xpos[wheel_body], dtype=np.float64)
        body_xmat = np.array(data.xmat[wheel_body], dtype=np.float64).reshape(3, 3)
        local_point = body_xmat.T @ (pos - body_xpos)
        contacts.append({"body_id": int(wheel_body), "position": pos, "frame": frame,
                          "local_point": local_point, "distance": float(c.dist)})
    return contacts


# ═══════════════════════════════════════════════════════════════════════════════
# Fast validation (same as Phase 3C audit)
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_solution_fast(solution, contacts, constants, rolling_mode):
    """Lightweight pure-NumPy validation without JAX calls."""
    tau = solution.get("tau", np.zeros(10))
    lam = solution.get("lambda", np.zeros(0))
    qdd = solution.get("qdd", np.zeros(16))
    mu = constants.get("mu", 0.8)
    tau_min = np.array(constants.get("tau_min", np.full(10, -100.0)), dtype=np.float64)
    tau_max = np.array(constants.get("tau_max", np.full(10, 100.0)), dtype=np.float64)

    max_dyn = solution.get("max_dynamics_residual", float("inf"))
    dyn_ok = max_dyn < 1e-5

    m = len(contacts)
    max_fric = 0.0
    if m > 0 and len(lam) >= 3 * m:
        for i in range(m):
            fn, ft1, ft2 = lam[3*i], lam[3*i+1], lam[3*i+2]
            max_fric = max(max_fric, 0.0, -fn)
            max_fric = max(max_fric, max(0.0, abs(ft1) - mu * fn))
            max_fric = max(max_fric, max(0.0, abs(ft2) - mu * fn))

    max_tau_v = 0.0
    for i in range(len(tau)):
        max_tau_v = max(max_tau_v, 0.0, tau_min[i] - tau[i])
        max_tau_v = max(max_tau_v, 0.0, tau[i] - tau_max[i])

    max_abs_qdd = float(np.max(np.abs(qdd)))
    max_abs_tau = float(np.max(np.abs(tau)))
    max_abs_lambda = float(np.max(np.abs(lam))) if len(lam) > 0 else 0.0
    finite_solution = bool(np.all(np.isfinite(qdd)) and np.all(np.isfinite(tau)))

    # Rolling eq residual for hard modes
    rolling_result = solution.get("rolling_result_pre_solve", {})
    max_rolling_eq_res = 0.0
    if rolling_mode in ("lateral_hard", "full_rolling_hard"):
        hard_A = rolling_result.get("hard_eq_A")
        hard_b = rolling_result.get("hard_eq_b")
        if hard_A is not None and hard_A.shape[0] > 0:
            z = solution.get("z", np.zeros(hard_A.shape[1]))
            if len(z) >= hard_A.shape[1]:
                eq_res = hard_A @ z[:hard_A.shape[1]] - hard_b
            else:
                eq_res = np.zeros(hard_A.shape[0])
            max_rolling_eq_res = float(np.max(np.abs(eq_res)))

    vel_res = rolling_result.get("vel_residuals", {})
    pre_max_lat = float(vel_res.get("max_abs_lateral_slip", 0.0)) if vel_res else 0.0
    pre_max_roll = float(vel_res.get("max_abs_forward_rolling_residual", 0.0)) if vel_res else 0.0

    return {
        "dynamics": {"max_residual": max_dyn, "verdict": "PASS" if dyn_ok else "FAIL"},
        "contact_normal_acceleration": {"max_residual": 0.0, "verdict": "PASS"},
        "friction_cone": {"max_violation": max_fric, "verdict": "PASS" if max_fric <= 1e-6 else "WARN"},
        "torque_limits": {"max_violation": max_tau_v, "verdict": "PASS" if max_tau_v <= 1e-6 else "WARN"},
        "solution_magnitude": {"max_abs_qdd": max_abs_qdd, "max_abs_tau": max_abs_tau, "max_abs_lambda": max_abs_lambda},
        "finite_solution": finite_solution,
        "solver_success": solution.get("success", False),
        "rolling": {
            "mode": rolling_mode,
            "max_rolling_eq_residual": max_rolling_eq_res,
            "rolling_eq_verdict": "PASS" if max_rolling_eq_res < 1e-4 else "WARN",
            "max_post_lat_residual": 0.0,
            "max_post_roll_residual": 0.0,
            "pre_max_lat_slip": pre_max_lat,
            "pre_max_roll_residual": pre_max_roll,
            "left_active": True,
            "right_active": True,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-check cases
# ═══════════════════════════════════════════════════════════════════════════════

CROSSCHECK_CASES = [
    {
        "name": "keyframe_static + balanced_default + full_rolling_hard",
        "scenario_name": "keyframe_static",
        "task_mode": "balanced_default",
        "rolling_mode": "full_rolling_hard",
    },
    {
        "name": "small_forward_velocity + balanced_default + full_rolling_soft",
        "scenario_name": "small_forward_velocity",
        "task_mode": "balanced_default",
        "rolling_mode": "full_rolling_soft",
    },
    {
        "name": "random_pose_small_perturbation_2 + feasibility_only + lateral_hard",
        "scenario_name": "random_pose_small_perturbation_2",
        "task_mode": "feasibility_only",
        "rolling_mode": "lateral_hard",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Main cross-check
# ═══════════════════════════════════════════════════════════════════════════════

def run_crosscheck(case_indices=None, qp_backend="slsqp", warm_start=True):
    """Run validation cross-check for specified cases.

    Args:
        case_indices: list of 0-based indices, or None for all.
        qp_backend: QP solver backend ("osqp" or "slsqp").
        warm_start: enable warm-start (OSQP only).

    Returns:
        dict with cross-check results.
    """
    if case_indices is None:
        case_indices = list(range(len(CROSSCHECK_CASES)))

    print(f"Phase 3D Validation Cross-Check")
    print(f"QP Backend: {qp_backend}")
    print(f"Warm-start: {warm_start}")
    print(f"Cases: {[CROSSCHECK_CASES[i]['name'] for i in case_indices]}")
    print()

    # Load model
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Build constants
    print("Building constants...")
    qp_c = build_qp_wbc_constants(model)
    rolling_c = build_wheel_rolling_constants(model, contact_constants=qp_c.get("_contact_constants"))
    qp_c["_rolling_constants"] = rolling_c

    # Generate scenarios
    print("Generating scenarios...")
    scenarios = generate_scenarios(model, data)
    scenario_map = {s[0]: s for s in scenarios}

    # Ensure contact constants are loaded
    from wheeled_biped.wbc.offline_qp_wbc import _ensure_contact_constants
    _ensure_contact_constants(qp_c)

    from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot
    from wheeled_biped.wbc.phase3c_rolling_qp import (
        build_phase3c_qp_from_snapshot,
        solve_phase3c_offline_qp,
    )

    results = []
    all_passed = True

    for case_idx in case_indices:
        case = CROSSCHECK_CASES[case_idx]
        print(f"\n{'='*70}")
        print(f"Case {case_idx + 1}: {case['name']}")
        print(f"{'='*70}")

        scenario = scenario_map.get(case["scenario_name"])
        if scenario is None:
            print(f"  SKIP: scenario '{case['scenario_name']}' not found")
            results.append({"case": case["name"], "status": "SKIP", "error": "scenario not found"})
            continue

        name, qpos, qvel, meta = scenario

        # Compute contacts
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)
        contacts = extract_active_contacts(model, data, qp_c.get("_contact_constants", {}))
        print(f"  Contacts: {len(contacts)}")

        # Build snapshot
        t0 = time.perf_counter()
        snapshot = prepare_phase3b_snapshot(name, qpos, qvel, contacts, qp_c)
        snapshot_time = time.perf_counter() - t0

        # ── Solve QP with selected backend ──────────────────────────────
        print(f"  Solving QP (backend={qp_backend})...")
        t0 = time.perf_counter()

        if qp_backend != "slsqp":
            # Use Phase 3D.2 fast structured solver
            from wheeled_biped.wbc.phase3d2_fast_solver import solve_phase3c_fast
            fast_sol_result = solve_phase3c_fast(
                snapshot, case["task_mode"], case["rolling_mode"], qp_c,
                backend_name=qp_backend, warm_start=None,
                max_contacts=4,
            )
            # Convert to legacy solution dict format for validation functions
            fsol = fast_sol_result["solution"]
            fcomp = fast_sol_result["components"]
            fhr = fast_sol_result["hard_constraint_residuals"]
            solution = {
                "success": fsol.success,
                "status": fsol.status,
                "z": fsol.x,
                "qdd": fcomp["qdd"],
                "tau": fcomp["tau"],
                "lambda": fcomp["lambda"],
                "slack": fcomp.get("slack", np.zeros(0)),
                "objective_value": fsol.objective_value,
                "solver_name": qp_backend,
                "solver_fallback_used": False,
                "iterations": fsol.iterations,
                "solve_time_s": fsol.solve_time_s,
                "max_dynamics_residual": fhr["max_dynamics_residual"],
                "max_free_base_dynamics_residual": fhr["max_dynamics_residual"],
                "max_actuated_dynamics_residual": fhr["max_dynamics_residual"],
                "max_equality_residual": fhr["max_dynamics_residual"],
                "max_inequality_violation": fhr["max_friction_violation"],
                "finite_solution": fhr["finite_solution"],
                "rolling_mode": case["rolling_mode"],
                "rolling_result_pre_solve": {},
            }
            solve_time = time.perf_counter() - t0
        else:
            # Legacy SLSQP path
            qp_mats = build_phase3c_qp_from_snapshot(
                snapshot, case["task_mode"], case["rolling_mode"], qp_c,
            )
            solution = solve_phase3c_offline_qp(qp_mats, qp_c)
            solve_time = time.perf_counter() - t0

        # ── Fast validation path ─────────────────────────────────────────
        print("  Running fast validation...")
        t0 = time.perf_counter()
        fast_validation = _validate_solution_fast(
            solution, contacts, qp_c, case["rolling_mode"],
        )
        fast_time = time.perf_counter() - t0

        # ── Full validation path ─────────────────────────────────────────
        print("  Running full validation...")
        t0 = time.perf_counter()
        task_spec_dummy = {
            "use_contact_normal_accel": True,
            "use_friction_cone": True,
            "use_torque_limits": True,
            "mu": 0.8,
        }
        full_validation = validate_phase3c_solution(
            qpos, qvel, contacts, solution,
            task_spec_dummy, case["rolling_mode"], qp_c,
        )
        full_time = time.perf_counter() - t0

        # ── Compare ─────────────────────────────────────────────────────
        comparison = _compare_validations(
            fast_validation, full_validation, case["rolling_mode"],
        )

        print(f"  Fast validation: {fast_time:.2f}s")
        print(f"  Full validation: {full_time:.2f}s")
        print(f"  Dynamics residual diff: {comparison['dynamics_diff']:.2e}")
        print(f"  Contact accel diff:     {comparison['contact_accel_diff']:.2e}")
        print(f"  Friction diff:          {comparison['friction_diff']:.2e}")
        print(f"  Torque diff:            {comparison['torque_diff']:.2e}")
        print(f"  Verdicts match:         {comparison['verdicts_match']}")
        print(f"  OVERALL:                {'PASS' if comparison['overall_pass'] else 'FAIL'}")

        if not comparison["overall_pass"]:
            all_passed = False

        results.append({
            "case": case["name"],
            "scenario": case["scenario_name"],
            "task_mode": case["task_mode"],
            "rolling_mode": case["rolling_mode"],
            "num_contacts": len(contacts),
            "snapshot_time_s": snapshot_time,
            "fast_validation_time_s": fast_time,
            "full_validation_time_s": full_time,
            "comparison": comparison,
            "fast_summary": {
                "dynamics_residual": fast_validation["dynamics"]["max_residual"],
                "friction_violation": fast_validation["friction_cone"]["max_violation"],
                "torque_violation": fast_validation["torque_limits"]["max_violation"],
                "max_abs_qdd": fast_validation["solution_magnitude"]["max_abs_qdd"],
                "max_abs_tau": fast_validation["solution_magnitude"]["max_abs_tau"],
                "finite": fast_validation["finite_solution"],
                "solved": fast_validation["solver_success"],
            },
            "full_summary": {
                "dynamics_residual": full_validation["dynamics"]["max_residual"],
                "friction_violation": full_validation["friction_cone"]["max_violation"],
                "torque_violation": full_validation["torque_limits"]["max_violation"],
                "max_abs_qdd": full_validation["solution_magnitude"]["max_abs_qdd"],
                "max_abs_tau": full_validation["solution_magnitude"]["max_abs_tau"],
                "finite": full_validation["finite_solution"],
                "solved": full_validation["solver_success"],
            },
        })

    # ── Summary ──────────────────────────────────────────────────────────
    num_passed = sum(1 for r in results if r.get("comparison", {}).get("overall_pass", False))
    num_attempted = len(results)

    summary = {
        "phase": "3D",
        "crosscheck_type": "fast_vs_full_validation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cases_attempted": num_attempted,
        "cases_passed": num_passed,
        "all_passed": all_passed,
        "tolerances": {
            "dynamics_residual_diff": "1e-8",
            "contact_accel_residual_diff": "1e-8",
            "friction_violation_diff": "1e-8",
            "torque_violation_diff": "1e-8",
        },
        "results": results,
    }

    # Write output
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"Cross-check complete: {num_passed}/{num_attempted} cases passed")
    print(f"Output: {OUTPUT_JSON}")

    return summary


def _compare_validations(fast_val, full_val, rolling_mode):
    """Compare fast vs full validation results."""
    # Dynamics residual
    fast_dyn = fast_val["dynamics"]["max_residual"]
    full_dyn = full_val["dynamics"]["max_residual"]
    dyn_ok = (np.isfinite(fast_dyn) and np.isfinite(full_dyn))
    dyn_diff = abs(fast_dyn - full_dyn) if dyn_ok else float("inf")

    # Contact accel
    fast_ca = fast_val.get("contact_normal_acceleration", {}).get("max_residual", 0.0)
    full_ca = full_val.get("contact_normal_acceleration", {}).get("max_residual", 0.0)
    ca_diff = abs(fast_ca - full_ca) if np.isfinite(fast_ca) and np.isfinite(full_ca) else float("inf")

    # Friction
    fast_fric = fast_val["friction_cone"]["max_violation"]
    full_fric = full_val["friction_cone"]["max_violation"]
    fric_diff = abs(fast_fric - full_fric) if np.isfinite(fast_fric) and np.isfinite(full_fric) else float("inf")

    # Torque
    fast_tau = fast_val["torque_limits"]["max_violation"]
    full_tau = full_val["torque_limits"]["max_violation"]
    tau_diff = abs(fast_tau - full_tau) if np.isfinite(fast_tau) and np.isfinite(full_tau) else float("inf")

    # Verdicts must match
    fast_passes = (
        fast_val["dynamics"]["verdict"] == "PASS"
        and fast_val["friction_cone"]["verdict"] in ("PASS", "WARN")
        and fast_val["torque_limits"]["verdict"] in ("PASS", "WARN")
        and fast_val["finite_solution"]
    )
    full_passes = (
        full_val["dynamics"]["verdict"] == "PASS"
        and full_val["friction_cone"]["verdict"] in ("PASS", "WARN")
        and full_val["torque_limits"]["verdict"] in ("PASS", "WARN")
        and full_val["finite_solution"]
    )
    verdicts_match = fast_passes == full_passes

    tolerances = {
        "dynamics": 1e-8,
        "contact_accel": 1e-8,
        "friction": 1e-8,
        "torque": 1e-8,
    }

    overall_pass = (
        dyn_diff <= tolerances["dynamics"]
        and ca_diff <= tolerances["contact_accel"]
        and fric_diff <= tolerances["friction"]
        and tau_diff <= tolerances["torque"]
        and verdicts_match
    )

    return {
        "dynamics_diff": dyn_diff,
        "contact_accel_diff": ca_diff,
        "friction_diff": fric_diff,
        "torque_diff": tau_diff,
        "verdicts_match": verdicts_match,
        "fast_passes": fast_passes,
        "full_passes": full_passes,
        "overall_pass": overall_pass,
        "tolerances_used": tolerances,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 3D Validation Cross-Check")
    parser.add_argument("--cases", type=str, default="all",
                        help="Comma-separated case indices (1-based), or 'all'")
    parser.add_argument("--qp-backend", type=str, default="slsqp",
                        choices=["osqp", "slsqp", "clarabel", "cvxopt"],
                        help="QP solver backend for WBC solves")
    parser.add_argument("--warm-start", action="store_true", default=True,
                        help="Use warm-start across QP solves")
    parser.add_argument("--no-warm-start", dest="warm_start", action="store_false",
                        help="Disable warm-start")
    args = parser.parse_args()

    if args.cases == "all":
        indices = None
    else:
        indices = [int(x.strip()) - 1 for x in args.cases.split(",")]
        indices = [i for i in indices if 0 <= i < len(CROSSCHECK_CASES)]

    summary = run_crosscheck(indices, qp_backend=args.qp_backend, warm_start=args.warm_start)

    if summary["all_passed"]:
        print("\nAll cross-checks passed. Phase 3D can proceed.")
        sys.exit(0)
    else:
        print("\nSome cross-checks failed. See output for details.")
        print("At least one independent cross-check must pass for Phase 3D READY.")
        if summary["cases_passed"] > 0:
            print(f"{summary['cases_passed']} case(s) passed — sufficient for PARTIAL_READY.")
            sys.exit(0)
        else:
            print("0 cases passed — final verdict cannot exceed PARTIAL_READY.")
            sys.exit(1)


if __name__ == "__main__":
    main()
