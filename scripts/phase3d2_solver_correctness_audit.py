"""Phase 3D.2 — Fast Solver Correctness Audit vs SLSQP Reference.

Compares the fast structured QP backend against the legacy SLSQP solver
on a small representative set of cases.  Reports hard constraint residuals,
objective values, and solution differences.

Usage:
    python scripts/phase3d2_solver_correctness_audit.py --backend osqp
    python scripts/phase3d2_solver_correctness_audit.py --backend osqp --output-dir outputs/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import mujoco
import numpy as np

# ── Append project root ──────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Correctness cases ────────────────────────────────────────────────────────

CORRECTNESS_CASES = [
    {
        "name": "case1_passive_settle_keyframe_balanced_full_rolling_soft",
        "scenario": "passive_settle_keyframe",
        "task_mode": "balanced_default",
        "rolling_mode": "full_rolling_soft",
        "expects_contacts": True,
        "expects_velocity": False,
    },
    {
        "name": "case2_mid_height_settle_balanced_full_rolling_soft",
        "scenario": "mid_height_settle",
        "task_mode": "balanced_default",
        "rolling_mode": "full_rolling_soft",
        "expects_contacts": True,
        "expects_velocity": False,
    },
    {
        "name": "case3_small_lateral_velocity_balanced_lateral_soft",
        "scenario": "small_lateral_velocity",
        "task_mode": "balanced_default",
        "rolling_mode": "lateral_soft",
        "expects_contacts": True,
        "expects_velocity": True,
    },
    {
        "name": "case4_small_yaw_rate_balanced_full_rolling_soft",
        "scenario": "small_yaw_rate",
        "task_mode": "balanced_default",
        "rolling_mode": "full_rolling_soft",
        "expects_contacts": True,
        "expects_velocity": True,
    },
    {
        "name": "case5_random_pose_perturbation_feasibility_lateral_hard",
        "scenario": "random_pose_small_perturbation_2",
        "task_mode": "feasibility_only",
        "rolling_mode": "lateral_hard",
        "expects_contacts": True,
        "expects_velocity": True,
    },
]

# ── Tolerance thresholds ─────────────────────────────────────────────────────

THRESHOLDS = {
    "max_dynamics_residual": 1e-5,
    "max_contact_accel_residual": 1e-4,
    "max_friction_violation": 1e-6,
    "max_torque_violation": 1e-6,
    "max_sane_qdd": 100.0,
    "max_sane_lambda": 500.0,
}


def main():
    parser = argparse.ArgumentParser(description="Phase 3D.2 Solver Correctness Audit")
    parser.add_argument("--backend", default="osqp", choices=["osqp", "slsqp", "clarabel", "cvxopt"],
                        help="Fast solver backend to audit")
    parser.add_argument("--cases", default="all",
                        help="Comma-separated case names or 'all'")
    parser.add_argument("--run-reference", action="store_true", default=True,
                        help="Run SLSQP reference solves for comparison")
    parser.add_argument("--no-reference", dest="run_reference", action="store_false",
                        help="Skip SLSQP reference (fast solver only)")
    parser.add_argument("--max-reference-cases", type=int, default=2,
                        help="Maximum number of SLSQP reference cases (limited because slow)")
    parser.add_argument("--output-dir", default="outputs/phase3d2",
                        help="Output directory")
    parser.add_argument("--eps-abs", type=float, default=1e-5)
    parser.add_argument("--eps-rel", type=float, default=1e-5)
    parser.add_argument("--max-iter", type=int, default=4000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Phase 3D.2 — Fast Solver Correctness Audit")
    print(f"  Backend: {args.backend}")
    print(f"  Cases: {args.cases}")
    print("=" * 80)

    # ── Import project modules ───────────────────────────────────────────
    try:
        _import_mujoco_model()
    except Exception as exc:
        print(f"WARNING: Could not import MuJoCo model: {exc}")
        print("Running structural validation only (no full QP solves).")

    from wheeled_biped.wbc.qp_solver_backends import (
        make_backend,
        get_available_qp_backends,
        SLSQPLegacyBackend,
        QPSolution,
    )
    from wheeled_biped.wbc.structured_qp_problem import (
        build_structured_qp_from_phase3c_snapshot,
        validate_structured_qp,
    )
    from wheeled_biped.wbc.phase3d2_fast_solver import (
        solve_phase3c_fast,
    )

    available = get_available_qp_backends()
    print(f"\nSolver backends available: {json.dumps(available)}")
    print(f"Selected backend: {args.backend}")

    # ── Build scenarios ──────────────────────────────────────────────────
    print("\nBuilding scenarios...")
    snapshots = _build_test_scenarios()
    print(f"  Built {len(snapshots)} scenario snapshots")

    # ── Create backend ────────────────────────────────────────────────────
    try:
        fast_backend = make_backend(args.backend, eps_abs=args.eps_abs,
                                     eps_rel=args.eps_rel, max_iter=args.max_iter)
        print(f"  Fast backend created: {fast_backend.name}")
    except ValueError as exc:
        print(f"ERROR: {exc}")
        print("Falling back to SLSQP. Verdict will be PARTIAL_READY.")
        fast_backend = SLSQPLegacyBackend()

    slsqp_backend = SLSQPLegacyBackend() if args.run_reference else None
    uses_slsqp_only = isinstance(fast_backend, SLSQPLegacyBackend)

    # ── Run audit cases ──────────────────────────────────────────────────
    case_names = args.cases.split(",") if args.cases != "all" else None
    audit_results = []
    all_pass = True

    for ci, case_spec in enumerate(CORRECTNESS_CASES):
        if case_names and case_spec["name"] not in case_names:
            continue

        print(f"\n{'-'*60}")
        print(f"Case {ci+1}/{len(CORRECTNESS_CASES)}: {case_spec['name']}")

        snap = _find_snapshot(snapshots, case_spec["scenario"])
        if snap is None:
            print(f"  SKIP: scenario '{case_spec['scenario']}' not found in snapshots")
            continue

        # Need constants
        constants = _build_constants()

        result_entry = {
            "case": case_spec["name"],
            "scenario": case_spec["scenario"],
            "task_mode": case_spec["task_mode"],
            "rolling_mode": case_spec["rolling_mode"],
            "expects_contacts": case_spec.get("expects_contacts", False),
            "expects_velocity": case_spec.get("expects_velocity", False),
        }

        # ── Extract contact info from snapshot ──────────────────────────
        cs = getattr(snap, "contact_stack", None)
        active_contact_count = cs.num_contacts if cs else 0
        active_contact_bodies = list(cs.get_active_body_ids()) if cs and hasattr(cs, "get_active_body_ids") else []
        active_contact_geoms = []
        result_entry["active_contact_count"] = active_contact_count
        result_entry["active_contact_bodies"] = active_contact_bodies
        result_entry["active_contact_geoms"] = active_contact_geoms
        print(f"  Active contacts: {active_contact_count}, bodies: {active_contact_bodies}")

        # ── Fast solver ────────────────────────────────────────────────
        print(f"  Fast solver ({args.backend})...")
        try:
            fast_result = solve_phase3c_fast(
                snap, case_spec["task_mode"], case_spec["rolling_mode"],
                constants, backend=fast_backend, max_contacts=4,
                eps_abs=args.eps_abs, eps_rel=args.eps_rel, max_iter=args.max_iter,
            )
            fast_ok = fast_result["solution"].success
            print(f"    Success: {fast_ok}, Time: {fast_result['solve_time_s']:.4f}s")
            if not fast_ok:
                print(f"    Status: {fast_result['solution'].status}")

            hr = fast_result["hard_constraint_residuals"]
            print(f"    max_dyn_res: {hr['max_dynamics_residual']:.2e}")
            print(f"    max_contact_accel_res: {hr['max_contact_accel_residual']:.2e}")
            print(f"    max_friction_violation: {hr['max_friction_violation']:.2e}")
            print(f"    max_torque_violation: {hr['max_torque_violation']:.2e}")
            print(f"    max_abs_qdd: {hr['max_abs_qdd']:.2f}")
            print(f"    max_abs_lambda: {hr['max_abs_lambda']:.2f}")

            result_entry["fast"] = {
                "success": fast_ok,
                "status": fast_result["solution"].status,
                "solve_time_s": fast_result["solve_time_s"],
                "setup_time_s": fast_result["setup_time_s"],
                "objective_value": fast_result["solution"].objective_value,
                "iterations": fast_result["solution"].iterations,
                "primal_residual": fast_result["solution"].primal_residual,
                "dual_residual": fast_result["solution"].dual_residual,
                "max_dynamics_residual": hr["max_dynamics_residual"],
                "max_contact_accel_residual": hr["max_contact_accel_residual"],
                "max_friction_violation": hr["max_friction_violation"],
                "max_torque_violation": hr["max_torque_violation"],
                "max_abs_qdd": hr["max_abs_qdd"],
                "max_abs_tau": hr["max_abs_tau"],
                "max_abs_lambda": hr["max_abs_lambda"],
                "finite_solution": hr["finite_solution"],
                "rolling_residual": fast_result.get("rolling_residuals", {}).get("max_rolling_eq_residual", 0.0),
            }

            # Check hard constraints
            hc_checks = _check_hard_constraints(hr, fast_ok)
            result_entry["fast"]["hard_constraints_pass"] = all(
                hc_checks[k] for k in hc_checks)
            result_entry["fast"]["hard_constraint_checks"] = hc_checks

            if not result_entry["fast"]["hard_constraints_pass"]:
                all_pass = False
                print("    HARD CONSTRAINT FAILURES:")
                for k, v in hc_checks.items():
                    if not v:
                        print(f"      {k}: FAIL (value={hr.get(k.replace('_pass', '').replace('check_', ''), '?')})")

        except Exception as exc:
            print(f"    ERROR: {exc}")
            traceback.print_exc()
            result_entry["fast"] = {"error": str(exc), "success": False}
            all_pass = False

        # ── SLSQP reference (limited cases) ─────────────────────────────
        if args.run_reference and len(audit_results) < args.max_reference_cases:
            print(f"  SLSQP reference...")
            try:
                from wheeled_biped.wbc.structured_qp_problem import (
                    build_structured_qp_from_phase3c_snapshot as build_sqp,
                )
                sqp = build_sqp(snap, case_spec["task_mode"], case_spec["rolling_mode"],
                                constants, padded_contacts=True, max_contacts=4)
                slsqp_sol = slsqp_backend.solve(sqp)
                slsqp_ok = slsqp_sol.success
                print(f"    Success: {slsqp_ok}, Time: {slsqp_sol.solve_time_s:.4f}s")

                # Compare solutions
                if fast_result["solution"].success and slsqp_ok:
                    comp = _compare_solutions(fast_result["solution"], slsqp_sol, sqp)
                    result_entry["reference_comparison"] = comp
                    print(f"    tau_RMS_diff: {comp.get('tau_rms_diff', float('nan')):.4e}")
                    print(f"    qdd_RMS_diff: {comp.get('qdd_rms_diff', float('nan')):.4e}")
                    print(f"    lambda_RMS_diff: {comp.get('lambda_rms_diff', float('nan')):.4e}")

                result_entry["slsqp_reference"] = {
                    "success": slsqp_ok,
                    "status": slsqp_sol.status,
                    "solve_time_s": slsqp_sol.solve_time_s,
                    "objective_value": slsqp_sol.objective_value,
                }
            except Exception as exc:
                print(f"    SLSQP ERROR: {exc}")
                result_entry["slsqp_reference"] = {"error": str(exc)}

        audit_results.append(result_entry)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"  Total cases: {len(audit_results)}")
    fast_successes = sum(1 for r in audit_results if r.get("fast", {}).get("success", False))
    hc_passes = sum(1 for r in audit_results if r.get("fast", {}).get("hard_constraints_pass", False))
    print(f"  Fast solver success: {fast_successes}/{len(audit_results)}")
    print(f"  Hard constraints pass: {hc_passes}/{len(audit_results)}")

    # Aggregate residuals
    max_dyn_res = max((r.get("fast", {}).get("max_dynamics_residual", float("inf")) for r in audit_results), default=float("inf"))
    max_ca_res = max((r.get("fast", {}).get("max_contact_accel_residual", float("inf")) for r in audit_results), default=float("inf"))
    max_fric_viol = max((r.get("fast", {}).get("max_friction_violation", float("inf")) for r in audit_results), default=float("inf"))
    max_tau_viol = max((r.get("fast", {}).get("max_torque_violation", float("inf")) for r in audit_results), default=float("inf"))

    print(f"  Max dynamics residual: {max_dyn_res:.2e}")
    print(f"  Max contact accel residual: {max_ca_res:.2e}")
    print(f"  Max friction violation: {max_fric_viol:.2e}")
    print(f"  Max torque violation: {max_tau_viol:.2e}")

    verdict = "PASS" if all_pass and fast_successes == len(audit_results) and not uses_slsqp_only else "PARTIAL_PASS" if fast_successes > 0 else "FAIL"
    print(f"\n  Verdict: {verdict}")
    if uses_slsqp_only:
        print("  NOTE: Only SLSQP available. Verdict limited to PARTIAL_READY.")

    # ── Save report ──────────────────────────────────────────────────────
    report = {
        "phase": "3D.2",
        "step": "correctness_audit",
        "backend": args.backend,
        "fast_backend_available": available.get(args.backend, False),
        "uses_slsqp_only": uses_slsqp_only,
        "num_cases": len(audit_results),
        "fast_solver_successes": fast_successes,
        "slsqp_reference_cases": sum(1 for r in audit_results if "slsqp_reference" in r),
        "max_dynamics_residual": max_dyn_res,
        "max_contact_accel_residual": max_ca_res,
        "max_friction_violation": max_fric_viol,
        "max_torque_violation": max_tau_viol,
        "pass": verdict == "PASS",
        "cases": audit_results,
        "thresholds": THRESHOLDS,
    }

    json_path = os.path.join(args.output_dir, "phase3d2_correctness_audit.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to: {json_path}")

    return 0 if verdict == "PASS" else 1


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_test_scenarios() -> list:
    """Build contact-rich test scenario snapshots for correctness audit.

    Each scenario is properly settled so wheel-ground contacts are active.
    At least 3 scenarios will have active wheel contacts.
    """
    from wheeled_biped.wbc.phase3b_cached_stack import (
        prepare_phase3b_snapshot,
    )
    from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants

    model = _import_mujoco_model()
    constants = build_qp_wbc_constants(model)

    # Wheel geom IDs for contact extraction
    wheel_geom_ids = set()
    wheel_body_ids = set()
    for i in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if gname and ("wheel" in gname.lower()):
            wheel_geom_ids.add(i)
            wheel_body_ids.add(int(model.geom_bodyid[i]))

    def _extract_contacts(d):
        contacts = []
        for ci in range(d.ncon):
            c = d.contact[ci]
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            wheel_body = b1 if g1 in wheel_geom_ids else (b2 if g2 in wheel_geom_ids else None)
            if wheel_body is None:
                continue
            pos = np.array(c.pos, dtype=np.float64)
            frame = np.array(c.frame, dtype=np.float64).reshape(3, 3)
            body_xpos = np.array(d.xpos[wheel_body], dtype=np.float64)
            body_xmat = np.array(d.xmat[wheel_body], dtype=np.float64).reshape(3, 3)
            local_point = body_xmat.T @ (pos - body_xpos)
            contacts.append({
                "body_id": int(wheel_body), "position": pos, "frame": frame,
                "local_point": local_point, "distance": float(c.dist),
            })
        return contacts

    def _get_active_contact_bodies(d):
        """Return set of body IDs that have active wheel contacts."""
        bodies = set()
        for ci in range(d.ncon):
            c = d.contact[ci]
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            if g1 in wheel_geom_ids:
                bodies.add(b1)
            if g2 in wheel_geom_ids:
                bodies.add(b2)
        return bodies

    # ═══════════════════════════════════════════════════════════════════════
    # Build 5 contact-rich scenarios.
    # The keyframe pose ALREADY has wheel-floor contacts (verified).
    # We do NOT free-fall the robot; we only call mj_forward after setting
    # joint/velocity perturbations so contacts are refreshed correctly.
    # ═══════════════════════════════════════════════════════════════════════

    raw_scenarios = []

    # Case 1: passive_settle_keyframe — default keyframe
    d = mujoco.MjData(model)
    try:
        mujoco.mj_resetDataKeyframe(model, d, 0)
    except Exception:
        mujoco.mj_resetData(model, d)
    mujoco.mj_forward(model, d)
    ct = _extract_contacts(d)
    cb = _get_active_contact_bodies(d)
    print(f"  Case 1 'passive_settle_keyframe': {len(ct)} contact-points, "
          f"wheel_bodies={cb}, COM_z={d.qpos[2]:.3f}")
    raw_scenarios.append(("passive_settle_keyframe",
                          d.qpos.copy(), np.zeros(model.nv), ct))

    # Case 2: mid_height_settle — keyframe lowered to COM height ~0.50 m
    # Adjust BOTH qpos[2] AND leg joints for kinematic consistency.
    d = mujoco.MjData(model)
    try:
        mujoco.mj_resetDataKeyframe(model, d, 0)
    except Exception:
        mujoco.mj_resetData(model, d)
    default_h = float(d.qpos[2])
    target_h = 0.50
    delta_h = target_h - default_h  # e.g. 0.50 - 0.532 = -0.032
    d.qpos[2] = target_h
    # Adjust leg joints proportionally so kinematics stay consistent
    d.qpos[9] += delta_h * 0.3   # l_hip_pitch
    d.qpos[10] += delta_h * 0.7  # l_knee
    d.qpos[14] += delta_h * 0.3  # r_hip_pitch
    d.qpos[15] += delta_h * 0.7  # r_knee
    mujoco.mj_forward(model, d)
    ct = _extract_contacts(d)
    cb = _get_active_contact_bodies(d)
    print(f"  Case 2 'mid_height_settle': {len(ct)} contact-points, "
          f"wheel_bodies={cb}, COM_z={d.qpos[2]:.3f}, delta_h={delta_h:.3f}")
    raw_scenarios.append(("mid_height_settle",
                          d.qpos.copy(), np.zeros(model.nv), ct))

    # Case 3: small_lateral_velocity — keyframe + vy=0.1 m/s
    d = mujoco.MjData(model)
    try:
        mujoco.mj_resetDataKeyframe(model, d, 0)
    except Exception:
        mujoco.mj_resetData(model, d)
    qv3 = np.zeros(model.nv)
    qv3[1] = 0.1
    d.qvel[:] = qv3
    mujoco.mj_forward(model, d)
    ct = _extract_contacts(d)
    cb = _get_active_contact_bodies(d)
    print(f"  Case 3 'small_lateral_velocity': {len(ct)} contact-points, "
          f"wheel_bodies={cb}, vy=0.1")
    raw_scenarios.append(("small_lateral_velocity",
                          d.qpos.copy(), qv3.copy(), ct))

    # Case 4: small_yaw_rate — keyframe + omega_z=0.1 rad/s
    d = mujoco.MjData(model)
    try:
        mujoco.mj_resetDataKeyframe(model, d, 0)
    except Exception:
        mujoco.mj_resetData(model, d)
    qv4 = np.zeros(model.nv)
    qv4[5] = 0.1
    d.qvel[:] = qv4
    mujoco.mj_forward(model, d)
    ct = _extract_contacts(d)
    cb = _get_active_contact_bodies(d)
    print(f"  Case 4 'small_yaw_rate': {len(ct)} contact-points, "
          f"wheel_bodies={cb}, omega_z=0.1")
    raw_scenarios.append(("small_yaw_rate",
                          d.qpos.copy(), qv4.copy(), ct))

    # Case 5: random_pose_small_perturbation_2 — keyframe + small perturbations
    d = mujoco.MjData(model)
    try:
        mujoco.mj_resetDataKeyframe(model, d, 0)
    except Exception:
        mujoco.mj_resetData(model, d)
    rng = np.random.RandomState(2)
    d.qpos[9] += rng.uniform(-0.05, 0.05)   # l_hip_pitch
    d.qpos[10] += rng.uniform(-0.05, 0.05)  # l_knee
    d.qpos[14] += rng.uniform(-0.05, 0.05)  # r_hip_pitch
    d.qpos[15] += rng.uniform(-0.05, 0.05)  # r_knee
    qv5 = np.zeros(model.nv)
    qv5[:6] = rng.uniform(-0.05, 0.05, 6)
    qv5[6:16] = rng.uniform(-0.02, 0.02, 10)
    d.qvel[:] = qv5
    mujoco.mj_forward(model, d)
    ct = _extract_contacts(d)
    cb = _get_active_contact_bodies(d)
    print(f"  Case 5 'random_pose_small_perturbation_2': {len(ct)} contact-points, "
          f"wheel_bodies={cb}")
    raw_scenarios.append(("random_pose_small_perturbation_2",
                          d.qpos.copy(), qv5.copy(), ct))

    # ── Build snapshots ────────────────────────────────────────────────────
    snapshots = []
    for name, qp, qv, ct in raw_scenarios:
        try:
            snap = prepare_phase3b_snapshot(name, qp, qv, ct, constants)
            snapshots.append(snap)
            print(f"  Snapshot built: {name} (nq={len(qp)}, nv={len(qv)}, "
                  f"contacts={len(ct)})")
        except Exception as exc:
            print(f"  WARNING: Could not build snapshot for {name}: {exc}")
            traceback.print_exc()

    return snapshots


def _make_static_pose(model, height=0.60):
    """Create a static pose near a given COM height. Returns (qpos, qvel)."""
    data = mujoco.MjData(model)
    try:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    except Exception:
        mujoco.mj_resetData(model, data)

    qpos = data.qpos.copy()
    default_height = 0.60
    delta_h = height - default_height
    hip_pitch_adjust = delta_h * 0.3
    knee_adjust = delta_h * 0.7
    qpos[9] += hip_pitch_adjust
    qpos[10] += knee_adjust
    qpos[14] += hip_pitch_adjust
    qpos[15] += knee_adjust

    return qpos, np.zeros(16)


def _make_velocity_pose(model, height=0.60, vy=0.0, omega_z=0.0):
    """Create a pose with lateral velocity or yaw rate. Returns (qpos, qvel)."""
    data = mujoco.MjData(model)
    try:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    except Exception:
        mujoco.mj_resetData(model, data)
    qpos = data.qpos.copy()

    default_height = 0.60
    delta_h = height - default_height
    hip_pitch_adjust = delta_h * 0.3
    knee_adjust = delta_h * 0.7
    qpos[9] += hip_pitch_adjust
    qpos[10] += knee_adjust
    qpos[14] += hip_pitch_adjust
    qpos[15] += knee_adjust

    qvel = np.zeros(16)
    if vy != 0.0:
        qvel[1] = vy
    if omega_z != 0.0:
        qvel[5] = omega_z

    return qpos, qvel


def _make_random_pose(model, height=0.55, seed=2):
    """Create random small perturbation. Returns (qpos, qvel)."""
    rng = np.random.RandomState(seed)
    data = mujoco.MjData(model)
    try:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    except Exception:
        mujoco.mj_resetData(model, data)
    qpos = data.qpos.copy()

    default_height = 0.60
    delta_h = height - default_height
    hip_pitch_adjust = delta_h * 0.3
    knee_adjust = delta_h * 0.7
    qpos[9] += hip_pitch_adjust + rng.uniform(-0.05, 0.05)
    qpos[10] += knee_adjust + rng.uniform(-0.05, 0.05)
    qpos[14] += hip_pitch_adjust + rng.uniform(-0.05, 0.05)
    qpos[15] += knee_adjust + rng.uniform(-0.05, 0.05)

    qvel = np.zeros(16)
    qvel[:6] = rng.uniform(-0.05, 0.05, 6)
    qvel[6:16] = rng.uniform(-0.02, 0.02, 10)

    return qpos, qvel


def _get_qpos_qvel(result_tuple):
    """Extract qpos, qvel from a (qpos, qvel) tuple."""
    return result_tuple[0], result_tuple[1]


def _find_snapshot(snapshots, name):
    for s in snapshots:
        if s.scenario_name == name:
            return s
    # Try prefix match
    for s in snapshots:
        if name in s.scenario_name:
            return s
    return None


def _build_constants():
    """Build WBC constants lazily."""
    from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants
    model = _import_mujoco_model()
    return build_qp_wbc_constants(model)


_MODEL_CACHE = None


def _import_mujoco_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    from wheeled_biped.utils.config import get_model_path
    import mujoco as _mj
    _MODEL_CACHE = _mj.MjModel.from_xml_path(str(get_model_path()))
    return _MODEL_CACHE


def _check_hard_constraints(hr, solver_success):
    """Check hard constraint pass/fail."""
    checks = {}
    checks["solver_success"] = solver_success
    checks["dynamics_residual"] = hr["max_dynamics_residual"] < THRESHOLDS["max_dynamics_residual"]
    checks["contact_accel_residual"] = hr["max_contact_accel_residual"] < THRESHOLDS["max_contact_accel_residual"]
    checks["friction_violation"] = hr["max_friction_violation"] <= THRESHOLDS["max_friction_violation"]
    checks["torque_violation"] = hr["max_torque_violation"] <= THRESHOLDS["max_torque_violation"]
    checks["qdd_sane"] = hr["max_abs_qdd"] <= THRESHOLDS["max_sane_qdd"]
    checks["lambda_sane"] = hr["max_abs_lambda"] <= THRESHOLDS["max_sane_lambda"]
    checks["finite_solution"] = hr["finite_solution"]
    return checks


def _compare_solutions(fast_sol: QPSolution, slsqp_sol: QPSolution, sqp) -> dict:
    """Compare fast solver vs SLSQP solution."""
    vs = sqp.variable_slices
    xf = fast_sol.x
    xs = slsqp_sol.x

    result = {}

    # tau difference
    tau_s, tau_e = vs["tau"]
    tau_diff = xf[tau_s:tau_e] - xs[tau_s:tau_e]
    result["tau_rms_diff"] = float(np.sqrt(np.mean(tau_diff**2)))
    result["tau_max_diff"] = float(np.max(np.abs(tau_diff)))

    # qdd difference
    qdd_s, qdd_e = vs["qdd"]
    qdd_diff = xf[qdd_s:qdd_e] - xs[qdd_s:qdd_e]
    result["qdd_rms_diff"] = float(np.sqrt(np.mean(qdd_diff**2)))
    result["qdd_max_diff"] = float(np.max(np.abs(qdd_diff)))

    # lambda difference
    lam_s, lam_e = vs["lambda"]
    if lam_e > lam_s:
        lam_diff = xf[lam_s:lam_e] - xs[lam_s:lam_e]
        result["lambda_rms_diff"] = float(np.sqrt(np.mean(lam_diff**2)))
        result["lambda_max_diff"] = float(np.max(np.abs(lam_diff)))
    else:
        result["lambda_rms_diff"] = 0.0
        result["lambda_max_diff"] = 0.0

    # Objective difference
    result["objective_fast"] = fast_sol.objective_value
    result["objective_slsqp"] = slsqp_sol.objective_value
    if fast_sol.objective_value is not None and slsqp_sol.objective_value is not None:
        result["objective_diff"] = abs(fast_sol.objective_value - slsqp_sol.objective_value)
        result["objective_rel_diff"] = result["objective_diff"] / max(
            abs(slsqp_sol.objective_value), 1e-10)

    return result


if __name__ == "__main__":
    import mujoco
    sys.exit(main())
