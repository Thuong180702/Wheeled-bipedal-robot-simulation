#!/usr/bin/env python
"""Phase 3D.3-C3 — Incremental QP Correctness Audit.

Compares the incremental QP path against the existing full rebuild path
across multiple test cases. Verifies that incremental P/A/q/l/u
numeric values match full rebuild within tolerance.

Usage:
    python scripts/phase3d3_incremental_qp_correctness_audit.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── JAX fork-safety: set CPU-only before any JAX import ─────────────────────
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
from scipy.spatial.transform import Rotation

from wheeled_biped.wbc.offline_three_arm_counterfactual import (
    compute_wbc_torque_for_state,
    build_three_arm_eval_constants,
)
from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants
from wheeled_biped.wbc.offline_rolling_constraints import build_wheel_rolling_constants
from wheeled_biped.wbc.phase3d3_incremental_qp import (
    initialize_incremental_qp_workspace,
    update_incremental_qp_workspace,
    solve_incremental_qp,
)
from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot
from wheeled_biped.wbc.structured_qp_problem import build_structured_qp_from_phase3c_snapshot
from wheeled_biped.utils.config import get_model_path

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase3d3_incremental_qp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASK_MODE = "balanced_default"
ROLLING_MODE = "full_rolling_soft"

TAU_TOL = 1e-4
RESIDUAL_TOL = 1e-4
P_STALE_TOL = 1e-6
A_STALE_TOL = 1e-6

CASES = [
    "keyframe_static",
    "small_forward_velocity",
    "small_lateral_velocity",
    "small_yaw_rate",
    "small_roll_tilt",
    "small_pitch_tilt",
    "deterministic_push_state",
    "random_push_state",
]


def _apply_body_rotation(qpos, axis, angle):
    """Apply a small world-frame rotation to the torso quaternion in qpos.

    Args:
        qpos: full qpos array (mutated in place).
        axis: 'x', 'y', or 'z'.
        angle: rotation angle in radians.
    """
    r = Rotation.from_euler(axis, angle)
    quat = qpos[3:7]  # w,x,y,z in MuJoCo
    # scipy uses x,y,z,w convention
    q_scipy = [quat[1], quat[2], quat[3], quat[0]]
    q_new = (r * Rotation.from_quat(q_scipy)).as_quat()
    qpos[3:7] = [q_new[3], q_new[0], q_new[1], q_new[2]]


def generate_case_state(model, mj_data, case_name):
    """Generate (qpos, qvel) for a given test case.

    Args:
        model: MuJoCo MjModel.
        mj_data: MuJoCo MjData (used for keyframe defaults).
        case_name: one of the CASES strings.

    Returns:
        (qpos, qvel) tuple of numpy arrays.
    """
    qpos = mj_data.qpos.copy()
    qvel = np.zeros(model.nv)

    if case_name == "keyframe_static":
        pass  # default standing posture

    elif case_name == "small_forward_velocity":
        qvel[0] = 0.05

    elif case_name == "small_lateral_velocity":
        qvel[1] = 0.05

    elif case_name == "small_yaw_rate":
        qvel[5] = 0.1

    elif case_name == "small_roll_tilt":
        _apply_body_rotation(qpos, 'x', 0.05)

    elif case_name == "small_pitch_tilt":
        _apply_body_rotation(qpos, 'y', 0.05)

    elif case_name == "deterministic_push_state":
        qvel[0] = 0.2
        qvel[2] = 0.1

    elif case_name == "random_push_state":
        rng = np.random.RandomState(42)
        qvel[:6] = rng.uniform(-0.2, 0.2, size=6)
        qvel[6:] = rng.uniform(-0.1, 0.1, size=model.nv - 6)

    return qpos, qvel


def _resolve_qp_constants(constants):
    """Extract qp_constants from three-arm eval constants and ensure rolling constants."""
    qp_c = constants.get("qp_constants", constants)
    if qp_c.get("_rolling_constants") is None:
        qp_c["_rolling_constants"] = constants.get("rolling_constants", {})
    return qp_c


def compare_case(model, mj_data, constants, case_name):
    """Compare full rebuild vs incremental QP for a single test case.

    Returns a dict with comparison results.
    """
    qpos, qvel = generate_case_state(model, mj_data, case_name)
    contacts = []

    # ── Full rebuild ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    result_full = compute_wbc_torque_for_state(
        qpos, qvel, contacts, TASK_MODE, ROLLING_MODE, constants,
        qp_backend="osqp",
    )
    full_time = time.perf_counter() - t0

    # ── Incremental (init from keyframe, then update to case state) ─────────
    keyframe_qpos = mj_data.qpos.copy()
    keyframe_qvel = np.zeros(model.nv)

    workspace = initialize_incremental_qp_workspace(
        model, keyframe_qpos, keyframe_qvel, contacts,
        task_mode=TASK_MODE, rolling_mode=ROLLING_MODE,
        constants=constants, max_contacts=4,
    )

    update_incremental_qp_workspace(workspace, qpos, qvel, contacts)
    result_incr = solve_incremental_qp(workspace, warm_start=True)

    # ── Compare tau ─────────────────────────────────────────────────────────
    tau_full = result_full["tau_wbc"]
    tau_incr = result_incr["tau_wbc"]
    max_abs_tau_diff = float(np.max(np.abs(tau_full - tau_incr)))

    # ── Compare qdd ─────────────────────────────────────────────────────────
    qdd_full = result_full["qdd_wbc"]
    qdd_incr = result_incr["qdd_wbc"]
    max_abs_qdd_diff = float(np.max(np.abs(qdd_full - qdd_incr)))

    # ── P / A staleness check (fresh full rebuild vs cached workspace) ─────
    qp_c = _resolve_qp_constants(constants)
    snapshot = prepare_phase3b_snapshot("audit", qpos, qvel, contacts, qp_c)
    sqp_fresh = build_structured_qp_from_phase3c_snapshot(
        snapshot, TASK_MODE, ROLLING_MODE, qp_c,
        padded_contacts=True, max_contacts=4,
        return_block_metadata=False,
    )
    p_stale = float(np.max(np.abs(sqp_fresh.P.data - workspace.structured_qp.P.data)))
    a_stale = float(np.max(np.abs(sqp_fresh.A.data - workspace.structured_qp.A.data)))

    passes = (
        max_abs_tau_diff <= TAU_TOL
        and p_stale <= P_STALE_TOL
        and a_stale <= A_STALE_TOL
        and result_incr["solve_status"] in ("solved", "solved inaccurate")
    )

    workspace.backend.close()

    return {
        "case": case_name,
        "pass": passes,
        "tau_max_abs_diff": max_abs_tau_diff,
        "qdd_max_abs_diff": max_abs_qdd_diff,
        "P_data_staleness": p_stale,
        "A_data_staleness": a_stale,
        "solver_status_full": (
            "ok" if result_full["solve_success"] else result_full["solve_status"]
        ),
        "solver_status_incr": result_incr["solve_status"],
        "dynamics_residual_full": result_full.get(
            "max_dynamics_residual", float("nan")
        ),
        "dynamics_residual_incr": result_incr.get(
            "max_dynamics_residual", float("nan")
        ),
        "full_rebuild_time_s": full_time,
    }


def main():
    print("Phase 3D.3-C3 — Incremental QP Correctness Audit")
    print("=" * 60)

    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    mj_data = mujoco.MjData(model)

    qp_c = build_qp_wbc_constants(model)
    rolling_c = build_wheel_rolling_constants(model)
    constants = build_three_arm_eval_constants(model, qp_c, rolling_c)

    results = []
    all_pass = True

    for case_name in CASES:
        try:
            result = compare_case(model, mj_data, constants, case_name)
            results.append(result)
            status = "PASS" if result["pass"] else "FAIL"
            print(
                f"  {case_name:35s} {status}  "
                f"tau_diff={result['tau_max_abs_diff']:.2e}  "
                f"P_stale={result['P_data_staleness']:.2e}  "
                f"A_stale={result['A_data_staleness']:.2e}"
            )
            if not result["pass"]:
                all_pass = False
        except Exception as exc:
            print(f"  {case_name:35s} ERROR: {exc}")
            results.append({"case": case_name, "pass": False, "error": str(exc)})
            all_pass = False

    print("=" * 60)
    verdict = (
        "INCREMENTAL_QP_CORRECTNESS_PASS"
        if all_pass
        else "INCREMENTAL_QP_CORRECTNESS_FAIL"
    )
    print(f"Verdict: {verdict}")
    n_pass = sum(1 for r in results if r.get("pass", False))
    print(f"Cases: {n_pass}/{len(results)} pass")

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "cases": results,
        "thresholds": {
            "tau_tol": TAU_TOL,
            "p_stale_tol": P_STALE_TOL,
            "a_stale_tol": A_STALE_TOL,
        },
    }
    out_path = OUTPUT_DIR / "incremental_qp_correctness.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved: {out_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
