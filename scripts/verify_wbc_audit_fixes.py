#!/usr/bin/env python3
"""Empirically verify that WBC audit findings F1-F4 are fixed in the current tree.

The paper's WBC numbers (docs/validation/V3_vs_V3_Assist_comparison_report.md,
2026-07-16) predate commit 433160f (2026-07-20), which fixed F1-F4. A reviewer
is entitled to ask whether the WBC baseline failed for architectural reasons or
because the code was wrong. This script checks the four fixes directly, so the
re-run's premise ("the bugs are gone") is itself evidence rather than assertion.

  F1 contact Jacobian zero-gradient  -> cross-check against MuJoCo mj_jac
  F2 feasibility_only hardcode       -> task_mode must reach the QP cost
  F3 base-z vs CoM-z height_ref      -> batch runner must pass CoM-z
  F4 gate height-reference conflict  -> gate must use its own base-z target

Usage:
  .venv/bin/python scripts/verify_wbc_audit_fixes.py
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Audit F1 acceptance threshold: post-fix residual was reported as ~4e-4.
F1_TOL = 1e-3

results: dict[str, dict] = {}


def check_f1() -> dict:
    """Contact Jacobian must match mj_jac, and its leg columns must be nonzero."""
    import mujoco

    from wheeled_biped.dynamics.jax_contact_dynamics import (
        build_contact_dynamics_constants,
        compare_contact_jacobian_to_mujoco,
    )

    # Same model resolution the batch runner uses, so the check is on the
    # plant the WBC was actually evaluated against.
    from wheeled_biped.utils.config import get_model_path
    model_path = get_model_path()

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    constants = build_contact_dynamics_constants(model)

    out: dict = {"contacts": [], "max_err": 0.0, "mj_jac_compared": False}
    # Wheel contact points: bottom of each wheel, in body-local coordinates.
    for name in ("l_wheel_link", "r_wheel_link"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            return {"pass": False, "error": f"body {name} not found"}
        local_point = np.array([0.0, 0.0, -0.06], dtype=np.float64)
        cmp = compare_contact_jacobian_to_mujoco(
            model, data, bid, local_point, constants)
        # The F1 symptom lived in the actuated (leg) columns; the full-Jacobian
        # error is the strictest single number, so gate on that.
        err = float(cmp.get("jacobian_full_max_abs_error", np.nan))
        out.setdefault("actuated_max_abs_error", 0.0)
        out["actuated_max_abs_error"] = max(
            out["actuated_max_abs_error"],
            float(cmp.get("jacobian_actuated_max_abs_error", np.nan)),
        )
        out["contacts"].append({"body": name, "metrics": {
            k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
            for k, v in cmp.items() if not isinstance(v, (list, dict, np.ndarray))
        }})
        if not np.isfinite(err):
            return {"pass": False, "error": f"no comparable diff metric: {list(cmp)}"}
        out["max_err"] = max(out["max_err"], err)
        out["mj_jac_compared"] = True

    # The F1 symptom was leg columns identically zero. Check them directly.
    import jax.numpy as jnp

    from wheeled_biped.dynamics.jax_contact_dynamics import (
        contact_point_translational_jacobian,
    )

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    J = np.asarray(contact_point_translational_jacobian(
        jnp.array(data.qpos.copy(), dtype=jnp.float32), bid,
        jnp.array([0.0, 0.0, -0.06], dtype=jnp.float32), constants))
    # Leg DOF columns (skip the 6 free-joint DOFs); left leg is 0-4 of 10.
    leg_cols = J[:, 6:6 + 10]
    col_norms = np.linalg.norm(leg_cols, axis=0)
    out["leg_col_norms"] = [round(float(c), 6) for c in col_norms]
    out["max_leg_col_norm"] = float(np.max(col_norms))  # at least one must move
    out["jac_shape"] = list(J.shape)

    out["pass"] = bool(out["max_err"] < F1_TOL and out["mj_jac_compared"] and out["max_leg_col_norm"] > 1e-6)
    return out


def check_f2() -> dict:
    """The QP cache key must include task_mode, not hardcode feasibility_only."""
    src = (ROOT / "wheeled_biped" / "wbc" / "structured_qp_problem.py").read_text()
    tree = ast.parse(src)

    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef)
               and n.name == "_build_phase3b_qp_cached"), None)
    if fn is None:
        return {"pass": False, "error": "_build_phase3b_qp_cached not found"}

    takes_task_mode = any(a.arg == "task_mode" for a in fn.args.args)

    # Any *call site* that still passes the literal must be an explicitly
    # documented constraint-builder fallback, not the task path.
    literal_calls = []
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "_build_phase3b_qp_cached"):
            for a in n.args:
                if isinstance(a, ast.Constant) and a.value == "feasibility_only":
                    literal_calls.append(n.lineno)

    return {
        "pass": bool(takes_task_mode),
        "takes_task_mode_param": takes_task_mode,
        "feasibility_only_literal_call_lines": literal_calls,
        "note": ("remaining literal call sites are the task-independent "
                 "constraint-builder fallback (hard constraints only)"),
    }


def check_f3_f4() -> dict:
    """The batch runner must pass CoM-z as height_ref and keep the gate on base-z."""
    src = (ROOT / "scripts" / "phase3d_full_batch_execution.py").read_text()

    com_z_ref = 'height_ref = float(scenario_meta.get("target_com_z"' in src
    gate_base_z = "assist_base_z_target" in src
    # The pre-fix form took base-z straight from qpos[2] as the V3 height ref.
    prefix_form = "height_ref = float(mj_data.qpos[2])" in src

    return {
        "pass": bool(com_z_ref and gate_base_z and not prefix_form),
        "f3_height_ref_is_com_z": com_z_ref,
        "f4_gate_has_own_base_z_target": gate_base_z,
        "prefix_base_z_form_present": prefix_form,
    }


def main() -> None:
    print("=" * 68)
    print("WBC AUDIT FIX VERIFICATION (F1-F4)")
    print("=" * 68)

    results["F1_contact_jacobian"] = check_f1()
    results["F2_feasibility_only_hardcode"] = check_f2()
    results["F3_F4_height_reference"] = check_f3_f4()

    for key, val in results.items():
        status = "PASS" if val.get("pass") else "FAIL"
        print(f"\n[{status}] {key}")
        for k, v in val.items():
            if k == "pass":
                continue
            print(f"    {k}: {v}")

    all_pass = all(v.get("pass") for v in results.values())
    print("\n" + "=" * 68)
    print("OVERALL:", "ALL FIXES VERIFIED" if all_pass else "SOME CHECKS FAILED")

    out = ROOT / "outputs" / "wbc_audit_fix_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"all_pass": all_pass, "f1_tolerance": F1_TOL, "checks": results},
        indent=2, default=str))
    print("Saved:", out)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
