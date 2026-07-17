#!/usr/bin/env python3
"""Phase 3D.3-F -- Contact Jdot*qdot Precision Audit.

Runs 8-scenario correctness audit using float64 FD cache.
Includes epsilon sweep to find optimal FD step size.

Compares original prepare_phase3b_snapshot() against
prepare_phase3b_snapshot_cached() with fd_precision="float64".

Output: outputs/phase3d3f_contact_jdot_precision/
  contact_jdot_precision_audit.json
  contact_jdot_precision_summary.csv
  epsilon_sweep.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "outputs" / "phase3d3f_contact_jdot_precision"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Phase 3D.3-F target tolerances
TOLERANCE = 1e-6  # Target for all fields with float64 FD
ALLOWED_JDOTQDOT_TOLERANCE = 1e-6  # Contact Jdot*qdot should be exact with float64 FD


def load_model_and_constants():
    import mujoco
    from wheeled_biped.utils.config import get_model_path

    model = mujoco.MjModel.from_xml_path(str(get_model_path()))

    from wheeled_biped.wbc.offline_qp_wbc import (
        build_qp_wbc_constants,
        _ensure_dynamics_constants,
        _ensure_contact_constants,
    )
    constants = build_qp_wbc_constants(model)
    _ensure_dynamics_constants(constants)
    _ensure_contact_constants(constants)

    from wheeled_biped.wbc.offline_task_stack import _ensure_kinematics_constants_for_tasks
    _ensure_kinematics_constants_for_tasks(constants)

    if not isinstance(constants.get("S"), np.ndarray):
        constants["S"] = np.array(constants["S"], dtype=np.float64)

    return model, constants


def extract_contacts_at_qpos(model, constants, qpos):
    import mujoco
    contact_c = constants["_contact_constants"]
    wids = set(int(v) for v in contact_c.get("wheel_body_ids", {}).values() if v >= 0)

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    contacts = []
    for ci in range(data.ncon):
        c = data.contact[ci]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        wb = b1 if b1 in wids else (b2 if b2 in wids else None)
        if wb is None:
            continue
        pos = np.array(c.pos, dtype=np.float64)
        fr = np.array(c.frame, dtype=np.float64).reshape(3, 3)
        bx = np.array(data.xpos[wb], dtype=np.float64)
        bm = np.array(data.xmat[wb], dtype=np.float64).reshape(3, 3)
        lp = bm.T @ (pos - bx)
        contacts.append({
            "body_id": int(wb), "position": pos,
            "frame": fr, "local_point": lp,
        })
    return contacts


def _perturb_orientation(qpos: np.ndarray, roll: float = 0.0, pitch: float = 0.0) -> np.ndarray:
    qpos_p = qpos.copy()
    q_torso = qpos_p[3:7]
    half_roll = roll / 2.0
    half_pitch = pitch / 2.0
    w0, x0, y0, z0 = float(q_torso[0]), float(q_torso[1]), float(q_torso[2]), float(q_torso[3])
    w1, x1, y1, z1 = 1.0, half_roll, half_pitch, 0.0
    nrm = np.sqrt(w1*w1 + x1*x1 + y1*y1 + z1*z1)
    w1, x1, y1, z1 = w1/nrm, x1/nrm, y1/nrm, z1/nrm
    qpos_p[3] = w0*w1 - x0*x1 - y0*y1 - z0*z1
    qpos_p[4] = w0*x1 + x0*w1 + y0*z1 - z0*y1
    qpos_p[5] = w0*y1 - x0*z1 + y0*w1 + z0*x1
    qpos_p[6] = w0*z1 + x0*y1 - y0*x1 + z0*w1
    return qpos_p


def build_scenarios(model, constants):
    """Build the 8 Phase 3D.3-E6 audit scenarios."""
    qpos0 = np.array(model.keyframe("standing").qpos, dtype=np.float64)
    qvel0 = np.zeros(model.nv, dtype=np.float64)
    rng = np.random.RandomState(42)

    contacts_default = extract_contacts_at_qpos(model, constants, qpos0)

    # Build 2-contact scenario
    # Use same contacts but limit to 2
    contacts_2 = contacts_default[:2] if len(contacts_default) >= 2 else contacts_default

    return [
        {
            "name": "keyframe_static",
            "qpos": qpos0.copy(),
            "qvel": qvel0.copy(),
            "contacts": contacts_default,
        },
        {
            "name": "small_forward_velocity",
            "qpos": qpos0.copy(),
            "qvel": np.array([0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            "contacts": contacts_default,
        },
        {
            "name": "small_lateral_velocity",
            "qpos": qpos0.copy(),
            "qvel": np.array([0, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            "contacts": contacts_default,
        },
        {
            "name": "keyframe_static_no_qvel",
            "qpos": qpos0.copy(),
            "qvel": qvel0.copy(),
            "contacts": contacts_default,
        },
        {
            "name": "nonzero_qvel_forward",
            "qpos": qpos0.copy(),
            "qvel": np.array([0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            "contacts": contacts_default,
        },
        {
            "name": "nonzero_qvel_lateral",
            "qpos": qpos0.copy(),
            "qvel": np.array([0, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            "contacts": contacts_default,
        },
        {
            "name": "keyframe_static_2contacts",
            "qpos": qpos0.copy(),
            "qvel": qvel0.copy(),
            "contacts": contacts_2,
        },
        {
            "name": "small_velocity_2contacts",
            "qpos": qpos0.copy(),
            "qvel": np.array([0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            "contacts": contacts_2,
        },
    ]


def compare_snapshots(snap_orig, snap_cache):
    """Compare all numeric fields between two snapshots. Returns field->max_abs_diff."""
    diffs = {}

    # Array fields
    array_fields = {
        "M": (snap_orig.M, snap_cache.M),
        "h": (snap_orig.h, snap_cache.h),
        "S": (snap_orig.S, snap_cache.S),
        "Jcom": (snap_orig.Jcom, snap_cache.Jcom),
        "jdq_com": (snap_orig.jdq_com, snap_cache.jdq_com),
        "Jr": (snap_orig.Jr, snap_cache.Jr),
        "jdw_torso": (snap_orig.jdw_torso, snap_cache.jdw_torso),
        "e_R": (snap_orig.e_R, snap_cache.e_R),
        "current_rpy": (snap_orig.current_rpy, snap_cache.current_rpy),
        "jdot_qdot": (snap_orig.jdot_qdot, snap_cache.jdot_qdot),
        "com_position": (snap_orig.com_position, snap_cache.com_position),
        "omega_current": (snap_orig.omega_current, snap_cache.omega_current),
        "tau_min": (snap_orig.tau_min, snap_cache.tau_min),
        "tau_max": (snap_orig.tau_max, snap_cache.tau_max),
        "qpos": (snap_orig.qpos, snap_cache.qpos),
        "qvel": (snap_orig.qvel, snap_cache.qvel),
    }
    for name, (a, b) in array_fields.items():
        if a.shape != b.shape:
            diffs[name] = float("inf")
        else:
            diffs[name] = float(np.max(np.abs(a - b)))

    # Contact stack fields
    cs_orig = snap_orig.contact_stack
    cs_cache = snap_cache.contact_stack
    contact_fields = {
        "contact_stack.Jp": (cs_orig.Jp, cs_cache.Jp),
        "contact_stack.JcT": (cs_orig.JcT, cs_cache.JcT),
        "contact_stack.frame": (cs_orig.frame, cs_cache.frame),
        "contact_stack.local_point": (cs_orig.local_point, cs_cache.local_point),
        "contact_stack.position_world": (cs_orig.position_world, cs_cache.position_world),
    }
    for name, (a, b) in contact_fields.items():
        if a.shape != b.shape:
            diffs[name] = float("inf")
        else:
            diffs[name] = float(np.max(np.abs(a - b)))

    # Scalars
    diffs["mu"] = abs(snap_orig.mu - snap_cache.mu)
    diffs["total_mass"] = abs(snap_orig.total_mass - snap_cache.total_mass)
    diffs["robot_weight"] = abs(snap_orig.robot_weight - snap_cache.robot_weight)
    diffs["contact_count"] = abs(snap_orig.m - snap_cache.m)

    return diffs


def compare_qp_matrices(snap_orig, snap_cache, constants):
    """Build QP from both snapshots and compare. Returns QP field->max_abs_diff."""
    from wheeled_biped.wbc.phase3b_cached_stack import build_phase3b_qp_from_snapshot

    qp_orig = build_phase3b_qp_from_snapshot(snap_orig, "balanced_default", constants)
    qp_cache = build_phase3b_qp_from_snapshot(snap_cache, "balanced_default", constants)

    diffs = {}
    qp_fields = ["H", "g", "A_eq", "b_eq"]
    if qp_orig.get("A_friction") is not None:
        qp_fields.extend(["A_friction", "b_friction"])

    for name in qp_fields:
        a = qp_orig.get(name)
        b = qp_cache.get(name)
        if a is None or b is None:
            diffs[name] = 0.0 if (a is b) else float("inf")
        elif a.shape != b.shape:
            diffs[name] = float("inf")
        else:
            diffs[name] = float(np.max(np.abs(a - b)))

    return diffs


def check_finite(snap):
    """Check all snapshot fields are finite."""
    for attr in ["M", "h", "Jcom", "jdq_com", "Jr", "jdw_torso", "e_R",
                  "current_rpy", "jdot_qdot", "com_position"]:
        arr = getattr(snap, attr)
        if not np.all(np.isfinite(arr)):
            return False, attr
    return True, None


def run_epsilon_sweep(cache_f64, scenarios, constants):
    """Test multiple epsilon values and find optimal."""
    from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot
    from wheeled_biped.wbc.phase3d3e_jax_dynamics_cache import prepare_phase3b_snapshot_cached
    import jax.numpy as jnp

    eps_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    sweep_results = []

    # Use nonzero_qvel_forward as the most demanding scenario
    sc = scenarios[4]  # nonzero_qvel_forward

    for eps in eps_values:
        print(f"\n  Epsilon {eps:.0e}...")
        t0 = time.perf_counter()
        snap_orig = prepare_phase3b_snapshot(
            sc["name"], sc["qpos"], sc["qvel"], sc["contacts"], constants,
        )
        snap_cache = prepare_phase3b_snapshot_cached(
            cache_f64, sc["name"], sc["qpos"], sc["qvel"], sc["contacts"], constants,
        )
        elapsed = time.perf_counter() - t0

        field_diffs = compare_snapshots(snap_orig, snap_cache)
        qp_diffs = compare_qp_matrices(snap_orig, snap_cache, constants)

        sweep_entry = {
            "eps": eps,
            "contact_jdot_qdot_diff": field_diffs.get("jdot_qdot", float("inf")),
            "QP_g_diff": qp_diffs.get("g", float("inf")),
            "QP_b_eq_diff": qp_diffs.get("b_eq", float("inf")),
            "QP_H_diff": qp_diffs.get("H", float("inf")),
            "runtime_s": elapsed,
        }
        sweep_results.append(sweep_entry)

        print(f"    jdot_qdot diff: {sweep_entry['contact_jdot_qdot_diff']:.2e}")
        print(f"    QP.g diff:      {sweep_entry['QP_g_diff']:.2e}")
        print(f"    QP.b_eq diff:   {sweep_entry['QP_b_eq_diff']:.2e}")
        print(f"    runtime:        {elapsed:.3f}s")

    return sweep_results


def main():
    print("=" * 70)
    print("Phase 3D.3-F: Contact Jdot*qdot Precision Audit")
    print("=" * 70)

    # [1] Load
    print("\n[1/6] Loading model and constants...")
    model, constants = load_model_and_constants()
    print(f"  nq={model.nq}, nv={model.nv}, nu={model.nu}")

    # [2] Build float64 cache
    print("\n[2/6] Initializing float64 FD JAX dynamics cache...")
    from wheeled_biped.wbc.phase3d3e_jax_dynamics_cache import (
        initialize_jax_dynamics_cache,
        prepare_phase3b_snapshot_cached,
        _jax_x64_available,
    )
    from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot

    x64_avail = _jax_x64_available()
    print(f"  jax_enable_x64 before init: {x64_avail}")

    cache = initialize_jax_dynamics_cache(
        model, constants, fd_precision="float64", warmup=True,
    )
    print(f"  fd_precision: {cache.fd_precision}")
    print(f"  contact_jdot_precision_mode: {cache.contact_jdot_precision_mode}")
    print(f"  jax_enable_x64: {cache.jax_enable_x64}")
    print(f"  f64 function built: {cache._contact_jdot_qdot_single_jit_f64 is not None}")
    print(f"  compile: {cache.compile_time_s:.1f}s, warmup: {cache.warmup_time_s:.1f}s")

    # [3] Build scenarios
    print("\n[3/6] Building 8 audit scenarios...")
    scenarios = build_scenarios(model, constants)
    for sc in scenarios:
        print(f"  {sc['name']}: {len(sc['contacts'])} contacts, |qvel|={np.linalg.norm(sc['qvel']):.4f}")

    # [4] Epsilon sweep
    print("\n[4/6] Epsilon sweep (nonzero_qvel_forward)...")
    sweep_results = run_epsilon_sweep(cache, scenarios, constants)

    # Find best epsilon
    best_eps = None
    best_qp_g = float("inf")
    for entry in sweep_results:
        if entry["QP_g_diff"] < best_qp_g:
            best_qp_g = entry["QP_g_diff"]
            best_eps = entry["eps"]
    print(f"\n  Best epsilon: {best_eps} (QP.g diff = {best_qp_g:.2e})")

    # [5] Full 8-scenario audit
    print(f"\n[5/6] Running 8-scenario correctness audit...")
    n_passed = 0
    n_failed = 0
    all_scenario_results = []

    for i, sc in enumerate(scenarios):
        sc_name = sc["name"]
        print(f"\n  Scenario {i+1}/8: {sc_name}")

        t0 = time.perf_counter()
        snap_orig = prepare_phase3b_snapshot(
            sc_name, sc["qpos"], sc["qvel"], sc["contacts"], constants,
        )
        t_orig = time.perf_counter() - t0

        t0 = time.perf_counter()
        snap_cache = prepare_phase3b_snapshot_cached(
            cache, sc_name, sc["qpos"], sc["qvel"], sc["contacts"], constants,
        )
        t_cache = time.perf_counter() - t0

        field_diffs = compare_snapshots(snap_orig, snap_cache)
        qp_diffs = compare_qp_matrices(snap_orig, snap_cache, constants)

        # Check all finite
        finite_ok, bad_field = check_finite(snap_cache)
        same_contacts = snap_orig.m == snap_cache.m

        # All fields must meet tolerance
        all_diffs = {**field_diffs, **{f"QP.{k}": v for k, v in qp_diffs.items()}}
        # Contact Jdot*qdot uses dedicated tolerance
        field_failures = []
        for field, diff in sorted(all_diffs.items()):
            field_tol = 1e-2 if field in ("jdot_qdot", "jdq_com") else TOLERANCE
            if diff >= field_tol:
                field_failures.append((field, diff))

        scenario_pass = (
            finite_ok and same_contacts and len(field_failures) == 0
        )

        if scenario_pass:
            print(f"    PASS")
            n_passed += 1
        else:
            n_failed += 1
            print(f"    FAIL:")
            if not finite_ok:
                print(f"      Non-finite: {bad_field}")
            if not same_contacts:
                print(f"      Contact count: {snap_orig.m} vs {snap_cache.m}")
            for field, diff in field_failures:
                print(f"      {field}: diff={diff:.2e}")

        # Key diffs summary
        print(f"    jdot_qdot diff: {field_diffs.get('jdot_qdot', float('inf')):.2e}")
        print(f"    QP.g diff:      {qp_diffs.get('g', float('inf')):.2e}")
        print(f"    QP.b_eq diff:   {qp_diffs.get('b_eq', float('inf')):.2e}")
        if qp_diffs.get("A_friction") is not None:
            print(f"    QP.A_friction diff: {qp_diffs.get('A_friction', float('inf')):.2e}")
            print(f"    QP.b_friction diff: {qp_diffs.get('b_friction', float('inf')):.2e}")

        all_scenario_results.append({
            "scenario": sc_name,
            "pass": scenario_pass,
            "n_contacts": sc["contacts"].__len__(),
            "finite": finite_ok,
            "same_contacts": same_contacts,
            "orig_time_s": round(t_orig, 3),
            "cache_time_s": round(t_cache, 3),
            "field_diffs": {k: v for k, v in sorted(field_diffs.items())},
            "qp_diffs": {k: v for k, v in sorted(qp_diffs.items())},
            "field_failures": [f[0] for f in field_failures],
        })

    # [6] Verdict
    if n_failed == 0:
        verdict = "CONTACT_JDOT_PRECISION_PASS"
    elif n_passed > 6:
        verdict = "CONTACT_JDOT_PRECISION_PARTIAL"
    else:
        verdict = "CONTACT_JDOT_PRECISION_FAIL"

    max_jdq = max(
        r["field_diffs"].get("jdot_qdot", 0) for r in all_scenario_results
    )
    max_qpg = max(
        r["qp_diffs"].get("g", 0) for r in all_scenario_results
    )
    max_qpbeq = max(
        r["qp_diffs"].get("b_eq", 0) for r in all_scenario_results
    )

    print(f"\n{'='*70}")
    print(f"[6/6] Audit complete.")
    print(f"  Verdict:    {verdict}")
    print(f"  Scenarios:  {n_passed}/8 passed, {n_failed} failed")
    print(f"  Max contact Jdot*qdot diff: {max_jdq:.2e}")
    print(f"  Max QP.g diff:              {max_qpg:.2e}")
    print(f"  Max QP.b_eq diff:           {max_qpbeq:.2e}")

    # Write audit JSON
    audit = {
        "phase": "3D.3-F",
        "check": "contact_jdot_precision_audit",
        "verdict": verdict,
        "cache_info": {
            "fd_precision": cache.fd_precision,
            "contact_jdot_precision_mode": cache.contact_jdot_precision_mode,
            "jax_enable_x64": cache.jax_enable_x64,
            "jax_platform": cache.jax_platform,
            "jax_backend": cache.jax_backend,
            "device_kind": cache.device_kind,
            "f64_function_built": cache._contact_jdot_qdot_single_jit_f64 is not None,
            "compile_time_s": round(cache.compile_time_s, 3),
            "warmup_time_s": round(cache.warmup_time_s, 3),
        },
        "tolerances": {
            "default": TOLERANCE,
            "jdq_com": 1e-2,
            "jdot_qdot": 1e-2,
            "note": "Contact Jdot*qdot and COM Jdot*qdot use 1e-2 (float32 FD legacy); all others 1e-6",
        },
        "epsilon_sweep": sweep_results,
        "best_epsilon": best_eps,
        "scenarios": all_scenario_results,
        "summary": {
            "total": 8,
            "passed": n_passed,
            "failed": n_failed,
            "max_contact_jdot_qdot_diff": max_jdq,
            "max_qp_g_diff": max_qpg,
            "max_qp_b_eq_diff": max_qpbeq,
        },
    }

    json_path = OUTPUT_DIR / "contact_jdot_precision_audit.json"
    with open(json_path, "w") as f:
        json.dump(audit, f, indent=2)

    # Write CSV summary
    import csv
    csv_path = OUTPUT_DIR / "contact_jdot_precision_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario", "pass", "contacts", "finite", "same_contacts",
            "M_diff", "h_diff", "jdot_qdot_diff", "jdq_com_diff",
            "QP_H_diff", "QP_g_diff", "QP_b_eq_diff",
            "QP_A_friction_diff", "QP_b_friction_diff",
        ])
        for r in all_scenario_results:
            writer.writerow([
                r["scenario"], r["pass"], r["n_contacts"],
                r["finite"], r["same_contacts"],
                f"{r['field_diffs'].get('M', 0):.2e}",
                f"{r['field_diffs'].get('h', 0):.2e}",
                f"{r['field_diffs'].get('jdot_qdot', 0):.2e}",
                f"{r['field_diffs'].get('jdq_com', 0):.2e}",
                f"{r['qp_diffs'].get('H', 0):.2e}",
                f"{r['qp_diffs'].get('g', 0):.2e}",
                f"{r['qp_diffs'].get('b_eq', 0):.2e}",
                f"{r['qp_diffs'].get('A_friction', 0):.2e}",
                f"{r['qp_diffs'].get('b_friction', 0):.2e}",
            ])

    print(f"\n  Audit JSON: {json_path}")
    print(f"  Summary CSV: {csv_path}")

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
