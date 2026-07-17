#!/usr/bin/env python3
"""Phase 3D.3-E6 -- JAX Dynamics Cache Correctness Audit.

Compares original ``prepare_phase3b_snapshot()`` against
``prepare_phase3b_snapshot_cached()`` across 8 scenarios, including
downstream QP matrix equivalence.

All snapshot fields are compared with field-specific tolerances:
  - 1e-6: most fields (M, h, Jcom, Jr, e_R, rpy, S, com_position, omega_current)
  - 1e-2: FD-computed fields (jdq_com, jdot_qdot) — float32 FD noise floor
  - jdw_torso: 1e-6 (torso quat Jacobian less sensitive to FD)

Output: outputs/phase3d3e_jax_dynamics/jax_dynamics_correctness.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "outputs" / "phase3d3e_jax_dynamics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH = OUTPUT_DIR / "jax_dynamics_correctness.json"

# ── Field-specific tolerances ──────────────────────────────────────────────
# Default tolerance for fields not listed: 1e-6
_FIELD_TOLERANCES: dict[str, float] = {
    # FD-computed fields — float32 noise floor ~1e-3 to 5e-3, use 1e-2
    "jdq_com": 1e-2,
    "jdot_qdot": 1e-2,
    # All other fields: 1e-6 (exact match expected, same JAX computation)
}


def _get_tolerance(field_name: str) -> float:
    """Return the tolerance for a given field name."""
    return _FIELD_TOLERANCES.get(field_name, 1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# Model and constants loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_model_and_constants():
    """Load MuJoCo model, build all constants, and ensure sub-constants exist.

    Returns:
        (model, constants) tuple.
    """
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

    # Ensure S is a numpy array (constants["S"] is a JAX array from build_qp_wbc_constants)
    if not isinstance(constants.get("S"), np.ndarray):
        constants["S"] = np.array(constants["S"], dtype=np.float64)

    return model, constants


# ═══════════════════════════════════════════════════════════════════════════════
# Contact extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_contacts_at_qpos(model, constants, qpos):
    """Extract active wheel contacts at a given qpos via MuJoCo forward.

    Returns:
        list of contact dicts with keys:
        body_id, position, frame, local_point.
    """
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
            "body_id": int(wb),
            "position": pos,
            "frame": fr,
            "local_point": lp,
        })
    return contacts


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario construction
# ═══════════════════════════════════════════════════════════════════════════════

def _perturb_orientation(qpos: np.ndarray, roll: float = 0.0, pitch: float = 0.0) -> np.ndarray:
    """Apply small roll/pitch perturbation to torso quaternion via Hamilton product.

    Args:
        qpos: (17,) generalized positions.
        roll: desired roll perturbation in radians.
        pitch: desired pitch perturbation in radians.

    Returns:
        (17,) perturbed qpos.
    """
    qpos_p = qpos.copy()
    q_torso = qpos_p[3:7]
    half_roll = roll / 2.0
    half_pitch = pitch / 2.0
    w0, x0, y0, z0 = float(q_torso[0]), float(q_torso[1]), float(q_torso[2]), float(q_torso[3])
    # Perturbation quaternion: (w≈1, half_roll, half_pitch, 0) normalized
    w1, x1, y1, z1 = 1.0, half_roll, half_pitch, 0.0
    nrm = np.sqrt(w1*w1 + x1*x1 + y1*y1 + z1*z1)
    w1, x1, y1, z1 = w1/nrm, x1/nrm, y1/nrm, z1/nrm
    # Hamilton product: q_new = q_torso * q_pert
    qpos_p[3] = w0*w1 - x0*x1 - y0*y1 - z0*z1
    qpos_p[4] = w0*x1 + x0*w1 + y0*z1 - z0*y1
    qpos_p[5] = w0*y1 - x0*z1 + y0*w1 + z0*x1
    qpos_p[6] = w0*z1 + x0*y1 - y0*x1 + z0*w1
    return qpos_p


def build_scenarios(model, constants):
    """Build 8 test scenarios with varied qpos/qvel.

    Returns:
        list of dicts with keys: name, qpos, qvel, contacts.
    """
    qpos0 = np.array(model.keyframe("standing").qpos, dtype=np.float64)
    qvel0 = np.zeros(model.nv, dtype=np.float64)
    rng = np.random.RandomState(42)

    contacts_default = extract_contacts_at_qpos(model, constants, qpos0)

    scenarios = [
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
            "name": "small_yaw_rate",
            "qpos": qpos0.copy(),
            "qvel": np.array([0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            "contacts": contacts_default,
        },
        {
            "name": "small_roll_tilt",
            "qpos": _perturb_orientation(qpos0, roll=0.05),
            "qvel": qvel0.copy(),
            "contacts": contacts_default,
        },
        {
            "name": "small_pitch_tilt",
            "qpos": _perturb_orientation(qpos0, pitch=0.05),
            "qvel": qvel0.copy(),
            "contacts": contacts_default,
        },
        {
            "name": "deterministic_push_state",
            "qpos": qpos0.copy(),
            "qvel": np.array([0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            "contacts": contacts_default,
        },
        {
            "name": "random_push_state",
            "qpos": qpos0.copy(),
            "qvel": np.array(list(rng.randn(6) * 0.2) + [0] * 10, dtype=np.float64),
            "contacts": contacts_default,
        },
    ]
    return scenarios


# ═══════════════════════════════════════════════════════════════════════════════
# Snapshot field comparison
# ═══════════════════════════════════════════════════════════════════════════════

def _compare_array(name: str, a: np.ndarray, b: np.ndarray) -> dict:
    """Compare two numpy arrays with field-specific tolerance.

    Returns:
        dict with keys: match, max_abs_diff, tolerance, shape, error (if any).
    """
    tol = _get_tolerance(name)
    if a.shape != b.shape:
        return {
            "match": False,
            "max_abs_diff": float("inf"),
            "tolerance": tol,
            "error": f"shape {a.shape} vs {b.shape}",
        }
    # Handle boolean arrays (e.g., active_mask) — convert to int for diff
    if a.dtype == bool and b.dtype == bool:
        max_diff = float(np.max(np.abs(a.astype(np.int32) - b.astype(np.int32))))
    else:
        max_diff = float(np.max(np.abs(a - b)))
    return {
        "match": max_diff < tol,
        "max_abs_diff": max_diff,
        "tolerance": tol,
        "shape": str(a.shape),
    }


def _compare_float(name: str, a: float, b: float) -> dict:
    """Compare two float scalars.

    Returns:
        dict with keys: match, abs_diff, tolerance.
    """
    tol = _get_tolerance(name)
    abs_diff = abs(a - b)
    return {
        "match": abs_diff < tol,
        "max_abs_diff": abs_diff,
        "tolerance": tol,
    }


def compare_snapshots(snap_orig, snap_cache) -> dict[str, dict]:
    """Compare all numeric ndarray and float fields between two snapshots.

    Includes snapshot-level fields and embedded contact stack fields.

    Returns:
        dict mapping field_name -> comparison result dict.
    """
    results = {}

    # ── Snapshot scalar fields ─────────────────────────────────────────
    scalar_fields = {
        "mu": (snap_orig.mu, snap_cache.mu),
        "total_mass": (snap_orig.total_mass, snap_cache.total_mass),
        "robot_weight": (snap_orig.robot_weight, snap_cache.robot_weight),
    }
    for name, (a, b) in scalar_fields.items():
        results[name] = _compare_float(name, a, b)

    # ── Snapshot array fields ──────────────────────────────────────────
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
        results[name] = _compare_array(name, a, b)

    # ── Contact stack fields ───────────────────────────────────────────
    cs_orig = snap_orig.contact_stack
    cs_cache = snap_cache.contact_stack

    contact_array_fields = {
        "contact_stack.Jp": (cs_orig.Jp, cs_cache.Jp),
        "contact_stack.Jr": (cs_orig.Jr, cs_cache.Jr),
        "contact_stack.JcT": (cs_orig.JcT, cs_cache.JcT),
        "contact_stack.frame": (cs_orig.frame, cs_cache.frame),
        "contact_stack.local_point": (cs_orig.local_point, cs_cache.local_point),
        "contact_stack.body_id": (cs_orig.body_id, cs_cache.body_id),
        "contact_stack.normal": (cs_orig.normal, cs_cache.normal),
        "contact_stack.position_world": (cs_orig.position_world, cs_cache.position_world),
        "contact_stack.active_mask": (cs_orig.active_mask, cs_cache.active_mask),
    }
    for name, (a, b) in contact_array_fields.items():
        results[name] = _compare_array(name, a, b)

    # ── Contact stack scalar ───────────────────────────────────────────
    results["contact_stack.num_contacts"] = _compare_float(
        "contact_stack.num_contacts",
        float(cs_orig.num_contacts),
        float(cs_cache.num_contacts),
    )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# QP matrix comparison
# ═══════════════════════════════════════════════════════════════════════════════

def compare_qp_matrices(snap_orig, snap_cache, constants) -> dict[str, dict]:
    """Build downstream QP from both snapshots and compare matrices.

    QP matrices are built from snapshot data using pure NumPy operations.
    Differences indicate propagated errors from snapshot field discrepancies.

    Returns:
        dict mapping QP matrix name -> comparison result dict.
    """
    from wheeled_biped.wbc.phase3b_cached_stack import build_phase3b_qp_from_snapshot

    qp_orig = build_phase3b_qp_from_snapshot(snap_orig, "balanced_default", constants)
    qp_cache = build_phase3b_qp_from_snapshot(snap_cache, "balanced_default", constants)

    qp_fields = [
        ("H", qp_orig.get("H"), qp_cache.get("H")),
        ("g", qp_orig.get("g"), qp_cache.get("g")),
        ("A_eq", qp_orig.get("A_eq"), qp_cache.get("A_eq")),
        ("b_eq", qp_orig.get("b_eq"), qp_cache.get("b_eq")),
    ]
    # Add friction inequality if present
    if qp_orig.get("A_friction") is not None and qp_cache.get("A_friction") is not None:
        qp_fields.append(("A_friction", qp_orig["A_friction"], qp_cache["A_friction"]))
        qp_fields.append(("b_friction", qp_orig["b_friction"], qp_cache.get("b_friction")))

    results = {}
    for name, a, b in qp_fields:
        if a is None or b is None:
            results[name] = {
                "match": a is b,
                "max_abs_diff": 0.0,
                "tolerance": 1e-6,
                "error": None if (a is b) else "one is None",
            }
        elif a.shape != b.shape:
            results[name] = {
                "match": False,
                "max_abs_diff": float("inf"),
                "tolerance": 1e-6,
                "error": f"shape {a.shape} vs {b.shape}",
                "shape": str(a.shape),
            }
        else:
            max_diff = float(np.max(np.abs(a - b)))
            results[name] = {
                "match": max_diff < 1e-6,
                "max_abs_diff": max_diff,
                "tolerance": 1e-6,
                "shape": str(a.shape),
            }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Phase 3D.3-E6: JAX Dynamics Cache Correctness Audit")
    print("=" * 70)

    # ── [1/4] Load model and constants ──────────────────────────────────
    print("\n[1/4] Loading model and constants...")
    t_load = time.perf_counter()
    model, constants = load_model_and_constants()
    load_time = time.perf_counter() - t_load
    print(f"  Model loaded in {load_time:.1f}s")
    print(f"  nq={model.nq}, nv={model.nv}, nu={model.nu}")

    from wheeled_biped.wbc.phase3d3e_jax_dynamics_cache import (
        initialize_jax_dynamics_cache,
        prepare_phase3b_snapshot_cached,
    )
    from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot

    # ── [2/4] Initialize JAX dynamics cache ────────────────────────────
    print("\n[2/4] Initializing JAX dynamics cache...")
    t0 = time.perf_counter()
    cache = initialize_jax_dynamics_cache(model, constants, warmup=True)
    init_time = time.perf_counter() - t0
    print(f"  Cache init: {init_time:.1f}s")
    print(f"  JAX platform: {cache.jax_platform}")
    print(f"  JAX backend:  {cache.jax_backend}")
    print(f"  Device:       {cache.device_kind} (x{cache.device_count})")
    print(f"  x64 enabled:  {cache.jax_enable_x64}")
    print(f"  Compile time: {cache.compile_time_s:.1f}s")
    print(f"  Warmup time:  {cache.warmup_time_s:.1f}s")

    # ── [3/4] Build scenarios ──────────────────────────────────────────
    scenarios = build_scenarios(model, constants)
    print(f"\n[3/4] Running {len(scenarios)} scenarios...")

    n_passed = 0
    n_failed = 0
    all_scenario_results = []

    for i, sc in enumerate(scenarios):
        sc_name = sc["name"]
        print(f"\n  Scenario {i+1}/{len(scenarios)}: {sc_name}")
        print(f"    contacts: {len(sc['contacts'])}")

        t_snap = time.perf_counter()

        # Original snapshot
        t_o = time.perf_counter()
        snap_orig = prepare_phase3b_snapshot(
            sc_name, sc["qpos"], sc["qvel"], sc["contacts"], constants,
        )
        orig_time = time.perf_counter() - t_o

        # Cached snapshot
        t_c = time.perf_counter()
        snap_cache = prepare_phase3b_snapshot_cached(
            cache, sc_name, sc["qpos"], sc["qvel"], sc["contacts"], constants,
        )
        cache_time = time.perf_counter() - t_c

        total_scenario_time = time.perf_counter() - t_snap
        print(f"    timing: orig={orig_time:.1f}s, cached={cache_time:.1f}s, "
              f"speedup={orig_time/cache_time:.1f}x" if cache_time > 0 else f"    timing: orig={orig_time:.1f}s, cached={cache_time:.1f}s")

        # Compare snapshot fields
        field_diffs = compare_snapshots(snap_orig, snap_cache)
        field_failures = {k: v for k, v in field_diffs.items() if not v["match"]}

        # Compare QP matrices
        qp_diffs = compare_qp_matrices(snap_orig, snap_cache, constants)
        qp_failures = {k: v for k, v in qp_diffs.items() if not v["match"]}

        scenario_pass = len(field_failures) == 0 and len(qp_failures) == 0

        if scenario_pass:
            print(f"    PASS (all {len(field_diffs)} fields, {len(qp_diffs)} QP matrices match)")
            n_passed += 1
        else:
            n_failed += 1
            print(f"    FAIL:")
            for name, d in field_failures.items():
                extra = d.get("error", "")
                print(f"      {name}: diff={d['max_abs_diff']:.2e} tol={d['tolerance']:.0e} {extra}")
            for name, d in qp_failures.items():
                extra = d.get("error", "")
                print(f"      QP.{name}: diff={d['max_abs_diff']:.2e} tol={d.get('tolerance', 1e-6):.0e} {extra}")

        # Store per-scenario results
        scenario_result = {
            "scenario": sc_name,
            "pass": scenario_pass,
            "n_contacts": len(sc["contacts"]),
            "orig_time_s": round(orig_time, 3),
            "cache_time_s": round(cache_time, 3),
            "speedup": round(orig_time / cache_time, 2) if cache_time > 0 else None,
            "field_diffs": {
                k: v["max_abs_diff"]
                for k, v in sorted(field_diffs.items())
            },
            "field_failures": sorted(field_failures.keys()),
            "qp_diffs": {
                k: v["max_abs_diff"]
                for k, v in sorted(qp_diffs.items())
            },
            "qp_failures": sorted(qp_failures.keys()),
        }
        all_scenario_results.append(scenario_result)

    # ── [4/4] Final report ─────────────────────────────────────────────
    verdict = "JAX_DYNAMICS_CACHE_CORRECTNESS_PASS" if n_failed == 0 else "JAX_DYNAMICS_CACHE_CORRECTNESS_FAIL"

    print(f"\n[4/4] Audit complete.")
    print(f"  Verdict:    {verdict}")
    print(f"  Scenarios:  {n_passed}/{len(scenarios)} passed, {n_failed} failed")
    print(f"  Total:      {n_passed + n_failed} scenarios, {len(scenarios[0]['contacts'])} contacts each")

    report = {
        "phase": "3D.3-E6",
        "verdict": verdict,
        "cache_info": {
            "init_time_s": round(init_time, 3),
            "compile_time_s": round(cache.compile_time_s, 3),
            "warmup_time_s": round(cache.warmup_time_s, 3),
            "jax_platform": cache.jax_platform,
            "jax_backend": cache.jax_backend,
            "jax_enable_x64": cache.jax_enable_x64,
            "device_kind": cache.device_kind,
            "device_count": cache.device_count,
            "dtype": cache.dtype_str,
            "max_contacts": cache.max_contacts,
        },
        "tolerances": dict(_FIELD_TOLERANCES),
        "scenarios": all_scenario_results,
        "summary": {
            "total": len(scenarios),
            "passed": n_passed,
            "failed": n_failed,
        },
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report: {RESULT_PATH}")

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
