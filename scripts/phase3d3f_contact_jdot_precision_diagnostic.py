#!/usr/bin/env python3
"""Phase 3D.3-F -- Contact Jdot*qdot Precision Diagnostic.

Compares original contact Jdot*qdot against cached float32 and float64
variants across multiple scenarios. Proves that float64 FD eliminates
the precision noise that causes QP.g/b_eq mismatches in nonzero-qvel
scenarios.

Output: outputs/phase3d3f_contact_jdot_precision/
  contact_jdot_precision_diagnostic.json
  contact_jdot_precision_timing.csv
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


def load_model_and_constants():
    """Load MuJoCo model and build all constants."""
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
    """Extract active wheel contacts at a given qpos via MuJoCo forward."""
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


def build_scenarios(model, constants):
    """Build diagnostic scenarios with varied qpos/qvel."""
    qpos0 = np.array(model.keyframe("standing").qpos, dtype=np.float64)
    qvel0 = np.zeros(model.nv, dtype=np.float64)
    rng = np.random.RandomState(42)

    contacts = extract_contacts_at_qpos(model, constants, qpos0)

    return [
        {
            "name": "keyframe_static",
            "qpos": qpos0.copy(),
            "qvel": qvel0.copy(),
            "contacts": contacts,
        },
        {
            "name": "nonzero_qvel_forward",
            "qpos": qpos0.copy(),
            "qvel": np.array([0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            "contacts": contacts,
        },
        {
            "name": "nonzero_qvel_lateral",
            "qpos": qpos0.copy(),
            "qvel": np.array([0, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            "contacts": contacts,
        },
        {
            "name": "random_push_state",
            "qpos": qpos0.copy(),
            "qvel": np.array(list(rng.randn(6) * 0.2) + [0] * 10, dtype=np.float64),
            "contacts": contacts,
        },
    ]


def compute_contact_jdot_qdot_original(
    qpos, qvel, contacts, contact_constants, eps=1e-5,
):
    """Original float64 NumPy path (from offline_qp_wbc)."""
    from wheeled_biped.wbc.offline_qp_wbc import compute_contact_jdot_qdot
    return compute_contact_jdot_qdot(qpos, qvel, contacts, contact_constants, eps=eps)


def compute_contact_jdot_qdot_cached_f32(
    cache, qpos, qvel, contacts,
):
    """Cached float32 FD path."""
    import jax.numpy as jnp
    qpos_jax = jnp.array(qpos, dtype=jnp.float32)
    qvel_jax = jnp.array(qvel, dtype=jnp.float32)
    m = len(contacts)
    jdot_qdot = np.zeros(3 * m, dtype=np.float64)
    for i, c in enumerate(contacts):
        bid = int(c["body_id"])
        lp_jax = jnp.array(c["local_point"], dtype=jnp.float32)
        jdq_i = np.array(
            cache._contact_jdot_qdot_single_jit(qpos_jax, qvel_jax, bid, lp_jax),
            dtype=np.float64,
        )
        jdot_qdot[3*i:3*i+3] = jdq_i
    return jdot_qdot


def compute_contact_jdot_qdot_cached_f64(
    cache, qpos, qvel, contacts,
):
    """Cached float64 FD path (Phase 3D.3-F)."""
    import jax.numpy as jnp
    qpos_f64 = jnp.array(qpos, dtype=jnp.float64)
    qvel_f64 = jnp.array(qvel, dtype=jnp.float64)
    m = len(contacts)
    jdot_qdot = np.zeros(3 * m, dtype=np.float64)
    for i, c in enumerate(contacts):
        bid = int(c["body_id"])
        lp_f64 = jnp.array(c["local_point"], dtype=jnp.float64)
        jdq_i = np.array(
            cache._contact_jdot_qdot_single_jit_f64(qpos_f64, qvel_f64, bid, lp_f64),
            dtype=np.float64,
        )
        jdot_qdot[3*i:3*i+3] = jdq_i
    return jdot_qdot


def compare_qp_from_snapshots(snap_orig, snap_cached, constants):
    """Build QP from both snapshots and compare g and b_eq."""
    from wheeled_biped.wbc.phase3b_cached_stack import build_phase3b_qp_from_snapshot

    qp_orig = build_phase3b_qp_from_snapshot(snap_orig, "balanced_default", constants)
    qp_cache = build_phase3b_qp_from_snapshot(snap_cached, "balanced_default", constants)

    results = {}
    for key in ["g", "b_eq", "b_friction"]:
        a = qp_orig.get(key)
        b = qp_cache.get(key)
        if a is not None and b is not None:
            results[f"qp_{key}_diff"] = float(np.max(np.abs(a - b)))
        else:
            results[f"qp_{key}_diff"] = None
    return results


def main():
    print("=" * 70)
    print("Phase 3D.3-F: Contact Jdot*qdot Precision Diagnostic")
    print("=" * 70)

    # Load
    print("\n[1/5] Loading model and constants...")
    model, constants = load_model_and_constants()
    print(f"  nq={model.nq}, nv={model.nv}, nu={model.nu}")

    # Build scenarios
    print("\n[2/5] Building diagnostic scenarios...")
    scenarios = build_scenarios(model, constants)
    print(f"  {len(scenarios)} scenarios: {[s['name'] for s in scenarios]}")

    # Build float32 cache
    print("\n[3/5] Building float32 FD cache...")
    from wheeled_biped.wbc.phase3d3e_jax_dynamics_cache import (
        initialize_jax_dynamics_cache,
        prepare_phase3b_snapshot_cached,
    )
    cache_f32 = initialize_jax_dynamics_cache(
        model, constants, fd_precision="float32", warmup=True,
    )
    print(f"  fd_precision={cache_f32.fd_precision}")
    print(f"  contact_jdot_precision_mode={cache_f32.contact_jdot_precision_mode}")
    print(f"  x64={cache_f32.jax_enable_x64}")

    # Build float64 cache
    print("\n[4/5] Building float64 FD cache...")
    cache_f64 = initialize_jax_dynamics_cache(
        model, constants, fd_precision="float64", warmup=True,
    )
    print(f"  fd_precision={cache_f64.fd_precision}")
    print(f"  contact_jdot_precision_mode={cache_f64.contact_jdot_precision_mode}")
    print(f"  x64={cache_f64.jax_enable_x64}")
    print(f"  f64 function built: {cache_f64._contact_jdot_qdot_single_jit_f64 is not None}")

    # Compare
    print("\n[5/5] Comparing contact Jdot*qdot across scenarios...")
    contact_c = constants["_contact_constants"]
    all_results = []
    timing_rows = []

    for sc in scenarios:
        sc_name = sc["name"]
        print(f"\n  Scenario: {sc_name}")
        print(f"    qvel norm: {np.linalg.norm(sc['qvel']):.4f}")

        # Original
        t0 = time.perf_counter()
        jdq_orig = compute_contact_jdot_qdot_original(
            sc["qpos"], sc["qvel"], sc["contacts"], contact_c,
        )
        t_orig = time.perf_counter() - t0

        # Cached float32
        t0 = time.perf_counter()
        jdq_f32 = compute_contact_jdot_qdot_cached_f32(
            cache_f32, sc["qpos"], sc["qvel"], sc["contacts"],
        )
        t_f32 = time.perf_counter() - t0

        # Cached float64
        t0 = time.perf_counter()
        jdq_f64 = compute_contact_jdot_qdot_cached_f64(
            cache_f64, sc["qpos"], sc["qvel"], sc["contacts"],
        )
        t_f64 = time.perf_counter() - t0

        max_diff_f32 = float(np.max(np.abs(jdq_orig - jdq_f32)))
        max_diff_f64 = float(np.max(np.abs(jdq_orig - jdq_f64)))

        print(f"    max_contact_jdot_diff_float32: {max_diff_f32:.2e}")
        print(f"    max_contact_jdot_diff_float64: {max_diff_f64:.2e}")
        print(f"    timing: orig={t_orig:.3f}s, f32={t_f32:.3f}s, f64={t_f64:.3f}s")

        result = {
            "scenario": sc_name,
            "qvel_norm": float(np.linalg.norm(sc["qvel"])),
            "max_contact_jdot_diff_float32": max_diff_f32,
            "max_contact_jdot_diff_float64": max_diff_f64,
            "timing_orig_s": t_orig,
            "timing_f32_s": t_f32,
            "timing_f64_s": t_f64,
        }

        # QP comparison
        from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot
        snap_orig = prepare_phase3b_snapshot(
            sc_name, sc["qpos"], sc["qvel"], sc["contacts"], constants,
        )
        snap_f64 = prepare_phase3b_snapshot_cached(
            cache_f64, sc_name, sc["qpos"], sc["qvel"], sc["contacts"], constants,
        )
        qp_diffs = compare_qp_from_snapshots(snap_orig, snap_f64, constants)
        result.update(qp_diffs)

        for k, v in qp_diffs.items():
            if v is not None:
                print(f"    {k}: {v:.2e}")

        all_results.append(result)
        timing_rows.append({
            "scenario": sc_name,
            "orig_s": round(t_orig, 4),
            "cached_f32_s": round(t_f32, 4),
            "cached_f64_s": round(t_f64, 4),
        })

    # Write JSON
    json_path = OUTPUT_DIR / "contact_jdot_precision_diagnostic.json"
    diagnostic_report = {
        "phase": "3D.3-F",
        "diagnostic": "contact_jdot_precision",
        "cache_f32": {
            "fd_precision": cache_f32.fd_precision,
            "contact_jdot_precision_mode": cache_f32.contact_jdot_precision_mode,
            "jax_enable_x64": cache_f32.jax_enable_x64,
        },
        "cache_f64": {
            "fd_precision": cache_f64.fd_precision,
            "contact_jdot_precision_mode": cache_f64.contact_jdot_precision_mode,
            "jax_enable_x64": cache_f64.jax_enable_x64,
            "f64_function_built": cache_f64._contact_jdot_qdot_single_jit_f64 is not None,
        },
        "scenarios": all_results,
    }
    with open(json_path, "w") as f:
        json.dump(diagnostic_report, f, indent=2)

    # Write CSV
    import csv
    csv_path = OUTPUT_DIR / "contact_jdot_precision_timing.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "orig_s", "cached_f32_s", "cached_f64_s"])
        writer.writeheader()
        writer.writerows(timing_rows)

    print(f"\nDiagnostic report: {json_path}")
    print(f"Timing CSV: {csv_path}")

    # Summary
    max_f32 = max(r["max_contact_jdot_diff_float32"] for r in all_results)
    max_f64 = max(r["max_contact_jdot_diff_float64"] for r in all_results)
    max_qpg_f64 = max(
        (r.get("qp_g_diff") or 0) for r in all_results
    )
    max_qpbeq_f64 = max(
        (r.get("qp_b_eq_diff") or 0) for r in all_results
    )

    print(f"\n{'='*70}")
    print("DIAGNOSTIC SUMMARY")
    print(f"  Max contact Jdot*qdot diff (float32): {max_f32:.2e}")
    print(f"  Max contact Jdot*qdot diff (float64): {max_f64:.2e}")
    print(f"  Max QP.g diff (float64 cache vs original): {max_qpg_f64:.2e}")
    print(f"  Max QP.b_eq diff (float64 cache vs original): {max_qpbeq_f64:.2e}")

    if max_f64 < 1e-6 and max_qpg_f64 < 1e-6 and max_qpbeq_f64 < 1e-6:
        print("\n  VERDICT: Float64 FD eliminates precision noise.")
        print("  Contact Jdot*qdot, QP.g, and QP.b_eq all match at < 1e-6.")
    else:
        print(f"\n  VERDICT: Float64 FD does NOT fully eliminate noise.")
        print(f"  Reported floor: jdq={max_f64:.2e}, QP.g={max_qpg_f64:.2e}, QP.b_eq={max_qpbeq_f64:.2e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
