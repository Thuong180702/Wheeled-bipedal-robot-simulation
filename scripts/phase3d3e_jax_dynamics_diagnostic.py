#!/usr/bin/env python3
"""Phase 3D.3-E1 -- JAX Dynamics Diagnostic (minimal, 2-call timing).

Times prepare_phase3b_snapshot() twice with the same state to detect
whether JAX re-traces on every call. Also records per-sub-operation
timing by running each major function once.

Output: outputs/phase3d3e_jax_dynamics/jax_dynamics_diagnostic.json
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
RESULT_PATH = OUTPUT_DIR / "jax_dynamics_diagnostic.json"


def record_jax_env():
    import jax
    info = {
        "jax_version": jax.__version__,
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "device_count": jax.device_count(),
    }
    try:
        info["jax_platform"] = str(jax.default_backend())
    except Exception:
        info["jax_platform"] = "unknown"
    try:
        from jax.extend.backend import get_backend
        info["jax_backend"] = str(get_backend().platform)
    except Exception:
        try:
            info["jax_backend"] = str(jax.lib.xla_bridge.get_backend().platform)
        except Exception:
            info["jax_backend"] = "unknown"
    info["devices"] = [str(d) for d in jax.devices()] if jax.device_count() > 0 else []
    info["device_kind"] = str(jax.devices()[0].device_kind) if jax.device_count() > 0 else "none"
    return info


def load_scenario():
    import mujoco
    from wheeled_biped.utils.config import get_model_path
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    from wheeled_biped.wbc.offline_qp_wbc import (
        build_qp_wbc_constants, _ensure_dynamics_constants,
        _ensure_contact_constants, build_actuator_selection_matrix_from_dims,
    )
    constants = build_qp_wbc_constants(model)
    if "S" not in constants:
        constants["S"] = build_actuator_selection_matrix_from_dims(model.nv, 10)
    _ensure_dynamics_constants(constants)
    _ensure_contact_constants(constants)
    from wheeled_biped.wbc.offline_task_stack import _ensure_kinematics_constants_for_tasks
    _ensure_kinematics_constants_for_tasks(constants)
    qpos0 = np.array(model.keyframe("standing").qpos, dtype=np.float64)
    qvel0 = np.zeros(model.nv, dtype=np.float64)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    mujoco.mj_forward(model, data)
    cc = constants["_contact_constants"]
    wids = set(int(v) for v in cc.get("wheel_body_ids", {}).values() if v >= 0)
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
        contacts.append({"body_id": int(wb), "position": pos, "frame": fr, "local_point": lp, "distance": float(c.dist)})
    return model, constants, qpos0, qvel0, contacts


def time_fn(fn, *args, **kw):
    t0 = time.perf_counter()
    try:
        _r = fn(*args, **kw)
        return time.perf_counter() - t0, None
    except Exception as e:
        return time.perf_counter() - t0, str(e)


def main():
    import jax
    import jax.numpy as jnp

    print("=" * 70)
    print("Phase 3D.3-E1: JAX Dynamics Diagnostic (minimal)")
    print("=" * 70)

    # 1. Environment
    print("\n[1] JAX environment...")
    env = record_jax_env()
    for k, v in env.items():
        print(f"  {k}: {v}")

    # 2. Load scenario
    print("\n[2] Loading scenario...")
    model, constants, qpos, qvel, contacts = load_scenario()
    n_c = len(contacts)
    print(f"  contacts: {n_c}")
    print(f"  qpos: {qpos.shape}, qvel: {qvel.shape}")

    mc = constants["_mass_matrix_constants"]
    bc = constants["_dynamics_constants"]
    cc = constants["_contact_constants"]
    kc = constants["_kinematics_constants"]
    qpj = jnp.array(qpos, dtype=jnp.float32)
    qvj = jnp.array(qvel, dtype=jnp.float32)

    results = {
        "phase": "3D.3-E1",
        "environment": env,
        "n_contacts": n_c,
        "qpos_shape": list(qpos.shape),
        "qvel_shape": list(qvel.shape),
    }

    # 3. Time major sub-operations (each once)
    print("\n[3] Timing major sub-operations...")
    sub_ops = {}

    # Skip per-contact Jacobian — was already ~53s in previous run
    # Skip contact_jdot_qdot — takes ~120s

    # Mass matrix
    from wheeled_biped.dynamics.jax_mass_matrix import jax_mass_matrix
    dt, err = time_fn(jax_mass_matrix, qpj, mc)
    sub_ops["mass_matrix_s"] = dt
    print(f"  mass_matrix:              {dt:.1f}s {'OK' if err is None else 'FAIL: '+err}")

    # Bias forces
    from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
    dt, err = time_fn(jax_bias_forces, qpj, qvj, bc)
    sub_ops["bias_forces_s"] = dt
    print(f"  bias_forces:              {dt:.1f}s {'OK' if err is None else 'FAIL: '+err}")

    # One contact Jacobian (representative)
    if contacts:
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_point_translational_jacobian
        c0 = contacts[0]
        lp0 = jnp.array(c0["local_point"], dtype=jnp.float32)
        dt, err = time_fn(contact_point_translational_jacobian, qpj, int(c0["body_id"]), lp0, cc)
        sub_ops["contact_jac_single_s"] = dt
        sub_ops["contact_jac_estimated_total_s"] = dt * n_c
        print(f"  contact_jac[0] (est 4x):  {dt:.1f}s (est {dt*n_c:.0f}s for {n_c}) OK" if err is None else f"  FAIL: {err}")

    # COM Jacobian
    from wheeled_biped.wbc.offline_task_stack import compute_com_jacobian
    dt, err = time_fn(compute_com_jacobian, qpos, kc)
    sub_ops["com_jacobian_s"] = dt
    print(f"  com_jacobian:             {dt:.1f}s {'OK' if err is None else 'FAIL: '+err}")

    # COM jdot_qdot
    from wheeled_biped.wbc.offline_task_stack import compute_com_jdot_qdot
    dt, err = time_fn(compute_com_jdot_qdot, qpos, qvel, kc)
    sub_ops["com_jdot_qdot_s"] = dt
    print(f"  com_jdot_qdot:            {dt:.1f}s {'OK' if err is None else 'FAIL: '+err}")

    # Torso angular velocity Jacobian
    from wheeled_biped.wbc.offline_task_stack import compute_torso_angular_velocity_jacobian
    dt, err = time_fn(compute_torso_angular_velocity_jacobian, qpos, kc)
    sub_ops["torso_ang_vel_jac_s"] = dt
    print(f"  torso_ang_vel_jac:        {dt:.1f}s {'OK' if err is None else 'FAIL: '+err}")

    # Torso jdotw_qdot
    from wheeled_biped.wbc.offline_task_stack import compute_torso_jdotw_qdot
    dt, err = time_fn(compute_torso_jdotw_qdot, qpos, qvel, kc)
    sub_ops["torso_jdotw_qdot_s"] = dt
    print(f"  torso_jdotw_qdot:         {dt:.1f}s {'OK' if err is None else 'FAIL: '+err}")

    # Torso orientation error
    from wheeled_biped.wbc.offline_task_stack import compute_torso_orientation_error
    dt, err = time_fn(compute_torso_orientation_error, qpos, kc)
    sub_ops["torso_orient_error_s"] = dt
    print(f"  torso_orient_error:       {dt:.1f}s {'OK' if err is None else 'FAIL: '+err}")

    results["sub_operation_timings_s"] = sub_ops

    # 4. Contact jdot_qdot (warning: SLOW)
    print("\n[4] Timing contact_jdot_qdot (WARNING: slow, ~120s)...")
    t0 = time.perf_counter()
    from wheeled_biped.wbc.offline_qp_wbc import compute_contact_jdot_qdot
    dt, err = time_fn(compute_contact_jdot_qdot, qpos, qvel, contacts, cc)
    wall_cjdq = time.perf_counter() - t0
    sub_ops["contact_jdot_qdot_s"] = dt
    print(f"  contact_jdot_qdot:        {dt:.1f}s (wall: {wall_cjdq:.1f}s) {'OK' if err is None else 'FAIL: '+err}")
    results["sub_operation_timings_s"] = sub_ops

    # 5. Full snapshot - first call
    print("\n[5] Full prepare_phase3b_snapshot (1st call)...")
    from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot
    t0 = time.perf_counter()
    dt1, err1 = time_fn(prepare_phase3b_snapshot, "diag_1st", qpos, qvel, contacts, constants)
    wall1 = time.perf_counter() - t0
    print(f"  1st call:  {dt1:.1f}s (wall: {wall1:.1f}s) {'OK' if err1 is None else 'FAIL: '+err1}")

    # 6. Full snapshot - second call (detect compilation)
    print("\n[6] Full prepare_phase3b_snapshot (2nd call, same state)...")
    t0 = time.perf_counter()
    dt2, err2 = time_fn(prepare_phase3b_snapshot, "diag_2nd", qpos, qvel, contacts, constants)
    wall2 = time.perf_counter() - t0
    ratio = dt1 / dt2 if dt2 > 0 else 0.0
    print(f"  2nd call:  {dt2:.1f}s (wall: {wall2:.1f}s, ratio: {ratio:.2f}x) {'OK' if err2 is None else 'FAIL: '+err2}")

    # 7. Report
    print("\n[7] Saving report...")
    results["full_snapshot_timings_s"] = {
        "first_call": dt1,
        "second_call": dt2,
        "first_to_second_ratio": round(ratio, 2),
    }
    results["summary"] = {
        "full_snapshot_first_s": dt1,
        "full_snapshot_second_s": dt2,
        "first_to_second_ratio": round(ratio, 2),
        "compile_overhead_estimate_s": round(dt1 - dt2, 1) if dt1 > dt2 else 0.0,
    }
    results["summary"].update(sub_ops)
    results["status"] = "complete"

    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print(f"{'='*70}")
    print(f"  JAX: {env.get('jax_platform','?')}/{env.get('jax_backend','?')}, device={env.get('device_kind','?')}, x64={env.get('jax_enable_x64','?')}")
    print(f"  Contacts: {n_c}")
    print(f"  1st full snapshot:  {dt1:.1f}s")
    print(f"  2nd full snapshot:  {dt2:.1f}s")
    print(f"  Ratio (1st/2nd):    {ratio:.2f}x")
    print(f"  Est. overhead:      {dt1 - dt2:.1f}s" if dt1 > dt2 else "  No overhead detected")
    print()

    # Top 3 slowest
    all_ops = [(k, v) for k, v in sub_ops.items()]
    all_ops.sort(key=lambda x: -x[1])
    print("  Top 3 slowest sub-operations:")
    for i, (name, t) in enumerate(all_ops[:3]):
        print(f"    {i+1}. {name}: {t:.1f}s")

    if ratio > 1.3:
        print(f"\n  VERDICT: JAX re-traces on first call (ratio={ratio:.1f}x). JIT needed.")
    elif abs(dt1 - dt2) / dt1 < 0.1:
        print(f"\n  VERDICT: Calls stable (ratio={ratio:.1f}x). No re-tracing detected.")
    else:
        print(f"\n  VERDICT: Variation detected (ratio={ratio:.1f}x). Investigate further.")

    print(f"\nReport: {RESULT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
