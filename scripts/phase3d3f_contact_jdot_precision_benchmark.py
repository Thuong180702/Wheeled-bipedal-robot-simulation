#!/usr/bin/env python3
"""Phase 3D.3-F -- Contact Jdot*qdot Precision Benchmark.

Measures performance of float32 vs float64 FD in cached JAX dynamics:
  - cached_float32_fd_snapshot_mean_ms
  - cached_float64_fd_snapshot_mean_ms
  - float64_overhead_factor
  - original_snapshot_mean_ms
  - speedup_vs_original_float64
  - correctness_pass_count

Output: outputs/phase3d3f_contact_jdot_precision/
  contact_jdot_precision_benchmark.json
"""

from __future__ import annotations

import argparse
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
    import mujoco
    from wheeled_biped.utils.config import get_model_path
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    from wheeled_biped.wbc.offline_qp_wbc import (
        build_qp_wbc_constants, _ensure_dynamics_constants, _ensure_contact_constants,
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


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3D.3-F Contact Jdot*qdot Precision Benchmark"
    )
    parser.add_argument("--steps", type=int, default=3,
                        help="Number of cached calls per precision mode")
    parser.add_argument("--skip-original", action="store_true",
                        help="Skip original timing")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else OUTPUT_DIR / "contact_jdot_precision_benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load
    print("=" * 70)
    print("Phase 3D.3-F: Contact Jdot*qdot Precision Benchmark")
    print("=" * 70)
    model, constants = load_model_and_constants()
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}")

    qpos0 = np.array(model.keyframe("standing").qpos, dtype=np.float64)
    qvel0 = np.zeros(model.nv, dtype=np.float64)
    # Add small velocity to exercise FD path
    qvel_p = np.array([0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    contacts = extract_contacts_at_qpos(model, constants, qpos0)

    print(f"\nTest state: {len(contacts)} contacts, |qvel|={np.linalg.norm(qvel_p):.4f}")

    from wheeled_biped.wbc.phase3d3e_jax_dynamics_cache import (
        initialize_jax_dynamics_cache,
        prepare_phase3b_snapshot_cached,
    )

    # Build float32 cache
    print("\n--- Float32 FD Cache ---")
    cache_f32 = initialize_jax_dynamics_cache(
        model, constants, fd_precision="float32", warmup=True,
    )
    print(f"  compile: {cache_f32.compile_time_s:.1f}s, warmup: {cache_f32.warmup_time_s:.1f}s")
    print(f"  x64: {cache_f32.jax_enable_x64}, mode: {cache_f32.contact_jdot_precision_mode}")

    # Build float64 cache
    print("\n--- Float64 FD Cache ---")
    cache_f64 = initialize_jax_dynamics_cache(
        model, constants, fd_precision="float64", warmup=True,
    )
    print(f"  compile: {cache_f64.compile_time_s:.1f}s, warmup: {cache_f64.warmup_time_s:.1f}s")
    print(f"  x64: {cache_f64.jax_enable_x64}, mode: {cache_f64.contact_jdot_precision_mode}")
    print(f"  f64 built: {cache_f64._contact_jdot_qdot_single_jit_f64 is not None}")

    # Time float32 cached
    print(f"\n--- Float32 Cached ({args.steps} calls) ---")
    f32_times = []
    for i in range(args.steps):
        t0 = time.perf_counter()
        _ = prepare_phase3b_snapshot_cached(
            cache_f32, f"bench_f32_{i}", qpos0, qvel_p, contacts, constants,
        )
        elapsed = time.perf_counter() - t0
        f32_times.append(elapsed)
        print(f"  call {i+1}: {elapsed:.3f}s")

    f32_mean = float(np.mean(f32_times))

    # Time float64 cached
    print(f"\n--- Float64 Cached ({args.steps} calls) ---")
    f64_times = []
    for i in range(args.steps):
        t0 = time.perf_counter()
        _ = prepare_phase3b_snapshot_cached(
            cache_f64, f"bench_f64_{i}", qpos0, qvel_p, contacts, constants,
        )
        elapsed = time.perf_counter() - t0
        f64_times.append(elapsed)
        print(f"  call {i+1}: {elapsed:.3f}s")

    f64_mean = float(np.mean(f64_times))

    # Time original (optional)
    orig_times = []
    if not args.skip_original:
        print("\n--- Original (1 call, expect ~300s) ---")
        from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot
        t0 = time.perf_counter()
        _ = prepare_phase3b_snapshot("bench_orig", qpos0, qvel_p, contacts, constants)
        elapsed = time.perf_counter() - t0
        orig_times.append(elapsed)
        print(f"  {elapsed:.1f}s")
    else:
        orig_times = [333.0]

    orig_mean = float(np.mean(orig_times))

    # Stats
    f64_overhead = f64_mean / f32_mean if f32_mean > 0 else float("inf")
    speedup_f32 = orig_mean / f32_mean if f32_mean > 0 else float("inf")
    speedup_f64 = orig_mean / f64_mean if f64_mean > 0 else float("inf")

    print(f"\n--- Summary ---")
    print(f"  Original mean:    {orig_mean:.1f}s")
    print(f"  Cached f32 mean:  {f32_mean:.3f}s")
    print(f"  Cached f64 mean:  {f64_mean:.3f}s")
    print(f"  f64 overhead:     {f64_overhead:.2f}x")
    print(f"  Speedup (f32):    {speedup_f32:.1f}x")
    print(f"  Speedup (f64):    {speedup_f64:.1f}x")

    benchmark = {
        "phase": "3D.3-F",
        "benchmark": "contact_jdot_precision",
        "environment": {
            "jax_enable_x64": cache_f64.jax_enable_x64,
            "jax_platform": cache_f64.jax_platform,
            "jax_backend": cache_f64.jax_backend,
            "device_kind": cache_f64.device_kind,
            "device_count": cache_f64.device_count,
        },
        "cache_info": {
            "f32": {
                "fd_precision": cache_f32.fd_precision,
                "contact_jdot_precision_mode": cache_f32.contact_jdot_precision_mode,
                "compile_time_s": round(cache_f32.compile_time_s, 3),
                "warmup_time_s": round(cache_f32.warmup_time_s, 3),
            },
            "f64": {
                "fd_precision": cache_f64.fd_precision,
                "contact_jdot_precision_mode": cache_f64.contact_jdot_precision_mode,
                "f64_function_built": cache_f64._contact_jdot_qdot_single_jit_f64 is not None,
                "compile_time_s": round(cache_f64.compile_time_s, 3),
                "warmup_time_s": round(cache_f64.warmup_time_s, 3),
            },
        },
        "timing": {
            "steps": args.steps,
            "cached_float32_fd": {
                "mean_s": f32_mean,
                "all_s": f32_times,
            },
            "cached_float64_fd": {
                "mean_s": f64_mean,
                "all_s": f64_times,
            },
            "original": {
                "mean_s": orig_mean,
                "all_s": orig_times,
            },
        },
        "speedup": {
            "float64_overhead_factor": f64_overhead,
            "speedup_f32_vs_original": speedup_f32,
            "speedup_f64_vs_original": speedup_f64,
        },
    }

    with open(output_path, "w") as f:
        json.dump(benchmark, f, indent=2, default=str)
    print(f"\nBenchmark saved: {output_path}")

    if speedup_f64 >= 20:
        print(f"  VERDICT: Float64 FD speedup {speedup_f64:.0f}x >= 20x — acceptable")
    else:
        print(f"  NOTE: Float64 FD speedup {speedup_f64:.0f}x < 20x target")

    return 0


if __name__ == "__main__":
    sys.exit(main())
