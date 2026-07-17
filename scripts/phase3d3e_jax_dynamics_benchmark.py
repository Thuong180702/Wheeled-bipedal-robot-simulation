#!/usr/bin/env python3
"""Phase 3D.3-E -- JAX Dynamics Cache Benchmark.

Measures:
  - Cache initialization time (compile + warmup)
  - First-call time (post-warmup)
  - Post-warmup mean, p50, p95, max
  - Speedup vs original prepare_phase3b_snapshot
  - Recompile count, fallback count
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import jax

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "outputs" / "phase3d3e_jax_dynamics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _build_scenario_states(model: mujoco.MjModel, n_states: int) -> list[dict[str, Any]]:
    """Build a small set of varied test states.

    Each state is a dict with qpos, qvel, contacts, and a scenario name.
    We generate states at different heights and with small velocity perturbations.
    """
    from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants, _ensure_contact_constants
    constants = build_qp_wbc_constants(model)
    _ensure_contact_constants(constants)
    contact_c = constants.get("_contact_constants", {})

    wheel_body_ids = contact_c.get("wheel_body_ids", {})
    wheel_ids_set = set(int(v) for v in wheel_body_ids.values() if v >= 0)

    def _extract_contacts(data: mujoco.MjData) -> list[dict[str, Any]]:
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
            contacts.append({
                "body_id": int(wheel_body),
                "position": pos,
                "frame": frame,
                "local_point": local_point,
                "distance": float(c.dist),
            })
        return contacts

    height_offsets = [0.0, -0.02, +0.02, -0.05, +0.05, 0.0, -0.03, +0.03]
    states = []

    for i in range(min(n_states, len(height_offsets))):
        data = mujoco.MjData(model)
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        offset = height_offsets[i]
        data.qpos[2] += offset

        # Small velocity perturbation for varied states
        if i % 3 == 1:
            data.qvel[0] = 0.02   # small forward velocity
        elif i % 3 == 2:
            data.qvel[1] = 0.02   # small lateral velocity

        mujoco.mj_forward(model, data)
        states.append({
            "scenario": f"bench_state_{i}_{offset:+0.3f}m",
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
            "contacts": _extract_contacts(data),
        })

    return states


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3D.3-E JAX Dynamics Cache Benchmark"
    )
    parser.add_argument("--states", type=int, default=2,
                        help="Number of test states (default: 2)")
    parser.add_argument("--steps", type=int, default=5,
                        help="Steps per state for cached path (default: 5)")
    parser.add_argument("--original-calls", type=int, default=1,
                        help="Number of original prepare_phase3b_snapshot calls to time (default: 1)")
    parser.add_argument("--max-contacts", type=int, default=4)
    parser.add_argument("--skip-original", action="store_true",
                        help="Skip original timing (if already known)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: outputs/phase3d3e_jax_dynamics/jax_dynamics_benchmark.json)")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else OUTPUT_DIR / "jax_dynamics_benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    from wheeled_biped.utils.config import get_model_path
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))

    # ── Build constants ───────────────────────────────────────────────────
    from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants, _ensure_contact_constants
    constants = build_qp_wbc_constants(model)
    _ensure_contact_constants(constants)

    # ── Environment info ──────────────────────────────────────────────────
    env_info = {
        "jax_version": jax.__version__,
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "device_count": jax.device_count(),
        "jax_platform": str(jax.default_backend()),
        "devices": [str(d.device_kind) for d in jax.devices()] if jax.device_count() > 0 else [],
    }
    try:
        from jax.extend.backend import get_backend
        env_info["jax_backend"] = str(get_backend().platform)
    except ImportError:
        env_info["jax_backend"] = str(jax.lib.xla_bridge.get_backend().platform)

    print("=" * 70)
    print("Phase 3D.3-E JAX Dynamics Cache Benchmark")
    print(f"  States: {args.states}, Steps: {args.steps}")
    print(f"  Original calls: {args.original_calls}")
    print(f"  Platform: {env_info['jax_platform']}, x64: {env_info['jax_enable_x64']}")
    print("=" * 70)

    # ── Generate test states ──────────────────────────────────────────────
    print("\nGenerating test states...")
    test_states = _build_scenario_states(model, args.states)
    print(f"  Generated {len(test_states)} states")
    for s in test_states:
        print(f"    {s['scenario']}: {len(s['contacts'])} contacts, "
              f"|qvel|={np.linalg.norm(s['qvel']):.4f}")

    # ── Initialize cache and measure compile + warmup ─────────────────────
    print("\n--- Cache Initialization ---")
    from wheeled_biped.wbc.phase3d3e_jax_dynamics_cache import (
        initialize_jax_dynamics_cache,
        prepare_phase3b_snapshot_cached,
    )

    t0 = time.perf_counter()
    cache = initialize_jax_dynamics_cache(
        model, constants,
        max_contacts=args.max_contacts,
        warmup=True,
    )
    total_init_time = time.perf_counter() - t0

    compile_time = cache.compile_time_s
    warmup_time = cache.warmup_time_s
    print(f"  Compile time:  {compile_time:.3f}s")
    print(f"  Warmup time:   {warmup_time:.3f}s")
    print(f"  Total init:    {total_init_time:.3f}s")
    print(f"  Platform:      {cache.jax_platform}")

    # ── Time first cached call (post-warmup) ──────────────────────────────
    print("\n--- First Cached Call (post-warmup) ---")
    first_state = test_states[0]
    qp_c = constants.get("qp_constants", constants)

    t_first = time.perf_counter()
    _snap_first = prepare_phase3b_snapshot_cached(
        cache, first_state["scenario"],
        first_state["qpos"], first_state["qvel"], first_state["contacts"],
        constants, max_contacts=args.max_contacts,
    )
    first_call_time = time.perf_counter() - t_first
    print(f"  First call time: {first_call_time:.3f}s")

    # ── Time post-warmup cached calls ─────────────────────────────────────
    print(f"\n--- Post-Warmup Cached ({len(test_states)} states x {args.steps} steps) ---")
    post_warmup_times: list[float] = []

    for state_idx, state in enumerate(test_states):
        for step in range(args.steps):
            t_s = time.perf_counter()
            _snap = prepare_phase3b_snapshot_cached(
                cache, f"{state['scenario']}_step{step}",
                state["qpos"], state["qvel"], state["contacts"],
                constants, max_contacts=args.max_contacts,
            )
            elapsed = time.perf_counter() - t_s
            post_warmup_times.append(elapsed)

        print(f"  State {state_idx} ({state['scenario']}): "
              f"mean={np.mean(post_warmup_times[-args.steps:]):.3f}s")

    pw_mean = float(np.mean(post_warmup_times))
    pw_p50 = float(np.percentile(post_warmup_times, 50))
    pw_p95 = float(np.percentile(post_warmup_times, 95))
    pw_max = float(np.max(post_warmup_times))
    pw_min = float(np.min(post_warmup_times))
    pw_std = float(np.std(post_warmup_times))

    print(f"\n  Post-warmup summary ({len(post_warmup_times)} calls):")
    print(f"    mean:  {pw_mean:.3f}s")
    print(f"    p50:   {pw_p50:.3f}s")
    print(f"    p95:   {pw_p95:.3f}s")
    print(f"    max:   {pw_max:.3f}s")
    print(f"    min:   {pw_min:.3f}s")
    print(f"    std:   {pw_std:.3f}s")

    # ── Time original prepare_phase3b_snapshot ────────────────────────────
    original_times: list[float] = []
    original_first: float | None = None

    if not args.skip_original:
        print(f"\n--- Original prepare_phase3b_snapshot ({args.original_calls} calls) ---")
        print("  (expect ~300s per call on CPU...)")
        from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot

        # Use first state only to keep runtime reasonable
        ref_state = test_states[0]
        for i in range(args.original_calls):
            print(f"  Call {i+1}/{args.original_calls}...", end=" ", flush=True)
            t_o = time.perf_counter()
            _snap_orig = prepare_phase3b_snapshot(
                f"orig_bench_{i}", ref_state["qpos"], ref_state["qvel"],
                ref_state["contacts"], constants, max_contacts=args.max_contacts,
            )
            elapsed = time.perf_counter() - t_o
            original_times.append(elapsed)
            if i == 0:
                original_first = elapsed
            print(f"{elapsed:.3f}s")
    else:
        print("\n--- Original timing SKIPPED (--skip-original) ---")
        original_times = [333.0]  # known value from E1 diagnostic
        original_first = 333.0

    orig_mean = float(np.mean(original_times)) if original_times else 0.0

    # ── Compute speedup ───────────────────────────────────────────────────
    speedup = orig_mean / pw_mean if pw_mean > 0 else float("inf")
    speedup_p95 = orig_mean / pw_p95 if pw_p95 > 0 else float("inf")

    print(f"\n--- Speedup ---")
    print(f"  Original mean:     {orig_mean:.3f}s")
    print(f"  Cached mean:       {pw_mean:.3f}s")
    print(f"  Speedup (mean):    {speedup:.1f}x")
    print(f"  Speedup (P95):     {speedup_p95:.1f}x")
    print(f"  First cached:      {first_call_time:.3f}s")
    print(f"  Compile overhead:  {compile_time:.3f}s")

    # ── Cache diagnostics ─────────────────────────────────────────────────
    diagnostics = {
        "compile_time_s": compile_time,
        "warmup_time_s": warmup_time,
        "call_count": cache.call_count,
        "recompile_count": cache.recompile_count,
        "fallback_count": cache.fallback_count,
        "cache_hit_count": cache.cache_hit_count,
        "cache_miss_count": cache.cache_miss_count,
    }

    print(f"\n--- Cache Diagnostics ---")
    print(f"  Total cached calls:  {cache.call_count}")
    print(f"  Recompiles:          {cache.recompile_count}")
    print(f"  Fallbacks:           {cache.fallback_count}")

    # ── Determine verdict ─────────────────────────────────────────────────
    correctness_pass = False   # 6/8 from E6 audit, not 8/8
    recompile_after_warmup = cache.recompile_count == 0
    speedup_adequate = speedup >= 20.0

    if speedup_adequate and recompile_after_warmup:
        perf_verdict = "JAX_DYNAMICS_PARTIAL_SPEEDUP"
    elif recompile_after_warmup:
        perf_verdict = "JAX_DYNAMICS_INSUFFICIENT_SPEEDUP"
    else:
        perf_verdict = "JAX_DYNAMICS_RECOMPILE_DETECTED"

    verdict = perf_verdict

    # ── Assemble output ───────────────────────────────────────────────────
    benchmark = {
        "phase": "3D.3-E7",
        "verdict": verdict,
        "environment": env_info,
        "config": {
            "n_states": args.states,
            "n_steps": args.steps,
            "original_calls": args.original_calls,
            "max_contacts": args.max_contacts,
        },
        "timing": {
            "compile_time_s": compile_time,
            "warmup_time_s": warmup_time,
            "total_init_time_s": total_init_time,
            "first_cached_call_s": first_call_time,
            "post_warmup": {
                "n_calls": len(post_warmup_times),
                "mean_s": pw_mean,
                "p50_s": pw_p50,
                "p95_s": pw_p95,
                "p99_s": float(np.percentile(post_warmup_times, 99)),
                "max_s": pw_max,
                "min_s": pw_min,
                "std_s": pw_std,
            },
            "original": {
                "n_calls": len(original_times),
                "mean_s": orig_mean,
                "first_s": original_first,
                "all_s": original_times,
            },
            "speedup": {
                "mean_vs_original": speedup,
                "p95_vs_original": speedup_p95,
            },
        },
        "cache_diagnostics": diagnostics,
        "correctness_note": "6/8 scenarios pass E6 audit; 2 fail from float32 FD noise in QP.g/b_eq",
    }

    # ── Write output ──────────────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2, default=str)

    print(f"\nBenchmark saved to: {output_path}")
    print(f"\n{'='*70}")
    print(f"PHASE 3D.3-E7 JAX DYNAMICS BENCHMARK RESULT")
    print(f"{'='*70}")
    print(f"Verdict:            {verdict}")
    print(f"Compile time:       {compile_time:.1f}s")
    print(f"Warmup time:        {warmup_time:.1f}s")
    print(f"Original mean:      {orig_mean:.0f}s")
    print(f"Cached mean:        {pw_mean:.1f}s")
    print(f"Cached P95:         {pw_p95:.1f}s")
    print(f"Speedup:            {speedup:.0f}x")
    print(f"Recompile count:    {cache.recompile_count} after warmup")
    print(f"Fallback count:     {cache.fallback_count}")
    print(f"Correctness:        6/8 PASS (see E6 audit)")


if __name__ == "__main__":
    main()
