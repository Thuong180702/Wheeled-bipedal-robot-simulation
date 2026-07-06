"""Phase 3D.2 — Fast Solver Performance Benchmark.

Benchmarks the fast structured QP backend across Phase 3C scenarios,
task modes, and rolling modes. Records timing, success rates, and residuals.

Minimum benchmark: 12 × 2 × 3 = 72 QP solves.
Preferred benchmark: 12 × 2 × 3 × 10 repeats with warm-start/cold-start.

Usage:
    python scripts/phase3d2_fast_solver_benchmark.py --backend osqp --warm-start --repeat 10
    python scripts/phase3d2_fast_solver_benchmark.py --backend osqp --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Benchmark config ──────────────────────────────────────────────────────────

TASK_MODES = ["feasibility_only", "balanced_default"]
ROLLING_MODES = ["normal_only", "lateral_soft", "full_rolling_soft"]

# Performance targets
BATCH_TARGET_MEAN_S = 0.050   # 50 ms
BATCH_TARGET_P95_S = 0.100    # 100 ms
REALTIME_TARGET_MEAN_S = 0.010  # 10 ms
REALTIME_TARGET_P95_S = 0.020   # 20 ms
REALTIME_TARGET_MAX_S = 0.050   # 50 ms


def main():
    parser = argparse.ArgumentParser(description="Phase 3D.2 Fast Solver Benchmark")
    parser.add_argument("--backend", default="osqp",
                        help="Solver backend to benchmark")
    parser.add_argument("--warm-start", action="store_true", default=True,
                        help="Use warm-start across solves")
    parser.add_argument("--no-warm-start", dest="warm_start", action="store_false",
                        help="Cold-start every solve")
    parser.add_argument("--repeat", type=int, default=10,
                        help="Number of repeats per case")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 2 scenarios, 2 task modes, 2 rolling modes, 1 repeat")
    parser.add_argument("--num-scenarios", type=int, default=12,
                        help="Number of scenarios to generate (default: 12)")
    parser.add_argument("--output-dir", default="outputs/phase3d2",
                        help="Output directory")
    parser.add_argument("--eps-abs", type=float, default=1e-5)
    parser.add_argument("--eps-rel", type=float, default=1e-5)
    parser.add_argument("--max-iter", type=int, default=4000)
    parser.add_argument("--save-jsonl", action="store_true", default=True,
                        help="Save per-solve results as JSONL")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Phase 3D.2 — Fast Solver Performance Benchmark")
    print(f"  Backend: {args.backend}")
    print(f"  Warm-start: {args.warm_start}")
    print(f"  Repeats: {args.repeat}")
    print("=" * 80)

    # ── Import modules ───────────────────────────────────────────────────
    import mujoco

    from wheeled_biped.wbc.qp_solver_backends import (
        make_backend,
        get_available_qp_backends,
        SLSQPLegacyBackend,
    )
    from wheeled_biped.wbc.structured_qp_problem import (
        build_structured_qp_from_phase3c_snapshot,
    )
    from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants
    from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot

    available = get_available_qp_backends()
    print(f"\nSolver backends: {json.dumps(available)}")

    # ── Create backend ───────────────────────────────────────────────────
    try:
        backend = make_backend(args.backend, eps_abs=args.eps_abs,
                                eps_rel=args.eps_rel, max_iter=args.max_iter)
        print(f"Backend: {backend.name}")
    except ValueError as exc:
        print(f"WARNING: {exc}")
        backend = SLSQPLegacyBackend()
        print(f"Fell back to: {backend.name}")

    uses_slsqp_only = isinstance(backend, SLSQPLegacyBackend)

    # ── Build scenarios ──────────────────────────────────────────────────
    if args.quick:
        num_scenarios = 2
        task_modes_local = ["feasibility_only", "balanced_default"]
        rolling_modes_local = ["normal_only", "full_rolling_soft"]
        repeats = 1
    else:
        num_scenarios = args.num_scenarios
        task_modes_local = TASK_MODES
        rolling_modes_local = ROLLING_MODES
        repeats = args.repeat

    print(f"\nBuilding {num_scenarios} scenarios...")
    constants, snapshots = _build_scenarios(num_scenarios)
    print(f"  Got {len(snapshots)} snapshots")
    total_solves = len(snapshots) * len(task_modes_local) * len(rolling_modes_local) * repeats
    print(f"  Total solves: {total_solves}")

    # ── Run benchmarks ───────────────────────────────────────────────────
    print(f"\nRunning benchmarks...")
    results = []
    warm_start_vec = None
    first_solve_setup_time = None

    total_start = time.perf_counter()
    solve_count = 0
    success_count = 0
    solve_times = []
    setup_times = []

    jsonl_path = os.path.join(args.output_dir, "phase3d2_fast_solver_benchmark.jsonl")
    jsonl_file = open(jsonl_path, "w") if args.save_jsonl else None

    for rep in range(repeats):
        if repeats > 1:
            print(f"\n  Repeat {rep+1}/{repeats}")

        for si, snap in enumerate(snapshots):
            scenario_name = getattr(snap, "scenario_name", f"scenario_{si}")

            for tm in task_modes_local:
                for rm in rolling_modes_local:
                    try:
                        t0 = time.perf_counter()
                        sqp = build_structured_qp_from_phase3c_snapshot(
                            snap, tm, rm, constants,
                            padded_contacts=True, max_contacts=4,
                        )
                        build_time = time.perf_counter() - t0

                        # Check if structure changed for setup timing
                        structure_key = (sqp.nx, sqp.nc)

                        t_solve_start = time.perf_counter()
                        sol = backend.solve(sqp, warm_start=(
                            warm_start_vec if args.warm_start else None))
                        solve_time = time.perf_counter() - t_solve_start

                        setup_time = sol.setup_time_s
                        if first_solve_setup_time is None and setup_time > 0:
                            first_solve_setup_time = setup_time

                        solve_count += 1
                        if sol.success:
                            success_count += 1
                            warm_start_vec = sol.x.copy()

                        solve_times.append(sol.solve_time_s)
                        if setup_time > 0:
                            setup_times.append(setup_time)

                        entry = {
                            "scenario": scenario_name,
                            "task_mode": tm,
                            "rolling_mode": rm,
                            "repeat": rep,
                            "backend": backend.name,
                            "success": sol.success,
                            "status": sol.status,
                            "solve_time_s": sol.solve_time_s,
                            "setup_time_s": setup_time,
                            "build_time_s": build_time,
                            "total_time_s": build_time + setup_time + sol.solve_time_s,
                            "iterations": sol.iterations,
                            "primal_residual": sol.primal_residual,
                            "dual_residual": sol.dual_residual,
                            "objective_value": sol.objective_value,
                            "warm_start_used": args.warm_start and rep > 0,
                        }

                        # Extract solution components
                        if sol.success:
                            vs = sqp.variable_slices
                            x = sol.x
                            qdd_s, qdd_e = vs["qdd"]
                            tau_s, tau_e = vs["tau"]
                            lam_s, lam_e = vs["lambda"]
                            entry["max_abs_qdd"] = float(np.max(np.abs(x[qdd_s:qdd_e])))
                            entry["max_abs_tau"] = float(np.max(np.abs(x[tau_s:tau_e])))
                            entry["max_abs_lambda"] = float(np.max(np.abs(x[lam_s:lam_e]))) if lam_e > lam_s else 0.0

                            # Hard constraint residuals
                            A = sqp.A.toarray()
                            l_vec = sqp.l
                            u_vec = sqp.u
                            Ax = A @ x
                            entry["primal_violation"] = float(
                                np.max(np.maximum(0, l_vec - Ax)) +
                                np.max(np.maximum(0, Ax - u_vec))
                            )

                        results.append(entry)

                        if jsonl_file:
                            jsonl_file.write(json.dumps(entry, default=str) + "\n")

                    except Exception as exc:
                        print(f"    ERROR: {scenario_name}/{tm}/{rm}: {exc}")
                        traceback.print_exc()
                        results.append({
                            "scenario": scenario_name,
                            "task_mode": tm,
                            "rolling_mode": rm,
                            "repeat": rep,
                            "backend": backend.name,
                            "success": False,
                            "error": str(exc),
                        })

        if repeats > 1 and (rep + 1) % 5 == 0:
            _print_progress(solve_count, total_solves, success_count, solve_times)

    total_elapsed = time.perf_counter() - total_start

    if jsonl_file:
        jsonl_file.close()

    # ── Summary statistics ───────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("SUMMARY")

    solve_times_arr = np.array(solve_times) if solve_times else np.array([float("inf")])
    success_rate = success_count / max(solve_count, 1)

    mean_st = float(np.mean(solve_times_arr))
    p50_st = float(np.median(solve_times_arr))
    p95_st = float(np.percentile(solve_times_arr, 95)) if len(solve_times_arr) >= 20 else mean_st
    max_st = float(np.max(solve_times_arr))
    min_st = float(np.min(solve_times_arr))

    print(f"  Solves: {solve_count}")
    print(f"  Successes: {success_count}")
    print(f"  Success rate: {success_rate:.4f}")
    print(f"  Total elapsed: {total_elapsed:.2f}s")
    print(f"  Mean solve time: {mean_st*1000:.2f} ms")
    print(f"  Median solve time: {p50_st*1000:.2f} ms")
    print(f"  P95 solve time: {p95_st*1000:.2f} ms")
    print(f"  Max solve time: {max_st*1000:.2f} ms")
    print(f"  Min solve time: {min_st*1000:.2f} ms")
    if setup_times:
        print(f"  Mean setup time: {np.mean(setup_times)*1000:.2f} ms")
        if first_solve_setup_time is not None:
            print(f"  First solve setup time: {first_solve_setup_time*1000:.2f} ms")

    meets_batch = (
        mean_st <= BATCH_TARGET_MEAN_S
        and p95_st <= BATCH_TARGET_P95_S
        and success_rate >= 0.99
    )
    meets_realtime = (
        mean_st <= REALTIME_TARGET_MEAN_S
        and p95_st <= REALTIME_TARGET_P95_S
        and max_st <= REALTIME_TARGET_MAX_S
        and success_rate >= 0.99
    )

    print(f"\n  Meets batch target (mean<=50ms, p95<=100ms, SR>=99%): {meets_batch}")
    print(f"  Meets realtime preferred (mean<=10ms, p95<=20ms, max<=50ms, SR>=99%): {meets_realtime}")
    if uses_slsqp_only:
        print("  NOTE: Only SLSQP available. Cannot achieve READY.")

    # ── Save summary ─────────────────────────────────────────────────────
    summary = {
        "phase": "3D.2",
        "step": "performance_benchmark",
        "backend": args.backend,
        "fast_backend_available": available.get(args.backend, False),
        "uses_slsqp_only": uses_slsqp_only,
        "warm_start_enabled": args.warm_start,
        "num_solves": solve_count,
        "success_rate": success_rate,
        "mean_solve_time_s": mean_st,
        "p50_solve_time_s": p50_st,
        "p95_solve_time_s": p95_st,
        "max_solve_time_s": max_st,
        "min_solve_time_s": min_st,
        "mean_setup_time_s": float(np.mean(setup_times)) if setup_times else None,
        "first_setup_time_s": first_solve_setup_time,
        "total_elapsed_s": total_elapsed,
        "meets_batch_target": meets_batch,
        "meets_realtime_preferred_target": meets_realtime,
        "num_scenarios": len(snapshots),
        "num_task_modes": len(task_modes_local),
        "num_rolling_modes": len(rolling_modes_local),
        "num_repeats": repeats,
        "total_solves_intended": total_solves,
    }

    summary_path = os.path.join(args.output_dir, "phase3d2_fast_solver_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to: {summary_path}")
    if args.save_jsonl:
        print(f"Per-solve results saved to: {jsonl_path}")

    return 0


def _print_progress(done, total, successes, times):
    arr = np.array(times)
    print(f"    [{done}/{total}] SR={successes/max(done,1):.3f}, "
          f"mean={np.mean(arr)*1000:.1f}ms, "
          f"p95={np.percentile(arr,95)*1000:.1f}ms")


def _build_scenarios(num_scenarios):
    """Build a diverse set of test scenarios."""
    import mujoco
    from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants
    from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot
    from wheeled_biped.utils.config import get_model_path

    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    constants = build_qp_wbc_constants(model)

    # Find wheel geom IDs for contact extraction
    wheel_geom_ids = set()
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name and ("wheel" in name.lower()):
            wheel_geom_ids.add(i)

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

    rng = np.random.RandomState(42)
    snapshots = []
    heights = np.linspace(0.40, 0.70, num_scenarios)
    qvel_scales = np.linspace(0.0, 0.3, num_scenarios)

    for i, (h, vscale) in enumerate(zip(heights, qvel_scales)):
        name = f"benchmark_scenario_{i}_h{h:.2f}"

        data = mujoco.MjData(model)
        try:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        except Exception:
            mujoco.mj_resetData(model, data)

        qpos = data.qpos.copy()
        default_h = 0.60
        delta_h = h - default_h
        qpos[9] += delta_h * 0.3   # l_hip_pitch
        qpos[10] += delta_h * 0.7   # l_knee
        qpos[14] += delta_h * 0.3   # r_hip_pitch
        qpos[15] += delta_h * 0.7   # r_knee

        # Small random perturbation
        qpos[9:11] += rng.uniform(-0.03, 0.03, 2)
        qpos[14:16] += rng.uniform(-0.03, 0.03, 2)

        qvel = np.zeros(16)
        qvel[:6] = rng.uniform(-vscale, vscale, 6)
        qvel[6:16] = rng.uniform(-vscale*0.5, vscale*0.5, 10)

        try:
            snap = prepare_phase3b_snapshot(name, qpos, qvel, _extract_contacts(data), constants)
            snapshots.append(snap)
        except Exception as exc:
            print(f"  WARNING: Could not build snapshot {name}: {exc}")

    return constants, snapshots


if __name__ == "__main__":
    sys.exit(main())
