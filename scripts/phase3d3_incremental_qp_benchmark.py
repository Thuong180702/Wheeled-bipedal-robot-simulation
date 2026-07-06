#!/usr/bin/env python
"""Phase 3D.3-C4 — Incremental QP Benchmark.

Measures full rebuild vs incremental QP timing across perturbed states.

Usage:
    python scripts/phase3d3_incremental_qp_benchmark.py
    python scripts/phase3d3_incremental_qp_benchmark.py --states 3 --steps 3
"""

from __future__ import annotations

import argparse
import csv
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
from wheeled_biped.utils.config import get_model_path

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase3d3_incremental_qp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASK_MODE = "balanced_default"
ROLLING_MODE = "full_rolling_soft"

# ── Allowed and forbidden verdicts ────────────────────────────────────────────

ALLOWED_VERDICTS = frozenset({
    "INCREMENTAL_QP_CORRECTNESS_PASS",
    "INCREMENTAL_QP_CORRECTNESS_FAIL",
    "PARTIAL_SPEEDUP_NOT_FULLY_UNBLOCKED",
    "CLOSED_LOOP_EVALUATION_UNBLOCKED",
    "REALTIME_CANDIDATE_STRONG",
    "INCREMENTAL_QP_INSUFFICIENT",
})

FORBIDDEN_VERDICTS = frozenset({
    "REALTIME_READY",
    "PRODUCTION_READY",
    "WBC_PROMOTED",
    "DEFAULT_CONTROLLER_UPDATED",
})


# ═══════════════════════════════════════════════════════════════════════════════
# Verdict logic
# ═══════════════════════════════════════════════════════════════════════════════

def compute_verdict(incr_times, full_times, correctness_pass):
    """Compute the incremental QP benchmark verdict.

    Args:
        incr_times: list of incremental per-step timing measurements (seconds).
        full_times: list of full-rebuild timing measurements (seconds).
        correctness_pass: bool from the correctness audit.

    Returns:
        dict with keys: verdict, incr_mean_ms, incr_p95_ms, full_mean_ms,
        speedup_ratio, thresholds_explanation.
    """
    incr_mean = float(np.mean(incr_times)) if incr_times else float("inf")
    incr_p95 = (
        float(np.percentile(incr_times, 95))
        if len(incr_times) >= 20
        else float("inf")
    )
    full_mean = float(np.mean(full_times)) if full_times else float("inf")
    speedup = full_mean / incr_mean if incr_mean > 0 else float("inf")

    if not correctness_pass:
        verdict = "INCREMENTAL_QP_CORRECTNESS_FAIL"
        explanation = "Correctness audit did not pass; timing is irrelevant."
    elif incr_mean < 0.030 and incr_p95 < 0.050:
        verdict = "REALTIME_CANDIDATE_STRONG"
        explanation = (
            f"Mean incremental solve {incr_mean*1000:.1f} ms, "
            f"P95 {incr_p95*1000:.1f} ms — within realtime budget."
        )
    elif incr_mean < 0.120:
        verdict = "CLOSED_LOOP_EVALUATION_UNBLOCKED"
        explanation = (
            f"Mean incremental solve {incr_mean*1000:.1f} ms — "
            f"fast enough for offline closed-loop evaluation."
        )
    elif speedup >= 50:
        verdict = "PARTIAL_SPEEDUP_NOT_FULLY_UNBLOCKED"
        explanation = (
            f"Speedup {speedup:.1f}x but mean {incr_mean*1000:.1f} ms "
            f"still above 120 ms — useful but not fully unblocked."
        )
    else:
        verdict = "INCREMENTAL_QP_INSUFFICIENT"
        explanation = (
            f"Mean {incr_mean*1000:.1f} ms, speedup {speedup:.1f}x — "
            f"insufficient for intended use."
        )

    return {
        "verdict": verdict,
        "incr_mean_ms": incr_mean * 1000.0,
        "incr_p95_ms": incr_p95 * 1000.0,
        "full_mean_ms": full_mean * 1000.0,
        "speedup_ratio": speedup,
        "thresholds_explanation": explanation,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# State perturbation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_perturbed_states(model, mj_data, num_states, seed=42):
    """Generate a list of perturbed (qpos, qvel) from the keyframe.

    Each perturbed state has a small random perturbation applied to both
    torso orientation and base velocity to exercise different parts of the
    QP cost and constraint structure.

    Args:
        model: MuJoCo MjModel.
        mj_data: MuJoCo MjData (keyframe state).
        num_states: number of perturbed states to generate.
        seed: RNG seed for reproducibility.

    Returns:
        list of (qpos, qvel) tuples.
    """
    rng = np.random.RandomState(seed)
    keyframe_qpos = mj_data.qpos.copy()
    keyframe_qvel = np.zeros(model.nv)

    states = []
    for i in range(num_states):
        # Start from keyframe
        qpos = keyframe_qpos.copy()
        qvel = keyframe_qvel.copy()

        # Small orientation perturbation (roll + pitch)
        roll = rng.uniform(-0.03, 0.03)
        pitch = rng.uniform(-0.03, 0.03)
        # Apply via small-angle approximation on the quaternion
        # Quaternion perturbation: q_new = q_delta * q_current
        from scipy.spatial.transform import Rotation
        quat = qpos[3:7]  # w,x,y,z in MuJoCo
        q_scipy = [quat[1], quat[2], quat[3], quat[0]]  # → x,y,z,w
        r_perturb = Rotation.from_euler('xy', [roll, pitch])
        q_new = (r_perturb * Rotation.from_quat(q_scipy)).as_quat()
        qpos[3:7] = [q_new[3], q_new[0], q_new[1], q_new[2]]

        # Small velocity perturbation
        qvel[0:3] = rng.uniform(-0.05, 0.05, size=3)   # linear velocity
        qvel[3:6] = rng.uniform(-0.1, 0.1, size=3)      # angular velocity

        states.append((qpos, qvel))

    return states


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark(model, mj_data, constants, num_states, num_steps):
    """Run the full-vs-incremental timing benchmark.

    For each perturbed state:
      - Full rebuild: time one ``compute_wbc_torque_for_state`` call.
      - Incremental: initialize workspace from keyframe once, then run
        ``num_steps`` update+solve cycles to that same perturbed state,
        recording per-step timing.

    Args:
        model: MuJoCo MjModel.
        mj_data: MuJoCo MjData (keyframe).
        constants: eval constants dict.
        num_states: number of perturbed states.
        num_steps: number of incremental steps per state.

    Returns:
        dict with keys: full_rebuild_results, incremental_results.
    """
    keyframe_qpos = mj_data.qpos.copy()
    keyframe_qvel = np.zeros(model.nv)
    contacts = []

    states = generate_perturbed_states(model, mj_data, num_states)

    full_results = []
    incr_results = []

    for state_idx, (qpos, qvel) in enumerate(states):
        # ── Full rebuild ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        result_full = compute_wbc_torque_for_state(
            qpos, qvel, contacts, TASK_MODE, ROLLING_MODE, constants,
            qp_backend="osqp",
        )
        full_time = time.perf_counter() - t0

        full_results.append({
            "state_index": state_idx,
            "path": "full_rebuild",
            "time_s": full_time,
            "solve_success": bool(result_full["solve_success"]),
            "solve_status": (
                "ok" if result_full["solve_success"]
                else result_full["solve_status"]
            ),
        })

        # ── Incremental (init once + N steps) ───────────────────────────────
        workspace = initialize_incremental_qp_workspace(
            model, keyframe_qpos, keyframe_qvel, contacts,
            task_mode=TASK_MODE, rolling_mode=ROLLING_MODE,
            constants=constants, max_contacts=4,
        )

        for step_idx in range(num_steps):
            t0 = time.perf_counter()
            update_incremental_qp_workspace(workspace, qpos, qvel, contacts)
            result_incr = solve_incremental_qp(workspace, warm_start=True)
            step_time = time.perf_counter() - t0

            incr_results.append({
                "state_index": state_idx,
                "step_index": step_idx,
                "path": "incremental",
                "time_s": step_time,
                "update_time_s": 0.0,  # filled by diagnostics if available
                "solve_time_s": result_incr.get("solve_time_s", step_time),
                "solve_success": bool(result_incr["solve_success"]),
                "solve_status": result_incr["solve_status"],
            })

        workspace.backend.close()

    return {
        "full_rebuild_results": full_results,
        "incremental_results": incr_results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Correctness check (lightweight inline re-run)
# ═══════════════════════════════════════════════════════════════════════════════

def check_correctness_pass(output_dir):
    """Check whether a correctness audit JSON already exists and passed.

    If the file does not exist, returns False (correctness not yet verified).
    """
    correctness_path = output_dir / "incremental_qp_correctness.json"
    if not correctness_path.exists():
        return False
    with open(correctness_path) as f:
        data = json.load(f)
    return data.get("verdict") == "INCREMENTAL_QP_CORRECTNESS_PASS"


# ═══════════════════════════════════════════════════════════════════════════════
# Output writers
# ═══════════════════════════════════════════════════════════════════════════════

def write_csv(incr_times, full_times, output_dir):
    """Write per-measurement timing CSV."""
    csv_path = output_dir / "incremental_qp_timing.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "time_s", "time_ms"])
        for t in incr_times:
            writer.writerow(["incremental", f"{t:.6f}", f"{t*1000:.3f}"])
        for t in full_times:
            writer.writerow(["full_rebuild", f"{t:.6f}", f"{t*1000:.3f}"])
    return csv_path


def write_benchmark_json(results, output_dir):
    """Write the full benchmark results JSON."""
    json_path = output_dir / "incremental_qp_benchmark.json"
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "task_mode": TASK_MODE,
            "rolling_mode": ROLLING_MODE,
        },
        **results,
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    return json_path


def write_verdict_json(verdict_result, output_dir):
    """Write the verdict JSON."""
    verdict_path = output_dir / "incremental_qp_verdict.json"
    with open(verdict_path, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **verdict_result,
        }, f, indent=2)
    return verdict_path


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 3D.3-C4 — Incremental QP Benchmark"
    )
    parser.add_argument(
        "--states", type=int, default=4,
        help="Number of perturbed states to benchmark (default: 4)."
    )
    parser.add_argument(
        "--steps", type=int, default=5,
        help="Number of incremental steps per state (default: 5)."
    )
    parser.add_argument(
        "--skip-correctness-check", action="store_true",
        help="Skip checking for a prior correctness pass."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Phase 3D.3-C4 — Incremental QP Benchmark")
    print("=" * 60)
    print(f"  States: {args.states}")
    print(f"  Steps per state: {args.steps}")
    print(f"  Total full rebuilds: {args.states}")
    print(f"  Total incremental solves: {args.states * args.steps}")

    # ── Correctness gate ────────────────────────────────────────────────────
    if not args.skip_correctness_check:
        correctness_pass = check_correctness_pass(OUTPUT_DIR)
        if not correctness_pass:
            print("\n[WARN] No prior correctness pass found.")
            print("  Run: python scripts/phase3d3_incremental_qp_correctness_audit.py")
            print("  Or use --skip-correctness-check to bypass.")
    else:
        correctness_pass = check_correctness_pass(OUTPUT_DIR)
        print(f"\n  Correctness check skipped. "
              f"Last known: {'PASS' if correctness_pass else 'UNKNOWN'}")

    # ── Load model and constants ────────────────────────────────────────────
    print("\n[1/3] Loading model and building constants ...")
    t_load = time.perf_counter()
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    mj_data = mujoco.MjData(model)

    qp_c = build_qp_wbc_constants(model)
    rolling_c = build_wheel_rolling_constants(model)
    constants = build_three_arm_eval_constants(model, qp_c, rolling_c)
    print(f"  Done in {time.perf_counter() - t_load:.1f}s")

    # ── Run benchmark ───────────────────────────────────────────────────────
    print(f"\n[2/3] Running benchmark ({args.states} states x {args.steps} steps) ...")
    t_bench = time.perf_counter()
    results = run_benchmark(model, mj_data, constants, args.states, args.steps)
    bench_time = time.perf_counter() - t_bench
    print(f"  Benchmark completed in {bench_time:.1f}s")

    # ── Collect timing arrays ───────────────────────────────────────────────
    incr_times = [r["time_s"] for r in results["incremental_results"]]
    full_times = [r["time_s"] for r in results["full_rebuild_results"]]

    incr_success = sum(1 for r in results["incremental_results"] if r["solve_success"])
    full_success = sum(1 for r in results["full_rebuild_results"] if r["solve_success"])

    print(f"\n  Full rebuild:   {full_success}/{len(full_times)} success  "
          f"mean={np.mean(full_times)*1000:.1f}ms")
    print(f"  Incremental:    {incr_success}/{len(incr_times)} success  "
          f"mean={np.mean(incr_times)*1000:.1f}ms")

    # ── Compute verdict ────────────────────────────────────────────────────
    verdict_result = compute_verdict(incr_times, full_times, correctness_pass)
    results["verdict"] = verdict_result

    # Sanity check verdicts
    if verdict_result["verdict"] in FORBIDDEN_VERDICTS:
        raise AssertionError(
            f"Forbidden verdict produced: {verdict_result['verdict']}"
        )
    if verdict_result["verdict"] not in ALLOWED_VERDICTS:
        raise AssertionError(
            f"Unknown verdict produced: {verdict_result['verdict']}"
        )

    # ── Write outputs ───────────────────────────────────────────────────────
    print(f"\n[3/3] Writing outputs ...")
    json_path = write_benchmark_json(results, OUTPUT_DIR)
    print(f"  JSON: {json_path}")

    csv_path = write_csv(incr_times, full_times, OUTPUT_DIR)
    print(f"  CSV:  {csv_path}")

    verdict_path = write_verdict_json(verdict_result, OUTPUT_DIR)
    print(f"  Verdict: {verdict_path}")

    print(f"\n{'='*60}")
    print(f"Verdict: {verdict_result['verdict']}")
    print(f"  Incremental mean: {verdict_result['incr_mean_ms']:.1f} ms")
    if verdict_result['incr_p95_ms'] != float('inf'):
        print(f"  Incremental P95:  {verdict_result['incr_p95_ms']:.1f} ms")
    print(f"  Full rebuild mean: {verdict_result['full_mean_ms']:.1f} ms")
    print(f"  Speedup:           {verdict_result['speedup_ratio']:.1f}x")
    print(f"  Explanation: {verdict_result['thresholds_explanation']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
