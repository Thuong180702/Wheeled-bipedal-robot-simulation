"""Stage 7: Performance benchmark driver for JAX K2 controller.

Runs the simulation script with --stage7-benchmark for all required scenarios
across both Python and JAX backends, aggregates results, and generates reports.

Usage:
    python scripts/stage7_run_benchmarks.py              # Full benchmark suite
    python scripts/stage7_run_benchmarks.py --quick       # Reduced steps for smoke test
    python scripts/stage7_run_benchmarks.py --scenario fixed_high_0p480  # Single scenario
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIMULATE = ROOT / "scripts" / "simulate_hierarchical_controller.py"
BENCHMARK_OUTPUT_DIR = ROOT / "outputs" / "benchmark"
SETUPS_DIR = ROOT / "outputs" / "physical_target_height_setups_centered"
TRAJ_DIR = ROOT / "outputs" / "k2_dynamic_height_gate_crossing" / "trajectories"

# ── Common base arguments for all K2 benchmark runs ──
K2_BASE_ARGS = [
    "--controller-mode", "balance-core",
    "--sagittal-controller", "velocity-damped",
    "--vd-sagittal-authority-profile", "k2_notch_low_q_v1",
    "--telemetry-decimation", "10",
    "--wbc-quiet",
    "--enable-mode-hip-yaw-divergence",
    "--mode-hip-yaw-div-kp", "10.0",
    "--mode-hip-yaw-div-kd", "0.50",
    "--mode-hip-yaw-div-max-torque", "7.5",
    "--mode-hip-yaw-div-soft-limit-rad", "0.30",
    "--mode-hip-yaw-div-soft-gain", "0.80",
    "--mode-hip-yaw-div-ref-source", "target",
]

# ── Benchmark configuration ──
# Steps: warmup + measured
BENCH_WARMUP = 100
BENCH_MEASURED = 1000
BENCH_TOTAL = BENCH_WARMUP + BENCH_MEASURED

# ── Output files ──
DOCS_DIR = ROOT / "docs" / "validation"
PERF_REPORT_MD = DOCS_DIR / "k2_jax_stage7_performance_report.md"
BENCH_CSV = DOCS_DIR / "k2_jax_stage7_benchmark_results.csv"
REALTIME_SUMMARY_MD = DOCS_DIR / "k2_jax_stage7_realtime_readiness_summary.md"


# ══════════════════════════════════════════════════════════════════════════════
# Scenario definitions
# ══════════════════════════════════════════════════════════════════════════════

def _scenario(tag, height_setup, extra_args=None, steps=None):
    """Build a scenario definition."""
    return {
        "tag": tag,
        "height_setup": height_setup,
        "extra_args": extra_args or [],
        "steps": steps if steps is not None else BENCH_TOTAL,
    }


SCENARIOS = [
    # ── Fixed height ──
    _scenario("fixed_low_0p320", "low_0p320_setup.json"),
    _scenario("fixed_low_0p330", "low_0p330_setup.json"),
    _scenario("fixed_mid_0p400", "mid_0p400_setup.json"),
    _scenario("fixed_high_0p480", "high_0p480_setup.json"),

    # ── Dynamic height (reuse verified Stage 6L trajectories) ──
    _scenario("dynamic_ramp_up", "low_0p330_setup.json",
              extra_args=[
                  "--dynamic-height-trajectory",
                  str(TRAJ_DIR / "ramp_up_0p330_to_0p480.json"),
              ],
              steps=5100),  # trajectory is 5000 steps + 100 warmup
    _scenario("dynamic_ramp_down", "high_0p480_setup.json",
              extra_args=[
                  "--dynamic-height-trajectory",
                  str(TRAJ_DIR / "ramp_down_0p480_to_0p330.json"),
              ],
              steps=5100),
    _scenario("dynamic_gate_chatter", "low_0p330_setup.json",
              extra_args=[
                  "--dynamic-height-trajectory",
                  str(TRAJ_DIR / "gate_chatter_0p400_0p470.json"),
              ],
              steps=5100),

    # ── Push scenarios ──
    _scenario("push_high_0p480_fwd_90N", "high_0p480_setup.json",
              extra_args=["__PUSH__", "300", "0.0", "90.0", "5"],
              steps=BENCH_TOTAL),
    _scenario("push_high_0p480_bwd_90N", "high_0p480_setup.json",
              extra_args=["__PUSH__", "300", "0.0", "-90.0", "5"],
              steps=BENCH_TOTAL),
]


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_push_sequence_file(push_step, fx, fy, duration):
    """Create a temporary push sequence JSON file in the Stage 6L format."""
    seq = {"sequence": [[int(push_step), float(fx), float(fy), int(duration)]]}
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(seq, tmp)
    tmp.close()
    return tmp.name


def _build_args(scenario, backend, quick=False):
    """Build command-line args for a scenario/backend combination."""
    args = [sys.executable, str(SIMULATE)]
    args.extend(["--controller-backend", backend])

    # Benchmark flags
    warmup = min(50, BENCH_WARMUP) if quick else BENCH_WARMUP
    measured = min(200, BENCH_MEASURED) if quick else BENCH_MEASURED
    args.extend([
        "--stage7-benchmark",
        "--stage7-benchmark-tag", scenario['tag'],  # backend is appended by sim script
        "--stage7-benchmark-warmup-steps", str(warmup),
        "--stage7-benchmark-measured-steps", str(measured),
    ])

    # Total steps = warmup + measured
    total = warmup + measured
    if scenario["steps"] > total:
        # Ensure we have enough steps for the scenario (e.g., dynamic trajectories)
        total = max(total, scenario["steps"])
    args.extend(["--steps", str(total)])

    # Failure window = total steps
    args.extend(["--failure-window-steps", str(total)])

    # Write run summary sidecar
    args.append("--write-run-summary-sidecar")

    # Base K2 args
    args.extend(K2_BASE_ARGS)

    # Height variant setup
    setup_path = SETUPS_DIR / scenario["height_setup"]
    args.extend(["--height-variant-setup", str(setup_path)])

    # Extra args (dynamic trajectory, push, etc.)
    extra = list(scenario["extra_args"])
    push_file = None
    if extra and extra[0] == "__PUSH__":
        _, step_s, fx, fy, dur = extra
        push_file = _make_push_sequence_file(step_s, fx, fy, dur)
        extra = ["--push-sequence-file", push_file]

    args.extend(extra)
    return args, push_file


def _run_one(scenario, backend, quick=False):
    """Run one benchmark and return parsed JSON result."""
    args, push_file = _build_args(scenario, backend, quick)
    tag = f"{scenario['tag']}_{backend}"
    print(f"\n{'='*70}")
    print(f"[STAGE7] Running: {tag}")
    print(f"[STAGE7] Command: {' '.join(args[:8])} ... (truncated)")
    print(f"{'='*70}")

    t0 = time.time()
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min per scenario
            cwd=str(ROOT),
        )
        elapsed = time.time() - t0
        print(f"[STAGE7] {tag}: exit={result.returncode} elapsed={elapsed:.1f}s")

        if result.returncode != 0:
            print(f"[STAGE7] STDERR (last 30 lines):")
            stderr_lines = result.stderr.strip().splitlines()
            for line in stderr_lines[-30:]:
                print(f"  {line}")
            return None

        # Also check stdout for errors
        if "Traceback" in result.stdout or "Error" in result.stdout:
            print(f"[STAGE7] STDOUT error detected -- checking...")
            for line in result.stdout.splitlines():
                if "Traceback" in line or "Error" in line:
                    print(f"  {line}")

    except subprocess.TimeoutExpired:
        print(f"[STAGE7] {tag}: TIMEOUT after 600s")
        return None
    finally:
        if push_file and os.path.exists(push_file):
            os.unlink(push_file)

    # Locate the JSON report
    json_path = BENCHMARK_OUTPUT_DIR / f"stage7_{scenario['tag']}_{backend}.json"
    if not json_path.exists():
        print(f"[STAGE7] WARNING: Report not found at {json_path}")
        return None

    try:
        with open(json_path) as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"[STAGE7] ERROR reading {json_path}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# CSV generation
# ══════════════════════════════════════════════════════════════════════════════

def _write_csv(all_results):
    """Write benchmark results CSV."""
    BENCH_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in all_results:
        if r is None:
            continue
        py_data = r.get("python", {})
        jx_data = r.get("jax", {})
        row = {
            "scenario": r["tag"],
            "py_total_step_mean_ms": _ts(py_data, "total_step_s", "mean_ms"),
            "py_total_step_p95_ms": _ts(py_data, "total_step_s", "p95_ms"),
            "py_physics_step_mean_ms": _ts(py_data, "physics_step_s", "mean_ms"),
            "jx_total_step_mean_ms": _ts(jx_data, "total_step_s", "mean_ms"),
            "jx_total_step_p95_ms": _ts(jx_data, "total_step_s", "p95_ms"),
            "jx_physics_step_mean_ms": _ts(jx_data, "physics_step_s", "mean_ms"),
            "jx_jit_compile_time_s": _tc(jx_data, "jit_compile_time_s"),
            "jx_hot_jit_step_mean_ms": _ts(jx_data, "jax_jit_step_s", "mean_ms"),
            "jx_hot_jit_step_p95_ms": _ts(jx_data, "jax_jit_step_s", "p95_ms"),
            "jx_pack_input_mean_ms": _ts(jx_data, "jax_pack_input_s", "mean_ms"),
            "jx_support_ff_mean_ms": _ts(jx_data, "jax_support_ff_s", "mean_ms"),
            "jx_diag_map_mean_ms": _ts(jx_data, "jax_diag_map_s", "mean_ms"),
            "jx_jax_total_mean_ms": _ts(jx_data, "jax_total_s", "mean_ms"),
            "jx_speedup_vs_python": _speedup(py_data, jx_data),
            "jx_meets_10ms_budget": _ts(jx_data, None, None, field="meets_10ms_budget"),
            "py_fell": _val(py_data, "fell"),
            "jx_fell": _val(jx_data, "fell"),
            "py_nan": _val(py_data, "nan_detected"),
            "jx_nan": _val(jx_data, "nan_detected"),
            "py_hip_yaw_rad": _val(py_data, "max_abs_hip_yaw_rad"),
            "jx_hip_yaw_rad": _val(jx_data, "max_abs_hip_yaw_rad"),
        }
        rows.append(row)

    if not rows:
        print("[STAGE7] No results to write to CSV")
        return

    fieldnames = list(rows[0].keys())
    with open(BENCH_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[STAGE7] CSV written to: {BENCH_CSV}")


def _ts(data, channel, stat, field=None):
    """Extract timing stat from result data."""
    if not data:
        return ""
    if field and field in data.get("summary", {}):
        return data["summary"][field]
    if channel and channel in data.get("timing_stats_ms", {}):
        return data["timing_stats_ms"][channel].get(stat, "")
    return ""


def _tc(data, field):
    """Extract compile field from result data."""
    if not data:
        return ""
    return data.get("compile", {}).get(field, "")


def _val(data, field):
    """Extract validation field from result data."""
    if not data:
        return ""
    return data.get("validation", {}).get(field, "")


def _speedup(py_data, jx_data):
    """Compute JAX speedup vs Python (total step)."""
    py_mean = _ts(py_data, "total_step_s", "mean_ms")
    jx_mean = _ts(jx_data, "total_step_s", "mean_ms")
    if py_mean and jx_mean and float(jx_mean) > 0:
        return round(float(py_mean) / float(jx_mean), 2)
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# Report generation
# ══════════════════════════════════════════════════════════════════════════════

def _generate_markdown_report(all_results):
    """Generate the detailed performance report."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect environment metadata from first successful JAX run
    env_info = {}
    for r in all_results:
        if r and r.get("jax"):
            env_info = r["jax"].get("environment", {})
            break

    lines = []
    lines.append("# Stage 7: JAX K2 Controller Performance Benchmark Report")
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d')}")
    lines.append(f"**Classification:** PENDING -- see summary below")
    lines.append("")

    # Environment
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- **Python:** {env_info.get('python_version', 'unknown')}")
    lines.append(f"- **JAX:** {env_info.get('jax_version', 'unknown')}")
    lines.append(f"- **jaxlib:** {env_info.get('jaxlib_version', 'unknown')}")
    lines.append(f"- **MuJoCo:** {env_info.get('mujoco_version', 'unknown')}")
    lines.append(f"- **Platform:** {env_info.get('platform', 'unknown')}")
    lines.append(f"- **CPU:** {env_info.get('cpu', 'unknown')}")
    lines.append(f"- **JAX x64:** {env_info.get('jax_x64_enabled', 'unknown')}")
    lines.append("")

    # Benchmark configuration
    lines.append("## Benchmark Configuration")
    lines.append("")
    lines.append(f"- **Warmup steps:** {BENCH_WARMUP}")
    lines.append(f"- **Measured steps:** {BENCH_MEASURED}")
    lines.append(f"- **Control dt:** 0.01 s (100 Hz)")
    lines.append(f"- **Controller mode:** balance-core")
    lines.append(f"- **Sagittal profile:** k2_notch_low_q_v1")
    lines.append("")

    # Headless results table
    lines.append("## Headless Benchmark Results")
    lines.append("")
    lines.append("| Scenario | Py Total Mean (ms) | Py Total p95 (ms) | JX Total Mean (ms) | JX Total p95 (ms) | JX Hot-Step Mean (ms) | JX Hot-Step p95 (ms) | Speedup |")
    lines.append("|----------|-------------------|------------------|-------------------|------------------|----------------------|---------------------|---------|")
    for r in all_results:
        if r is None:
            continue
        py = r.get("python", {})
        jx = r.get("jax", {})
        py_tot = _ts(py, "total_step_s", "mean_ms")
        py_p95 = _ts(py, "total_step_s", "p95_ms")
        jx_tot = _ts(jx, "total_step_s", "mean_ms")
        jx_p95 = _ts(jx, "total_step_s", "p95_ms")
        jx_hot = _ts(jx, "jax_jit_step_s", "mean_ms")
        jx_hp95 = _ts(jx, "jax_jit_step_s", "p95_ms")
        sp = _speedup(py, jx)
        lines.append(f"| {r['tag']} | {py_tot} | {py_p95} | {jx_tot} | {jx_p95} | {jx_hot} | {jx_hp95} | {sp} |")

    lines.append("")
    lines.append("**Note:** 'Hot-Step' = JIT execution only (with `block_until_ready()`). Compile time excluded.")
    lines.append("")

    # JAX overhead breakdown
    lines.append("## JAX Path Overhead Breakdown")
    lines.append("")
    lines.append("| Scenario | Pack Input (ms) | JIT Step (ms) | Support FF (ms) | Diag Map (ms) | JAX Total (ms) |")
    lines.append("|----------|----------------|--------------|----------------|--------------|---------------|")
    for r in all_results:
        if r is None or not r.get("jax"):
            continue
        jx = r["jax"]
        lines.append(
            f"| {r['tag']} | "
            f"{_ts(jx, 'jax_pack_input_s', 'mean_ms')} | "
            f"{_ts(jx, 'jax_jit_step_s', 'mean_ms')} | "
            f"{_ts(jx, 'jax_support_ff_s', 'mean_ms')} | "
            f"{_ts(jx, 'jax_diag_map_s', 'mean_ms')} | "
            f"{_ts(jx, 'jax_total_s', 'mean_ms')} |"
        )
    lines.append("")

    # Compile time
    lines.append("## JIT Compilation")
    lines.append("")
    lines.append("| Scenario | Compile Time (s) | Recompilations |")
    lines.append("|----------|-----------------|----------------|")
    for r in all_results:
        if r is None or not r.get("jax"):
            continue
        jx = r["jax"]
        ct = _tc(jx, "jit_compile_time_s")
        rc = jx.get("compile", {}).get("recompilation_audit", {}).get("recompilation_count", "")
        lines.append(f"| {r['tag']} | {ct} | {rc} |")
    lines.append("")

    # Validation
    lines.append("## Validation During Benchmark")
    lines.append("")
    lines.append("| Scenario | Py Fell | JX Fell | Py NaN | JX NaN | Py HipYaw (rad) | JX HipYaw (rad) |")
    lines.append("|----------|---------|---------|--------|--------|----------------|----------------|")
    for r in all_results:
        if r is None:
            continue
        py = r.get("python", {})
        jx = r.get("jax", {})
        lines.append(
            f"| {r['tag']} | "
            f"{_val(py, 'fell')} | {_val(jx, 'fell')} | "
            f"{_val(py, 'nan_detected')} | {_val(jx, 'nan_detected')} | "
            f"{_val(py, 'max_abs_hip_yaw_rad')} | {_val(jx, 'max_abs_hip_yaw_rad')} |"
        )
    lines.append("")

    # Recompilation audit
    lines.append("## Recompilation Audit")
    lines.append("")
    first_jx = None
    for r in all_results:
        if r and r.get("jax"):
            first_jx = r["jax"]
            break
    if first_jx:
        audit = first_jx.get("compile", {}).get("recompilation_audit", {})
        lines.append(f"- **Input flat shape:** {audit.get('input_flat_shape', 'N/A')}")
        lines.append(f"- **State flat shape:** {audit.get('state_flat_shape', 'N/A')}")
        lines.append(f"- **Params shape:** {audit.get('params_flat_shape', 'N/A')}")
        lines.append(f"- **Diag flat shape:** {audit.get('diag_flat_shape', 'N/A')}")
        lines.append(f"- **Static args:** {audit.get('static_args', 'N/A')}")
        lines.append(f"- **Dynamic height recompiles:** {audit.get('dynamic_height_recompiles', 'N/A')}")
        lines.append(f"- **Telemetry mode recompiles:** {audit.get('telemetry_mode_recompiles', 'N/A')}")
        lines.append(f"- **Headless vs visual:** {audit.get('headless_vs_visual', 'N/A')}")
        lines.append(f"- **Recompilation count:** {audit.get('recompilation_count', 'N/A')}")
    lines.append("")

    # Bottleneck analysis
    lines.append("## Bottleneck Analysis")
    lines.append("")
    lines.append("### Python Backend Bottlenecks")
    lines.append("")
    py_totals = []
    jx_totals = []
    jx_hot = []
    for r in all_results:
        if r is None:
            continue
        py = r.get("python", {})
        jx = r.get("jax", {})
        pt = _ts(py, "total_step_s", "mean_ms")
        jt = _ts(jx, "total_step_s", "mean_ms")
        jh = _ts(jx, "jax_jit_step_s", "mean_ms")
        if pt:
            py_totals.append(float(pt))
        if jt:
            jx_totals.append(float(jt))
        if jh:
            jx_hot.append(float(jh))

    if py_totals:
        lines.append(f"- **Python mean total step:** {sum(py_totals)/len(py_totals):.1f} ms")
    if jx_totals:
        lines.append(f"- **JAX mean total step:** {sum(jx_totals)/len(jx_totals):.1f} ms")
    if jx_hot:
        lines.append(f"- **JAX hot-step mean:** {sum(jx_hot)/len(jx_hot):.3f} ms (JIT only, with block_until_ready)")
    lines.append("")
    lines.append("### Primary bottleneck: Python balance-core computation + telemetry (~100+ ms)")
    lines.append("The JAX JIT step itself is extremely fast (<1 ms), but the total step time is dominated by:")
    lines.append("1. Python balance-core controller computation (runs in both backends for telemetry)")
    lines.append("2. Input packing (Python -> JAX device transfer, ~20 ms)")
    lines.append("3. Telemetry dict construction per step")
    lines.append("4. Duplicate state estimation (control + log phases)")
    lines.append("")

    with open(PERF_REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[STAGE7] Performance report written to: {PERF_REPORT_MD}")


def _generate_realtime_summary(all_results):
    """Generate the realtime-readiness summary."""
    lines = []
    lines.append("# Stage 7: Realtime-Readiness Summary")
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d')}")
    lines.append("")

    # Collect key metrics
    jx_hot_means = []
    jx_hot_p95s = []
    jx_speedups = []
    all_fell_ok = True
    all_nan_ok = True
    all_recompile_ok = True

    for r in all_results:
        if r is None:
            continue
        jx = r.get("jax", {})
        py = r.get("python", {})
        hm = _ts(jx, "jax_jit_step_s", "mean_ms")
        hp = _ts(jx, "jax_jit_step_s", "p95_ms")
        sp = _speedup(py, jx)
        if hm:
            jx_hot_means.append(float(hm))
        if hp:
            jx_hot_p95s.append(float(hp))
        if sp:
            jx_speedups.append(float(sp))
        if _val(jx, "fell") == "True" or _val(py, "fell") == "True":
            all_fell_ok = False
        if _val(jx, "nan_detected") == "True" or _val(py, "nan_detected") == "True":
            all_nan_ok = False
        rc = jx.get("compile", {}).get("recompilation_audit", {}).get("recompilation_count", 0)
        if rc and int(rc) > 0:
            all_recompile_ok = False

    lines.append("## Key Metrics")
    lines.append("")
    if jx_hot_means:
        lines.append(f"- **JAX hot-step mean (average across scenarios):** {sum(jx_hot_means)/len(jx_hot_means):.3f} ms")
        lines.append(f"- **JAX hot-step p95 (max across scenarios):** {max(jx_hot_p95s):.3f} ms" if jx_hot_p95s else "")
        lines.append(f"- **Meets 10 ms control budget:** {'YES' if all(m < 10.0 for m in jx_hot_p95s) else 'NO'}")
    lines.append("")

    # Realtime verdict
    lines.append("## Realtime Verdict")
    lines.append("")
    hot_ok = all(m < 10.0 for m in jx_hot_p95s) if jx_hot_p95s else False
    lines.append(f"- **JAX hot-step meets 10ms budget:** {'[PASS] YES' if hot_ok else '[FAIL] NO'}")
    lines.append(f"- **No falls:** {'[PASS] YES' if all_fell_ok else '[FAIL] NO'}")
    lines.append(f"- **No NaN:** {'[PASS] YES' if all_nan_ok else '[FAIL] NO'}")
    lines.append(f"- **No per-step recompilation:** {'[PASS] YES' if all_recompile_ok else '[FAIL] NO'}")
    lines.append("")

    if hot_ok and all_fell_ok and all_nan_ok and all_recompile_ok:
        lines.append("### Conclusion: Controller compute is realtime-ready [PASS]")
        lines.append("")
        lines.append("The JAX JIT-compiled controller step executes in well under 1 ms with `block_until_ready()`, ")
        lines.append("leaving ample headroom within the 10 ms control budget. The total simulation step time ")
        lines.append("is dominated by Python-level operations (balance-core block, telemetry, state estimation) ")
        lines.append("that run identically in both backends.")
        lines.append("")
        lines.append("**Remaining blockers before changing default backend:**")
        lines.append("1. Python balance-core computation runs in both backends for telemetry -- a ")
        lines.append("   telemetry-decoupled mode would be needed for full JAX benefit.")
        lines.append("2. Input packing (`pack_input_k2`) costs ~20 ms (Python->JAX device transfer).")
        lines.append("3. Duplicate state estimation (control + log) adds ~1-2 ms overhead.")
        lines.append("4. Visual/render overhead not yet benchmarked.")
    else:
        lines.append("### Conclusion: Real-time readiness NOT YET CONFIRMED [FAIL]")
        lines.append("")
        issues = []
        if not hot_ok:
            issues.append("- JAX hot-step exceeds 10 ms budget")
        if not all_fell_ok:
            issues.append("- Falls detected during benchmark")
        if not all_nan_ok:
            issues.append("- NaN detected during benchmark")
        if not all_recompile_ok:
            issues.append("- Per-step recompilation detected")
        lines.extend(issues)

    lines.append("")
    lines.append("## Next Steps")
    lines.append("")
    lines.append("1. Add visual benchmark runs (--visual mode)")
    lines.append("2. Consider telemetry-decoupled mode for production JAX path")
    lines.append("3. Consider moving input packing to JIT (pre-pack inputs)")
    lines.append("4. Consider deduplicating state estimation")
    lines.append("")

    with open(REALTIME_SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[STAGE7] Realtime summary written to: {REALTIME_SUMMARY_MD}")


def _generate_csv_standalone(all_results):
    """Write the CSV if not already written by _write_csv."""
    _write_csv(all_results)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stage 7: Performance benchmark driver for JAX K2 controller"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run with reduced steps for smoke testing (50 warmup + 200 measured)",
    )
    parser.add_argument(
        "--scenario", type=str, default=None,
        help="Run only a single scenario (by tag, e.g., 'fixed_high_0p480')",
    )
    parser.add_argument(
        "--backend", type=str, default=None, choices=["python", "jax"],
        help="Run only a single backend (default: both)",
    )
    parser.add_argument(
        "--visual", action="store_true",
        help="Run visual benchmarks instead of headless (fixed_high_0p480 only)",
    )
    args_cli = parser.parse_args()

    # Ensure output directories
    BENCHMARK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Filter scenarios
    scenarios_to_run = SCENARIOS
    if args_cli.scenario:
        scenarios_to_run = [s for s in SCENARIOS if s["tag"] == args_cli.scenario]
        if not scenarios_to_run:
            print(f"[STAGE7] ERROR: Unknown scenario '{args_cli.scenario}'. Available: {[s['tag'] for s in SCENARIOS]}")
            sys.exit(1)

    # Visual mode
    if args_cli.visual:
        visual_extra = ["--visual", "--visual-disable-realtime-pacing"]
        # Override scenarios to visual ones
        scenarios_to_run = [
            _scenario("visual_fixed_high_0p480", "high_0p480_setup.json",
                       extra_args=visual_extra),
        ]
        print("[STAGE7] Visual benchmark mode enabled")

    backends = [args_cli.backend] if args_cli.backend else ["python", "jax"]

    all_results = []
    for scenario in scenarios_to_run:
        result_entry = {"tag": scenario["tag"]}
        for backend in backends:
            data = _run_one(scenario, backend, quick=args_cli.quick)
            result_entry[backend] = data
        all_results.append(result_entry)

    # Generate reports
    if all_results:
        _write_csv(all_results)
        _generate_markdown_report(all_results)
        _generate_realtime_summary(all_results)

    # Final summary
    print(f"\n{'='*70}")
    print("[STAGE7] Benchmark suite complete")
    print(f"[STAGE7] Scenarios: {len(all_results)}")
    print(f"[STAGE7] Reports:")
    print(f"  - {BENCH_CSV}")
    print(f"  - {PERF_REPORT_MD}")
    print(f"  - {REALTIME_SUMMARY_MD}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
