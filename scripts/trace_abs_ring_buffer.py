"""Phase 0: Trace ABS ring buffer state for failing both-synced scenarios.

Captures detailed Python and JAX ABS state at each step in the critical window.
Runs simulate_hierarchical_controller.py with extended instrumentation.
"""

import argparse, json, subprocess, sys, time, re, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM = str(ROOT / "scripts" / "simulate_hierarchical_controller.py")
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups_centered"

# Critical windows for each scenario
CRITICAL_WINDOWS = {
    "fixed_low_0p330": (220, 280),
    "push_fwd_90N": (240, 300),
    "push_bwd_90N": (250, 310),
}

SCENARIO_CONFIGS = {
    "fixed_low_0p330": {
        "steps": 500,
        "extra": ["--steps", "500", "--height-variant-setup",
                  str(SETUP_DIR / "low_0p330_setup.json")],
    },
    "push_fwd_90N": {
        "steps": 500,
        "extra": ["--steps", "500",
                  "--height-variant-setup", str(SETUP_DIR / "high_0p480_setup.json"),
                  "--push-force-n", "90",
                  "--push-interval-steps", "250",
                  "--push-start-step", "20",
                  "--sagittal-push-only"],
    },
    "push_bwd_90N": {
        "steps": 500,
        "extra": ["--steps", "500",
                  "--height-variant-setup", str(SETUP_DIR / "high_0p480_setup.json"),
                  "--push-force-n", "-90",
                  "--push-interval-steps", "250",
                  "--push-start-step", "20",
                  "--sagittal-push-only"],
    },
}

BASE_CMD = [
    sys.executable, SIM,
    "--controller-mode", "balance-core",
    "--sagittal-controller", "velocity-damped",
    "--vd-sagittal-authority-profile", "k2_notch_low_q_v1",
    "--controller-backend", "both-synced",
    "--wbc-quiet",
    # Enable verbose synced output every step (we need all steps for trace)
    # We'll capture the stdout and parse per-step diagnostics
]


def run_traced_scenario(name, output_dir):
    """Run a scenario and capture full stdout for trace analysis."""
    config = SCENARIO_CONFIGS[name]
    cmd = list(BASE_CMD) + config["extra"]

    print(f"  [{name}] Running simulation...", flush=True)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                          timeout=600)
    except subprocess.TimeoutExpired:
        return None, time.time() - t0
    elapsed = time.time() - t0

    stdout = r.stdout
    stderr = r.stderr

    # Save full output
    out_path = output_dir / f"{name}_stdout.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(stdout)
    if stderr:
        err_path = output_dir / f"{name}_stderr.txt"
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(stderr)

    # Parse the output to extract per-step diagnostics
    trace = parse_synced_trace(stdout, name)

    trace_path = output_dir / f"{name}_trace.json"
    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=2, default=str)

    print(f"  [{name}] Saved {len(trace)} trace entries to {trace_path}")
    print(f"  [{name}] Elapsed: {elapsed:.0f}s")

    return trace, elapsed


def parse_synced_trace(stdout, name):
    """Parse the per-step SYNCED diagnostics from stdout."""
    trace = []
    win_start, win_end = CRITICAL_WINDOWS[name]

    # Pattern: [SYNCED@{step}] max_abs_diff=...
    lines = stdout.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'\[SYNCED@(\d+)\]\s+max_abs_diff=([\d.e+\-]+)\s+first_divergent_idx=(\d+)\s+val=([\d.e+\-]+)', line)
        if m:
            step = int(m.group(1))

            # For push scenarios, capture 20 steps before push trigger too
            expanded = (name.startswith("push_"))
            in_window = win_start <= step <= win_end
            if expanded and step >= win_start - 20 and step <= win_end:
                in_window = True

            entry = {
                "step": step,
                "max_abs_diff": float(m.group(2)),
                "first_divergent_idx": int(m.group(3)),
                "first_divergent_val": float(m.group(4)),
            }

            # Read subsequent lines for this step
            i += 1
            while i < len(lines) and not lines[i].startswith("[SYNCED@") and not lines[i].startswith("[SYNCED"):
                detail_line = lines[i]

                # PY_tau
                m_tau = re.match(r'\s+PY_tau=\[(.+)\]', detail_line)
                if m_tau:
                    entry["py_tau"] = [float(x.strip()) for x in m_tau.group(1).split(",")]

                # JX_tau
                m_tau = re.match(r'\s+JX_tau=\[(.+)\]', detail_line)
                if m_tau:
                    entry["jx_tau"] = [float(x.strip()) for x in m_tau.group(1).split(",")]

                # DIFF
                m_diff = re.match(r'\s+DIFF=\s*\[(.+)\]', detail_line)
                if m_diff:
                    entry["tau_diff"] = [float(x.strip()) for x in m_diff.group(1).split(",")]

                # PY_STATE
                m_abs = re.match(r'\s+PY_STATE: abs_trim=([\d.e+\-]+)\s+abs_hold=(\d+)\s+abs_err_sign=(-?\d+)\s+abs_zc=(\d+)\s+abs_guard=(\d+)', detail_line)
                if m_abs:
                    entry["py_abs_trim_tau"] = float(m_abs.group(1))
                    entry["py_abs_hold_steps"] = int(m_abs.group(2))
                    entry["py_abs_err_sign"] = int(m_abs.group(3))
                    entry["py_abs_zc_count"] = int(m_abs.group(4))
                    entry["py_abs_guard_trigger"] = int(m_abs.group(5))

                # ABS hist len
                m_hist = re.match(r'\s+PY_STATE: abs_hist_len=(\d+)\s+abs_slow_sum=([\d.e+\-]+)', detail_line)
                if m_hist:
                    entry["py_abs_slow_len"] = int(m_hist.group(1))
                    entry["py_abs_slow_sum"] = float(m_hist.group(2))

                # JX diagnostic values from [JX@{step}]
                m_jx = re.match(r'\[JX@(\d+)\]\s+slow_mean=([\d.e+\-]+)\s+fast_mean=([\d.e+\-]+)\s+zc_count=([\d.e+\-]+)\s+abs_trim=([\d.e+\-]+)', detail_line)
                if m_jx:
                    entry["jx_slow_mean"] = float(m_jx.group(2))
                    entry["jx_fast_mean"] = float(m_jx.group(3))
                    entry["jx_zc_count"] = float(m_jx.group(4))
                    entry["jx_abs_trim"] = float(m_jx.group(5))

                # py_abs_trim_trace
                m_pytr = re.match(r'\s+PY_ABS_TRACE:\s+err=([\d.e+\-]+)\s+mean=([\d.e+\-]+)\s+fast=([\d.e+\-]+)\s+target=([\d.e+\-]+)\s+clipped=([\d.e+\-]+)\s+rate=([\d.e+\-]+)\s+trim_to_apply=([\d.e+\-]+)', detail_line)
                if m_pytr:
                    entry["py_abs_trace"] = {
                        "signed_error": float(m_pytr.group(1)),
                        "mean_err": float(m_pytr.group(2)),
                        "fast_mean_err": float(m_pytr.group(3)),
                        "target": float(m_pytr.group(4)),
                        "clipped": float(m_pytr.group(5)),
                        "rate": float(m_pytr.group(6)),
                        "trim_to_apply": float(m_pytr.group(7)),
                    }

                # JX_ABS_DIAG
                m_jxdiag = re.match(r'\s+JX_ABS_DIAG:\s+slow=([\d.e+\-]+)\s+fast=([\d.e+\-]+)\s+zc=([\d.e+\-]+)\s+abs_trim=([\d.e+\-]+)\s+raw_target=([\d.e+\-]+)\s+clipped=([\d.e+\-]+)\s+rate=([\d.e+\-]+)\s+trim_to_apply=([\d.e+\-]+)', detail_line)
                if m_jxdiag:
                    entry["jx_abs_diag"] = {
                        "slow_mean": float(m_jxdiag.group(1)),
                        "fast_mean": float(m_jxdiag.group(2)),
                        "zc_count": float(m_jxdiag.group(3)),
                        "abs_trim": float(m_jxdiag.group(4)),
                        "raw_target": float(m_jxdiag.group(5)),
                        "clipped": float(m_jxdiag.group(6)),
                        "rate": float(m_jxdiag.group(7)),
                        "trim_to_apply": float(m_jxdiag.group(8)),
                    }

                i += 1

            if in_window:
                trace.append(entry)
        else:
            i += 1

    # Sort by step
    trace.sort(key=lambda x: x["step"])
    return trace


def analyze_trace(trace, name):
    """Analyze a trace to find the first ABS scalar divergence."""
    win_start, win_end = CRITICAL_WINDOWS[name]
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {name} (steps {win_start}-{win_end})")
    print(f"{'='*60}")

    # Find step where max_abs_diff exceeds threshold
    first_bad_step = None
    for entry in trace:
        if entry["max_abs_diff"] > 1e-6:
            first_bad_step = entry["step"]
            break

    if first_bad_step is None:
        print("No divergence found in window!")
        return

    print(f"First step exceeding 1e-6: {first_bad_step}")

    # Show divergence growth
    print(f"\nDivergence growth in window:")
    print(f"{'Step':>6} {'max_diff':>14} {'actuator':>10} {'py_trim':>14} {'jx_trim':>14} {'trim_diff':>14}")
    for entry in trace:
        if "py_abs_trim_tau" in entry and "jx_abs_trim" in entry:
            trim_diff = entry["py_abs_trim_tau"] - entry.get("jx_abs_trim", 0)
            print(f"{entry['step']:>6} {entry['max_abs_diff']:>14.6e} {entry['first_divergent_idx']:>10} "
                  f"{entry['py_abs_trim_tau']:>14.6e} {entry.get('jx_abs_trim', 0):>14.6e} {trim_diff:>14.6e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/k2_jax_abs_trim_phase0_trace")
    p.add_argument("--scenarios", nargs="*",
                   default=["fixed_low_0p330", "push_fwd_90N", "push_bwd_90N"])
    p.add_argument("--analyze-only", action="store_true",
                   help="Only analyze existing traces")
    args = p.parse_args()

    od = Path(args.output_dir); od.mkdir(parents=True, exist_ok=True)

    for name in args.scenarios:
        if args.analyze_only:
            trace_path = od / f"{name}_trace.json"
            if trace_path.exists():
                with open(trace_path) as f:
                    trace = json.load(f)
                analyze_trace(trace, name)
            else:
                print(f"  [{name}] No trace found at {trace_path}")
            continue

        trace, elapsed = run_traced_scenario(name, od)
        if trace is not None:
            analyze_trace(trace, name)

    print(f"\nTraces saved to {od}")


if __name__ == "__main__":
    main()
