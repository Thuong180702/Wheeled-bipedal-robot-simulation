"""Phase 0: APCR1ND wheel damping override push failure detailed diagnostic.

Runs push_fwd_90N and push_bwd_90N with both-synced mode, capturing
detailed APCR1ND state from both Python and JAX at each step.

Captures:
A. APCR1ND activation/state: recenter_active, startup guard, drift direction,
   direct/soft entry, hold/release, converging steps, recenter_held,
   safety_pass, step counter, prev_error
B. APCR1ND inputs: sagittal_position_error_m, support_velocity, com_z, pitch, roll,
   contact_valid, tau_position, effective_max_position_tau, boosted_cap
C. Wheel damping override: active, scale, min clamp, tau_wheel_vel left/right,
   raw vs final wheel damping
D. Torque propagation: final tau[4,9], full 10-dim output
"""
import argparse, json, subprocess, sys, time, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM = str(ROOT / "scripts" / "simulate_hierarchical_controller.py")
OUT_DIR = ROOT / "outputs" / "k2_jax_apcr1nd_wd_parity_phase0"


def run_push_trace(scenario_name, push_dir, out_dir, trace_start, trace_end, steps=350):
    """Run push scenario with both-synced backend and capture detailed traces."""
    trace_steps = f"{trace_start}-{trace_end}"

    cmd = [
        sys.executable, SIM,
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", "k2_notch_low_q_v1",
        "--controller-backend", "both-synced",
        "--wbc-quiet",
        "--steps", str(steps),
        "--height-variant-setup", str(ROOT / "outputs" / "physical_target_height_setups_centered" / "fixed_high_0p480.json"),
        "--push-enabled",
        "--push-magnitude-n", "90",
        "--push-duration-steps", "5",
        "--push-interval-steps", "500",
        "--push-start-step", "20",
        "--synced-trace-steps", trace_steps,
    ]

    if push_dir == "forward":
        cmd.append("--sagittal-push-only")

    print(f"Running {scenario_name} with trace steps {trace_start}-{trace_end}...")
    print(f"  Command: {' '.join(cmd[-8:])}", flush=True)

    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0

    with open(out_dir / f"{scenario_name}_stdout.txt", "w", encoding="utf-8") as f:
        f.write(r.stdout)
    with open(out_dir / f"{scenario_name}_stderr.txt", "w", encoding="utf-8") as f:
        f.write(r.stderr)

    # Parse max diff from stdout
    import re
    max_diff = None
    max_step = None
    max_act = None
    m = re.search(r'Worst max_abs_diff:\s*([\d.e+\-]+)', r.stdout)
    if m:
        max_diff = float(m.group(1))
    m = re.search(r'at step\s+(\d+),\s*actuator index\s+(\d+)', r.stdout)
    if m:
        max_step = int(m.group(1))
        max_act = int(m.group(2))

    classification = None
    m = re.search(r'Classification:\s*(\S+)', r.stdout)
    if m:
        classification = m.group(1)

    result = {
        "scenario": scenario_name,
        "max_abs_diff": max_diff,
        "max_step": max_step,
        "max_actuator": max_act,
        "classification": classification,
        "returncode": r.returncode,
        "elapsed_s": elapsed,
    }

    # Extract APCR1ND diagnostic lines
    apcr1nd_lines = []
    sag_diff_lines = []
    for line in r.stdout.split("\n"):
        if "APCR1ND:" in line or "apcr1nd" in line.lower():
            apcr1nd_lines.append(line.strip())
        if "SAG_TERMS:" in line and ("tau_wheel" in line or "tau_position" in line):
            sag_diff_lines.append(line.strip())

    result["apcr1nd_diag_lines"] = len(apcr1nd_lines)
    result["sag_diff_lines"] = len(sag_diff_lines)

    return result


def main():
    parser = argparse.ArgumentParser(description="APCR1ND wheel damping override push diagnostics")
    parser.add_argument("--scenario", choices=["push_fwd_90N", "push_bwd_90N", "both"],
                        default="both")
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    if args.scenario in ("push_fwd_90N", "both"):
        results["push_fwd_90N"] = run_push_trace(
            "push_fwd_90N", "forward", out_dir,
            trace_start=100, trace_end=300, steps=args.steps)

    if args.scenario in ("push_bwd_90N", "both"):
        results["push_bwd_90N"] = run_push_trace(
            "push_bwd_90N", "backward", out_dir,
            trace_start=60, trace_end=300, steps=args.steps)

    # Print summary
    print("\n" + "="*60)
    print("Phase 0: APCR1ND Push Failure Reproduction Summary")
    print("="*60)
    for name, r in results.items():
        status = "PASS" if (r["max_abs_diff"] is not None and r["max_abs_diff"] < 1e-5) else "FAIL"
        print(f"  {name}: {status} max_diff={r['max_abs_diff']:.6e} "
              f"step={r['max_step']} act={r['max_actuator']} "
              f"class={r['classification']} "
              f"apcr1nd_lines={r['apcr1nd_diag_lines']}")

    with open(out_dir / "phase0_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
