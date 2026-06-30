"""Phase 0: ABS trim state/timing failure trace — ramp_up steps 0-200.

Captures Python and JAX ABS trim state at every step to identify
the first divergent scalar.
"""
import json, sys, time, math, os, re, subprocess
from pathlib import Path
from collections import OrderedDict


def main():
    print("=" * 80)
    print("PHASE 0: ABS TRIM STATE/TIMING FAILURE TRACE")
    print("=" * 80)

    # Import the simulate_hierarchical_controller module and use its setup
    import scripts.simulate_hierarchical_controller as shc_mod

    # Parse args for setup
    sys.argv = [
        "simulate_hierarchical_controller.py",
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", "k2_notch_low_q_v1",
        "--controller-backend", "both-synced",
        "--wbc-quiet",
        "--enable-mode-hip-yaw-divergence",
        "--mode-hip-yaw-div-kp", "10.0",
        "--mode-hip-yaw-div-kd", "0.50",
        "--mode-hip-yaw-div-max-torque", "7.5",
        "--mode-hip-yaw-div-soft-limit-rad", "0.30",
        "--mode-hip-yaw-div-soft-gain", "0.80",
        "--mode-hip-yaw-div-ref-source", "target",
        "--steps", "200",
        "--height-variant-setup", str(ROOT / "outputs" / "physical_target_height_setups_centered" / "low_0p330_setup.json"),
        "--dynamic-height-trajectory", str(ROOT / "outputs" / "k2_jax_abs_trim_phase6" / "trajectories" / "ramp_up_0p330_to_0p480.json"),
    ]

    # We need to run the actual simulation. Instead of replicating the entire setup,
    # let's use subprocess to run the simulate script and capture verbose output.
    print("\nRunning both-synced simulation for ramp_up (200 steps)...")
    print("This will capture detailed ABS trim diagnostics.")

    import subprocess
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", "k2_notch_low_q_v1",
        "--controller-backend", "both-synced",
        "--wbc-quiet",
        "--enable-mode-hip-yaw-divergence",
        "--mode-hip-yaw-div-kp", "10.0",
        "--mode-hip-yaw-div-kd", "0.50",
        "--mode-hip-yaw-div-max-torque", "7.5",
        "--mode-hip-yaw-div-soft-limit-rad", "0.30",
        "--mode-hip-yaw-div-soft-gain", "0.80",
        "--mode-hip-yaw-div-ref-source", "target",
        "--steps", "200",
        "--height-variant-setup", str(ROOT / "outputs" / "physical_target_height_setups_centered" / "low_0p330_setup.json"),
        "--dynamic-height-trajectory", str(ROOT / "outputs" / "k2_jax_abs_trim_phase6" / "trajectories" / "ramp_up_0p330_to_0p480.json"),
    ]

    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=600)

    stdout = result.stdout
    stderr = result.stderr

    # Save full output
    out_dir = ROOT / "outputs" / "k2_jax_abs_trim_phase0"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "ramp_up_full_stdout.txt", "w", encoding="utf-8") as f:
        f.write(stdout)
    with open(out_dir / "ramp_up_full_stderr.txt", "w", encoding="utf-8") as f:
        f.write(stderr)

    # Parse SYNCED diagnostics from stdout
    synced_lines = []
    in_synced_block = False
    synced_block_lines = []
    for line in stdout.splitlines():
        if line.startswith("[SYNCED@"):
            if synced_block_lines:
                synced_lines.append(synced_block_lines)
            synced_block_lines = [line]
        elif synced_block_lines:
            synced_block_lines.append(line)
            if len(synced_block_lines) > 30:  # max lines per synced block
                synced_lines.append(synced_block_lines)
                synced_block_lines = []
    if synced_block_lines:
        synced_lines.append(synced_block_lines)

    # Also capture TAU_COMP blocks (detailed when diff > 1e-7)
    tau_comp_lines = []
    for line in stdout.splitlines():
        if "TAU_COMP@" in line:
            tau_comp_lines.append(line)
        elif any(x in line for x in ["ABS:", "py_trim=", "jx_trim="]):
            tau_comp_lines.append(line)

    # Parse BOTH synced blocks for early steps (step < 20)
    both_lines = []
    for line in stdout.splitlines():
        if "[BOTH@" in line or "[SYNCED@" in line:
            both_lines.append(line)

    # Find divergence onset
    print(f"\nCaptured {len(synced_lines)} SYNCED diagnostic blocks")
    print(f"Captured {len(tau_comp_lines)} TAU_COMP detail lines")

    # Extract key metrics from synced blocks
    steps_with_abs_info = []
    for block in synced_lines:
        for line in block:
            if line.startswith("[SYNCED@"):
                import re
                m = re.search(r'\[SYNCED@(\d+)\]', line)
                if m:
                    step = int(m.group(1))
                m = re.search(r'max_abs_diff=([\d.e+\-]+)', line)
                if m:
                    max_diff = float(m.group(1))
                m = re.search(r'first_divergent_idx=(\d+)', line)
                if m:
                    first_div = int(m.group(1))
                m = re.search(r'val=([\d.e+\-]+)', line)
                if m:
                    first_val = float(m.group(1))
                steps_with_abs_info.append({
                    "step": step,
                    "max_diff": max_diff,
                    "first_divergent": first_div,
                    "first_val": first_val,
                })

    # Print divergence timeline
    print(f"\n{'='*80}")
    print("DIVERGENCE TIMELINE (steps with SYNCED diagnostics)")
    print(f"{'='*80}")
    print(f"{'Step':>6s} {'MaxDiff':>14s} {'FirstAct':>8s} {'FirstVal':>14s}")
    print("-" * 50)
    for entry in steps_with_abs_info:
        md_val = entry.get('max_diff', 0)
        fd_val = entry.get('first_divergent', 0)
        fv_val = entry.get('first_val', 0)
        if md_val > 1e-10:
            print(f"{entry['step']:6d} {md_val:14.6e} {fd_val:8d} {fv_val:14.6e}")

    # Search for ABS trim values in output
    print(f"\n{'='*80}")
    print("ABS TRIM VALUES (from SYNCED diagnostics)")
    print(f"{'='*80}")
    for line in stdout.splitlines():
        if "abs_trim=" in line or "jx_trim=" in line or "ABS:" in line:
            # Extract step if nearby
            print(f"  {line.strip()}")

    # Check for worst-case summary
    for line in stdout.splitlines():
        if "Worst max_abs_diff" in line or "Classification" in line:
            print(f"  {line.strip()}")

    print(f"\nFull output saved to: {out_dir}")
    print(f"Done. Return code: {result.returncode}")


if __name__ == "__main__":
    main()
