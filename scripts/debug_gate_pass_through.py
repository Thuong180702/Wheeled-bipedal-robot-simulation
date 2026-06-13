#!/usr/bin/env python3
"""Gate pass-through debug script.

Verifies that A0 and B1 candidates produce different gate values
at nominal height, proving the z_low/z_high parameters are passed correctly.

Usage:
    python scripts/debug_gate_pass_through.py
"""

import subprocess
import sys
import json
from pathlib import Path

OUTPUT_DIR = Path("outputs/hip_yaw_divergence_fix_authority_eval/gate_pass_through_debug")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_gate_debug(candidate: str, height: str, steps: int = 100) -> dict:
    """Run a single simulation and extract gate telemetry."""
    configs = {
        "A0": {"k": 5.0, "kd": 1.0, "tau_max": 0.5, "z_low": 0.300, "z_high": 0.393},
        "B1": {"k": 5.0, "kd": 1.0, "tau_max": 1.0, "z_low": 0.300, "z_high": 0.500},
    }

    height_setups = {
        "nominal": None,
        "low_0p300": "outputs/physical_target_height_setups/low_0p300_setup.json",
        "high_0p480": "outputs/physical_target_height_setups/high_0p480_setup.json",
    }

    config = configs[candidate]
    setup = height_setups[height]

    # Build command
    cmd = [
        sys.executable, "scripts/simulate_hierarchical_controller.py",
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", "J3",
        "--steps", str(steps),
        "--enable-hip-yaw-divergence-damping",
        "--hip-yaw-divergence-k", str(config["k"]),
        "--hip-yaw-divergence-kd", str(config["kd"]),
        "--hip-yaw-divergence-tau-max", str(config["tau_max"]),
        "--hip-yaw-divergence-z-low", str(config["z_low"]),
        "--hip-yaw-divergence-z-high", str(config["z_high"]),
        "--telemetry-decimation", "1",
    ]

    if setup:
        cmd.extend(["--height-variant-setup", setup])

    print(f"  Running {candidate}_{height} (z_high={config['z_high']})...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return {
                "candidate": candidate,
                "height": height,
                "z_high": config["z_high"],
                "error": f"Sim failed: {result.stderr[-500:]}",
            }

        # Find telemetry file
        telemetry_path = None
        for line in result.stdout.split('\n'):
            if 'Telemetry saved to:' in line:
                telemetry_path = line.split('Telemetry saved to:')[-1].strip()
                break

        if not telemetry_path or not Path(telemetry_path).exists():
            return {
                "candidate": candidate,
                "height": height,
                "z_high": config["z_high"],
                "error": "No telemetry found",
            }

        # Parse telemetry
        import csv
        gate_values = []
        z_low_values = []
        z_high_values = []
        divergences = []

        with open(telemetry_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    gate_values.append(float(row.get('hip_yaw_div_height_gate', 0)))
                    z_low_values.append(float(row.get('hip_yaw_div_z_low', 0)))
                    z_high_values.append(float(row.get('hip_yaw_div_z_high', 0)))
                    divergences.append(float(row.get('hip_yaw_divergence', 0)))
                except (ValueError, TypeError):
                    pass

        if not gate_values:
            return {
                "candidate": candidate,
                "height": height,
                "z_high": config["z_high"],
                "error": "No gate data in telemetry",
            }

        return {
            "candidate": candidate,
            "height": height,
            "z_high": config["z_high"],
            "z_low": config["z_low"],
            "gate_min": min(gate_values),
            "gate_max": max(gate_values),
            "gate_mean": sum(gate_values) / len(gate_values),
            "gate_final": gate_values[-1],
            "div_min": min(divergences) if divergences else 0,
            "div_max": max(divergences) if divergences else 0,
            "div_mean": sum(divergences) / len(divergences) if divergences else 0,
            "telemetry": telemetry_path,
            "n_samples": len(gate_values),
        }

    except subprocess.TimeoutExpired:
        return {"candidate": candidate, "height": height, "z_high": config["z_high"], "error": "Timeout"}
    except Exception as e:
        return {"candidate": candidate, "height": height, "z_high": config["z_high"], "error": str(e)}


def main():
    print("=" * 80)
    print("HY2-DIV Gate Pass-Through Debug")
    print("=" * 80)

    candidates = ["A0", "B1"]
    heights = ["nominal", "low_0p300", "high_0p480"]

    results = []
    for candidate in candidates:
        for height in heights:
            result = run_gate_debug(candidate, height, steps=100)
            results.append(result)
            print(f"    -> gate_mean={result.get('gate_mean', 'N/A'):.3f}, div_mean={result.get('div_mean', 'N/A'):.4f}")

    # Print summary table
    print("\n" + "=" * 80)
    print("GATE PASS-THROUGH SUMMARY")
    print("=" * 80)
    print(f"{'Candidate':<10} {'Height':<12} {'z_high':<8} {'gate_min':<10} {'gate_max':<10} {'gate_mean':<10} {'div_mean':<10}")
    print("-" * 80)

    for r in results:
        if "error" in r:
            print(f"{r['candidate']:<10} {r['height']:<12} {r.get('z_high', 'N/A'):<8} ERROR: {r['error'][:40]}")
        else:
            print(f"{r['candidate']:<10} {r['height']:<12} {r['z_high']:<8.3f} {r['gate_min']:<10.3f} {r['gate_max']:<10.3f} {r['gate_mean']:<10.3f} {r['div_mean']:<10.4f}")

    print("=" * 80)

    # Verify expectations
    print("\nVERIFICATION:")
    a0_nominal = next((r for r in results if r["candidate"] == "A0" and r["height"] == "nominal"), None)
    b1_nominal = next((r for r in results if r["candidate"] == "B1" and r["height"] == "nominal"), None)

    if a0_nominal and b1_nominal and "error" not in a0_nominal and "error" not in b1_nominal:
        gate_diff = abs(a0_nominal["gate_mean"] - b1_nominal["gate_mean"])

        print(f"  A0 nominal gate: {a0_nominal['gate_mean']:.3f} (expected ~0.0)")
        print(f"  B1 nominal gate: {b1_nominal['gate_mean']:.3f} (expected >0.0)")
        print(f"  Gate difference: {gate_diff:.3f}")

        if gate_diff > 0.1:
            print("\n  PASS: Candidates produce DIFFERENT gate values at nominal height.")
            print("  -> z_low/z_high parameters ARE being passed correctly.")
            verdict = "PASS"
        else:
            print("\n  FAIL: Candidates produce IDENTICAL gate values at nominal height.")
            print("  -> z_low/z_high parameters are NOT being passed (bug remains).")
            verdict = "FAIL"
    else:
        print("  Could not verify (simulation errors)")
        verdict = "ERROR"

    # Save results
    output_file = OUTPUT_DIR / "gate_pass_through_debug.json"
    with open(output_file, 'w') as f:
        json.dump({
            "verdict": verdict,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Save markdown report
    md_file = OUTPUT_DIR / "gate_pass_through_debug.md"
    with open(md_file, 'w') as f:
        f.write("# HY2-DIV Gate Pass-Through Debug Report\n\n")
        f.write(f"**Verdict: {verdict}**\n\n")
        f.write("## Gate Values by Candidate/Height\n\n")
        f.write("| Candidate | Height | z_high | gate_min | gate_max | gate_mean | div_mean |\n")
        f.write("|-----------|--------|--------|----------|----------|-----------|----------|\n")
        for r in results:
            if "error" in r:
                f.write(f"| {r['candidate']} | {r['height']} | {r.get('z_high', 'N/A')} | ERROR |\n")
            else:
                f.write(f"| {r['candidate']} | {r['height']} | {r['z_high']:.3f} | {r['gate_min']:.3f} | {r['gate_max']:.3f} | {r['gate_mean']:.3f} | {r['div_mean']:.4f} |\n")

        f.write("\n## Expected Behavior\n\n")
        f.write("- **A0** (z_high=0.393): Gate ≈ 0.0 at nominal (z=0.404) and high_0p480 (z=0.480)\n")
        f.write("- **B1** (z_high=0.500): Gate > 0.0 at nominal and high_0p480\n")
        f.write("- **Both**: Gate ≈ 1.0 at low_0p300 (z=0.300)\n\n")
        f.write(f"## Conclusion\n\n")
        if verdict == "PASS":
            f.write("Candidates produce DIFFERENT gate values - z_low/z_high parameters ARE being passed.\n")
        else:
            f.write("Candidates produce IDENTICAL gate values - z_low/z_high parameters are NOT being passed.\n")
    print(f"Markdown report: {md_file}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
