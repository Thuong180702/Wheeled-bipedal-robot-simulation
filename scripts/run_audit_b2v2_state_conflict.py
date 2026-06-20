"""Batch-run 2000-step simulations for state/torque conflict audit.

Runs B2v2 + centered posture at all 10 heights, saves telemetry to
outputs/audit_b2v2/. Also runs baseline B + original posture for comparison.
"""

import subprocess, sys, time, os, json
from pathlib import Path

BASE = ["python", "scripts/simulate_hierarchical_controller.py"]
BC = ["--controller-mode", "balance-core", "--sagittal-controller", "velocity-damped"]
STEPS = ["--steps", "2000", "--telemetry-decimation", "1"]

HEIGHTS = {
    "low_0p300": "low_0p300_setup.json",
    "low_0p320": "low_0p320_setup.json",
    "low_0p330": "low_0p330_setup.json",
    "low_0p340": "low_0p340_setup.json",
    "low_0p360": "low_0p360_setup.json",
    "low_0p380": "low_0p380_setup.json",
    "high_0p430": "high_0p430_setup.json",
    "high_0p450": "high_0p450_setup.json",
    "high_0p465": "high_0p465_setup.json",
    "high_0p480": "high_0p480_setup.json",
}

CENTRAL_SETUP_DIR = "outputs/physical_target_height_setups_centered"

def run_sim(profile, setup_name, setup_file, out_dir):
    cmd = BASE + BC + STEPS + [
        "--vd-sagittal-authority-profile", profile,
        "--height-variant-setup", str(Path(CENTRAL_SETUP_DIR) / setup_file),
        "--output-dir", out_dir,
    ]
    print(f"[{setup_name}] {' '.join(cmd[-8:])}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    elapsed = time.time() - t0
    survived = "Completed full simulation" in result.stdout
    terminated = "TERMINATED" in result.stdout
    reason = ""
    if terminated:
        for line in result.stdout.split("\n"):
            if "terminated" in line.lower() or "TERMINATED" in line:
                reason = line.strip()
    print(f"[{setup_name}] {'PASS' if survived else 'FAIL'} ({elapsed:.1f}s)")
    if reason:
        print(f"  -> {reason}")
    return {
        "setup": setup_name,
        "profile": profile,
        "elapsed_s": elapsed,
        "survived": survived,
        "terminated": terminated,
        "reason": reason,
    }

def main():
    os.chdir(Path(__file__).parent.parent)
    os.makedirs("outputs/audit_b2v2", exist_ok=True)
    os.makedirs("outputs/audit_baseline_b", exist_ok=True)

    results = []

    # Run B2v2 + centered posture
    print("=" * 60)
    print("B2v2 + Centered Posture — All 10 heights")
    print("=" * 60)
    for setup_name, setup_file in HEIGHTS.items():
        r = run_sim(
            "calibrated_support_position_outer_loop_pitch_ref_v2",
            setup_name, setup_file,
            "outputs/audit_b2v2",
        )
        results.append(("B2v2", r))

    # Run B baseline + original posture (HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM uses original setups)
    # Actually B = support_position_outer_loop_pitch_ref uses the SAME centered posture if we use centered setups
    # For a proper comparison, let's run B with original posture setups
    print("=" * 60)
    print("Support Position Outer Loop (B) + Original Posture — comparison")
    print("=" * 60)

    # Use original setups for B (for fair comparison to the existing best)
    ORIG_SETUP_DIR = "outputs/physical_target_height_setups"
    for setup_name, setup_file in HEIGHTS.items():
        r = run_sim(
            "support_position_outer_loop_pitch_ref",
            setup_name, setup_file,
            "outputs/audit_baseline_b",
        )
        results.append(("B_original", r))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = all(r["survived"] for _, r in results)
    for profile, r in results:
        status = "✓" if r["survived"] else "✗"
        print(f"  {status} {profile} {r['setup']}: {r['elapsed_s']:.1f}s")
    print(f"\nAll passed: {all_pass}")

    # Save summary
    summary = {f"{p}_{r['setup']}": r for p, r in results}
    with open("outputs/audit_batch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary saved to outputs/audit_batch_summary.json")

if __name__ == "__main__":
    main()
