"""Simple HY-FF evaluation - run baseline and sign determination only."""

import subprocess
import json
from pathlib import Path
import pandas as pd


def run_sim(variant, setup, enable_ff, k, sign):
    """Run single simulation."""
    cmd = [
        "python", "scripts/simulate_hierarchical_controller.py",
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--steps", "1000",
    ]

    if setup:
        cmd.extend(["--height-variant-setup", setup])

    if enable_ff:
        cmd.append("--enable-hip-yaw-support-feedforward")
        cmd.extend(["--hip-yaw-support-k", str(k)])
        cmd.extend(["--hip-yaw-support-tau-max", "1.0"])
        cmd.extend(["--hip-yaw-support-sign", str(sign)])

    print(f"Running: {variant}, ff={enable_ff}, k={k}, sign={sign}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[:200]}")
        return None

    # Get telemetry
    sim_dir = Path("outputs/hierarchical_controller_sim")
    telem = sorted(sim_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime)[-1]
    df = pd.read_csv(telem)

    hip_yaw = float(df["hip_yaw_abs_max"].max())
    support = float(df["support_position_error_m"].max())

    print(f"  hip_yaw={hip_yaw:.4f}, support={support:.4f}")

    return {"variant": variant, "enable_ff": enable_ff, "k": k, "sign": sign,
            "hip_yaw": hip_yaw, "support": support}


def main():
    output_dir = Path("outputs/hip_yaw_hy_ff_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Baseline at low_0p300
    print("\n=== BASELINE low_0p300 ===")
    r = run_sim("low_0p300", "outputs/physical_target_height_setups/low_0p300_setup.json",
                False, 0.0, 1.0)
    if r:
        results.append(r)

    # Sign +1 at low_0p300
    print("\n=== SIGN +1.0 low_0p300 ===")
    r = run_sim("low_0p300", "outputs/physical_target_height_setups/low_0p300_setup.json",
                True, 2.0, 1.0)
    if r:
        results.append(r)

    # Sign -1 at low_0p300
    print("\n=== SIGN -1.0 low_0p300 ===")
    r = run_sim("low_0p300", "outputs/physical_target_height_setups/low_0p300_setup.json",
                True, 2.0, -1.0)
    if r:
        results.append(r)

    # Save results
    with open(output_dir / "simple_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Analysis
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    for r in results:
        status = "PASS" if r["hip_yaw"] <= 0.07 else "FAIL"
        print(f"{r['variant']:15} ff={r['enable_ff']:5} k={r['k']:3.1f} sign={r['sign']:+4.1f}: "
              f"hip_yaw={r['hip_yaw']:.4f} ({status})")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
