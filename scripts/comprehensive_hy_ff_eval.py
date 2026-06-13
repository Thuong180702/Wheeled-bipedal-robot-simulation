"""Comprehensive HY-FF evaluation with sign +1.0."""

import subprocess
import json
from pathlib import Path
import pandas as pd


def run_sim(label, variant, setup, enable_ff, k, tau_max, sign):
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
        cmd.extend(["--hip-yaw-support-tau-max", str(tau_max)])
        cmd.extend(["--hip-yaw-support-sign", str(sign)])

    print(f"Running: {label}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[:200]}")
        return None

    # Get telemetry
    sim_dir = Path("outputs/hierarchical_controller_sim")
    telem = sorted(sim_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime)[-1]
    df = pd.read_csv(telem)

    metrics = {
        "label": label,
        "variant": variant,
        "enable_ff": enable_ff,
        "k": k,
        "tau_max": tau_max,
        "sign": sign,
        "hip_yaw_abs_max": float(df["hip_yaw_abs_max"].max()),
        "support_error_max": float(df["support_position_error_m"].max()),
        "pitch_x_max": float(df["pitch_x"].abs().max()),
        "roll_y_max": float(df["roll_y"].abs().max()),
        "contact_valid_pct": 100.0 * (df["contact_force_valid"] == True).sum() / len(df),
        "wbc_applied": bool(df["wbc_applied_any"].any()) if "wbc_applied_any" in df.columns else False,
        "hip_yaw_over_010_pct": 100.0 * (df["hip_yaw_abs_max"] > 0.10).sum() / len(df),
    }

    status = "PASS" if metrics["hip_yaw_abs_max"] <= 0.07 else "FAIL"
    print(f"  hip_yaw={metrics['hip_yaw_abs_max']:.4f} ({status}), support={metrics['support_error_max']:.4f}")

    return metrics


def main():
    output_dir = Path("outputs/hip_yaw_hy_ff_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Test matrix with sign +1.0
    tests = [
        # Baselines
        ("A_baseline_low", "low_0p300", "outputs/physical_target_height_setups/low_0p300_setup.json", False, 0.0, 1.0, 1.0),
        ("A_baseline_high", "high_0p480", "outputs/physical_target_height_setups/high_0p480_setup.json", False, 0.0, 1.0, 1.0),
        ("A_baseline_nominal", "nominal", None, False, 0.0, 1.0, 1.0),
        # k=2.0
        ("B_k2_low", "low_0p300", "outputs/physical_target_height_setups/low_0p300_setup.json", True, 2.0, 1.0, 1.0),
        ("B_k2_high", "high_0p480", "outputs/physical_target_height_setups/high_0p480_setup.json", True, 2.0, 1.0, 1.0),
        ("B_k2_nominal", "nominal", None, True, 2.0, 1.0, 1.0),
        # k=4.0
        ("D_k4_low", "low_0p300", "outputs/physical_target_height_setups/low_0p300_setup.json", True, 4.0, 1.0, 1.0),
        ("D_k4_high", "high_0p480", "outputs/physical_target_height_setups/high_0p480_setup.json", True, 4.0, 1.0, 1.0),
        ("D_k4_nominal", "nominal", None, True, 4.0, 1.0, 1.0),
        # k=6.0
        ("E_k6_low", "low_0p300", "outputs/physical_target_height_setups/low_0p300_setup.json", True, 6.0, 2.0, 1.0),
        ("E_k6_high", "high_0p480", "outputs/physical_target_height_setups/high_0p480_setup.json", True, 6.0, 2.0, 1.0),
        ("E_k6_nominal", "nominal", None, True, 6.0, 2.0, 1.0),
        # k=8.0
        ("F_k8_low", "low_0p300", "outputs/physical_target_height_setups/low_0p300_setup.json", True, 8.0, 2.0, 1.0),
        ("F_k8_high", "high_0p480", "outputs/physical_target_height_setups/high_0p480_setup.json", True, 8.0, 2.0, 1.0),
        ("F_k8_nominal", "nominal", None, True, 8.0, 2.0, 1.0),
    ]

    for test in tests:
        r = run_sim(*test)
        if r:
            results.append(r)

    # Save results
    with open(output_dir / "comprehensive_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Analysis
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS")
    print("="*80)

    for variant in ["low_0p300", "high_0p480", "nominal"]:
        print(f"\n{variant}:")
        variant_results = [r for r in results if r["variant"] == variant]
        for r in variant_results:
            status = "✓" if r["hip_yaw_abs_max"] <= 0.07 else "✗"
            print(f"  {r['label']:20} k={r['k']:3.1f}: hip_yaw={r['hip_yaw_abs_max']:.4f} {status}, "
                  f"support={r['support_error_max']:.4f}")

    # Best result
    passing = [r for r in results if r["hip_yaw_abs_max"] <= 0.07]
    if passing:
        print(f"\n✓ PASSING CANDIDATES: {len(passing)}")
        for p in passing:
            print(f"  {p['label']}: hip_yaw={p['hip_yaw_abs_max']:.4f}")
    else:
        print(f"\n✗ NO CANDIDATES PASS hip_yaw <= 0.07 gate")

        best_low = min([r for r in results if r["variant"] == "low_0p300"], key=lambda x: x["hip_yaw_abs_max"])
        print(f"\nBest at low_0p300:")
        print(f"  {best_low['label']}: k={best_low['k']}, hip_yaw={best_low['hip_yaw_abs_max']:.4f}, "
              f"support={best_low['support_error_max']:.4f}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
