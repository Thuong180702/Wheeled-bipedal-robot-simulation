"""Zero-Crossing Support Recenter — Phase 6: Staged validation.

Runs zero_crossing_support_recenter at high_0p480 for 1200, 2000, 5000 steps.
Compare against adaptive_support_centering_trim baseline.
"""
import csv, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups"
OUT_BASE = ROOT / "outputs" / "step_e_extreme_support_fix_eval" / "active_pitch_crossing"
PROFILE_ZC = "zero_crossing_support_recenter"
PROFILE_ADAPTIVE = "adaptive_support_centering_trim"


def run_simulation(profile, steps, label):
    """Run simulation and return path to telemetry."""
    setup_path = SETUP_DIR / "high_0p480_setup.json"
    out_dir = OUT_BASE / f"zc_{steps}_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not setup_path.exists():
        print(f"ERROR: Setup not found: {setup_path}")
        return None

    args = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", profile,
        "--height-variant-setup", str(setup_path),
        "--steps", str(steps),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(steps),
        "--write-run-summary-sidecar",
    ]

    print(f"\n{'='*60}")
    print(f"Running: {profile} @ high_0p480 for {steps} steps")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")

    result = subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"ERROR: Simulation failed (return code {result.returncode})")
        return None

    # Find the telemetry CSV (timestamped)
    csv_files = sorted(
        (ROOT / "outputs" / "hierarchical_controller_sim").glob("telemetry_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if csv_files:
        latest = csv_files[0]
        # Copy to our output dir
        import shutil
        target = out_dir / f"telemetry_{steps}.csv"
        shutil.copy(latest, target)
        return target

    return None


def analyze_telemetry(csv_path, label):
    """Analyze telemetry and return stats."""
    if not csv_path or not csv_path.exists():
        return None

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"\n{label} Results ({len(rows)} rows):")

    # Check termination
    terminated = rows[-1].get("terminated", "").lower() == "true"
    term_reason = rows[-1].get("termination_reason", "N/A") if terminated else "completed"
    print(f"  Status: {'TERMINATED (' + term_reason + ')' if terminated else 'Completed'}")

    # ZC telemetry
    zc_active = sum(1 for r in rows if r.get("zc_active", "").lower() == "true")
    zc_enter = rows[-1].get("zc_enter_event", "0")
    zc_exit = rows[-1].get("zc_exit_event", "0")
    zc_episode = rows[-1].get("zc_episode_id", "0")
    zc_state = rows[-1].get("zc_state", "N/A")
    print(f"  ZC active steps: {zc_active}/{len(rows)} ({100*zc_active/len(rows):.1f}%)")
    print(f"  ZC enter events: {zc_enter}, exit events: {zc_exit}, episodes: {zc_episode}")
    print(f"  ZC final state: {zc_state}")

    # Drift stats
    drift_col = "active_pitch_crossing_signed_error_m"
    if drift_col in rows[0]:
        drift = [float(r[drift_col]) for r in rows if r.get(drift_col)]
        min_d = min(drift)
        max_d = max(drift)
        mean_d = sum(drift) / len(drift)
        pos = sum(1 for d in drift if d > 0)
        neg = sum(1 for d in drift if d < 0)
        zc = sum(1 for i in range(1, len(drift)) if drift[i-1] >= 0 and drift[i] < 0)
        pos_area = sum(max(d, 0) for d in drift)
        neg_area = sum(abs(min(d, 0)) for d in drift)

        print(f"  Drift: min={min_d:.4f}, max={max_d:.4f}, mean={mean_d:.4f}")
        print(f"  Drift: positive={pos} ({100*pos/len(drift):.1f}%), negative={neg} ({100*neg/len(drift):.1f}%)")
        print(f"  Zero crossings: {zc}")
        print(f"  Area: pos={pos_area:.2f}, neg={neg_area:.2f}, ratio={pos_area/neg_area:.2f}")

        return {
            "rows": len(rows),
            "terminated": terminated,
            "term_reason": term_reason,
            "zc_active": zc_active,
            "zc_enter": int(zc_enter),
            "zc_exit": int(zc_exit),
            "zc_episode": int(zc_episode),
            "zc_state": zc_state,
            "drift_min": min_d,
            "drift_max": max_d,
            "drift_mean": mean_d,
            "drift_pos_pct": 100 * pos / len(drift),
            "drift_neg_pct": 100 * neg / len(drift),
            "drift_zero_crossings": zc,
            "drift_pos_area": pos_area,
            "drift_neg_area": neg_area,
        }

    return None


def main():
    stages = [1200, 2000, 5000]

    results = {}

    for steps in stages:
        # Run ZC
        csv_path = run_simulation(PROFILE_ZC, steps, "high_0p480")
        if csv_path:
            results[f"zc_{steps}"] = analyze_telemetry(csv_path, f"ZC {steps} steps")
        else:
            results[f"zc_{steps}"] = None

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY: Zero-Crossing Support Recenter at high_0p480")
    print("="*80)
    print(f"{'Steps':>8} | {'Status':>10} | {'Drift Mean':>10} | {'Pos%':>6} | {'Neg%':>6} | {'ZC':>4} | {'Enter':>5} | {'Exit':>5}")
    print("-"*80)

    for steps in stages:
        r = results.get(f"zc_{steps}")
        if r:
            status = "TERM" if r["terminated"] else "OK"
            print(f"{steps:>8} | {status:>10} | {r['drift_mean']:>10.4f} | "
                  f"{r['drift_pos_pct']:>5.1f}% | {r['drift_neg_pct']:>5.1f}% | "
                  f"{r['drift_zero_crossings']:>4} | {r['zc_enter']:>5} | {r['zc_exit']:>5}")
        else:
            print(f"{steps:>8} | {'FAILED':>10}")

    # Comparison with adaptive baseline (from Phase 1 audit)
    adaptive_baseline = {
        5000: {"drift_mean": 0.080, "drift_pos_pct": 92.2, "drift_neg_pct": 7.7, "drift_zero_crossings": 26},
    }

    print("\n" + "="*80)
    print("COMPARISON: ZC vs Adaptive (5000 steps)")
    print("="*80)
    if results.get("zc_5000") and adaptive_baseline.get(5000):
        zc = results["zc_5000"]
        ad = adaptive_baseline[5000]
        print(f"{'Metric':>25} | {'Adaptive':>12} | {'ZC':>12} | {'Change':>12}")
        print("-"*80)
        print(f"{'Drift Mean (m)':>25} | {ad['drift_mean']:>12.4f} | {zc['drift_mean']:>12.4f} | "
              f"{(zc['drift_mean']-ad['drift_mean']):>+12.4f}")
        print(f"{'Positive %':>25} | {ad['drift_pos_pct']:>12.1f} | {zc['drift_pos_pct']:>12.1f} | "
              f"{(zc['drift_pos_pct']-ad['drift_pos_pct']):>+12.1f}")
        print(f"{'Negative %':>25} | {ad['drift_neg_pct']:>12.1f} | {zc['drift_neg_pct']:>12.1f} | "
              f"{(zc['drift_neg_pct']-ad['drift_neg_pct']):>+12.1f}")
        print(f"{'Zero Crossings':>25} | {ad['drift_zero_crossings']:>12} | {zc['drift_zero_crossings']:>12} | "
              f"{(zc['drift_zero_crossings']-ad['drift_zero_crossings']):>+12}")

    # Save results
    output_path = OUT_BASE / "zc_staged_validation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()