"""Early Zero-Crossing Support Recenter — Phase 5: 500-step diagnostic.

Runs early_zero_crossing_recenter at high_0p480 for 500 steps.
Compare against adaptive_support_centering_trim and zero_crossing_support_recenter baselines.
"""
import csv, json, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups"
OUT_BASE = ROOT / "outputs" / "step_e_extreme_support_fix_eval" / "active_pitch_crossing"
PROFILE_EZC = "early_zero_crossing_recenter"
PROFILE_ZC = "zero_crossing_support_recenter"
PROFILE_ADAPTIVE = "adaptive_support_centering_trim"


def run_simulation(profile, steps, label):
    """Run simulation and return path to telemetry."""
    setup_path = SETUP_DIR / "high_0p480_setup.json"
    out_dir = OUT_BASE / f"ezc_{steps}_{label}"
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
        if result.stdout:
            print(result.stdout[-1000:])
        if result.stderr:
            print(result.stderr[-1000:])
        return None

    # Find the telemetry CSV (timestamped)
    csv_files = sorted(
        (ROOT / "outputs" / "hierarchical_controller_sim").glob("telemetry_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if csv_files:
        latest = csv_files[0]
        target = out_dir / f"telemetry_{steps}.csv"
        shutil.copy(latest, target)
        return target

    return None


def analyze_telemetry(csv_path, label, profile):
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
    print(f"  Termination: {term_reason}")

    # Drift column priority
    drift_cols = [
        'active_pitch_crossing_signed_error_m',
        'sagittal_position_error_m',
        'support_position_error_m',
        'hip_yaw_comp_support_error_m'
    ]

    drift = None
    drift_col = None
    for col in drift_cols:
        if col in rows[0] and rows[0][col]:
            try:
                drift = [float(r[col]) for r in rows if r[col]]
                drift_col = col
                break
            except (ValueError, TypeError):
                continue

    if drift is None:
        print("  ERROR: No drift column found")
        return None

    print(f"  Drift column: {drift_col}")
    print(f"  min drift: {min(drift):.6f} m")
    print(f"  max drift: {max(drift):.6f} m")
    print(f"  P2P: {max(drift) - min(drift):.6f} m")
    print(f"  max abs: {max(abs(d) for d in drift):.6f} m")
    print(f"  mean: {sum(drift)/len(drift):.6f} m")

    pos_count = sum(1 for d in drift if d > 0)
    neg_count = sum(1 for d in drift if d < 0)
    print(f"  positive %: {pos_count/len(drift)*100:.1f}%")
    print(f"  negative %: {neg_count/len(drift)*100:.1f}%")

    # Zero crossings
    signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in drift]
    crossings = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i-1])
    print(f"  zero crossings: {crossings}")

    # Positive/negative areas
    pos_area = sum(d for d in drift if d > 0)
    neg_area = abs(sum(d for d in drift if d < 0))
    print(f"  positive area: {pos_area:.4f}")
    print(f"  negative area: {neg_area:.4f}")
    if neg_area > 0:
        print(f"  symmetry ratio: {pos_area/neg_area:.3f}")

    # Band analysis
    in_003 = sum(1 for d in drift if -0.03 <= d <= 0.03)
    in_005 = sum(1 for d in drift if -0.05 <= d <= 0.05)
    in_008 = sum(1 for d in drift if -0.08 <= d <= 0.08)
    out_008 = sum(1 for d in drift if d < -0.08 or d > 0.08)
    out_010 = sum(1 for d in drift if d < -0.10 or d > 0.10)
    out_015 = sum(1 for d in drift if d < -0.15 or d > 0.15)
    print(f"  time inside ±0.03: {in_003/len(drift)*100:.1f}%")
    print(f"  time inside ±0.05: {in_005/len(drift)*100:.1f}%")
    print(f"  time inside ±0.08: {in_008/len(drift)*100:.1f}%")
    print(f"  time outside ±0.08: {out_008/len(drift)*100:.1f}%")
    print(f"  time outside ±0.10: {out_010/len(drift)*100:.1f}%")
    print(f"  time outside ±0.15: {out_015/len(drift)*100:.1f}%")

    # EZC-specific telemetry
    if 'ezc_state_id' in rows[0]:
        ezc_active_count = sum(1 for r in rows if r.get('ezc_active', '').lower() == 'true')
        ezc_enter_events = int(float(rows[-1].get('ezc_enter_event', 0)))
        ezc_zero_cross_exits = int(float(rows[-1].get('ezc_zero_cross_exit_event', 0)))
        ezc_safety_exits = int(float(rows[-1].get('ezc_safety_exit_event', 0)))
        print(f"  EZC active steps: {ezc_active_count}/{len(rows)}")
        print(f"  EZC enter events: {ezc_enter_events}")
        print(f"  EZC zero-cross exits: {ezc_zero_cross_exits}")
        print(f"  EZC safety exits: {ezc_safety_exits}")

    return {
        'steps': len(rows),
        'terminated': terminated,
        'term_reason': term_reason,
        'min_drift': min(drift),
        'max_drift': max(drift),
        'p2p': max(drift) - min(drift),
        'max_abs': max(abs(d) for d in drift),
        'mean_drift': sum(drift)/len(drift),
        'pos_pct': pos_count/len(drift)*100,
        'neg_pct': neg_count/len(drift)*100,
        'zero_crossings': crossings,
        'pos_area': pos_area,
        'neg_area': neg_area,
        'symmetry_ratio': pos_area/neg_area if neg_area > 0 else float('inf'),
        'in_003_pct': in_003/len(drift)*100,
        'in_005_pct': in_005/len(drift)*100,
        'in_008_pct': in_008/len(drift)*100,
        'out_008_pct': out_008/len(drift)*100,
        'out_010_pct': out_010/len(drift)*100,
        'out_015_pct': out_015/len(drift)*100,
    }


def main():
    STEPS = 500
    print(f"\n{'='*70}")
    print(f"EARLY ZERO-CROSSING RECENTER — Phase 5: 500-step Diagnostic")
    print(f"Height: high_0p480 | Steps: {STEPS}")
    print(f"{'='*70}")

    results = {}

    # Run EZC
    csv_ezc = run_simulation(PROFILE_EZC, STEPS, "high_0p480")
    if csv_ezc:
        results['ezc'] = analyze_telemetry(csv_ezc, "EARLY_ZC", PROFILE_EZC)

    # Run old ZC for comparison
    csv_zc = run_simulation(PROFILE_ZC, STEPS, "high_0p480")
    if csv_zc:
        results['zc'] = analyze_telemetry(csv_zc, "OLD_ZC", PROFILE_ZC)

    # Run adaptive for baseline
    csv_adp = run_simulation(PROFILE_ADAPTIVE, STEPS, "high_0p480")
    if csv_adp:
        results['adaptive'] = analyze_telemetry(csv_adp, "ADAPTIVE", PROFILE_ADAPTIVE)

    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'ADAPTIVE':<12} {'OLD_ZC':<12} {'EARLY_ZC':<12}")
    print("-"*65)
    if 'adaptive' in results:
        r = results['adaptive']
        print(f"{'min drift':<25} {r['min_drift']:>10.4f}")
    if 'zc' in results:
        r = results['zc']
        print(f"{'min drift':<25} {'':<12} {r['min_drift']:>10.4f}")
    if 'ezc' in results:
        r = results['ezc']
        print(f"{'min drift':<25} {'':<12} {'':<12} {r['min_drift']:>10.4f}")

    if 'adaptive' in results:
        r = results['adaptive']
        print(f"{'max drift':<25} {r['max_drift']:>10.4f}")
    if 'zc' in results:
        r = results['zc']
        print(f"{'max drift':<25} {'':<12} {r['max_drift']:>10.4f}")
    if 'ezc' in results:
        r = results['ezc']
        print(f"{'max drift':<25} {'':<12} {'':<12} {r['max_drift']:>10.4f}")

    if 'adaptive' in results:
        r = results['adaptive']
        print(f"{'P2P':<25} {r['p2p']:>10.4f}")
    if 'zc' in results:
        r = results['zc']
        print(f"{'P2P':<25} {'':<12} {r['p2p']:>10.4f}")
    if 'ezc' in results:
        r = results['ezc']
        print(f"{'P2P':<25} {'':<12} {'':<12} {r['p2p']:>10.4f}")

    if 'adaptive' in results:
        r = results['adaptive']
        print(f"{'positive %':<25} {r['pos_pct']:>10.1f}%")
    if 'zc' in results:
        r = results['zc']
        print(f"{'positive %':<25} {'':<12} {r['pos_pct']:>10.1f}%")
    if 'ezc' in results:
        r = results['ezc']
        print(f"{'positive %':<25} {'':<12} {'':<12} {r['pos_pct']:>10.1f}%")

    if 'adaptive' in results:
        r = results['adaptive']
        print(f"{'negative %':<25} {r['neg_pct']:>10.1f}%")
    if 'zc' in results:
        r = results['zc']
        print(f"{'negative %':<25} {'':<12} {r['neg_pct']:>10.1f}%")
    if 'ezc' in results:
        r = results['ezc']
        print(f"{'negative %':<25} {'':<12} {'':<12} {r['neg_pct']:>10.1f}%")

    if 'adaptive' in results:
        r = results['adaptive']
        print(f"{'zero crossings':<25} {r['zero_crossings']:>10}")
    if 'zc' in results:
        r = results['zc']
        print(f"{'zero crossings':<25} {'':<12} {r['zero_crossings']:>10}")
    if 'ezc' in results:
        r = results['ezc']
        print(f"{'zero crossings':<25} {'':<12} {'':<12} {r['zero_crossings']:>10}")

    if 'adaptive' in results:
        r = results['adaptive']
        print(f"{'symmetry ratio':<25} {r['symmetry_ratio']:>10.1f}")
    if 'zc' in results:
        r = results['zc']
        print(f"{'symmetry ratio':<25} {'':<12} {r['symmetry_ratio']:>10.1f}")
    if 'ezc' in results:
        r = results['ezc']
        print(f"{'symmetry ratio':<25} {'':<12} {'':<12} {r['symmetry_ratio']:>10.1f}")

    if 'adaptive' in results:
        r = results['adaptive']
        print(f"{'outside ±0.08':<25} {r['out_008_pct']:>10.1f}%")
    if 'zc' in results:
        r = results['zc']
        print(f"{'outside ±0.08':<25} {'':<12} {r['out_008_pct']:>10.1f}%")
    if 'ezc' in results:
        r = results['ezc']
        print(f"{'outside ±0.08':<25} {'':<12} {'':<12} {r['out_008_pct']:>10.1f}%")

    if 'adaptive' in results:
        r = results['adaptive']
        print(f"{'outside ±0.15':<25} {r['out_015_pct']:>10.1f}%")
    if 'zc' in results:
        r = results['zc']
        print(f"{'outside ±0.15':<25} {'':<12} {r['out_015_pct']:>10.1f}%")
    if 'ezc' in results:
        r = results['ezc']
        print(f"{'outside ±0.15':<25} {'':<12} {'':<12} {r['out_015_pct']:>10.1f}%")

    print(f"\n{'='*70}")
    print("Phase 5 Classification: PASS / PASS_WITH_MONITORING / FAIL")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()