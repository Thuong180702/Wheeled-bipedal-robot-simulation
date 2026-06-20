"""Early Zero-Crossing Support Recenter — Phase 6: Staged validation.

Runs early_zero_crossing_recenter at high_0p480 for 1200, 2000, 5000 steps.
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
        ezc_episodes = int(float(rows[-1].get('ezc_episode_id', 0)))
        print(f"  EZC active steps: {ezc_active_count}/{len(rows)}")
        print(f"  EZC enter events: {ezc_enter_events}")
        print(f"  EZC zero-cross exits: {ezc_zero_cross_exits}")
        print(f"  EZC safety exits: {ezc_safety_exits}")
        print(f"  EZC episodes: {ezc_episodes}")

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


def print_comparison_table(results, steps_list):
    """Print a comparison table across all steps and profiles."""
    print(f"\n{'='*90}")
    print("STAGED VALIDATION COMPARISON")
    print(f"{'='*90}")

    for steps in steps_list:
        print(f"\n### {steps}-Step Results ###")
        print(f"{'Metric':<25} {'ADAPTIVE':<15} {'OLD_ZC':<15} {'EARLY_ZC':<15}")
        print("-"*70)

        for key in ['min_drift', 'max_drift', 'p2p', 'max_abs', 'mean_drift',
                    'pos_pct', 'neg_pct', 'zero_crossings', 'symmetry_ratio',
                    'in_005_pct', 'in_008_pct', 'out_008_pct', 'out_015_pct']:
            label = key
            vals = []
            for profile in ['adaptive', 'zc', 'ezc']:
                if profile in results and steps in results[profile]:
                    r = results[profile][steps]
                    v = r.get(key, float('nan'))
                    if key == 'symmetry_ratio':
                        vals.append(f"{v:.1f}" if v != float('inf') else "inf")
                    elif 'pct' in key:
                        vals.append(f"{v:.1f}%")
                    elif key in ['min_drift', 'max_drift', 'p2p', 'max_abs', 'mean_drift']:
                        vals.append(f"{v:.4f} m")
                    else:
                        vals.append(str(v))
                else:
                    vals.append("N/A")

            print(f"{label:<25} {vals[0]:<15} {vals[1]:<15} {vals[2]:<15}")


def main():
    STEPS_LIST = [1200, 2000, 5000]
    print(f"\n{'='*70}")
    print(f"EARLY ZERO-CROSSING RECENTER — Phase 6: Staged Validation")
    print(f"Height: high_0p480 | Steps: {STEPS_LIST}")
    print(f"{'='*70}")

    results = {}

    for steps in STEPS_LIST:
        print(f"\n{'='*60}")
        print(f"STEP {steps}")
        print(f"{'='*60}")

        # Run EZC
        csv_ezc = run_simulation(PROFILE_EZC, steps, "high_0p480")
        if csv_ezc:
            if 'ezc' not in results:
                results['ezc'] = {}
            results['ezc'][steps] = analyze_telemetry(csv_ezc, "EARLY_ZC", PROFILE_EZC)

        # Run old ZC for comparison
        csv_zc = run_simulation(PROFILE_ZC, steps, "high_0p480")
        if csv_zc:
            if 'zc' not in results:
                results['zc'] = {}
            results['zc'][steps] = analyze_telemetry(csv_zc, "OLD_ZC", PROFILE_ZC)

        # Run adaptive for baseline
        csv_adp = run_simulation(PROFILE_ADAPTIVE, steps, "high_0p480")
        if csv_adp:
            if 'adaptive' not in results:
                results['adaptive'] = {}
            results['adaptive'][steps] = analyze_telemetry(csv_adp, "ADAPTIVE", PROFILE_ADAPTIVE)

    # Print comparison table
    print_comparison_table(results, STEPS_LIST)

    # Final assessment
    print(f"\n{'='*70}")
    print("FINAL ASSESSMENT")
    print(f"{'='*70}")

    # Check 5000-step results
    if 'ezc' in results and 5000 in results['ezc']:
        r = results['ezc'][5000]
        print(f"\nEZC @ 5000 steps:")
        print(f"  Survived: {'YES' if not r['terminated'] else 'NO (' + r['term_reason'] + ')'}")
        print(f"  max abs drift: {r['max_abs']:.4f} m (target: < 0.22 m)")
        print(f"  P2P: {r['p2p']:.4f} m")
        print(f"  positive %: {r['pos_pct']:.1f}%")
        print(f"  negative %: {r['neg_pct']:.1f}%")
        print(f"  symmetry ratio: {r['symmetry_ratio']:.1f}")
        print(f"  zero crossings: {r['zero_crossings']}")

    print(f"\nPhase 6 Classification:")
    print(f"  PASS_BETTER_THAN_ADAPTIVE / PASS_WITH_MONITORING / NOT_BETTER / FAIL")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()