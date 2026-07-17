"""Early Zero-Crossing Support Recenter — Phase 7: Height Ladder Sanity.

Runs early_zero_crossing_recenter at multiple heights for 2000 steps.
Compare against adaptive_support_centering_trim and zero_crossing_support_recenter baselines.
"""
import csv, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups"
OUT_BASE = ROOT / "outputs" / "step_e_extreme_support_fix_eval" / "active_pitch_crossing"
PROFILE_EZC = "early_zero_crossing_recenter"
PROFILE_ZC = "zero_crossing_support_recenter"
PROFILE_ADAPTIVE = "adaptive_support_centering_trim"


def run_simulation(profile, steps, height_label, setup_name):
    """Run simulation and return path to telemetry."""
    setup_path = SETUP_DIR / setup_name
    out_dir = OUT_BASE / f"ezc_{steps}_{height_label}"
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
    print(f"Running: {profile} @ {height_label} for {steps} steps")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")

    result = subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"ERROR: Simulation failed (return code {result.returncode})")
        if result.stdout:
            print(result.stdout[-500:])
        if result.stderr:
            print(result.stderr[-500:])
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


def analyze_height(csv_path, height_label, profile):
    """Analyze telemetry for one height."""
    if not csv_path or not csv_path.exists():
        return None

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    terminated = rows[-1].get("terminated", "").lower() == "true"
    term_reason = rows[-1].get("termination_reason", "N/A") if terminated else "completed"

    # Drift column
    drift_cols = ['active_pitch_crossing_signed_error_m', 'sagittal_position_error_m', 'support_position_error_m']
    drift = None
    for col in drift_cols:
        if col in rows[0] and rows[0][col]:
            try:
                drift = [float(r[col]) for r in rows if r[col]]
                break
            except:
                continue

    if drift is None:
        return {'terminated': terminated, 'term_reason': term_reason, 'error': 'no drift column'}

    return {
        'steps': len(rows),
        'terminated': terminated,
        'term_reason': term_reason,
        'min_drift': min(drift),
        'max_drift': max(drift),
        'p2p': max(drift) - min(drift),
        'max_abs': max(abs(d) for d in drift),
        'pos_pct': sum(1 for d in drift if d > 0) / len(drift) * 100,
        'neg_pct': sum(1 for d in drift if d < 0) / len(drift) * 100,
        'out_015_pct': sum(1 for d in drift if d < -0.15 or d > 0.15) / len(drift) * 100,
    }


def main():
    STEPS = 2000
    HEIGHTS = [
        ('low_0p300', 'low_0p300_setup.json'),
        ('low_0p320', 'low_0p320_setup.json'),
        ('low_0p340', 'low_0p340_setup.json'),
        ('low_0p360', 'low_0p360_setup.json'),
        ('low_0p380', 'low_0p380_setup.json'),
        ('extreme_height', 'extreme_height_setup.json'),
        ('high_0p430', 'high_0p430_setup.json'),
        ('high_0p450', 'high_0p450_setup.json'),
        ('high_0p465', 'high_0p465_setup.json'),
        ('high_0p480', 'high_0p480_setup.json'),
    ]

    print(f"\n{'='*70}")
    print(f"EARLY ZERO-CROSSING RECENTER — Phase 7: Height Ladder Sanity")
    print(f"Steps per height: {STEPS}")
    print(f"{'='*70}")

    results = {}

    for height_label, setup_name in HEIGHTS:
        print(f"\n{'='*50}")
        print(f"HEIGHT: {height_label}")
        print(f"{'='*50}")

        results[height_label] = {}

        # Run EZC
        csv_ezc = run_simulation(PROFILE_EZC, STEPS, height_label, setup_name)
        if csv_ezc:
            results[height_label]['ezc'] = analyze_height(csv_ezc, height_label, PROFILE_EZC)

        # Run old ZC
        csv_zc = run_simulation(PROFILE_ZC, STEPS, height_label, setup_name)
        if csv_zc:
            results[height_label]['zc'] = analyze_height(csv_zc, height_label, PROFILE_ZC)

        # Run adaptive
        csv_adp = run_simulation(PROFILE_ADAPTIVE, STEPS, height_label, setup_name)
        if csv_adp:
            results[height_label]['adaptive'] = analyze_height(csv_adp, height_label, PROFILE_ADAPTIVE)

    # Summary table
    print(f"\n{'='*90}")
    print("HEIGHT LADDER SUMMARY")
    print(f"{'='*90}")
    print(f"{'Height':<15} {'Profile':<10} {'Survived':<10} {'min_drift':<12} {'max_drift':<12} {'P2P':<10} {'max_abs':<10} {'pos%':<8} {'neg%':<8} {'out_15%':<8}")
    print("-"*115)

    for height_label, _ in HEIGHTS:
        if height_label not in results:
            continue
        for profile in ['adaptive', 'zc', 'ezc']:
            if profile not in results[height_label]:
                continue
            r = results[height_label][profile]
            if 'error' in r:
                print(f"{height_label:<15} {profile:<10} {'ERROR':<10}")
                continue
            survived = 'YES' if not r['terminated'] else f"NO ({r['term_reason'][:8]})"
            print(f"{height_label:<15} {profile:<10} {survived:<10} {r['min_drift']:<12.4f} {r['max_drift']:<12.4f} {r['p2p']:<10.4f} {r['max_abs']:<10.4f} {r['pos_pct']:<8.1f} {r['neg_pct']:<8.1f} {r['out_015_pct']:<8.1f}")

    # Check for failures
    print(f"\n{'='*70}")
    print("FAILURE ANALYSIS")
    print(f"{'='*70}")
    failures = []
    for height_label, _ in HEIGHTS:
        if height_label not in results:
            continue
        for profile in ['ezc', 'zc', 'adaptive']:
            if profile in results[height_label]:
                r = results[height_label][profile]
                if r.get('terminated', False):
                    failures.append((height_label, profile, r['term_reason']))

    if failures:
        print("Failures detected:")
        for h, p, reason in failures:
            print(f"  {h} ({p}): {reason}")
    else:
        print("No failures detected! All heights survived.")

    print(f"\nPhase 7 Classification:")
    print(f"  PASS / PASS_WITH_MONITORING / FAIL")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()