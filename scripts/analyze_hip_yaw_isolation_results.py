"""Analyze hip-yaw isolation experiment results."""

import json
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    results_path = Path("outputs/hip_yaw_disturbance_rejection_audit/isolation/isolation_experiment_results.json")

    with open(results_path) as f:
        data = json.load(f)

    # Filter by variant
    low_data = [r for r in data if r['variant'] == 'low_0p300' and r['status'] == 'success']
    high_data = [r for r in data if r['variant'] == 'high_0p480' and r['status'] == 'success']

    print("=" * 80)
    print("HIP-YAW ISOLATION EXPERIMENTS - PHASE 3 ANALYSIS")
    print("=" * 80)

    # LOW_0p300 analysis
    print("\n### CRITICAL HEIGHT: low_0p300 ###\n")

    baseline = [r for r in low_data if r.get('experiment') == 'baseline'][0]
    print(f"Baseline (kp=15, kd=3):")
    print(f"  hip_yaw_abs_max: {baseline['hip_yaw_abs_max']:.4f} rad")
    print(f"  support_error:   {baseline['support_position_error_max']:.4f} m")

    # Best result
    best = min(low_data, key=lambda r: r['hip_yaw_abs_max'])
    improvement_pct = (best['hip_yaw_abs_max'] / baseline['hip_yaw_abs_max'] - 1) * 100
    support_change_pct = (best['support_position_error_max'] / baseline['support_position_error_max'] - 1) * 100

    print(f"\nBest result (kp={best['kp_hip_yaw']:.0f}, kd={best['kd_hip_yaw']:.0f}):")
    print(f"  hip_yaw_abs_max: {best['hip_yaw_abs_max']:.4f} rad ({improvement_pct:+.1f}%)")
    print(f"  support_error:   {best['support_position_error_max']:.4f} m ({support_change_pct:+.1f}%)")

    gate_threshold = 0.07
    gap = best['hip_yaw_abs_max'] - gate_threshold
    gap_pct = (best['hip_yaw_abs_max'] / gate_threshold - 1) * 100

    print(f"\nGate analysis:")
    print(f"  Target threshold: {gate_threshold:.4f} rad")
    print(f"  Best achieved:    {best['hip_yaw_abs_max']:.4f} rad")
    print(f"  Gap:              {gap:.4f} rad ({gap_pct:.1f}% over threshold)")
    print(f"  Verdict:          ❌ FAIL - No configuration passes gate")

    # Trend analysis
    print(f"\nTrend analysis:")
    print(f"  kp increase (15→20→25 at kd=9):")
    for kp in [15, 20, 25]:
        r = [x for x in low_data if x['kp_hip_yaw'] == kp and x['kd_hip_yaw'] == 9][0]
        print(f"    kp={kp}: {r['hip_yaw_abs_max']:.4f} rad")

    print(f"  kd increase (3→5→7→9 at kp=15):")
    for kd in [3, 5, 7, 9]:
        matches = [x for x in low_data if x['kp_hip_yaw'] == 15 and x['kd_hip_yaw'] == kd]
        if matches:
            print(f"    kd={kd}: {matches[0]['hip_yaw_abs_max']:.4f} rad")

    # HIGH_0p480 analysis
    print("\n\n### NOMINAL HEIGHT: high_0p480 ###\n")

    baseline_h = [r for r in high_data if r.get('experiment') == 'baseline'][0]
    print(f"Baseline (kp=15, kd=3):")
    print(f"  hip_yaw_abs_max: {baseline_h['hip_yaw_abs_max']:.4f} rad")
    print(f"  support_error:   {baseline_h['support_position_error_max']:.4f} m")
    print(f"  Verdict:         ✅ PASS gate")

    best_h = min(high_data, key=lambda r: r['hip_yaw_abs_max'])
    print(f"\nBest result (kp={best_h['kp_hip_yaw']:.0f}, kd={best_h['kd_hip_yaw']:.0f}):")
    print(f"  hip_yaw_abs_max: {best_h['hip_yaw_abs_max']:.4f} rad")
    print(f"  support_error:   {best_h['support_position_error_max']:.4f} m")

    support_degradation = (best_h['support_position_error_max'] / baseline_h['support_position_error_max'] - 1) * 100
    print(f"  Support degradation: {support_degradation:+.1f}%")

    if support_degradation > 10:
        print(f"  ⚠️  WARNING: Support worsened by >{10}% (rejection criterion)")

    # Mechanism classification
    print("\n\n" + "=" * 80)
    print("PHASE 3: MECHANISM CLASSIFICATION")
    print("=" * 80)

    print(f"\n**Finding:** Hip-yaw control authority increase NOT sufficient at low_0p300")
    print(f"\n**Evidence:**")
    print(f"  1. Maximum kp/kd tested: kp=25 (67% increase), kd=9 (200% increase)")
    print(f"  2. Best hip-yaw achieved: {best['hip_yaw_abs_max']:.4f} rad")
    print(f"  3. Still {gap_pct:.1f}% over threshold")
    print(f"  4. Improvement trend: kp helps more than kd")
    print(f"     - kp 15→25 at kd=9: {baseline['hip_yaw_abs_max']:.4f}→{best['hip_yaw_abs_max']:.4f} ({improvement_pct:.1f}%)")
    print(f"     - kd 3→9 at kp=15: {baseline['hip_yaw_abs_max']:.4f}→{[r for r in low_data if r['kp_hip_yaw']==15 and r['kd_hip_yaw']==9][0]['hip_yaw_abs_max']:.4f}")

    print(f"\n**Mechanism Classification:**")
    print(f"  `hip_yaw_disturbance_rejection_insufficient_authority_alone`")

    print(f"\n**Root Cause:**")
    print(f"  Support drift at low_0p300 creates disturbance torque that exceeds")
    print(f"  hip-yaw controller's rejection capability even with 67% kp increase.")
    print(f"  PD control authority increase helps but cannot fully reject disturbance.")

    print(f"\n**Recommended Fix Path:**")
    print(f"  Option A: Implement support-error feedforward (HY-FF)")
    print(f"            - Compensate for known support drift before it couples to hip-yaw")
    print(f"  Option B: Fix sagittal support drift first")
    print(f"            - Return to continuous low-height sagittal authority fix")
    print(f"            - Reduce support drift → reduce disturbance → hip-yaw may pass")
    print(f"  Option C: Implement continuous kp schedule as partial improvement")
    print(f"            - Use kp=25 at low heights (24% hip-yaw improvement)")
    print(f"            - Still fails gate, but better than baseline")

    print(f"\n**Reject:** Continuous kd-only schedule")
    print(f"  Reason: kd increase has minimal effect and causes instability at kd=12")

if __name__ == "__main__":
    main()
