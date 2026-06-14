#!/usr/bin/env python3
"""
Phase 1: APCR1nD band-control failure audit.

Analyzes why APCR1nD fails to keep support drift inside ±0.08 m target band.
Uses correct physical signed drift metrics only.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path


def classify_band_failure(
    outside_08_count,
    outside_08_inactive,
    early_release_count,
    late_entry_count,
    authority_weak_count,
    damping_rare_pct
):
    """Classify primary band failure mode."""

    causes = []

    # Late entry: many crossings happen before activation
    if late_entry_count > 10:
        causes.append("LATE_ENTRY")

    # Early release: feature turns off while still outside band
    if early_release_count > 20:
        causes.append("EARLY_RELEASE")

    # Moving-away gate too strict: outside band but not active because converging
    if outside_08_inactive > 100:
        causes.append("MOVING_AWAY_GATING")

    # Authority too weak: active but not enough to return quickly
    if authority_weak_count > 50:
        causes.append("WEAK_AUTHORITY")

    # Damping override too rare
    if damping_rare_pct < 5.0:
        causes.append("DAMPING_TOO_RARE")

    if len(causes) == 0:
        return "APCR1ND_BAND_FAIL_INCONCLUSIVE"
    elif len(causes) == 1:
        return f"APCR1ND_BAND_FAIL_{causes[0]}"
    else:
        return "APCR1ND_BAND_FAIL_MIXED_CAUSES"


def main():
    telemetry_path = Path("outputs/hierarchical_controller_sim/telemetry_1781226281.csv")

    if not telemetry_path.exists():
        print(f"ERROR: Telemetry file not found: {telemetry_path}")
        return

    print(f"Loading telemetry: {telemetry_path}")
    df = pd.read_csv(telemetry_path)

    print(f"Total steps: {len(df)}")

    # Use correct physical drift
    drift_col = None
    for col in ['active_pitch_crossing_signed_error_m',
                'sagittal_position_error_m',
                'support_position_error_m',
                'hip_yaw_comp_support_error_m']:
        if col in df.columns:
            drift_col = col
            break

    if drift_col is None:
        print("ERROR: No physical drift column found")
        return

    print(f"Using drift column: {drift_col}")

    e = df[drift_col].values
    abs_e = np.abs(e)

    # APCR1nD feature columns
    active = df['apcr1nd_direct_recenter_priority_active'].values if 'apcr1nd_direct_recenter_priority_active' in df.columns else np.zeros(len(df))
    eligible = df['apcr1nd_direct_recenter_eligible'].values if 'apcr1nd_direct_recenter_eligible' in df.columns else np.zeros(len(df))
    moving_away = df['apcr1nd_moving_away'].values if 'apcr1nd_moving_away' in df.columns else np.zeros(len(df))

    # Compute error rate
    e_dot = np.diff(e, prepend=e[0])

    # Position cap boost (check if column exists)
    if 'apcr1n_position_cap_boost_active' in df.columns:
        cap_boost_active = df['apcr1n_position_cap_boost_active'].values
    else:
        cap_boost_active = np.zeros(len(df))

    # Wheel damping override
    if 'apcr1n_wheel_damping_fights_drift' in df.columns:
        damping_override = df['apcr1n_wheel_damping_fights_drift'].values
    else:
        damping_override = np.zeros(len(df))

    # Band thresholds
    band_05 = abs_e > 0.05
    band_08 = abs_e > 0.08
    band_10 = abs_e > 0.10
    band_12 = abs_e > 0.12
    band_15 = abs_e > 0.15

    # Outside band counts
    outside_05 = np.sum(band_05)
    outside_08 = np.sum(band_08)
    outside_10 = np.sum(band_10)
    outside_12 = np.sum(band_12)
    outside_15 = np.sum(band_15)

    # Outside band AND active/inactive
    outside_08_active = np.sum(band_08 & (active > 0.5))
    outside_08_inactive = np.sum(band_08 & (active < 0.5))

    outside_10_active = np.sum(band_10 & (active > 0.5))
    outside_10_inactive = np.sum(band_10 & (active < 0.5))

    # Converging vs moving away
    converging = (e * e_dot) < 0  # opposite signs
    moving_away_flag = (e * e_dot) > 0

    outside_08_converging = np.sum(band_08 & converging)
    outside_08_moving_away = np.sum(band_08 & moving_away_flag)

    # Band crossings
    crossings_05 = []
    crossings_08 = []
    crossings_10 = []
    crossings_12 = []
    crossings_15 = []

    for i in range(1, len(df)):
        if abs_e[i-1] <= 0.05 and abs_e[i] > 0.05:
            crossings_05.append(i)
        if abs_e[i-1] <= 0.08 and abs_e[i] > 0.08:
            crossings_08.append(i)
        if abs_e[i-1] <= 0.10 and abs_e[i] > 0.10:
            crossings_10.append(i)
        if abs_e[i-1] <= 0.12 and abs_e[i] > 0.12:
            crossings_12.append(i)
        if abs_e[i-1] <= 0.15 and abs_e[i] > 0.15:
            crossings_15.append(i)

    # Analyze crossing events
    crossing_events = []
    for i in crossings_08:
        event = {
            'step': int(i),
            'abs_e': float(abs_e[i]),
            'e': float(e[i]),
            'e_dot': float(e_dot[i]),
            'moving_away': bool(moving_away[i] > 0.5) if i < len(moving_away) else False,
            'active': bool(active[i] > 0.5) if i < len(active) else False,
            'eligible': bool(eligible[i] > 0.5) if i < len(eligible) else False,
        }
        crossing_events.append(event)

    # Early release detection: feature turns off while abs(e) > threshold
    early_release_08 = 0
    early_release_10 = 0
    early_release_12 = 0

    for i in range(1, len(df)):
        # If was active, now inactive, but still outside band
        if active[i-1] > 0.5 and active[i] < 0.5:
            if abs_e[i] > 0.08:
                early_release_08 += 1
            if abs_e[i] > 0.10:
                early_release_10 += 1
            if abs_e[i] > 0.12:
                early_release_12 += 1

    # Late entry detection: crosses band before activation
    late_entry_events = []
    for cross_step in crossings_08:
        # Check if was inactive before crossing
        if cross_step > 0 and active[cross_step-1] < 0.5:
            # Find how long until activation
            activation_delay = None
            for j in range(cross_step, min(cross_step + 50, len(df))):
                if active[j] > 0.5:
                    activation_delay = j - cross_step
                    break

            late_entry_events.append({
                'cross_step': int(cross_step),
                'activation_delay': int(activation_delay) if activation_delay else None,
                'abs_e_at_cross': float(abs_e[cross_step])
            })

    # Authority weakness: active but error not decreasing
    authority_weak_count = 0
    for i in range(10, len(df)):
        if active[i] > 0.5 and abs_e[i] > 0.08:
            # Check if error decreased in last 10 steps
            if abs_e[i] >= abs_e[i-10]:
                authority_weak_count += 1

    # Damping override stats
    damping_active_pct = 100.0 * np.sum(damping_override > 0.5) / len(df)

    # Classification
    classification = classify_band_failure(
        outside_08,
        outside_08_inactive,
        early_release_08,
        len(late_entry_events),
        authority_weak_count,
        damping_active_pct
    )

    # Summary
    summary = {
        'telemetry_file': str(telemetry_path),
        'total_steps': len(df),
        'drift_column_used': drift_col,
        'max_abs_e': float(abs_e.max()),
        'p2p': float(e.max() - e.min()),
        'mean_abs_e': float(abs_e.mean()),
        'final_e': float(e[-1]),
        'band_crossings': {
            'cross_05': len(crossings_05),
            'cross_08': len(crossings_08),
            'cross_10': len(crossings_10),
            'cross_12': len(crossings_12),
            'cross_15': len(crossings_15)
        },
        'outside_band_counts': {
            'outside_05': int(outside_05),
            'outside_08': int(outside_08),
            'outside_10': int(outside_10),
            'outside_12': int(outside_12),
            'outside_15': int(outside_15)
        },
        'outside_band_percent': {
            'outside_05_pct': 100.0 * outside_05 / len(df),
            'outside_08_pct': 100.0 * outside_08 / len(df),
            'outside_10_pct': 100.0 * outside_10 / len(df),
            'outside_12_pct': 100.0 * outside_12 / len(df),
            'outside_15_pct': 100.0 * outside_15 / len(df)
        },
        'outside_band_active_inactive': {
            'outside_08_active': int(outside_08_active),
            'outside_08_inactive': int(outside_08_inactive),
            'outside_08_active_pct': 100.0 * outside_08_active / max(1, outside_08),
            'outside_08_inactive_pct': 100.0 * outside_08_inactive / max(1, outside_08),
            'outside_10_active': int(outside_10_active),
            'outside_10_inactive': int(outside_10_inactive),
            'outside_10_active_pct': 100.0 * outside_10_active / max(1, outside_10),
            'outside_10_inactive_pct': 100.0 * outside_10_inactive / max(1, outside_10)
        },
        'converging_vs_moving_away': {
            'outside_08_converging': int(outside_08_converging),
            'outside_08_moving_away': int(outside_08_moving_away),
            'outside_08_converging_pct': 100.0 * outside_08_converging / max(1, outside_08),
            'outside_08_moving_away_pct': 100.0 * outside_08_moving_away / max(1, outside_08)
        },
        'feature_activation': {
            'direct_recenter_active_pct': 100.0 * np.sum(active > 0.5) / len(df),
            'direct_recenter_eligible_pct': 100.0 * np.sum(eligible > 0.5) / len(df),
            'position_cap_boost_active_pct': 100.0 * np.sum(cap_boost_active > 0.5) / len(df),
            'wheel_damping_override_pct': damping_active_pct
        },
        'release_behavior': {
            'early_release_while_outside_08': int(early_release_08),
            'early_release_while_outside_10': int(early_release_10),
            'early_release_while_outside_12': int(early_release_12)
        },
        'entry_behavior': {
            'late_entry_count': len(late_entry_events),
            'late_entry_events': late_entry_events[:10]  # First 10
        },
        'authority_analysis': {
            'authority_weak_count': int(authority_weak_count),
            'authority_weak_pct': 100.0 * authority_weak_count / len(df)
        },
        'classification': classification
    }

    # Save results
    output_dir = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "apcr1nd_band_control_failure_audit.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved audit: {json_path}")

    # Save crossing events
    events_df = pd.DataFrame(crossing_events)
    csv_path = output_dir / "apcr1nd_band_control_failure_events.csv"
    events_df.to_csv(csv_path, index=False)
    print(f"Saved events: {csv_path}")

    # Print summary
    print("\n" + "="*60)
    print("APCR1nD BAND CONTROL FAILURE AUDIT")
    print("="*60)
    print(f"\nMax |e|: {summary['max_abs_e']:.4f} m")
    print(f"P2P: {summary['p2p']:.4f} m")
    print(f"Mean |e|: {summary['mean_abs_e']:.4f} m")
    print(f"Final e: {summary['final_e']:.4f} m")
    print(f"\nOutside ±0.08: {outside_08}/{len(df)} steps ({summary['outside_band_percent']['outside_08_pct']:.1f}%)")
    print(f"Outside ±0.10: {outside_10}/{len(df)} steps ({summary['outside_band_percent']['outside_10_pct']:.1f}%)")
    print(f"Outside ±0.15: {outside_15}/{len(df)} steps ({summary['outside_band_percent']['outside_15_pct']:.1f}%)")
    print(f"\nBand crossings:")
    print(f"  +0.05: {len(crossings_05)} times")
    print(f"  +0.08: {len(crossings_08)} times")
    print(f"  +0.10: {len(crossings_10)} times")
    print(f"  +0.15: {len(crossings_15)} times")
    print(f"\nOutside ±0.08 behavior:")
    print(f"  Active: {outside_08_active}/{outside_08} ({summary['outside_band_active_inactive']['outside_08_active_pct']:.1f}%)")
    print(f"  Inactive: {outside_08_inactive}/{outside_08} ({summary['outside_band_active_inactive']['outside_08_inactive_pct']:.1f}%)")
    print(f"  Converging: {outside_08_converging}/{outside_08} ({summary['converging_vs_moving_away']['outside_08_converging_pct']:.1f}%)")
    print(f"  Moving away: {outside_08_moving_away}/{outside_08} ({summary['converging_vs_moving_away']['outside_08_moving_away_pct']:.1f}%)")
    print(f"\nFeature activation:")
    print(f"  Direct recenter: {summary['feature_activation']['direct_recenter_active_pct']:.1f}%")
    print(f"  Position cap boost: {summary['feature_activation']['position_cap_boost_active_pct']:.1f}%")
    print(f"  Wheel damping override: {summary['feature_activation']['wheel_damping_override_pct']:.1f}%")
    print(f"\nEarly release events: {early_release_08} (while outside ±0.08)")
    print(f"Late entry events: {len(late_entry_events)}")
    print(f"Authority weak count: {authority_weak_count}")
    print(f"\nClassification: {classification}")
    print("="*60)


if __name__ == '__main__':
    main()
