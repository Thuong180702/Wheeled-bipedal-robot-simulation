#!/usr/bin/env python3
"""Analyze APCR1nD 2000-step telemetry to verify feature activation."""

import pandas as pd
import json
from pathlib import Path

def main():
    # Find the most recent APCR1nD telemetry
    telemetry_files = list(Path("outputs/hierarchical_controller_sim").glob("telemetry_*.csv"))
    if not telemetry_files:
        print("ERROR: No telemetry files found")
        return

    # Get most recent
    latest = max(telemetry_files, key=lambda p: p.stat().st_mtime)
    print(f"Analyzing: {latest}")

    df = pd.read_csv(latest)
    print(f"Total steps: {len(df)}")

    # Check APCR1nD columns
    apcr1nd_cols = [c for c in df.columns if 'apcr1nd' in c.lower()]
    print(f"\nAPCR1nD columns ({len(apcr1nd_cols)}):")
    for col in sorted(apcr1nd_cols):
        print(f"  - {col}")

    if not apcr1nd_cols:
        print("\nERROR: No APCR1nD columns found - direct trigger not working?")
        return

    # Analyze feature activation
    print("\n=== Feature Activation Analysis ===")

    # Check direct recenter priority
    if 'apcr1nd_direct_recenter_priority_active' in df.columns:
        active_count = df['apcr1nd_direct_recenter_priority_active'].sum()
        print(f"Direct recenter priority active: {active_count}/{len(df)} steps")

    if 'apcr1nd_direct_recenter_eligible' in df.columns:
        eligible_count = df['apcr1nd_direct_recenter_eligible'].sum()
        print(f"Direct recenter eligible: {eligible_count}/{len(df)} steps")

    # Check block reasons
    if 'apcr1nd_direct_recenter_block_reason' in df.columns:
        print("\nBlock reasons distribution:")
        print(df['apcr1nd_direct_recenter_block_reason'].value_counts())

    # Check moving away
    if 'apcr1nd_moving_away' in df.columns:
        moving_away_count = df['apcr1nd_moving_away'].sum()
        print(f"\nMoving away: {moving_away_count}/{len(df)} steps")

    # Check abs error
    if 'apcr1nd_abs_error' in df.columns:
        print(f"\nAPCR1nD abs error range: {df['apcr1nd_abs_error'].min():.4f} - {df['apcr1nd_abs_error'].max():.4f} m")

    # Check APCR1n features (should be inactive since no APC)
    print("\n=== APCR1n Features (should be inactive) ===")
    if 'apcr1n_recenter_priority_active' in df.columns:
        apcr1n_active = df['apcr1n_recenter_priority_active'].sum()
        print(f"APCR1n recenter priority active: {apcr1n_active}/{len(df)} steps")

    # Save summary
    summary = {
        "profile": "APCR1nD_direct_support_recenter_features",
        "total_steps": len(df),
        "apcr1nd_columns": apcr1nd_cols,
        "feature_activation": {
            "direct_recenter_priority_active": int(df['apcr1nd_direct_recenter_priority_active'].sum()) if 'apcr1nd_direct_recenter_priority_active' in df.columns else 0,
            "direct_recenter_eligible": int(df['apcr1nd_direct_recenter_eligible'].sum()) if 'apcr1nd_direct_recenter_eligible' in df.columns else 0,
        },
    }

    output_path = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1nd_2000_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {output_path}")

if __name__ == "__main__":
    main()