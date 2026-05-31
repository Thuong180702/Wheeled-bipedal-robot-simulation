"""Analyze capture gate telemetry from hierarchical controller simulation.

Checks:
1. Whether capture gate activated during transient
2. Support position error peak and timing
3. Gate factor transitions
4. Conflict detection accuracy
"""

import pandas as pd
import numpy as np
import sys
import json
from pathlib import Path


def analyze_capture_gate(csv_path: str, output_dir: str = None, prefix: str = "analysis"):
    """Analyze capture gate telemetry."""

    # Load telemetry
    df = pd.read_csv(csv_path)

    print("=" * 80)
    print("CAPTURE GATE TELEMETRY ANALYSIS")
    print("=" * 80)
    print(f"Telemetry file: {csv_path}")
    print(f"Total steps: {len(df)}")
    print()

    # Check if capture gate columns exist
    capture_gate_cols = [col for col in df.columns if 'capture_gate' in col.lower()]

    if not capture_gate_cols:
        print("[WARNING] No capture gate columns found in telemetry!")
        print("Available columns:", df.columns.tolist())
        return

    print(f"[OK] Found {len(capture_gate_cols)} capture gate columns")
    print()

    # Extract key metrics
    if 'sagittal_support_position_error_m' in df.columns:
        spe = df['sagittal_support_position_error_m'].values
        max_spe = np.max(np.abs(spe))
        max_spe_step = np.argmax(np.abs(spe))
        final_spe = spe[-1]

        print("=" * 80)
        print("SUPPORT POSITION ERROR")
        print("=" * 80)
        print(f"Max SPE: {max_spe:.3f} m at step {max_spe_step}")
        print(f"Final SPE: {final_spe:.3f} m")
        print()
    else:
        print("[WARNING] No support position error column found")
        max_spe_step = None

    # Analyze capture gate activation
    if 'capture_gate_active' in df.columns:
        gate_active = df['capture_gate_active'].values
        gate_active_steps = np.where(gate_active)[0]

        print("=" * 80)
        print("CAPTURE GATE ACTIVATION")
        print("=" * 80)
        print(f"Gate active: {len(gate_active_steps)} / {len(df)} steps ({100*len(gate_active_steps)/len(df):.1f}%)")

        if len(gate_active_steps) > 0:
            print(f"First activation: step {gate_active_steps[0]}")
            print(f"Last activation: step {gate_active_steps[-1]}")
            print()

            # Check gate factor
            if 'capture_gate_factor' in df.columns:
                gate_factor = df['capture_gate_factor'].values
                print(f"Gate factor range: {gate_factor.min():.3f} - {gate_factor.max():.3f}")
                print(f"Gate factor at max SPE (step {max_spe_step}): {gate_factor[max_spe_step]:.3f}")
                print()
        else:
            print("[INFO] Gate never activated")
            print()

    # Analyze conflict detection
    if 'capture_gate_position_opposes_capture' in df.columns:
        conflicts = df['capture_gate_position_opposes_capture'].values
        conflict_steps = np.where(conflicts)[0]

        print("=" * 80)
        print("CONFLICT DETECTION")
        print("=" * 80)
        print(f"Conflicts detected: {len(conflict_steps)} / {len(df)} steps ({100*len(conflict_steps)/len(df):.1f}%)")

        if len(conflict_steps) > 0:
            print(f"First conflict: step {conflict_steps[0]}")
            print(f"Last conflict: step {conflict_steps[-1]}")

            if max_spe_step is not None and max_spe_step in conflict_steps:
                print(f"[OK] Conflict detected at max SPE step {max_spe_step}")
            elif max_spe_step is not None:
                print(f"[WARNING] No conflict at max SPE step {max_spe_step}")
            print()

    # Transient window analysis (steps 1300-1420)
    if len(df) >= 1420:
        print("=" * 80)
        print("TRANSIENT WINDOW ANALYSIS (steps 1300-1420)")
        print("=" * 80)

        window = df.iloc[1300:1421].copy()

        if 'sagittal_support_position_error_m' in window.columns:
            window_spe = window['sagittal_support_position_error_m'].values
            window_max_spe = np.max(np.abs(window_spe))
            window_max_spe_idx = np.argmax(np.abs(window_spe))
            window_max_spe_step = 1300 + window_max_spe_idx

            print(f"Window max SPE: {window_max_spe:.3f} m at step {window_max_spe_step}")

            if 'capture_gate_active' in window.columns:
                window_gate_active = window['capture_gate_active'].values
                window_gate_active_count = np.sum(window_gate_active)
                print(f"Gate active in window: {window_gate_active_count} / {len(window)} steps")

                if 'capture_gate_factor' in window.columns:
                    window_gate_factor = window['capture_gate_factor'].values
                    print(f"Gate factor at window max SPE: {window_gate_factor[window_max_spe_idx]:.3f}")
            print()

    # Save summary
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        summary = {
            "telemetry_file": str(csv_path),
            "total_steps": len(df),
            "capture_gate_columns": capture_gate_cols,
        }

        if 'sagittal_support_position_error_m' in df.columns:
            summary["max_spe_m"] = float(max_spe)
            summary["max_spe_step"] = int(max_spe_step)
            summary["final_spe_m"] = float(final_spe)

        if 'capture_gate_active' in df.columns:
            summary["gate_active_steps"] = int(len(gate_active_steps))
            summary["gate_active_percent"] = float(100 * len(gate_active_steps) / len(df))

        if 'capture_gate_position_opposes_capture' in df.columns:
            summary["conflict_steps"] = int(len(conflict_steps))
            summary["conflict_percent"] = float(100 * len(conflict_steps) / len(df))

        summary_path = output_path / f"{prefix}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"[OK] Summary saved to: {summary_path}")

    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_capture_gate_telemetry.py <telemetry.csv> [output_dir] [prefix]")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    prefix = sys.argv[3] if len(sys.argv) > 3 else "analysis"

    analyze_capture_gate(csv_path, output_dir, prefix)
