#!/usr/bin/env python3
"""Quick analysis of APC1 simulation failure"""
import pandas as pd
import sys

# Read the telemetry files
files = [
    ("APC1", "outputs/hierarchical_controller_sim/telemetry_1780913786.csv"),
    ("D2 baseline", "outputs/hierarchical_controller_sim/telemetry_1780913944.csv"),
]

for name, path in files:
    print(f"\n{'='*60}")
    print(f"{name}: {path}")
    print('='*60)

    try:
        df = pd.read_csv(path)
        print(f"Total rows: {len(df)}")
        print(f"Terminated: {df['terminated'].iloc[-1] if len(df) > 0 else 'N/A'}")
        print(f"Termination reason: {df['termination_reason'].iloc[-1] if len(df) > 0 else 'N/A'}")

        # Key columns
        cols = ['step', 'com_z', 'robot_pitch_x', 'robot_roll_y', 'height_error_m']
        available = [c for c in cols if c in df.columns]
        if available:
            print(f"\nFirst 5 steps:")
            print(df[available].head())
            print(f"\nLast 3 steps:")
            print(df[available].tail(3))

        # Check sagittal_position_error
        if 'sagittal_position_error_m' in df.columns:
            print(f"\nSagittal position error stats:")
            print(f"  Min: {df['sagittal_position_error_m'].min():.4f}")
            print(f"  Max: {df['sagittal_position_error_m'].max():.4f}")
            print(f"  Mean: {df['sagittal_position_error_m'].mean():.4f}")

        # Check contact state
        if 'left_contact_active' in df.columns:
            print(f"\nContact state:")
            print(f"  Left contact active (last): {df['left_contact_active'].iloc[-1]}")
            print(f"  Right contact active (last): {df['right_contact_active'].iloc[-1]}")

    except Exception as e:
        print(f"Error reading {path}: {e}")
