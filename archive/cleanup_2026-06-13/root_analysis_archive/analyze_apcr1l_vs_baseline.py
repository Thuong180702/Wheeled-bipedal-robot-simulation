"""
APCR1l Validation Analysis Script
Compares APCR1l with APCR1i/APCR1j baseline using MuJoCo telemetry.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Paths
APCR1L_CSV = "outputs/hierarchical_controller_sim/telemetry_1781085977.csv"  # Latest APCR1l with height variant
APCR1I_CSV = "outputs/hierarchical_controller_sim/telemetry_1781058071.csv"  # APCR1i baseline
OUTPUT_DIR = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1l_1000_validation")


def load_telemetry(path):
    """Load telemetry CSV."""
    df = pd.read_csv(path)
    print(f"Loaded {path}: {len(df)} rows")
    return df


def analyze_telemetry(df, name):
    """Analyze telemetry for APCR metrics."""
    results = {'name': name}

    # Basic simulation stats
    results['num_steps'] = len(df)
    results['sim_time_s'] = df['time'].max() if 'time' in df.columns else len(df) * 0.01

    # Check if robot survived
    if 'terminated' in df.columns:
        results['survived'] = not df['terminated'].any()
        if results['survived']:
            terminated_rows = df[df['terminated'] == True]
            if len(terminated_rows) > 0:
                results['termination_reason'] = terminated_rows.iloc[0]['termination_reason']
                results['termination_step'] = terminated_rows.iloc[0]['step']
    else:
        results['survived'] = True

    # Position error metrics
    if 'sagittal_position_error_m' in df.columns:
        e = pd.to_numeric(df['sagittal_position_error_m'], errors='coerce').dropna()
        if len(e) > 0:
            results['min_e_m'] = float(e.min())
            results['max_e_m'] = float(e.max())
            results['max_abs_e_m'] = float(e.abs().max())
            results['mean_e_m'] = float(e.mean())
            results['abs_mean_e_m'] = float(e.abs().mean())
            results['final_e_m'] = float(e.iloc[-1])

    # Height metrics
    if 'com_z' in df.columns:
        com_z = pd.to_numeric(df['com_z'], errors='coerce').dropna()
        if len(com_z) > 0:
            results['height_min_m'] = float(com_z.min())
            results['height_max_m'] = float(com_z.max())
            results['height_mean_m'] = float(com_z.mean())

    # Pitch metrics (convert to degrees)
    if 'robot_pitch_x' in df.columns:
        pitch = pd.to_numeric(df['robot_pitch_x'], errors='coerce').dropna()
        if len(pitch) > 0:
            pitch_deg = np.degrees(pitch.values)
            results['pitch_min_deg'] = float(pitch_deg.min())
            results['pitch_max_deg'] = float(pitch_deg.max())
            results['pitch_rms_deg'] = float(np.sqrt(np.mean(pitch_deg**2)))

    # APCR hysteresis state analysis
    if 'active_pitch_crossing_hysteresis_state' in df.columns:
        states = df['active_pitch_crossing_hysteresis_state'].fillna('UNKNOWN')
        results['hysteresis_state_counts'] = states.value_counts().to_dict()

    # APCR torque analysis
    if 'active_pitch_crossing_tau_clipped' in df.columns:
        apc_tau = pd.to_numeric(df['active_pitch_crossing_tau_clipped'], errors='coerce').dropna()
        if len(apc_tau) > 0:
            results['apc_tau_mean'] = float(apc_tau.mean())
            results['apc_tau_max_abs'] = float(apc_tau.abs().max())
            results['apc_tau_active_count'] = int((apc_tau.abs() > 0.01).sum())

    # Final wheel torque analysis
    if 'final_wheel_tau_with_apc' in df.columns:
        tau = pd.to_numeric(df['final_wheel_tau_with_apc'], errors='coerce').dropna()
        if len(tau) > 0:
            results['tau_with_apc_mean'] = float(tau.mean())
            results['tau_with_apc_max_abs'] = float(tau.abs().max())

    # Torque direction analysis
    if 'sagittal_position_error_m' in df.columns and 'final_wheel_tau_with_apc' in df.columns:
        e = pd.to_numeric(df['sagittal_position_error_m'], errors='coerce')
        tau = pd.to_numeric(df['final_wheel_tau_with_apc'], errors='coerce')
        valid = ~(e.isna() | tau.isna())
        e_valid = e[valid].values
        tau_valid = tau[valid].values

        if len(e_valid) > 0:
            # For positive drift, need negative torque
            # For negative drift, need positive torque
            correct_direction = (
                (e_valid > 0) & (tau_valid < 0) |  # Positive drift, negative torque (correct)
                (e_valid < 0) & (tau_valid > 0)     # Negative drift, positive torque (correct)
            )
            results['torque_correct_direction_count'] = int(correct_direction.sum())
            results['torque_correct_direction_pct'] = float(correct_direction.mean() * 100)

    return results


def main():
    print("=" * 70)
    print("APCR1l Validation Analysis")
    print("=" * 70)
    print()

    # Load telemetry files
    print("Loading telemetry files...")
    try:
        df_apcr1l = load_telemetry(APCR1L_CSV)
    except Exception as e:
        print(f"Error loading APCR1l: {e}")
        return

    try:
        df_apcr1i = load_telemetry(APCR1I_CSV)
    except Exception as e:
        print(f"Error loading APCR1i: {e}")
        return

    # Analyze each
    print("\nAnalyzing APCR1i (baseline)...")
    results_apcr1i = analyze_telemetry(df_apcr1i, "APCR1i")

    print("\nAnalyzing APCR1l (with pitch suppression)...")
    results_apcr1l = analyze_telemetry(df_apcr1l, "APCR1l")

    # Print results
    print("\n" + "=" * 70)
    print("SIMULATION STATUS COMPARISON")
    print("=" * 70)

    for name, r in [("APCR1i", results_apcr1i), ("APCR1l", results_apcr1l)]:
        print(f"\n{name}:")
        print(f"  Steps:      {r.get('num_steps', 'N/A')}")
        print(f"  Survived:  {r.get('survived', 'N/A')}")
        if r.get('termination_reason'):
            print(f"  Terminated: {r.get('termination_reason')} at step {r.get('termination_step')}")

    print("\n" + "=" * 70)
    print("POSITION ERROR COMPARISON")
    print("=" * 70)

    print(f"\nMetric              | APCR1i     | APCR1l")
    print("-" * 50)
    print(f"max_abs_e (m)      | {results_apcr1i.get('max_abs_e_m', 'N/A'):.4f}     | {results_apcr1l.get('max_abs_e_m', 'N/A'):.4f}")
    print(f"abs_mean_e (m)     | {results_apcr1i.get('abs_mean_e_m', 'N/A'):.4f}     | {results_apcr1l.get('abs_mean_e_m', 'N/A'):.4f}")
    print(f"min_e (m)           | {results_apcr1i.get('min_e_m', 'N/A'):.4f}     | {results_apcr1l.get('min_e_m', 'N/A'):.4f}")
    print(f"max_e (m)           | {results_apcr1i.get('max_e_m', 'N/A'):.4f}     | {results_apcr1l.get('max_e_m', 'N/A'):.4f}")

    print("\n" + "=" * 70)
    print("HEIGHT COMPARISON")
    print("=" * 70)

    print(f"\nMetric              | APCR1i     | APCR1l")
    print("-" * 50)
    print(f"height_min (m)     | {results_apcr1i.get('height_min_m', 'N/A'):.4f}     | {results_apcr1l.get('height_min_m', 'N/A'):.4f}")
    print(f"height_max (m)     | {results_apcr1i.get('height_max_m', 'N/A'):.4f}     | {results_apcr1l.get('height_max_m', 'N/A'):.4f}")
    print(f"height_mean (m)     | {results_apcr1i.get('height_mean_m', 'N/A'):.4f}     | {results_apcr1l.get('height_mean_m', 'N/A'):.4f}")

    print("\n" + "=" * 70)
    print("PITCH COMPARISON")
    print("=" * 70)

    print(f"\nMetric               | APCR1i     | APCR1l")
    print("-" * 50)
    print(f"pitch_min (deg)     | {results_apcr1i.get('pitch_min_deg', 'N/A'):.2f}     | {results_apcr1l.get('pitch_min_deg', 'N/A'):.2f}")
    print(f"pitch_max (deg)     | {results_apcr1i.get('pitch_max_deg', 'N/A'):.2f}     | {results_apcr1l.get('pitch_max_deg', 'N/A'):.2f}")
    print(f"pitch_rms (deg)     | {results_apcr1i.get('pitch_rms_deg', 'N/A'):.2f}     | {results_apcr1l.get('pitch_rms_deg', 'N/A'):.2f}")

    print("\n" + "=" * 70)
    print("APCR TORQUE ANALYSIS")
    print("=" * 70)

    print(f"\nMetric                  | APCR1i     | APCR1l")
    print("-" * 50)
    print(f"apc_tau_mean (Nm)      | {results_apcr1i.get('apc_tau_mean', 'N/A'):.4f}     | {results_apcr1l.get('apc_tau_mean', 'N/A'):.4f}")
    print(f"apc_tau_max_abs (Nm)   | {results_apcr1i.get('apc_tau_max_abs', 'N/A'):.4f}     | {results_apcr1l.get('apc_tau_max_abs', 'N/A'):.4f}")
    print(f"apc_tau_active_count   | {results_apcr1i.get('apc_tau_active_count', 'N/A')}     | {results_apcr1l.get('apc_tau_active_count', 'N/A')}")

    print("\n" + "=" * 70)
    print("TORQUE DIRECTION CORRECTNESS")
    print("=" * 70)

    print(f"\nMetric                  | APCR1i     | APCR1l")
    print("-" * 50)
    print(f"correct_direction_count | {results_apcr1i.get('torque_correct_direction_count', 'N/A')}     | {results_apcr1l.get('torque_correct_direction_count', 'N/A')}")
    print(f"correct_direction_pct  | {results_apcr1i.get('torque_correct_direction_pct', 'N/A'):.1f}%    | {results_apcr1l.get('torque_correct_direction_pct', 'N/A'):.1f}%")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {
        'apcr1i': results_apcr1i,
        'apcr1l': results_apcr1l,
    }

    results_path = OUTPUT_DIR / "apcr1l_comparison_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
