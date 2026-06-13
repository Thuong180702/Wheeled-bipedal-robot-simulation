"""
APCR1l Validation Analysis Script
Compares APCR1l with APCR1j baseline using MuJoCo telemetry.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Paths
APCR1L_CSV = "outputs/hierarchical_controller_sim/telemetry_1781085575.csv"
APCR1J_CSV = "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1j_1000_episode_table.csv"
OUTPUT_DIR = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1l_1000_validation")

def load_apcr1l_telemetry():
    """Load APCR1l MuJoCo telemetry."""
    df = pd.read_csv(APCR1L_CSV)
    print(f"Loaded APCR1l telemetry: {len(df)} rows")
    return df

def analyze_apcr1l_telemetry(df):
    """Analyze APCR1l telemetry."""
    results = {}

    # Basic simulation stats
    results['num_steps'] = len(df)
    results['sim_time_s'] = df['time'].max()

    # Check if robot survived
    results['survived'] = not df['terminated'].any()
    if 'termination_reason' in df.columns:
        terminated_rows = df[df['terminated'] == True]
        if len(terminated_rows) > 0:
            results['termination_reason'] = terminated_rows['termination_reason'].iloc[0]
            results['termination_step'] = terminated_rows['step'].iloc[0]
        else:
            results['termination_reason'] = None
            results['termination_step'] = None

    # Position error metrics
    if 'sagittal_position_error_m' in df.columns:
        e = df['sagittal_position_error_m'].values
        e = e[~np.isnan(e)]  # Remove NaN
        results['min_e_m'] = float(np.min(e))
        results['max_e_m'] = float(np.max(e))
        results['max_abs_e_m'] = float(np.max(np.abs(e)))
        results['mean_e_m'] = float(np.mean(e))
        results['abs_mean_e_m'] = float(np.mean(np.abs(e)))
        results['final_e_m'] = float(e[-1]) if len(e) > 0 else 0.0

    # Pitch metrics
    if 'pitch_x' in df.columns:
        pitch = df['pitch_x'].values
        pitch_deg = np.degrees(pitch)
        results['pitch_min_deg'] = float(np.min(pitch_deg))
        results['pitch_max_deg'] = float(np.max(pitch_deg))
        results['pitch_rms_deg'] = float(np.sqrt(np.mean(pitch_deg**2)))

    # Height metrics
    if 'com_z' in df.columns:
        com_z = df['com_z'].values
        com_z = com_z[~np.isnan(com_z)]
        if len(com_z) > 0:
            results['height_min_m'] = float(np.min(com_z))
            results['height_max_m'] = float(np.max(com_z))
            results['height_mean_m'] = float(np.mean(com_z))

    # APCR1l specific telemetry
    if 'apcr1l_pitch_suppress_active' in df.columns:
        pitch_suppress = df['apcr1l_pitch_suppress_active'].values
        pitch_suppress = pitch_suppress[pitch_suppress == True] if len(pitch_suppress) > 0 else []
        results['pitch_suppress_count'] = int(len(pitch_suppress))
        results['pitch_suppress_pct'] = float(len(pitch_suppress) / len(df) * 100)

    if 'apcr1l_recenter_state' in df.columns:
        recenter_states = df['apcr1l_recenter_state'].values
        results['recenter_counts'] = {
            'NEUTRAL': int(np.sum(recenter_states == 'NEUTRAL')),
            'RECENTER_FROM_POSITIVE': int(np.sum(recenter_states == 'RECENTER_FROM_POSITIVE')),
            'RECENTER_FROM_NEGATIVE': int(np.sum(recenter_states == 'RECENTER_FROM_NEGATIVE')),
        }

    # Torque analysis
    if 'final_wheel_tau_with_apc' in df.columns:
        tau_with_apc = df['final_wheel_tau_with_apc'].values
        tau_with_apc = tau_with_apc[~np.isnan(tau_with_apc)]
        if len(tau_with_apc) > 0:
            results['tau_with_apc_mean'] = float(np.mean(tau_with_apc))
            results['tau_with_apc_max'] = float(np.max(np.abs(tau_with_apc)))

    if 'final_wheel_tau_without_apc' in df.columns:
        tau_without_apc = df['final_wheel_tau_without_apc'].values
        tau_without_apc = tau_without_apc[~np.isnan(tau_without_apc)]
        if len(tau_without_apc) > 0:
            results['tau_without_apc_mean'] = float(np.mean(tau_without_apc))
            results['tau_without_apc_max'] = float(np.max(np.abs(tau_without_apc)))

    # APCR contribution analysis
    if 'active_pitch_crossing_tau_clipped' in df.columns:
        apc_tau = df['active_pitch_crossing_tau_clipped'].values
        apc_tau = apc_tau[~np.isnan(apc_tau)]
        if len(apc_tau) > 0:
            results['apc_tau_mean'] = float(np.mean(apc_tau))
            results['apc_tau_max'] = float(np.max(np.abs(apc_tau)))
            results['apc_tau_active_count'] = int(np.sum(np.abs(apc_tau) > 0.01))

    # Torque direction analysis
    if 'sagittal_position_error_m' in df.columns and 'final_wheel_tau_with_apc' in df.columns:
        e = df['sagittal_position_error_m'].values
        tau = df['final_wheel_tau_with_apc'].values

        # Valid steps (non-NaN)
        valid = ~(np.isnan(e) | np.isnan(tau))
        e_valid = e[valid]
        tau_valid = tau[valid]

        if len(e_valid) > 0:
            # Check if final torque opposes drift
            # For positive drift, need negative torque
            # For negative drift, need positive torque
            correct_direction = (
                (e_valid > 0) & (tau_valid < 0) |  # Positive drift, negative torque (correct)
                (e_valid < 0) & (tau_valid > 0)     # Negative drift, positive torque (correct)
            )
            results['torque_correct_direction_count'] = int(np.sum(correct_direction))
            results['torque_correct_direction_pct'] = float(np.mean(correct_direction) * 100)

    return results

def load_apcr1j_baseline():
    """Load APCR1j baseline episode data."""
    df = pd.read_csv(APCR1J_CSV)
    print(f"Loaded APCR1j baseline: {len(df)} episodes")
    return df

def compare_with_baseline(results, baseline_df):
    """Compare APCR1l results with APCR1j baseline."""
    comparison = {}

    # Get APCR1j episode metrics
    if len(baseline_df) > 0:
        baseline_max_e = baseline_df['max_e'].max()
        baseline_mean_max_e = baseline_df['max_e'].mean()
        baseline_mean_tau = baseline_df['max_tau'].mean()

        comparison['apcr1j_max_max_e'] = float(baseline_max_e)
        comparison['apcr1j_mean_max_e'] = float(baseline_mean_max_e)
        comparison['apcr1j_mean_max_tau'] = float(baseline_mean_tau)

        # Compare with APCR1l
        if 'max_abs_e_m' in results:
            comparison['apcr1l_max_abs_e'] = results['max_abs_e_m']
            comparison['improvement_vs_apcr1j'] = (baseline_max_e - results['max_abs_e_m']) / baseline_max_e * 100

    return comparison

def main():
    print("=" * 70)
    print("APCR1l Validation Analysis")
    print("=" * 70)
    print()

    # Load and analyze APCR1l telemetry
    print("Loading APCR1l telemetry...")
    df = load_apcr1l_telemetry()

    print("\nAnalyzing APCR1l results...")
    results = analyze_apcr1l_telemetry(df)

    # Load baseline
    print("\nLoading APCR1j baseline...")
    baseline_df = load_apcr1j_baseline()

    # Compare
    print("\nComparing with baseline...")
    comparison = compare_with_baseline(results, baseline_df)

    # Print results
    print("\n" + "=" * 70)
    print("APCR1l VALIDATION RESULTS")
    print("=" * 70)
    print(f"\nSimulation Status:")
    print(f"  Steps:        {results.get('num_steps', 'N/A')}")
    print(f"  Sim time:    {results.get('sim_time_s', 'N/A'):.2f} s")
    print(f"  Survived:    {results.get('survived', 'N/A')}")
    if results.get('termination_reason'):
        print(f"  Termination: {results.get('termination_reason')} at step {results.get('termination_step')}")

    print(f"\nPosition Error Metrics:")
    print(f"  min_e:       {results.get('min_e_m', 'N/A'):.4f} m")
    print(f"  max_e:       {results.get('max_e_m', 'N/A'):.4f} m")
    print(f"  max_abs_e:   {results.get('max_abs_e_m', 'N/A'):.4f} m")
    print(f"  mean_e:      {results.get('mean_e_m', 'N/A'):.4f} m")
    print(f"  abs_mean_e:  {results.get('abs_mean_e_m', 'N/A'):.4f} m")
    print(f"  final_e:     {results.get('final_e_m', 'N/A'):.4f} m")

    if 'pitch_rms_deg' in results:
        print(f"\nPitch Metrics:")
        print(f"  Range:       [{results.get('pitch_min_deg', 'N/A'):.2f}, {results.get('pitch_max_deg', 'N/A'):.2f}] deg")
        print(f"  RMS:         {results.get('pitch_rms_deg', 'N/A'):.2f} deg")

    print(f"\nAPCR1l Pitch Suppression:")
    print(f"  Suppression count: {results.get('pitch_suppress_count', 'N/A')} / {results.get('num_steps', 'N/A')}")
    print(f"  Suppression pct:   {results.get('pitch_suppress_pct', 'N/A'):.1f}%")
    if 'recenter_counts' in results:
        print(f"  Recenter states:   {results.get('recenter_counts')}")

    if 'torque_correct_direction_pct' in results:
        print(f"\nTorque Direction Analysis:")
        print(f"  Correct direction: {results.get('torque_correct_direction_count', 'N/A')} / {results.get('num_steps', 'N/A')}")
        print(f"  Correct pct:       {results.get('torque_correct_direction_pct', 'N/A'):.1f}%")

    print("\n" + "=" * 70)
    print("COMPARISON WITH APCR1j BASELINE")
    print("=" * 70)
    print(f"\nMetric              | APCR1j     | APCR1l     | Change")
    print("-" * 60)
    print(f"max_abs_e (m)       | {comparison.get('apcr1j_max_max_e', 'N/A'):.4f}     | "
          f"{comparison.get('apcr1l_max_abs_e', 'N/A'):.4f}     | "
          f"{comparison.get('improvement_vs_apcr1j', 'N/A'):+.1f}%")
    print(f"mean_max_e (m)      | {comparison.get('apcr1j_mean_max_e', 'N/A'):.4f}     | -            | -")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {
        'apcr1l_results': results,
        'comparison': comparison,
    }

    results_path = OUTPUT_DIR / "apcr1l_analysis_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return results, comparison

if __name__ == "__main__":
    main()
