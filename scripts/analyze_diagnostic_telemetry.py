"""Quick analysis of Phase B.8 diagnostic telemetry to identify hierarchical controller failure mode."""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_controller(csv_path: Path, controller_name: str):
    """Analyze single controller telemetry."""
    df = pd.read_csv(csv_path)

    print(f"\n{'='*60}")
    print(f"{controller_name} - {csv_path.name}")
    print(f"{'='*60}")
    print(f"Survival time: {df['time'].iloc[-1]:.3f}s")
    print(f"Fell: {df['fell'].iloc[-1]}")
    print(f"\nFirst 5 timesteps:")

    # Key signals
    cols = ['time', 'pitch', 'pitch_rate', 'roll', 'roll_rate',
            'raw_wheel_cmd_l', 'filtered_wheel_cmd_l',
            'base_action_4', 'base_action_9',  # wheel actions
            'vmc_height_correction', 'vmc_com_correction',
            'action_saturation_rate']

    print(df[cols].head(5).to_string(index=False))

    # Statistics
    print(f"\nStatistics (first 5 steps):")
    print(f"  Max |pitch_rate|: {df['pitch_rate'].abs().head(5).max():.2f} rad/s")
    print(f"  Max |roll_rate|: {df['roll_rate'].abs().head(5).max():.2f} rad/s")
    print(f"  Max |wheel_cmd|: {df['raw_wheel_cmd_l'].abs().head(5).max():.2f} rad/s")
    print(f"  Mean saturation: {df['action_saturation_rate'].head(5).mean():.2%}")

    return df

def main():
    output_dir = Path("outputs/phase_b8_diagnostics")

    # Compare h=0.70, ep=0 for both controllers
    baseline_path = output_dir / "telemetry_height_scheduled_dynamic_lqr_h0.70_ep0.csv"
    hierarchical_path = output_dir / "telemetry_hierarchical_vmc_lqr_h0.70_ep0.csv"

    df_baseline = analyze_controller(baseline_path, "Baseline (height_scheduled_dynamic_lqr)")
    df_hierarchical = analyze_controller(hierarchical_path, "Hierarchical (hierarchical_vmc_lqr)")

    # Direct comparison
    print(f"\n{'='*60}")
    print("DIRECT COMPARISON (t=0.02s)")
    print(f"{'='*60}")

    t_idx = 1  # t=0.02s

    print(f"\nPitch dynamics:")
    print(f"  Baseline:     pitch={df_baseline['pitch'].iloc[t_idx]:+.3f}, rate={df_baseline['pitch_rate'].iloc[t_idx]:+.2f}")
    print(f"  Hierarchical: pitch={df_hierarchical['pitch'].iloc[t_idx]:+.3f}, rate={df_hierarchical['pitch_rate'].iloc[t_idx]:+.2f}")

    print(f"\nRoll dynamics:")
    print(f"  Baseline:     roll={df_baseline['roll'].iloc[t_idx]:+.3f}, rate={df_baseline['roll_rate'].iloc[t_idx]:+.2f}")
    print(f"  Hierarchical: roll={df_hierarchical['roll'].iloc[t_idx]:+.3f}, rate={df_hierarchical['roll_rate'].iloc[t_idx]:+.2f}")

    print(f"\nWheel commands:")
    print(f"  Baseline:     raw={df_baseline['raw_wheel_cmd_l'].iloc[t_idx]:+.2f}, filtered={df_baseline['filtered_wheel_cmd_l'].iloc[t_idx]:+.2f}")
    print(f"  Hierarchical: raw={df_hierarchical['raw_wheel_cmd_l'].iloc[t_idx]:+.2f}, filtered={df_hierarchical['filtered_wheel_cmd_l'].iloc[t_idx]:+.2f}")

    print(f"\nWheel actions (normalized):")
    print(f"  Baseline:     L={df_baseline['base_action_4'].iloc[t_idx]:+.2f}, R={df_baseline['base_action_9'].iloc[t_idx]:+.2f}")
    print(f"  Hierarchical: L={df_hierarchical['base_action_4'].iloc[t_idx]:+.2f}, R={df_hierarchical['base_action_9'].iloc[t_idx]:+.2f}")

    print(f"\nVMC corrections:")
    print(f"  Baseline:     height={df_baseline['vmc_height_correction'].iloc[t_idx]:+.3f}, com={df_baseline['vmc_com_correction'].iloc[t_idx]:+.3f}")
    print(f"  Hierarchical: height={df_hierarchical['vmc_height_correction'].iloc[t_idx]:+.3f}, com={df_hierarchical['vmc_com_correction'].iloc[t_idx]:+.3f}")

    print(f"\n{'='*60}")
    print("ROOT CAUSE HYPOTHESIS")
    print(f"{'='*60}")
    print("1. Hierarchical controller generates large wheel commands immediately (5.83 rad/s)")
    print("2. Baseline controller uses 0.0 wheel command at same state")
    print("3. Roll rate explodes in hierarchical (7.07 vs 2.66 rad/s)")
    print("4. Actions saturate immediately in hierarchical (20% saturation)")
    print("\nLikely issues:")
    print("- LQR gains too aggressive (especially pitch/pitch_rate)")
    print("- Sign error in wheel command or pitch feedback")
    print("- VMC corrections adding to instability")
    print("- No wheel command filtering working properly")

if __name__ == "__main__":
    main()
