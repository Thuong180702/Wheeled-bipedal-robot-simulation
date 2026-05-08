"""Create summary CSV and diagnostic plots for Phase B.8 Task 2."""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def create_summary_csv(output_dir: Path):
    """Create summary.csv with key metrics for all runs."""

    summary_rows = []

    for csv_file in sorted(output_dir.glob("telemetry_*.csv")):
        df = pd.read_csv(csv_file)

        # Parse filename
        parts = csv_file.stem.split("_")
        if "hierarchical" in csv_file.stem:
            controller = "hierarchical_vmc_lqr"
            height_idx = parts.index("lqr") + 1
        else:
            controller = "height_scheduled_dynamic_lqr"
            height_idx = parts.index("lqr") + 1

        height_str = parts[height_idx]
        episode_str = parts[-1]
        height = float(height_str[1:])
        episode = int(episode_str[2:])

        # Compute metrics
        survival_time = df['time'].iloc[-1]
        fell = df['fell'].iloc[-1]

        # First 5 timesteps
        df_early = df.head(5)
        max_pitch_rate = df_early['pitch_rate'].abs().max()
        max_roll_rate = df_early['roll_rate'].abs().max()
        max_wheel_cmd = df_early['raw_wheel_cmd_l'].abs().max()
        mean_saturation = df_early['action_saturation_rate'].mean()

        # Full episode
        mean_pitch = df['pitch'].abs().mean()
        mean_roll = df['roll'].abs().mean()
        max_com_error = df['com_y_error'].abs().max()

        summary_rows.append({
            'controller': controller,
            'height': height,
            'episode': episode,
            'survival_time_s': survival_time,
            'fell': fell,
            'max_pitch_rate_early': max_pitch_rate,
            'max_roll_rate_early': max_roll_rate,
            'max_wheel_cmd_early': max_wheel_cmd,
            'mean_saturation_early': mean_saturation,
            'mean_pitch_abs': mean_pitch,
            'mean_roll_abs': mean_roll,
            'max_com_error': max_com_error,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(['controller', 'height', 'episode'])

    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Created {summary_path}")

    return summary_df

def create_diagnostic_plots(output_dir: Path):
    """Create time-series diagnostic plots."""

    # Compare h=0.70, ep=0 for both controllers
    baseline_path = output_dir / "telemetry_height_scheduled_dynamic_lqr_h0.70_ep0.csv"
    hierarchical_path = output_dir / "telemetry_hierarchical_vmc_lqr_h0.70_ep0.csv"

    df_baseline = pd.read_csv(baseline_path)
    df_hierarchical = pd.read_csv(hierarchical_path)

    fig, axes = plt.subplots(5, 1, figsize=(12, 15))

    # Plot 1: Pitch
    axes[0].plot(df_baseline['time'], np.rad2deg(df_baseline['pitch']),
                 label='Baseline', linewidth=2)
    axes[0].plot(df_hierarchical['time'], np.rad2deg(df_hierarchical['pitch']),
                 label='Hierarchical', linewidth=2)
    axes[0].set_ylabel('Pitch (deg)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Phase B.8 Diagnostic: Controller Comparison (h=0.70m)')

    # Plot 2: Height tracking
    axes[1].plot(df_baseline['time'], df_baseline['height_actual'],
                 label='Baseline', linewidth=2)
    axes[1].plot(df_hierarchical['time'], df_hierarchical['height_actual'],
                 label='Hierarchical', linewidth=2)
    axes[1].axhline(0.70, color='k', linestyle='--', alpha=0.5, label='Command')
    axes[1].set_ylabel('Height (m)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: CoM error
    axes[2].plot(df_baseline['time'], df_baseline['com_y_error'],
                 label='Baseline', linewidth=2)
    axes[2].plot(df_hierarchical['time'], df_hierarchical['com_y_error'],
                 label='Hierarchical', linewidth=2)
    axes[2].set_ylabel('CoM Error (m)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Plot 4: Wheel command
    axes[3].plot(df_baseline['time'], df_baseline['raw_wheel_cmd_l'],
                 label='Baseline (raw)', linewidth=2)
    axes[3].plot(df_hierarchical['time'], df_hierarchical['raw_wheel_cmd_l'],
                 label='Hierarchical (raw)', linewidth=2)
    axes[3].plot(df_hierarchical['time'], df_hierarchical['filtered_wheel_cmd_l'],
                 label='Hierarchical (filtered)', linewidth=2, linestyle='--')
    axes[3].set_ylabel('Wheel Cmd (rad/s)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    # Plot 5: Action saturation
    axes[4].plot(df_baseline['time'], df_baseline['action_saturation_rate'],
                 label='Baseline', linewidth=2)
    axes[4].plot(df_hierarchical['time'], df_hierarchical['action_saturation_rate'],
                 label='Hierarchical', linewidth=2)
    axes[4].set_ylabel('Saturation Rate')
    axes[4].set_xlabel('Time (s)')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "diagnostic_comparison.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Created {plot_path}")
    plt.close()

def main():
    output_dir = Path("outputs/phase_b8_diagnostics")

    print("Creating Phase B.8 Task 2 summary and plots...")

    summary_df = create_summary_csv(output_dir)

    print("\nSummary statistics:")
    print(summary_df.groupby('controller')[['survival_time_s', 'max_wheel_cmd_early',
                                             'mean_saturation_early']].mean())

    create_diagnostic_plots(output_dir)

    print("\nPhase B.8 Task 2 complete.")
    print(f"Outputs in: {output_dir}")

if __name__ == "__main__":
    main()
