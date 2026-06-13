#!/usr/bin/env python3
"""
Render offline video replays for low_0p300 visual inspection.

Attempts to use MuJoCo offscreen renderer to create video files without requiring
live OpenGL viewer window.

Outputs:
- baseline_low_0p300.mp4
- J2_low_0p300.mp4
- J3_low_0p300.mp4

If offscreen rendering fails, falls back to telemetry-based plots.
"""

import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional


OUTPUT_DIR = Path("outputs/visual_inspection_low_0p300")
VIDEO_DIR = OUTPUT_DIR / "videos"
FALLBACK_DIR = OUTPUT_DIR / "replay_fallback"

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
FALLBACK_DIR.mkdir(parents=True, exist_ok=True)


PROFILES = [
    ("baseline", "Baseline (no schedule)"),
    ("J2", "J2 (k_pos=80, k_vel=30)"),
    ("J3", "J3 (k_pos=80, k_vel=25)"),
]


def try_offscreen_video_render(profile: str, profile_label: str) -> bool:
    """
    Attempt to use MuJoCo offscreen rendering to create video.

    Returns True if successful, False if offscreen rendering unavailable.
    """
    print(f"\n[{profile}] Attempting offscreen video render...")

    # MuJoCo python bindings support offscreen rendering via mujoco.Renderer
    # This requires mujoco >= 2.3.0
    try:
        import mujoco
        print(f"  MuJoCo version: {mujoco.__version__}")
    except ImportError:
        print("  [SKIP] MuJoCo python bindings not available")
        return False

    # Check if we can create an offscreen renderer
    try:
        # This would need the actual model XML path
        # Since we don't have easy access to the model here, we'll skip
        # the actual rendering and just document the approach
        print("  [INFO] Offscreen rendering requires model XML and simulation setup")
        print("  [INFO] This would need integration with simulate_hierarchical_controller.py")
        print("  [SKIP] Not implemented in this script - use telemetry fallback")
        return False
    except Exception as e:
        print(f"  [ERROR] Offscreen renderer initialization failed: {e}")
        return False


def create_telemetry_animation(profile: str, profile_label: str) -> Path:
    """
    Create animation from telemetry data showing key metrics over time.
    """
    print(f"\n[{profile}] Creating telemetry-based animation...")

    csv_path = OUTPUT_DIR / f"{profile}_telemetry.csv"
    if not csv_path.exists():
        print(f"  [ERROR] Telemetry file not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    # Extract key metrics
    time = df["sim_time_s"].values
    pitch = df["pitch_x_rad"].values * 57.3  # convert to degrees
    hip_yaw_left = df["l_hip_yaw_pos"].values * 57.3
    hip_yaw_right = df["r_hip_yaw_pos"].values * 57.3
    hip_yaw_max = np.maximum(np.abs(hip_yaw_left), np.abs(hip_yaw_right))
    support_error = df["support_position_error_m"].values
    com_z = df["com_z_m"].values

    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"{profile_label} - low_0p300 Telemetry Animation", fontsize=14, fontweight='bold')

    # Plot 1: Pitch
    ax1 = axes[0]
    ax1.set_ylabel("Pitch (deg)", fontsize=10)
    ax1.axhline(5.7, color='red', linestyle='--', linewidth=1, label='Gate (5.7°)')
    ax1.axhline(-5.7, color='red', linestyle='--', linewidth=1)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-12, 12)
    line1, = ax1.plot([], [], 'b-', linewidth=2, label='Pitch')
    ax1.legend(loc='upper right')

    # Plot 2: Hip Yaw
    ax2 = axes[1]
    ax2.set_ylabel("Hip Yaw (deg)", fontsize=10)
    ax2.axhline(4.0, color='red', linestyle='--', linewidth=1, label='Gate (4.0°)')
    ax2.axhline(-4.0, color='red', linestyle='--', linewidth=1)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-15, 15)
    line2_left, = ax2.plot([], [], 'g-', linewidth=1.5, alpha=0.6, label='Left')
    line2_right, = ax2.plot([], [], 'orange', linewidth=1.5, alpha=0.6, label='Right')
    line2_max, = ax2.plot([], [], 'r-', linewidth=2, label='Max')
    ax2.legend(loc='upper right')

    # Plot 3: Support Error
    ax3 = axes[2]
    ax3.set_ylabel("Support Error (m)", fontsize=10)
    ax3.axhline(0.15, color='red', linestyle='--', linewidth=1, label='Gate (0.15m)')
    ax3.axhline(-0.15, color='red', linestyle='--', linewidth=1)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.3, 0.3)
    line3, = ax3.plot([], [], 'purple', linewidth=2, label='Support')
    ax3.legend(loc='upper right')

    # Plot 4: CoM Height
    ax4 = axes[3]
    ax4.set_ylabel("CoM Height (m)", fontsize=10)
    ax4.set_xlabel("Time (s)", fontsize=10)
    ax4.axhline(0.300, color='red', linestyle='--', linewidth=1, label='Target (0.300m)')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0.25, 0.35)
    line4, = ax4.plot([], [], 'teal', linewidth=2, label='CoM Z')
    ax4.legend(loc='upper right')

    plt.tight_layout()

    # Animation update function
    def update(frame):
        idx = frame
        if idx >= len(time):
            idx = len(time) - 1

        # Update lines
        line1.set_data(time[:idx], pitch[:idx])
        line2_left.set_data(time[:idx], hip_yaw_left[:idx])
        line2_right.set_data(time[:idx], hip_yaw_right[:idx])
        line2_max.set_data(time[:idx], hip_yaw_max[:idx])
        line3.set_data(time[:idx], support_error[:idx])
        line4.set_data(time[:idx], com_z[:idx])

        # Update x-axis limits to follow the data
        if idx > 0:
            for ax in axes:
                ax.set_xlim(0, time[idx] + 0.5)

        return line1, line2_left, line2_right, line2_max, line3, line4

    # Create animation
    num_frames = len(time)
    skip = max(1, num_frames // 200)  # Limit to ~200 frames for reasonable file size

    anim = animation.FuncAnimation(
        fig, update, frames=range(0, num_frames, skip),
        interval=50, blit=True, repeat=True
    )

    # Save as mp4
    output_path = VIDEO_DIR / f"{profile}_low_0p300_telemetry.mp4"

    try:
        print(f"  Saving animation to {output_path}...")
        anim.save(str(output_path), writer='ffmpeg', fps=20, dpi=100)
        print(f"  [OK] Animation saved: {output_path}")
        plt.close(fig)
        return output_path
    except Exception as e:
        print(f"  [ERROR] Failed to save animation: {e}")
        print(f"  [INFO] Trying pillow writer instead...")
        try:
            anim.save(str(output_path.with_suffix('.gif')), writer='pillow', fps=20)
            print(f"  [OK] GIF animation saved: {output_path.with_suffix('.gif')}")
            plt.close(fig)
            return output_path.with_suffix('.gif')
        except Exception as e2:
            print(f"  [ERROR] GIF save also failed: {e2}")
            plt.close(fig)
            return None


def create_static_summary_plot(profile: str, profile_label: str) -> Path:
    """
    Create static plot showing peak frames and key metrics.
    """
    print(f"\n[{profile}] Creating static summary plot...")

    csv_path = OUTPUT_DIR / f"{profile}_telemetry.csv"
    if not csv_path.exists():
        print(f"  [ERROR] Telemetry file not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    # Find peak frames
    pitch_peak_idx = df["pitch_x_rad"].abs().idxmax()
    hip_yaw_peak_idx = df[["l_hip_yaw_pos", "r_hip_yaw_pos"]].abs().max(axis=1).idxmax()
    support_peak_idx = df["support_position_error_m"].abs().idxmax()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{profile_label} - low_0p300 Key Metrics", fontsize=14, fontweight='bold')

    # Plot 1: Pitch over time with peak marked
    ax1 = axes[0, 0]
    time = df["sim_time_s"].values
    pitch = df["pitch_x_rad"].values * 57.3
    ax1.plot(time, pitch, 'b-', linewidth=1.5, label='Pitch')
    ax1.axhline(5.7, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Gate (5.7°)')
    ax1.axhline(-5.7, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax1.scatter([time[pitch_peak_idx]], [pitch[pitch_peak_idx]],
                color='red', s=100, zorder=10, label=f'Peak: {pitch[pitch_peak_idx]:.1f}°')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Pitch (deg)")
    ax1.set_title("Pitch Excursion")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Hip Yaw over time with peak marked
    ax2 = axes[0, 1]
    hip_yaw_left = df["l_hip_yaw_pos"].values * 57.3
    hip_yaw_right = df["r_hip_yaw_pos"].values * 57.3
    hip_yaw_max = np.maximum(np.abs(hip_yaw_left), np.abs(hip_yaw_right))
    ax2.plot(time, hip_yaw_left, 'g-', linewidth=1, alpha=0.6, label='Left')
    ax2.plot(time, hip_yaw_right, 'orange', linewidth=1, alpha=0.6, label='Right')
    ax2.plot(time, hip_yaw_max, 'r-', linewidth=1.5, label='Max')
    ax2.axhline(4.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Gate (4.0°)')
    ax2.axhline(-4.0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax2.scatter([time[hip_yaw_peak_idx]], [hip_yaw_max[hip_yaw_peak_idx]],
                color='red', s=100, zorder=10, label=f'Peak: {hip_yaw_max[hip_yaw_peak_idx]:.1f}°')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Hip Yaw (deg)")
    ax2.set_title("Hip Yaw Excursion")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Support error over time with peak marked
    ax3 = axes[1, 0]
    support_error = df["support_position_error_m"].values
    ax3.plot(time, support_error, 'purple', linewidth=1.5, label='Support Error')
    ax3.axhline(0.15, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Gate (0.15m)')
    ax3.axhline(-0.15, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax3.scatter([time[support_peak_idx]], [support_error[support_peak_idx]],
                color='red', s=100, zorder=10, label=f'Peak: {support_error[support_peak_idx]:.3f}m')
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Support Error (m)")
    ax3.set_title("Support Position Error")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Calculate metrics
    support_max = abs(support_error).max()
    hip_yaw_max_val = hip_yaw_max.max()
    pitch_max_val = abs(pitch).max()

    support_pass = support_max <= 0.15
    hip_yaw_pass = hip_yaw_max_val <= 4.0
    pitch_pass = pitch_max_val <= 5.7

    table_data = [
        ["Metric", "Value", "Gate", "Status"],
        ["Support Error", f"{support_max:.4f} m", "≤0.15 m", "✓ PASS" if support_pass else "✗ FAIL"],
        ["Hip Yaw Max", f"{hip_yaw_max_val:.2f}°", "≤4.0°", "✓ PASS" if hip_yaw_pass else "✗ FAIL"],
        ["Pitch Max", f"{pitch_max_val:.2f}°", "≤5.7°", "✓ PASS" if pitch_pass else "✗ FAIL"],
        ["", "", "", ""],
        ["Steps Survived", f"{len(df)}", "1000", "✓" if len(df) >= 1000 else "✗"],
        ["Non-wheel Contacts", f"{(df['non_wheel_floor_contacts'] > 0).sum()}", "0", "✓" if (df['non_wheel_floor_contacts'] > 0).sum() == 0 else "✗"],
    ]

    table = ax4.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.35, 0.25, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color status cells
    for i in range(1, 4):
        status = table_data[i][3]
        if "PASS" in status:
            table[(i, 3)].set_facecolor('#C8E6C9')
        else:
            table[(i, 3)].set_facecolor('#FFCDD2')

    ax4.set_title("Summary Metrics", fontweight='bold', pad=20)

    plt.tight_layout()

    output_path = FALLBACK_DIR / f"{profile}_low_0p300_summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  [OK] Static plot saved: {output_path}")
    return output_path


def main():
    print("="*80)
    print("low_0p300 Visual Replay Generation")
    print("="*80)
    print()
    print("Attempting to create visual artifacts for posture inspection:")
    print("  1. Try offscreen video rendering (requires MuJoCo renderer)")
    print("  2. Fall back to telemetry-based animations")
    print("  3. Generate static summary plots")
    print()

    results = {}

    for profile, profile_label in PROFILES:
        print("\n" + "="*80)
        print(f"Processing: {profile_label}")
        print("="*80)

        # Try offscreen rendering first
        offscreen_success = try_offscreen_video_render(profile, profile_label)

        if not offscreen_success:
            # Fall back to telemetry animation
            anim_path = create_telemetry_animation(profile, profile_label)
            results[profile] = {"animation": anim_path}

        # Always create static summary
        summary_path = create_static_summary_plot(profile, profile_label)
        if profile in results:
            results[profile]["summary"] = summary_path
        else:
            results[profile] = {"summary": summary_path}

    # Print summary
    print("\n" + "="*80)
    print("Generation Complete")
    print("="*80)
    print()
    print("Generated artifacts:")
    for profile, paths in results.items():
        print(f"\n{profile}:")
        for artifact_type, path in paths.items():
            if path:
                print(f"  {artifact_type}: {path}")
            else:
                print(f"  {artifact_type}: FAILED")

    print()
    print("="*80)
    print("Manual Visual Inspection Commands")
    print("="*80)
    print()
    print("If you have a machine with OpenGL support, run these commands locally:")
    print()

    for profile, profile_label in PROFILES:
        print(f"# {profile_label}")
        print(f"python scripts/simulate_hierarchical_controller.py \\")
        print(f"  --controller-mode balance-core \\")
        print(f"  --sagittal-controller velocity-damped \\")
        print(f"  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \\")
        print(f"  --vd-sagittal-authority-profile {profile} \\")
        print(f"  --steps 1000 \\")
        print(f"  --visual")
        print()

    print("="*80)
    print("OpenGL Troubleshooting")
    print("="*80)
    print()
    print("If you see 'WGL: The driver does not appear to support OpenGL':")
    print()
    print("1. Update GPU driver:")
    print("   - NVIDIA: Download from nvidia.com/drivers")
    print("   - AMD: Download from amd.com/support")
    print("   - Intel: Download from intel.com/content/www/us/en/download-center")
    print()
    print("2. Check display adapter:")
    print("   - Open Device Manager -> Display adapters")
    print("   - If 'Microsoft Basic Display Adapter', GPU driver not installed")
    print()
    print("3. Avoid Remote Desktop:")
    print("   - Remote Desktop disables OpenGL acceleration")
    print("   - Use VNC, TeamViewer, or physical console instead")
    print()
    print("4. Run locally:")
    print("   - WSL/headless environments may not support OpenGL")
    print("   - Need physical or virtual machine with GPU access")
    print()
    print("5. Check OpenGL version:")
    print("   - Run: glxinfo | grep 'OpenGL version' (Linux)")
    print("   - Or use GPU-Z on Windows")
    print("   - MuJoCo requires OpenGL 3.3+")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
