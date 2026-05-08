"""Phase B.9 Task 6 (corrected): Build empirical IK manifold via contact-aware FK.

Previous version bug: measured fixed root z instead of actual achievable height.

Corrected approach:
1. Set hip_pitch and knee joint angles
2. Call mj_forward to compute wheel positions
3. Find lowest wheel z-coordinate
4. Shift root z so lowest wheel touches ground (z=0)
5. Call mj_forward again
6. Measure torso height (root z after ground contact adjustment)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import mujoco
import numpy as np
from rich.console import Console
from rich.progress import track
from rich.table import Table
from scipy.interpolate import interp1d

from wheeled_biped.utils.config import get_model_path

console = Console()


def compute_height_with_ground_contact(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    hip_pitch: float,
    knee: float,
) -> Dict[str, float]:
    """Compute achievable height with wheels on ground.

    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data (will be modified)
        hip_pitch: Hip pitch angle [rad]
        knee: Knee angle [rad]

    Returns:
        Dictionary with height and stability metrics
    """
    # Set root pose upright at arbitrary height
    mj_data.qpos[0] = 0.0  # x
    mj_data.qpos[1] = 0.0  # y
    mj_data.qpos[2] = 1.0  # z (arbitrary, will be corrected)
    mj_data.qpos[3] = 1.0  # qw
    mj_data.qpos[4] = 0.0  # qx
    mj_data.qpos[5] = 0.0  # qy
    mj_data.qpos[6] = 0.0  # qz

    # Set symmetric leg configuration
    mj_data.qpos[7] = 0.0  # l_hip_roll
    mj_data.qpos[8] = 0.0  # l_hip_yaw
    mj_data.qpos[9] = hip_pitch  # l_hip_pitch
    mj_data.qpos[10] = knee  # l_knee
    mj_data.qpos[11] = 0.0  # l_wheel

    mj_data.qpos[12] = 0.0  # r_hip_roll
    mj_data.qpos[13] = 0.0  # r_hip_yaw
    mj_data.qpos[14] = hip_pitch  # r_hip_pitch
    mj_data.qpos[15] = knee  # r_knee
    mj_data.qpos[16] = 0.0  # r_wheel

    # Zero velocities
    mj_data.qvel[:] = 0.0

    # Forward kinematics to get wheel positions
    mujoco.mj_forward(mj_model, mj_data)

    # Get wheel body positions
    l_wheel_id = mj_model.body("l_wheel_link").id
    r_wheel_id = mj_model.body("r_wheel_link").id
    l_wheel_pos = mj_data.xpos[l_wheel_id].copy()
    r_wheel_pos = mj_data.xpos[r_wheel_id].copy()

    # Find lowest wheel z (should touch ground)
    lowest_wheel_z = min(l_wheel_pos[2], r_wheel_pos[2])

    # Shift root z so lowest wheel is at ground level (z=0)
    # Assume wheel radius is small or wheel center should be at ground
    # For more accuracy, could subtract wheel radius
    root_z_correction = -lowest_wheel_z
    mj_data.qpos[2] += root_z_correction

    # Recompute FK with corrected root z
    mujoco.mj_forward(mj_model, mj_data)

    # Now measure torso height (root z after ground contact)
    torso_height = float(mj_data.qpos[2])

    # Get CoM position
    mujoco.mj_subtreeVel(mj_model, mj_data)
    mujoco.mj_comPos(mj_model, mj_data)
    com_pos = mj_data.subtree_com[0]

    # Get wheel contact positions (should now be at ground)
    l_wheel_pos = mj_data.xpos[l_wheel_id]
    r_wheel_pos = mj_data.xpos[r_wheel_id]
    wheel_contact_y = (l_wheel_pos[1] + r_wheel_pos[1]) / 2.0

    # CoM offset from wheel contact (sagittal plane)
    com_offset_y = float(com_pos[1] - wheel_contact_y)

    # Check joint limits
    is_valid = True
    for i, jname in enumerate(["l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee", "l_wheel",
                                "r_hip_roll", "r_hip_yaw", "r_hip_pitch", "r_knee", "r_wheel"]):
        jid = mj_model.joint(jname).id
        jrange = mj_model.jnt_range[jid]
        qpos_val = mj_data.qpos[7 + i]
        if qpos_val < jrange[0] or qpos_val > jrange[1]:
            is_valid = False
            break

    # Stability score: prefer CoM near wheel contact
    stability_score = 1.0 / (1.0 + abs(com_offset_y) * 10.0)

    # Sanity check: wheel positions after correction
    wheel_z_min = min(l_wheel_pos[2], r_wheel_pos[2])
    wheel_z_max = max(l_wheel_pos[2], r_wheel_pos[2])

    return {
        "height": torso_height,
        "com_offset_y": com_offset_y,
        "stability_score": stability_score,
        "is_valid": is_valid,
        "wheel_z_min": float(wheel_z_min),
        "wheel_z_max": float(wheel_z_max),
    }


def sweep_joint_space(
    mj_model: mujoco.MjModel,
    hip_pitch_samples: int = 50,
    knee_samples: int = 50,
) -> List[Dict[str, float]]:
    """Sweep joint space with contact-aware FK.

    Args:
        mj_model: MuJoCo model
        hip_pitch_samples: Number of hip pitch samples
        knee_samples: Number of knee samples

    Returns:
        List of configuration dictionaries
    """
    mj_data = mujoco.MjData(mj_model)

    # Get joint limits
    hip_pitch_jid = mj_model.joint("l_hip_pitch").id
    knee_jid = mj_model.joint("l_knee").id
    hip_pitch_range = mj_model.jnt_range[hip_pitch_jid]
    knee_range = mj_model.jnt_range[knee_jid]

    # Constrain to standing configurations (positive angles only)
    # Negative hip_pitch = leaning backward, not suitable for standing balance
    hip_pitch_min = max(0.0, hip_pitch_range[0])
    hip_pitch_max = min(1.5, hip_pitch_range[1])
    knee_min = max(0.0, knee_range[0])
    knee_max = min(2.5, knee_range[1])

    # Sample joint space
    hip_pitch_values = np.linspace(hip_pitch_min, hip_pitch_max, hip_pitch_samples)
    knee_values = np.linspace(knee_min, knee_max, knee_samples)

    configurations = []

    for hip_pitch in track(hip_pitch_values, description="Sweeping joint space"):
        for knee in knee_values:
            metrics = compute_height_with_ground_contact(mj_model, mj_data, hip_pitch, knee)

            config = {
                "hip_pitch": float(hip_pitch),
                "knee": float(knee),
                "height": metrics["height"],
                "com_offset_y": metrics["com_offset_y"],
                "stability_score": metrics["stability_score"],
                "is_valid": metrics["is_valid"],
                "wheel_z_min": metrics["wheel_z_min"],
                "wheel_z_max": metrics["wheel_z_max"],
            }
            configurations.append(config)

    return configurations


def build_ik_interpolator(
    configurations: List[Dict[str, float]],
    stability_threshold: float = 0.5,
) -> Tuple[interp1d, interp1d, float, float]:
    """Build interpolators for height → (hip_pitch, knee) mapping.

    Args:
        configurations: List of configuration dictionaries
        stability_threshold: Minimum stability score to include

    Returns:
        (hip_pitch_interpolator, knee_interpolator, min_height, max_height)
    """
    # Filter to valid and stable configurations
    valid_configs = [
        c for c in configurations
        if c["is_valid"] and c["stability_score"] >= stability_threshold
    ]

    if not valid_configs:
        raise ValueError("No valid configurations found!")

    # Sort by height
    valid_configs.sort(key=lambda c: c["height"])

    # Extract arrays
    heights = np.array([c["height"] for c in valid_configs])
    hip_pitches = np.array([c["hip_pitch"] for c in valid_configs])
    knees = np.array([c["knee"] for c in valid_configs])

    # Remove duplicate heights (keep first occurrence)
    unique_heights, unique_indices = np.unique(heights, return_index=True)
    unique_hip_pitches = hip_pitches[unique_indices]
    unique_knees = knees[unique_indices]

    min_height = float(unique_heights[0])
    max_height = float(unique_heights[-1])

    # Build interpolators
    hip_pitch_interp = interp1d(
        unique_heights,
        unique_hip_pitches,
        kind="linear",
        bounds_error=False,
        fill_value=(unique_hip_pitches[0], unique_hip_pitches[-1]),
    )

    knee_interp = interp1d(
        unique_heights,
        unique_knees,
        kind="linear",
        bounds_error=False,
        fill_value=(unique_knees[0], unique_knees[-1]),
    )

    return hip_pitch_interp, knee_interp, min_height, max_height


def main():
    parser = argparse.ArgumentParser(
        description="Build empirical IK manifold via contact-aware FK (Phase B.9 Task 6 corrected)"
    )
    parser.add_argument(
        "--hip-pitch-samples",
        type=int,
        default=50,
        help="Number of hip pitch samples",
    )
    parser.add_argument(
        "--knee-samples",
        type=int,
        default=50,
        help="Number of knee samples",
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=0.5,
        help="Minimum stability score",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase_b9_task6_empirical_ik_corrected"),
        help="Output directory",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold cyan]Phase B.9 Task 6 (Corrected): Build Empirical IK Manifold[/bold cyan]\n")
    console.print("[yellow]Using contact-aware FK: adjusting root z for ground contact[/yellow]\n")

    # Load model
    model_path = get_model_path()
    mj_model = mujoco.MjModel.from_xml_path(str(model_path))

    # Sweep joint space
    console.print(f"Sweeping {args.hip_pitch_samples} x {args.knee_samples} = "
                  f"{args.hip_pitch_samples * args.knee_samples} configurations...")
    configurations = sweep_joint_space(mj_model, args.hip_pitch_samples, args.knee_samples)

    # Save raw configurations
    raw_file = args.output_dir / "raw_configurations.json"
    with open(raw_file, "w") as f:
        json.dump(configurations, f, indent=2)
    console.print(f"[green]Saved raw configurations to: {raw_file}[/green]")

    # Filter to valid and stable
    valid_configs = [
        c for c in configurations
        if c["is_valid"] and c["stability_score"] >= args.stability_threshold
    ]
    console.print(f"\n[yellow]Valid configurations: {len(valid_configs)}/{len(configurations)}[/yellow]")

    if not valid_configs:
        console.print("[red]No valid configurations found! Try lowering stability threshold.[/red]")
        return

    # Analyze height range
    heights = [c["height"] for c in valid_configs]
    min_height = min(heights)
    max_height = max(heights)
    height_std = np.std(heights)
    console.print(f"[yellow]Achievable height range: [{min_height:.3f}, {max_height:.3f}] m[/yellow]")
    console.print(f"[yellow]Height std dev: {height_std:.4f} m[/yellow]")

    # Build interpolators
    hip_pitch_interp, knee_interp, interp_min_h, interp_max_h = build_ik_interpolator(
        valid_configs, args.stability_threshold
    )

    # Test interpolators at target heights
    test_heights = [0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40]
    table = Table(title="Empirical IK Targets (Contact-Aware)")
    table.add_column("Target Height [m]", justify="right")
    table.add_column("Achievable", justify="center")
    table.add_column("Hip Pitch [rad]", justify="right")
    table.add_column("Knee [rad]", justify="right")

    ik_targets = []
    for h in test_heights:
        if interp_min_h <= h <= interp_max_h:
            hip_pitch = float(hip_pitch_interp(h))
            knee = float(knee_interp(h))
            achievable = "Y"
        else:
            # Clamp to achievable range
            h_clamped = np.clip(h, interp_min_h, interp_max_h)
            hip_pitch = float(hip_pitch_interp(h_clamped))
            knee = float(knee_interp(h_clamped))
            achievable = "N (clamped)"

        table.add_row(
            f"{h:.2f}",
            achievable,
            f"{hip_pitch:.3f}",
            f"{knee:.3f}",
        )

        ik_targets.append({
            "target_height": h,
            "achievable": achievable == "Y",
            "hip_pitch": hip_pitch,
            "knee": knee,
        })

    console.print(table)

    # Save IK targets
    ik_file = args.output_dir / "empirical_ik_targets.json"
    with open(ik_file, "w") as f:
        json.dump({
            "min_height": interp_min_h,
            "max_height": interp_max_h,
            "targets": ik_targets,
        }, f, indent=2)
    console.print(f"\n[green]Saved IK targets to: {ik_file}[/green]")

    # Compare with static equilibrium
    console.print("\n[bold cyan]Comparison with Static Equilibrium (Task 5)[/bold cyan]\n")
    target_hip = 0.256
    target_knee = 0.538
    close_configs = [
        c for c in valid_configs
        if abs(c["hip_pitch"] - target_hip) < 0.05
        and abs(c["knee"] - target_knee) < 0.05
    ]
    if close_configs:
        console.print(f"[green]Found {len(close_configs)} configs near static equilibrium[/green]")
        console.print(f"  Static equilibrium: hip={target_hip:.3f}, knee={target_knee:.3f}, height~0.71m")
        for c in close_configs[:3]:
            console.print(f"  Empirical: hip={c['hip_pitch']:.3f}, knee={c['knee']:.3f}, "
                          f"height={c['height']:.3f}m, stability={c['stability_score']:.3f}")

    console.print("\n[bold green]Phase B.9 Task 6 (corrected) complete![/bold green]")


if __name__ == "__main__":
    main()
