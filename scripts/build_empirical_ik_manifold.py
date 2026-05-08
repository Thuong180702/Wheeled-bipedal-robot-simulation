"""Phase B.9 Task 6: Build empirical IK manifold via forward kinematics sweep.

Instead of analytical IK or simulation, systematically sweep the joint space
and use forward kinematics to find which configurations achieve which heights.

Approach:
1. Sample hip_pitch and knee over their valid ranges
2. For each configuration, compute forward kinematics to get height
3. Check stability: CoM over base, joint limits, reasonable posture
4. Build empirical mapping: height → (hip_pitch, knee)
5. Fit interpolator for new IK module
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


def compute_height_and_stability(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    hip_pitch: float,
    knee: float,
) -> Dict[str, float]:
    """Compute height and stability metrics for a joint configuration.

    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data (will be modified)
        hip_pitch: Hip pitch angle [rad]
        knee: Knee angle [rad]

    Returns:
        Dictionary with height and stability metrics
    """
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

    # Forward kinematics
    mujoco.mj_forward(mj_model, mj_data)

    # Get torso height
    height = float(mj_data.qpos[2])

    # Get CoM position
    mujoco.mj_subtreeVel(mj_model, mj_data)
    mujoco.mj_comPos(mj_model, mj_data)
    com_pos = mj_data.subtree_com[0]

    # Get wheel contact positions
    l_wheel_id = mj_model.body("l_wheel_link").id
    r_wheel_id = mj_model.body("r_wheel_link").id
    l_wheel_pos = mj_data.xpos[l_wheel_id]
    r_wheel_pos = mj_data.xpos[r_wheel_id]
    wheel_contact_y = (l_wheel_pos[1] + r_wheel_pos[1]) / 2.0

    # CoM offset from wheel contact
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

    return {
        "height": height,
        "com_offset_y": com_offset_y,
        "stability_score": stability_score,
        "is_valid": is_valid,
    }


def sweep_joint_space(
    mj_model: mujoco.MjModel,
    hip_pitch_samples: int = 50,
    knee_samples: int = 50,
) -> List[Dict[str, float]]:
    """Sweep joint space and record height for each configuration.

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

    # Sample joint space
    hip_pitch_values = np.linspace(hip_pitch_range[0], hip_pitch_range[1], hip_pitch_samples)
    knee_values = np.linspace(knee_range[0], knee_range[1], knee_samples)

    configurations = []

    total = hip_pitch_samples * knee_samples
    for hip_pitch in track(hip_pitch_values, description="Sweeping joint space"):
        for knee in knee_values:
            metrics = compute_height_and_stability(mj_model, mj_data, hip_pitch, knee)

            config = {
                "hip_pitch": float(hip_pitch),
                "knee": float(knee),
                "height": metrics["height"],
                "com_offset_y": metrics["com_offset_y"],
                "stability_score": metrics["stability_score"],
                "is_valid": metrics["is_valid"],
            }
            configurations.append(config)

    return configurations


def build_ik_interpolator(
    configurations: List[Dict[str, float]],
    stability_threshold: float = 0.5,
) -> Tuple[interp1d, interp1d]:
    """Build interpolators for height → (hip_pitch, knee) mapping.

    Args:
        configurations: List of configuration dictionaries
        stability_threshold: Minimum stability score to include

    Returns:
        (hip_pitch_interpolator, knee_interpolator) tuple
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

    return hip_pitch_interp, knee_interp


def main():
    parser = argparse.ArgumentParser(
        description="Build empirical IK manifold via forward kinematics (Phase B.9 Task 6)"
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
        default=Path("outputs/phase_b9_task6_empirical_ik"),
        help="Output directory",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold cyan]Phase B.9 Task 6: Build Empirical IK Manifold[/bold cyan]\n")

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
    console.print(f"[yellow]Achievable height range: [{min_height:.3f}, {max_height:.3f}] m[/yellow]")

    # Build interpolators
    hip_pitch_interp, knee_interp = build_ik_interpolator(valid_configs, args.stability_threshold)

    # Test interpolators at target heights
    test_heights = [0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40]
    table = Table(title="Empirical IK Targets")
    table.add_column("Target Height [m]", justify="right")
    table.add_column("Achievable", justify="center")
    table.add_column("Hip Pitch [rad]", justify="right")
    table.add_column("Knee [rad]", justify="right")

    ik_targets = []
    for h in test_heights:
        if min_height <= h <= max_height:
            hip_pitch = float(hip_pitch_interp(h))
            knee = float(knee_interp(h))
            achievable = "Y"
        else:
            # Clamp to achievable range
            h_clamped = np.clip(h, min_height, max_height)
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
            "min_height": min_height,
            "max_height": max_height,
            "targets": ik_targets,
        }, f, indent=2)
    console.print(f"\n[green]Saved IK targets to: {ik_file}[/green]")

    # Compare with old IK
    console.print("\n[bold cyan]Comparison with Old IK[/bold cyan]\n")
    console.print("Old IK assumed robot could squat to any height (0.40-0.70m)")
    console.print(f"Empirical IK shows achievable range: [{min_height:.3f}, {max_height:.3f}] m")
    console.print(f"Heights below {min_height:.3f}m are KINEMATICALLY IMPOSSIBLE")

    console.print("\n[bold green]Phase B.9 Task 6 complete![/bold green]")
    console.print("\n[yellow]Next steps:[/yellow]")
    console.print("1. Replace old IK module with empirical interpolators")
    console.print("2. Add height clamping to prevent infeasible commands")
    console.print("3. Re-evaluate controller with corrected IK")


if __name__ == "__main__":
    main()
