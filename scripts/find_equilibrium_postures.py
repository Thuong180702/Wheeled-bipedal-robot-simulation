"""Phase B.9 Task 5: Find static equilibrium postures at each height.

For each commanded height, find joint configurations where:
1. Torso height matches the command
2. CoM is balanced over wheel contact (zero pitch moment)
3. Joint torques are minimal (gravity-compensated)
4. Configuration is kinematically valid

This reveals whether the IK targets are physically reasonable as static poses.
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import jax.numpy as jnp
import mujoco
import numpy as np
from rich.console import Console
from rich.table import Table
from scipy.optimize import minimize

from wheeled_biped.utils.config import get_model_path

console = Console()


def compute_static_metrics(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    hip_pitch: float,
    knee: float,
    target_height: float,
) -> Dict[str, float]:
    """Compute static equilibrium metrics for a given leg configuration.

    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data (will be modified)
        hip_pitch: Hip pitch angle [rad]
        knee: Knee angle [rad]
        target_height: Desired torso height [m]

    Returns:
        Dictionary of metrics:
        - height_error: |actual_height - target_height| [m]
        - com_offset_y: CoM y-position relative to wheel contact [m]
        - pitch_moment: Torque needed to prevent pitch rotation [Nm]
        - joint_torques_rms: RMS of joint torques [Nm]
        - is_valid: Whether configuration is kinematically valid
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
    actual_height = float(mj_data.qpos[2])
    height_error = abs(actual_height - target_height)

    # Get CoM position
    mujoco.mj_subtreeVel(mj_model, mj_data)
    mujoco.mj_comPos(mj_model, mj_data)
    com_pos = mj_data.subtree_com[0]  # Root body's subtree COM

    # Get wheel contact positions (average of left and right)
    l_wheel_id = mj_model.body("l_wheel_link").id
    r_wheel_id = mj_model.body("r_wheel_link").id
    l_wheel_pos = mj_data.xpos[l_wheel_id]
    r_wheel_pos = mj_data.xpos[r_wheel_id]
    wheel_contact_y = (l_wheel_pos[1] + r_wheel_pos[1]) / 2.0

    # CoM offset from wheel contact (sagittal plane)
    com_offset_y = float(com_pos[1] - wheel_contact_y)

    # Compute pitch moment (torque needed to prevent rotation)
    # Moment = mass * g * com_offset_y
    total_mass = np.sum(mj_model.body_mass)
    gravity = 9.81
    pitch_moment = abs(total_mass * gravity * com_offset_y)

    # Compute joint torques needed for static equilibrium
    # Set qacc=0 (static), compute inverse dynamics
    mj_data.qacc[:] = 0.0
    mujoco.mj_inverse(mj_model, mj_data)
    joint_torques = mj_data.qfrc_inverse[6:16]  # 10 actuated joints
    joint_torques_rms = float(np.sqrt(np.mean(joint_torques**2)))

    # Check kinematic validity
    is_valid = True
    for i, jname in enumerate(["l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee", "l_wheel",
                                "r_hip_roll", "r_hip_yaw", "r_hip_pitch", "r_knee", "r_wheel"]):
        jid = mj_model.joint(jname).id
        jrange = mj_model.jnt_range[jid]
        qpos_val = mj_data.qpos[7 + i]
        if qpos_val < jrange[0] or qpos_val > jrange[1]:
            is_valid = False
            break

    return {
        "height_error": height_error,
        "com_offset_y": com_offset_y,
        "pitch_moment": pitch_moment,
        "joint_torques_rms": joint_torques_rms,
        "is_valid": is_valid,
        "actual_height": actual_height,
    }


def find_equilibrium(
    mj_model: mujoco.MjModel,
    target_height: float,
    initial_guess: Tuple[float, float] = None,
) -> Dict[str, float]:
    """Find static equilibrium configuration for a given height.

    Args:
        mj_model: MuJoCo model
        target_height: Desired torso height [m]
        initial_guess: Optional (hip_pitch, knee) starting point [rad]

    Returns:
        Dictionary with equilibrium configuration and metrics
    """
    mj_data = mujoco.MjData(mj_model)

    # Get joint limits
    hip_pitch_jid = mj_model.joint("l_hip_pitch").id
    knee_jid = mj_model.joint("l_knee").id
    hip_pitch_range = mj_model.jnt_range[hip_pitch_jid]
    knee_range = mj_model.jnt_range[knee_jid]

    # Initial guess: if not provided, use mid-range
    if initial_guess is None:
        x0 = np.array([
            (hip_pitch_range[0] + hip_pitch_range[1]) / 2.0,
            (knee_range[0] + knee_range[1]) / 2.0,
        ])
    else:
        x0 = np.array(initial_guess)

    # Objective: minimize weighted sum of errors
    def objective(x):
        hip_pitch, knee = x
        metrics = compute_static_metrics(mj_model, mj_data, hip_pitch, knee, target_height)

        if not metrics["is_valid"]:
            return 1e6  # Large penalty for invalid configs

        # Weighted objective
        w_height = 100.0  # Height error is critical
        w_com = 10.0      # CoM balance is critical
        w_torque = 0.1    # Minimize joint torques (secondary)

        cost = (
            w_height * metrics["height_error"]**2 +
            w_com * metrics["com_offset_y"]**2 +
            w_torque * metrics["joint_torques_rms"]**2
        )
        return cost

    # Bounds
    bounds = [
        (hip_pitch_range[0], hip_pitch_range[1]),
        (knee_range[0], knee_range[1]),
    ]

    # Optimize
    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-8},
    )

    # Extract solution
    hip_pitch_eq, knee_eq = result.x
    metrics = compute_static_metrics(mj_model, mj_data, hip_pitch_eq, knee_eq, target_height)

    return {
        "hip_pitch": hip_pitch_eq,
        "knee": knee_eq,
        "height_error": metrics["height_error"],
        "actual_height": metrics["actual_height"],
        "com_offset_y": metrics["com_offset_y"],
        "pitch_moment": metrics["pitch_moment"],
        "joint_torques_rms": metrics["joint_torques_rms"],
        "is_valid": metrics["is_valid"],
        "optimization_success": result.success,
        "optimization_cost": result.fun,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Find static equilibrium postures at each height (Phase B.9 Task 5)"
    )
    parser.add_argument(
        "--heights",
        type=float,
        nargs="+",
        default=[0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40],
        help="Heights to evaluate [m]",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase_b9_task5_equilibrium"),
        help="Output directory",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold cyan]Phase B.9 Task 5: Find Static Equilibrium Postures[/bold cyan]\n")

    # Load model
    model_path = get_model_path()
    mj_model = mujoco.MjModel.from_xml_path(str(model_path))

    # Find equilibrium for each height
    results = []

    for height in args.heights:
        console.print(f"Finding equilibrium at h={height:.2f}m...")

        # Use previous solution as initial guess (warm start)
        initial_guess = None
        if results:
            prev = results[-1]
            initial_guess = (prev["hip_pitch"], prev["knee"])

        eq = find_equilibrium(mj_model, height, initial_guess)
        eq["target_height"] = height
        results.append(eq)

        console.print(
            f"  hip_pitch={eq['hip_pitch']:.3f} rad, knee={eq['knee']:.3f} rad, "
            f"height_err={eq['height_error']:.4f}m, com_offset={eq['com_offset_y']:.4f}m"
        )

    # Display results table
    table = Table(title="Static Equilibrium Postures")
    table.add_column("Height [m]", justify="right")
    table.add_column("Hip Pitch [rad]", justify="right")
    table.add_column("Knee [rad]", justify="right")
    table.add_column("Height Error [m]", justify="right")
    table.add_column("CoM Offset [m]", justify="right")
    table.add_column("Pitch Moment [Nm]", justify="right")
    table.add_column("Torque RMS [Nm]", justify="right")
    table.add_column("Valid", justify="center")

    for r in results:
        table.add_row(
            f"{r['target_height']:.2f}",
            f"{r['hip_pitch']:.3f}",
            f"{r['knee']:.3f}",
            f"{r['height_error']:.4f}",
            f"{r['com_offset_y']:.4f}",
            f"{r['pitch_moment']:.2f}",
            f"{r['joint_torques_rms']:.2f}",
            "Y" if r["is_valid"] else "N",
        )

    console.print(table)

    # Save results
    import json
    output_file = args.output_dir / "equilibrium_postures.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[green]Results saved to: {output_file}[/green]")

    # Compare with current IK targets
    console.print("\n[bold cyan]Comparison with Current IK Targets[/bold cyan]\n")

    # Load current IK from height_scheduled_dynamic_lqr config
    import yaml
    config_path = Path("configs/controllers/height_scheduled_dynamic_lqr.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        ik_heights = config.get("ik", {}).get("heights", [])
        ik_hip_pitch = config.get("ik", {}).get("hip_pitch_targets", [])
        ik_knee = config.get("ik", {}).get("knee_targets", [])

        if ik_heights and ik_hip_pitch and ik_knee:
            comp_table = Table(title="IK vs Equilibrium Comparison")
            comp_table.add_column("Height [m]", justify="right")
            comp_table.add_column("IK Hip Pitch", justify="right")
            comp_table.add_column("Eq Hip Pitch", justify="right")
            comp_table.add_column("Δ Hip [rad]", justify="right")
            comp_table.add_column("IK Knee", justify="right")
            comp_table.add_column("Eq Knee", justify="right")
            comp_table.add_column("Δ Knee [rad]", justify="right")

            for r in results:
                h = r["target_height"]
                # Find closest IK height
                idx = min(range(len(ik_heights)), key=lambda i: abs(ik_heights[i] - h))

                if abs(ik_heights[idx] - h) < 0.01:  # Match within 1cm
                    ik_hp = ik_hip_pitch[idx]
                    ik_kn = ik_knee[idx]
                    eq_hp = r["hip_pitch"]
                    eq_kn = r["knee"]

                    comp_table.add_row(
                        f"{h:.2f}",
                        f"{ik_hp:.3f}",
                        f"{eq_hp:.3f}",
                        f"{eq_hp - ik_hp:+.3f}",
                        f"{ik_kn:.3f}",
                        f"{eq_kn:.3f}",
                        f"{eq_kn - ik_kn:+.3f}",
                    )

            console.print(comp_table)

    console.print("\n[bold green]Phase B.9 Task 5 complete![/bold green]")


if __name__ == "__main__":
    main()
