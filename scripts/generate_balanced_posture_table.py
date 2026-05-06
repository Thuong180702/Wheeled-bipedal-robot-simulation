"""Generate human-like knee-forward, CoM-aware balanced posture table.

This script optimizes postures for each height in the grid to satisfy:
1. Torso height approximately matches height_cmd
2. Both knees bend forward relative to hip (human-like)
3. Whole-body CoM is aligned with wheel contact (balanced)
4. Left and right legs remain symmetric
5. Joint limits are respected

The optimization uses scipy.optimize.minimize with soft constraints.
"""

import numpy as np
import mujoco
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import Tuple
import yaml
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from wheeled_biped.utils.config import get_model_path


@dataclass
class PostureSolution:
    """Solution for a single height."""
    height_cmd: float
    hip_pitch: float
    knee: float
    torso_pitch_bias: float
    knee_forward_margin_left: float
    knee_forward_margin_right: float
    com_y_error: float
    height_error: float
    torso_z: float
    feasible: bool
    objective_value: float


def get_body_com(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Compute whole-body center of mass position."""
    total_mass = 0.0
    com_weighted = np.zeros(3)

    for i in range(model.nbody):
        body_mass = model.body_mass[i]
        if body_mass > 0:
            body_com = data.xipos[i]
            com_weighted += body_mass * body_com
            total_mass += body_mass

    return com_weighted / total_mass if total_mass > 0 else np.zeros(3)


def evaluate_posture(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hip_pitch: float,
    knee: float,
    height_cmd: float,
    torso_pitch_bias: float = 0.0,
) -> Tuple[float, float, float, float, float, float]:
    """Evaluate a candidate posture.

    Returns:
        (torso_z, knee_fwd_l, knee_fwd_r, com_y_error, wheel_contact_y, com_y)
    """
    # Reset and set posture
    mujoco.mj_resetData(model, data)

    # Start with base at origin
    data.qpos[0:3] = [0, 0, 0.5]  # Initial guess for base height

    # Apply torso pitch bias if any
    if abs(torso_pitch_bias) > 1e-6:
        # Convert pitch bias to quaternion
        quat = np.array([
            np.cos(torso_pitch_bias / 2),
            0,
            np.sin(torso_pitch_bias / 2),
            0
        ])
        quat = quat / np.linalg.norm(quat)
        data.qpos[3:7] = quat
    else:
        data.qpos[3:7] = [1, 0, 0, 0]

    # Symmetric leg posture
    data.qpos[7:17] = [
        0, 0, hip_pitch, knee, 0,  # left leg
        0, 0, hip_pitch, knee, 0,  # right leg
    ]

    # Forward kinematics to get initial wheel positions
    mujoco.mj_forward(model, data)

    # Get wheel body IDs
    l_wheel_body_id = model.body("l_wheel_link").id
    r_wheel_body_id = model.body("r_wheel_link").id

    # Find lowest wheel Z position
    l_wheel_z = data.xpos[l_wheel_body_id][2]
    r_wheel_z = data.xpos[r_wheel_body_id][2]
    min_wheel_z = min(l_wheel_z, r_wheel_z)

    # Adjust base Z so lowest wheel is at ground level (Z=0)
    data.qpos[2] -= min_wheel_z

    # Forward kinematics with corrected base height
    mujoco.mj_forward(model, data)

    # Get body IDs
    torso_body_id = model.body("torso").id
    l_hip_yaw_body_id = model.body("l_hip_yaw_link").id
    l_knee_link_body_id = model.body("l_knee_link").id
    l_wheel_body_id = model.body("l_wheel_link").id
    r_hip_yaw_body_id = model.body("r_hip_yaw_link").id
    r_knee_link_body_id = model.body("r_knee_link").id
    r_wheel_body_id = model.body("r_wheel_link").id

    # Body positions
    torso_pos = data.xpos[torso_body_id]
    l_hip_pos = data.xpos[l_hip_yaw_body_id]
    l_knee_pos = data.xpos[l_knee_link_body_id]
    l_wheel_pos = data.xpos[l_wheel_body_id]
    r_hip_pos = data.xpos[r_hip_yaw_body_id]
    r_knee_pos = data.xpos[r_knee_link_body_id]
    r_wheel_pos = data.xpos[r_wheel_body_id]

    # Compute whole-body CoM
    com_pos = get_body_com(model, data)

    # Measurements in world frame (Y-axis is sagittal/forward-backward)
    torso_z = torso_pos[2]
    wheel_contact_y = (l_wheel_pos[1] + r_wheel_pos[1]) / 2.0

    # Knee-forward margin (positive = knee forward of hip)
    knee_forward_margin_l = l_knee_pos[1] - l_hip_pos[1]
    knee_forward_margin_r = r_knee_pos[1] - r_hip_pos[1]

    # CoM error (positive = CoM forward of wheels)
    com_y_error = com_pos[1] - wheel_contact_y

    return (
        torso_z,
        knee_forward_margin_l,
        knee_forward_margin_r,
        com_y_error,
        wheel_contact_y,
        com_pos[1],
    )


def get_height_dependent_margin_min(height: float) -> float:
    """Get minimum knee-forward margin for a given height.

    Taller postures (near standing) can have smaller forward bend.
    Lower postures (deep crouch) should have more forward bend.
    """
    if height >= 0.68:
        return 0.01  # near standing, minimal forward bend
    elif height >= 0.60:
        return 0.04  # moderate crouch
    elif height >= 0.50:
        return 0.07  # low crouch
    else:
        return 0.09  # very low crouch


def objective_function(
    x: np.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    height_cmd: float,
    weights: dict,
) -> float:
    """Objective function for posture optimization.

    Args:
        x: [hip_pitch, knee, torso_pitch_bias]
        model: MuJoCo model
        data: MuJoCo data
        height_cmd: Target height
        weights: Weight dictionary

    Returns:
        Objective value (lower is better)
    """
    hip_pitch, knee, torso_pitch_bias = x

    # Get joint limits
    hip_pitch_min, hip_pitch_max = -0.5, 1.8
    knee_min, knee_max = -0.5, 2.7

    # Hard constraint violations (large penalty)
    penalty = 0.0

    # Joint limit violations
    if hip_pitch < hip_pitch_min or hip_pitch > hip_pitch_max:
        penalty += 1000.0 * (
            max(0, hip_pitch_min - hip_pitch)**2 +
            max(0, hip_pitch - hip_pitch_max)**2
        )

    if knee < knee_min or knee > knee_max:
        penalty += 1000.0 * (
            max(0, knee_min - knee)**2 +
            max(0, knee - knee_max)**2
        )

    # Torso pitch bias should be moderate
    if abs(torso_pitch_bias) > 0.5:  # ~29 degrees
        penalty += 1000.0 * (abs(torso_pitch_bias) - 0.5)**2

    # Evaluate posture
    try:
        torso_z, knee_fwd_l, knee_fwd_r, com_y_error, _, _ = evaluate_posture(
            model, data, hip_pitch, knee, height_cmd, torso_pitch_bias
        )
    except Exception:
        return 1e6 + penalty

    # Height error
    height_error = torso_z - height_cmd
    cost_height = weights['w_height'] * height_error**2

    # CoM alignment error
    cost_com = weights['w_com'] * com_y_error**2

    # Knee-forward constraint
    margin_min = get_height_dependent_margin_min(height_cmd)
    knee_backward_penalty_l = max(0, margin_min - knee_fwd_l)**2
    knee_backward_penalty_r = max(0, margin_min - knee_fwd_r)**2
    cost_knee_forward = weights['w_knee_forward'] * (
        knee_backward_penalty_l + knee_backward_penalty_r
    )

    # Torso pitch bias regularization
    cost_torso_pitch = weights['w_torso_pitch'] * torso_pitch_bias**2

    # Joint regularization (prefer moderate joint angles)
    hip_pitch_center = 0.65  # mid-range
    knee_center = 1.1  # mid-range
    cost_joint_reg = weights['w_joint_reg'] * (
        (hip_pitch - hip_pitch_center)**2 +
        (knee - knee_center)**2
    )

    # Left-right symmetry (should be perfect by construction, but check)
    cost_symmetry = weights['w_symmetry'] * (knee_fwd_l - knee_fwd_r)**2

    total_cost = (
        cost_height +
        cost_com +
        cost_knee_forward +
        cost_torso_pitch +
        cost_joint_reg +
        cost_symmetry +
        penalty
    )

    return total_cost


def optimize_posture(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    height_cmd: float,
    weights: dict,
    initial_guess: np.ndarray = None,
) -> PostureSolution:
    """Optimize posture for a given height.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        height_cmd: Target height
        weights: Weight dictionary
        initial_guess: Initial guess [hip_pitch, knee, torso_pitch_bias]

    Returns:
        PostureSolution
    """
    from scipy.optimize import minimize

    # Bounds
    bounds = [
        (-0.5, 1.8),  # hip_pitch
        (-0.5, 2.7),  # knee
        (-0.5, 0.5),  # torso_pitch_bias (relaxed to allow more CoM adjustment)
    ]

    # Try multiple initial guesses to escape local minima
    best_result = None
    best_cost = float('inf')

    # Initial guess 1: From previous solution or geometric IK
    if initial_guess is None:
        hip_pitch_init = max(-0.5, min(1.8, (0.70 - height_cmd) * 4.0))
        knee_init = max(-0.5, min(2.7, 2.0 * hip_pitch_init))
        torso_pitch_bias_init = 0.0
        x0 = np.array([hip_pitch_init, knee_init, torso_pitch_bias_init])
    else:
        x0 = initial_guess

    # Try initial guess
    result = minimize(
        objective_function,
        x0,
        args=(model, data, height_cmd, weights),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-6},
    )
    if result.fun < best_cost:
        best_cost = result.fun
        best_result = result

    # Try with forward torso lean
    x0_forward = x0.copy()
    x0_forward[2] = 0.2  # Forward lean
    result = minimize(
        objective_function,
        x0_forward,
        args=(model, data, height_cmd, weights),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-6},
    )
    if result.fun < best_cost:
        best_cost = result.fun
        best_result = result

    # Try with backward torso lean
    x0_backward = x0.copy()
    x0_backward[2] = -0.2  # Backward lean
    result = minimize(
        objective_function,
        x0_backward,
        args=(model, data, height_cmd, weights),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-6},
    )
    if result.fun < best_cost:
        best_cost = result.fun
        best_result = result

    result = best_result

    # Extract solution
    hip_pitch_opt, knee_opt, torso_pitch_bias_opt = result.x

    # Evaluate final posture
    torso_z, knee_fwd_l, knee_fwd_r, com_y_error, _, _ = evaluate_posture(
        model, data, hip_pitch_opt, knee_opt, height_cmd, torso_pitch_bias_opt
    )

    height_error = torso_z - height_cmd

    # Check feasibility
    margin_min = get_height_dependent_margin_min(height_cmd)
    knee_forward_ok = (knee_fwd_l >= margin_min * 0.5 and knee_fwd_r >= margin_min * 0.5)
    com_ok = abs(com_y_error) < 0.03  # 3 cm tolerance
    height_ok = abs(height_error) < 0.05  # 5 cm tolerance
    feasible = knee_forward_ok and com_ok and height_ok and result.success

    return PostureSolution(
        height_cmd=height_cmd,
        hip_pitch=float(hip_pitch_opt),
        knee=float(knee_opt),
        torso_pitch_bias=float(torso_pitch_bias_opt),
        knee_forward_margin_left=float(knee_fwd_l),
        knee_forward_margin_right=float(knee_fwd_r),
        com_y_error=float(com_y_error),
        height_error=float(height_error),
        torso_z=float(torso_z),
        feasible=feasible,
        objective_value=float(result.fun),
    )


def main():
    print("=" * 80)
    print("Balanced Posture Table Generator")
    print("=" * 80)

    # Load model
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Height grid (exploring achievable range)
    # Lower heights (0.40-0.55m) appear kinematically infeasible with knee-forward + CoM balance
    height_grid = [0.75, 0.70, 0.65, 0.60, 0.55]

    # Optimization weights
    # Prioritize CoM balance and knee-forward over exact height tracking
    weights = {
        'w_height': 50.0,  # Relaxed - accept height deviations for better balance
        'w_com': 500.0,  # Critical for balance
        'w_knee_forward': 500.0,  # Critical for human-like posture
        'w_torso_pitch': 1.0,
        'w_joint_reg': 1.0,
        'w_symmetry': 100.0,
    }

    print("\nOptimization weights:")
    for key, val in weights.items():
        print(f"  {key}: {val}")

    print("\nOptimizing postures...")
    print("-" * 80)

    solutions = []
    prev_solution = None

    for height_cmd in height_grid:
        print(f"\nHeight {height_cmd:.2f}m:")

        # Use previous solution as initial guess for smoothness
        initial_guess = None
        if prev_solution is not None:
            initial_guess = np.array([
                prev_solution.hip_pitch,
                prev_solution.knee,
                prev_solution.torso_pitch_bias,
            ])

        solution = optimize_posture(
            model, data, height_cmd, weights, initial_guess
        )
        solutions.append(solution)
        prev_solution = solution

        print(f"  hip_pitch: {solution.hip_pitch:.4f} rad ({np.degrees(solution.hip_pitch):.2f}°)")
        print(f"  knee:      {solution.knee:.4f} rad ({np.degrees(solution.knee):.2f}°)")
        print(f"  torso_pitch_bias: {solution.torso_pitch_bias:.4f} rad ({np.degrees(solution.torso_pitch_bias):.2f}°)")
        print(f"  knee_forward_margin_left:  {solution.knee_forward_margin_left:.4f} m")
        print(f"  knee_forward_margin_right: {solution.knee_forward_margin_right:.4f} m")
        print(f"  com_y_error: {solution.com_y_error:.4f} m")
        print(f"  height_error: {solution.height_error:.4f} m")
        print(f"  torso_z: {solution.torso_z:.4f} m")
        print(f"  objective: {solution.objective_value:.2f}")
        print(f"  feasible: {solution.feasible}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    feasible_count = sum(1 for s in solutions if s.feasible)
    print(f"\nFeasible solutions: {feasible_count}/{len(solutions)}")

    knee_forward_count = sum(
        1 for s in solutions
        if s.knee_forward_margin_left > 0 and s.knee_forward_margin_right > 0
    )
    print(f"Knee-forward postures: {knee_forward_count}/{len(solutions)}")

    avg_com_error = np.mean([abs(s.com_y_error) for s in solutions])
    max_com_error = max([abs(s.com_y_error) for s in solutions])
    print(f"\nCoM alignment:")
    print(f"  Average |CoM error|: {avg_com_error:.4f} m")
    print(f"  Maximum |CoM error|: {max_com_error:.4f} m")

    avg_height_error = np.mean([abs(s.height_error) for s in solutions])
    max_height_error = max([abs(s.height_error) for s in solutions])
    print(f"\nHeight tracking:")
    print(f"  Average |height error|: {avg_height_error:.4f} m")
    print(f"  Maximum |height error|: {max_height_error:.4f} m")

    # Save to YAML
    output_path = Path(__file__).parent.parent / "configs" / "controllers" / "balanced_posture_table.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    posture_table = {
        'posture_table_type': 'human_like_knee_forward_com_aware',
        'forward_axis': 'Y (sagittal, world frame)',
        'generated_date': datetime.now().strftime('%Y-%m-%d'),
        'optimization_weights': weights,
        'height_grid_m': height_grid,
        'entries': {}
    }

    for sol in solutions:
        posture_table['entries'][f"{sol.height_cmd:.2f}"] = {
            'hip_pitch': float(sol.hip_pitch),
            'knee': float(sol.knee),
            'torso_pitch_bias': float(sol.torso_pitch_bias),
            'knee_forward_margin_left': float(sol.knee_forward_margin_left),
            'knee_forward_margin_right': float(sol.knee_forward_margin_right),
            'com_y_error': float(sol.com_y_error),
            'height_error': float(sol.height_error),
            'torso_z': float(sol.torso_z),
            'feasible': bool(sol.feasible),
            'objective_value': float(sol.objective_value),
        }

    with open(output_path, 'w') as f:
        yaml.dump(posture_table, f, default_flow_style=False, sort_keys=False)

    print(f"\nSaved posture table to: {output_path}")

    # Save diagnostics
    diag_dir = Path(__file__).parent.parent / "outputs" / "diagnostics" / "posture_geometry"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = diag_dir / "posture_table_diagnostics.csv"
    with open(csv_path, 'w') as f:
        f.write("height,hip_pitch,knee,torso_pitch_bias,knee_fwd_l,knee_fwd_r,com_y_error,height_error,feasible\n")
        for sol in solutions:
            f.write(f"{sol.height_cmd:.2f},{sol.hip_pitch:.4f},{sol.knee:.4f},"
                   f"{sol.torso_pitch_bias:.4f},{sol.knee_forward_margin_left:.4f},"
                   f"{sol.knee_forward_margin_right:.4f},{sol.com_y_error:.4f},"
                   f"{sol.height_error:.4f},{int(sol.feasible)}\n")

    print(f"Saved diagnostics CSV to: {csv_path}")

    # Summary text
    summary_path = diag_dir / "posture_table_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Balanced Posture Table Generation Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Feasible solutions: {feasible_count}/{len(solutions)}\n")
        f.write(f"Knee-forward postures: {knee_forward_count}/{len(solutions)}\n\n")
        f.write(f"CoM alignment:\n")
        f.write(f"  Average |CoM error|: {avg_com_error:.4f} m\n")
        f.write(f"  Maximum |CoM error|: {max_com_error:.4f} m\n\n")
        f.write(f"Height tracking:\n")
        f.write(f"  Average |height error|: {avg_height_error:.4f} m\n")
        f.write(f"  Maximum |height error|: {max_height_error:.4f} m\n\n")
        f.write("Detailed Results:\n")
        f.write("-" * 80 + "\n")
        for sol in solutions:
            f.write(f"\nHeight {sol.height_cmd:.2f}m:\n")
            f.write(f"  hip_pitch: {sol.hip_pitch:.4f} rad ({np.degrees(sol.hip_pitch):.2f}°)\n")
            f.write(f"  knee: {sol.knee:.4f} rad ({np.degrees(sol.knee):.2f}°)\n")
            f.write(f"  torso_pitch_bias: {sol.torso_pitch_bias:.4f} rad ({np.degrees(sol.torso_pitch_bias):.2f}°)\n")
            f.write(f"  knee_forward_margin_left: {sol.knee_forward_margin_left:.4f} m\n")
            f.write(f"  knee_forward_margin_right: {sol.knee_forward_margin_right:.4f} m\n")
            f.write(f"  com_y_error: {sol.com_y_error:.4f} m\n")
            f.write(f"  height_error: {sol.height_error:.4f} m\n")
            f.write(f"  feasible: {sol.feasible}\n")

    print(f"Saved summary to: {summary_path}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
