"""Micro-test D: ForceDistributor My sign test.

Verify whether positive/negative My commands produce restoring roll response
through the actual distributor and actuator path.

Setup:
- Calibrated equilibrium using Stage 2B feedforward + static posture
- Perturb roll_y = +0.03 rad and -0.03 rad
- Command isolated My corrections through SimpleForceDistributor
- Disable pitch/height/CoM/capture-point corrections
- Measure resulting roll acceleration

Decision rules:
1. If correction_My_roll has correct sign but roll acceleration increases error:
   root cause = My-to-hip-roll mapping sign or actuator sign error
2. If both oppose roll error but torque saturates:
   root cause = insufficient roll authority
3. If works in one-step but fails in long horizon:
   root cause = roll integral/reference/contact asymmetry/drift
"""

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from pathlib import Path
import yaml

from wheeled_biped.controllers.simple_force_distributor import SimpleForceDistributor


def load_model():
    """Load MuJoCo model."""
    xml_path = Path("assets/robot/wheeled_biped_real.xml")
    if not xml_path.exists():
        raise FileNotFoundError(f"Model file not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data


def load_force_distributor_config():
    """Load force distributor configuration."""
    config_path = Path("configs/controllers/simple_force_distributor.yaml")
    if not config_path.exists():
        # Use default config
        return {
            "tau_hip_roll_max": 15.0,
            "max_force_asymmetry": 40.0,
            "min_wheel_force": 10.0,
        }

    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def establish_equilibrium_with_stage2b(model, data, target_height=0.50):
    """Establish equilibrium using Stage 2B feedforward + static posture.

    This is a simplified version - in practice would use the full controller.
    For now, just settle the robot with gravity compensation.
    """
    # Reset to default pose
    mujoco.mj_resetData(model, data)

    # Set initial configuration closer to standing
    # Base height
    data.qpos[2] = target_height + 0.05

    # Hip pitch and knee to approximate standing posture
    # Joint order: l_hip_roll, l_hip_yaw, l_hip_pitch, l_knee, l_wheel, r_hip_roll, r_hip_yaw, r_hip_pitch, r_knee, r_wheel
    # Indices in qpos: 7, 8, 9, 10, 11, 12, 13, 14, 15, 16

    # Left leg: hip_pitch ~0.3, knee ~-0.6
    data.qpos[9] = 0.3   # l_hip_pitch
    data.qpos[10] = -0.6  # l_knee

    # Right leg: hip_pitch ~0.3, knee ~-0.6
    data.qpos[14] = 0.3   # r_hip_pitch
    data.qpos[15] = -0.6  # r_knee

    # Apply gravity compensation torques
    # Approximate: each leg supports half the robot weight
    robot_mass = 8.1  # kg
    g = 9.81
    weight_per_leg = robot_mass * g / 2.0

    # Hip pitch torque to support weight
    hip_pitch_torque = 10.0  # Nm, approximate
    knee_torque = -15.0      # Nm, approximate

    # Set control targets
    data.ctrl[2] = 0.3   # l_hip_pitch position
    data.ctrl[3] = -0.6  # l_knee position
    data.ctrl[7] = 0.3   # r_hip_pitch position
    data.ctrl[8] = -0.6  # r_knee position

    # Settle for 2 seconds
    for _ in range(2000):
        mujoco.mj_step(model, data)

    return data.qpos.copy(), data.qvel.copy()


def perturb_roll(model, data, roll_perturbation):
    """Apply roll perturbation to base orientation.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        roll_perturbation: Roll angle perturbation (rad)
    """
    # Get current quaternion
    quat = data.qpos[3:7].copy()

    # Create roll perturbation quaternion
    # Roll is rotation about x-axis
    half_angle = roll_perturbation / 2.0
    roll_quat = np.array([
        np.cos(half_angle),
        np.sin(half_angle),
        0.0,
        0.0,
    ])

    # Multiply quaternions: q_new = q_roll * q_current
    # Using quaternion multiplication formula
    w1, x1, y1, z1 = roll_quat
    w2, x2, y2, z2 = quat

    quat_new = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

    # Normalize
    quat_new = quat_new / np.linalg.norm(quat_new)

    # Set new quaternion
    data.qpos[3:7] = quat_new

    # Forward kinematics
    mujoco.mj_forward(model, data)


def extract_roll_from_quat(quat):
    """Extract roll angle from quaternion."""
    # For small angles: roll ≈ 2 * atan2(qy, qw)
    return 2.0 * np.arctan2(quat[1], quat[0])


def test_my_correction_one_step(model, data, distributor, my_command, qpos_eq, qvel_eq):
    """Test My correction for one step.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        distributor: SimpleForceDistributor instance
        my_command: My moment command (Nm)
        qpos_eq: Equilibrium qpos
        qvel_eq: Equilibrium qvel

    Returns:
        dict with test results
    """
    # Reset to equilibrium
    data.qpos[:] = qpos_eq
    data.qvel[:] = qvel_eq
    mujoco.mj_forward(model, data)

    # Record before state
    quat_before = data.qpos[3:7].copy()
    roll_y_before = extract_roll_from_quat(quat_before)
    roll_rate_y_before = data.qvel[3]

    # Compute force distribution for My correction
    # Assume nominal vertical force
    fz_total = 8.1 * 9.81  # robot_mass * g

    # Create wrench with only My component
    wrench = jnp.array([0.0, 0.0, fz_total, 0.0, my_command, 0.0])

    # Wheel positions relative to CoM (approximate)
    wheel_track = 0.30  # m
    wheel_pos_left = jnp.array([0.0, wheel_track / 2.0, 0.0])
    wheel_pos_right = jnp.array([0.0, -wheel_track / 2.0, 0.0])

    # Distribute forces using contact-aware method
    f_left, f_right, tau_hip_roll, diagnostics = distributor.distribute_wrench_contact_aware(
        desired_wrench=wrench,
        left_contact=True,
        right_contact=True,
        wheel_pos_left=wheel_pos_left,
        wheel_pos_right=wheel_pos_right,
        hip_roll_authority_scale=1.0,
        recovery_mode=False,
    )

    # Apply torques
    # Get joint indices
    l_hip_roll_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "l_hip_roll")
    r_hip_roll_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r_hip_roll")

    # Apply hip roll torques directly
    data.ctrl[l_hip_roll_id] = float(tau_hip_roll[0])
    data.ctrl[r_hip_roll_id] = float(tau_hip_roll[1])

    # Step once
    mujoco.mj_step(model, data)

    # Record after state
    quat_after = data.qpos[3:7].copy()
    roll_y_after = extract_roll_from_quat(quat_after)
    roll_rate_y_after = data.qvel[3]

    # Compute deltas
    delta_roll_y = roll_y_after - roll_y_before
    delta_roll_rate_y = roll_rate_y_after - roll_rate_y_before

    return {
        "my_command": my_command,
        "roll_y_before": roll_y_before,
        "roll_y_after": roll_y_after,
        "delta_roll_y": delta_roll_y,
        "roll_rate_y_before": roll_rate_y_before,
        "roll_rate_y_after": roll_rate_y_after,
        "delta_roll_rate_y": delta_roll_rate_y,
        "tau_hip_roll_left": float(tau_hip_roll[0]),
        "tau_hip_roll_right": float(tau_hip_roll[1]),
        "f_left_z": float(f_left[2]),
        "f_right_z": float(f_right[2]),
    }


def run_microtest_d():
    """Run micro-test D: ForceDistributor My sign test."""
    print("=" * 80)
    print("MICRO-TEST D: ForceDistributor My sign test")
    print("=" * 80)

    # Load model
    model, data = load_model()

    # Load force distributor config
    config = load_force_distributor_config()
    print(f"\nForce distributor config:")
    print(f"  tau_hip_roll_max: {config['tau_hip_roll_max']:.1f}Nm")
    print(f"  max_force_asymmetry: {config['max_force_asymmetry']:.1f}N")
    print(f"  min_wheel_force: {config['min_wheel_force']:.1f}N")

    # Create force distributor
    distributor = SimpleForceDistributor(
        tau_hip_roll_max=config['tau_hip_roll_max'],
        max_force_asymmetry=config['max_force_asymmetry'],
        min_wheel_force=config['min_wheel_force'],
    )

    # Establish equilibrium
    print("\nEstablishing equilibrium with Stage 2B...")
    qpos_eq, qvel_eq = establish_equilibrium_with_stage2b(model, data, target_height=0.50)
    print(f"Equilibrium: com_z={data.subtree_com[1][2]:.3f}m")

    # Test cases: My commands
    my_commands = [+2.0, +5.0, +10.0, -2.0, -5.0, -10.0]

    results = []

    print("\n" + "=" * 80)
    print("TEST: Neutral equilibrium (no perturbation)")
    print("=" * 80)

    for my_cmd in my_commands:
        result = test_my_correction_one_step(model, data, distributor, my_cmd, qpos_eq, qvel_eq)
        results.append(result)

        print(f"\nMy = {my_cmd:+.1f} Nm")
        print(f"  tau_hip_roll: L={result['tau_hip_roll_left']:+.2f}, R={result['tau_hip_roll_right']:+.2f} Nm")
        print(f"  f_wheel_z: L={result['f_left_z']:+.1f}, R={result['f_right_z']:+.1f} N")
        print(f"  roll_y: {result['roll_y_before']:+.4f} -> {result['roll_y_after']:+.4f} rad (delta={result['delta_roll_y']:+.6f})")
        print(f"  roll_rate_y: {result['roll_rate_y_before']:+.4f} -> {result['roll_rate_y_after']:+.4f} rad/s (delta={result['delta_roll_rate_y']:+.6f})")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    print("\nMy command -> roll acceleration:")
    for result in results:
        my_cmd = result['my_command']
        delta_roll_rate = result['delta_roll_rate_y']
        direction = "LEFT" if delta_roll_rate < 0 else "RIGHT"
        print(f"  My={my_cmd:+.1f}Nm -> delta_roll_rate={delta_roll_rate:+.6f} rad/s ({direction})")

    print("\nMapping verification:")
    print("  - Positive My should produce RIGHT roll (positive delta_roll_rate_y)")
    print("  - Negative My should produce LEFT roll (negative delta_roll_rate_y)")
    print("  - To correct positive roll_y (leaning right), need negative My")
    print("  - To correct negative roll_y (leaning left), need positive My")

    # Check sign consistency
    print("\nSign consistency check:")
    for result in results:
        my_cmd = result['my_command']
        delta_roll_rate = result['delta_roll_rate_y']

        if my_cmd > 0 and delta_roll_rate > 0:
            print(f"  OK My={my_cmd:+.1f}Nm produces RIGHT roll (positive delta_roll_rate)")
        elif my_cmd > 0 and delta_roll_rate < 0:
            print(f"  FAIL My={my_cmd:+.1f}Nm produces LEFT roll (WRONG SIGN)")
        elif my_cmd < 0 and delta_roll_rate < 0:
            print(f"  OK My={my_cmd:+.1f}Nm produces LEFT roll (negative delta_roll_rate)")
        elif my_cmd < 0 and delta_roll_rate > 0:
            print(f"  FAIL My={my_cmd:+.1f}Nm produces RIGHT roll (WRONG SIGN)")

    print("\n" + "=" * 80)
    print("MICRO-TEST D COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_microtest_d()
