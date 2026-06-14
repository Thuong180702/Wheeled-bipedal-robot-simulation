"""Diagnostic script to test lateral control authority via hip_roll.

Tests whether hip_roll commands actually affect torso roll angle and lateral stability.
Logs commanded actions, actual joint positions, torques, and torso roll over time.
"""

import numpy as np
import mujoco
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wheeled_biped.utils.config import get_model_path
from wheeled_biped.controllers.lqr_ik_prior import create_lqr_ik_prior


def run_open_loop_test(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    action: np.ndarray,
    duration_s: float = 2.0,
    dt: float = 0.01,
) -> dict:
    """Run open-loop test with fixed action command.

    Args:
        model: MuJoCo model
        data: MuJoCo data (will be reset)
        action: Normalized action [-1, 1]^10 to apply
        duration_s: Test duration in seconds
        dt: Control timestep

    Returns:
        Dict with logged signals over time
    """
    # Reset to standing pose
    mujoco.mj_resetData(model, data)

    # Joint limits for denormalization
    joint_mins = np.array([
        -0.7, -0.4, -0.5, -0.5, -20.0,  # left leg
        -0.7, -0.4, -0.5, -0.5, -20.0,  # right leg
    ])
    joint_maxs = np.array([
        0.7, 0.4, 1.8, 2.7, 20.0,  # left leg
        0.7, 0.4, 1.8, 2.7, 20.0,  # right leg
    ])

    # PID gains (from low_level_pid.yaml)
    kp = 100.0
    kd = 10.0

    # Wheel mask (1 for wheels, 0 for legs)
    wheel_mask = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

    # Denormalize action to joint targets
    pos_target = joint_mins + (action + 1.0) * 0.5 * (joint_maxs - joint_mins)
    vel_target_whl = action * 20.0  # wheel velocity limit

    # Logging arrays
    n_steps = int(duration_s / dt)
    log = {
        "time": np.zeros(n_steps),
        "torso_roll_deg": np.zeros(n_steps),
        "torso_pitch_deg": np.zeros(n_steps),
        "torso_z": np.zeros(n_steps),
        "l_hip_roll_cmd": np.zeros(n_steps),
        "r_hip_roll_cmd": np.zeros(n_steps),
        "l_hip_roll_pos": np.zeros(n_steps),
        "r_hip_roll_pos": np.zeros(n_steps),
        "l_hip_roll_torque": np.zeros(n_steps),
        "r_hip_roll_torque": np.zeros(n_steps),
        "contact_left": np.zeros(n_steps),
        "contact_right": np.zeros(n_steps),
    }

    # Run simulation
    for step in range(n_steps):
        # Get current joint state
        joint_pos = data.qpos[7:17]
        joint_vel = data.qvel[6:16]

        # Compute PID control
        pos_err = pos_target - joint_pos
        error = (1.0 - wheel_mask) * pos_err + wheel_mask * (vel_target_whl - joint_vel)
        d_error = (1.0 - wheel_mask) * (-joint_vel)

        ctrl = kp * error + kd * d_error
        ctrl = np.clip(ctrl, -100.0, 100.0)  # torque limits

        # Apply control
        data.ctrl[:] = ctrl

        # Step simulation
        mujoco.mj_step(model, data)

        # Log signals
        log["time"][step] = step * dt

        # Torso orientation (from quaternion)
        quat = data.qpos[3:7]
        # Convert to roll/pitch (approximate for small angles)
        # roll = atan2(2*(w*x + y*z), 1 - 2*(x^2 + y^2))
        # pitch = asin(2*(w*y - z*x))
        w, x, y, z = quat
        roll_rad = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
        pitch_rad = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))

        log["torso_roll_deg"][step] = np.degrees(roll_rad)
        log["torso_pitch_deg"][step] = np.degrees(pitch_rad)
        log["torso_z"][step] = data.qpos[2]

        # Hip roll commands and positions
        log["l_hip_roll_cmd"][step] = pos_target[0]
        log["r_hip_roll_cmd"][step] = pos_target[5]
        log["l_hip_roll_pos"][step] = joint_pos[0]
        log["r_hip_roll_pos"][step] = joint_pos[5]

        # Hip roll torques
        log["l_hip_roll_torque"][step] = ctrl[0]
        log["r_hip_roll_torque"][step] = ctrl[5]

        # Contact forces (sum of all contact forces on left/right feet)
        # This is approximate - would need to identify foot geoms
        log["contact_left"][step] = 0.0  # placeholder
        log["contact_right"][step] = 0.0  # placeholder

    return log


def print_test_summary(name: str, log: dict):
    """Print summary statistics for a test."""
    print(f"\n{name}:")
    print(f"  Final torso roll: {log['torso_roll_deg'][-1]:.2f}°")
    print(f"  Max |roll|: {np.max(np.abs(log['torso_roll_deg'])):.2f}°")
    print(f"  Final torso z: {log['torso_z'][-1]:.3f} m")
    print(f"  L hip_roll cmd: {log['l_hip_roll_cmd'][0]:.3f} rad")
    print(f"  R hip_roll cmd: {log['r_hip_roll_cmd'][0]:.3f} rad")
    print(f"  L hip_roll final pos: {log['l_hip_roll_pos'][-1]:.3f} rad")
    print(f"  R hip_roll final pos: {log['r_hip_roll_pos'][-1]:.3f} rad")
    print(f"  L hip_roll RMS torque: {np.sqrt(np.mean(log['l_hip_roll_torque']**2)):.2f} Nm")
    print(f"  R hip_roll RMS torque: {np.sqrt(np.mean(log['r_hip_roll_torque']**2)):.2f} Nm")


def main():
    print("=" * 80)
    print("Lateral Control Authority Diagnostic")
    print("=" * 80)

    # Load model
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Test 1: Neutral standing (all zeros)
    print("\nTest 1: Neutral standing (all zero commands)")
    action_neutral = np.zeros(10)
    log_neutral = run_open_loop_test(model, data, action_neutral, duration_s=2.0)
    print_test_summary("Neutral", log_neutral)

    # Test 2: Symmetric positive hip_roll (both hips roll outward)
    print("\nTest 2: Symmetric positive hip_roll (+0.3 normalized)")
    action_sym_pos = np.zeros(10)
    action_sym_pos[0] = 0.3  # l_hip_roll
    action_sym_pos[5] = 0.3  # r_hip_roll
    log_sym_pos = run_open_loop_test(model, data, action_sym_pos, duration_s=2.0)
    print_test_summary("Symmetric +hip_roll", log_sym_pos)

    # Test 3: Symmetric negative hip_roll (both hips roll inward)
    print("\nTest 3: Symmetric negative hip_roll (-0.3 normalized)")
    action_sym_neg = np.zeros(10)
    action_sym_neg[0] = -0.3  # l_hip_roll
    action_sym_neg[5] = -0.3  # r_hip_roll
    log_sym_neg = run_open_loop_test(model, data, action_sym_neg, duration_s=2.0)
    print_test_summary("Symmetric -hip_roll", log_sym_neg)

    # Test 4: Antisymmetric hip_roll (left out, right in)
    print("\nTest 4: Antisymmetric hip_roll (L=+0.3, R=-0.3)")
    action_antisym_lr = np.zeros(10)
    action_antisym_lr[0] = 0.3   # l_hip_roll
    action_antisym_lr[5] = -0.3  # r_hip_roll
    log_antisym_lr = run_open_loop_test(model, data, action_antisym_lr, duration_s=2.0)
    print_test_summary("Antisymmetric L+/R-", log_antisym_lr)

    # Test 5: Antisymmetric hip_roll (left in, right out)
    print("\nTest 5: Antisymmetric hip_roll (L=-0.3, R=+0.3)")
    action_antisym_rl = np.zeros(10)
    action_antisym_rl[0] = -0.3  # l_hip_roll
    action_antisym_rl[5] = 0.3   # r_hip_roll
    log_antisym_rl = run_open_loop_test(model, data, action_antisym_rl, duration_s=2.0)
    print_test_summary("Antisymmetric L-/R+", log_antisym_rl)

    # Test 6: LQR/IK prior at nominal height
    print("\nTest 6: LQR/IK prior controller at h=0.65m")
    config_path = Path(__file__).parent.parent / "configs" / "controllers" / "gain_scheduled_lqr.yaml"
    controller = create_lqr_ik_prior(config_path, model)

    # Reset and create fake observation
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Build observation (42-dim)
    from wheeled_biped.utils.math_utils import get_gravity_in_body_frame
    quat = data.qpos[3:7]
    g_body = get_gravity_in_body_frame(quat)
    body_ang_vel = data.qvel[3:6]
    body_lin_vel = data.qvel[0:3]
    qpos = data.qpos[7:17]
    qvel = data.qvel[6:16]
    prev_action = np.zeros(10)
    height_cmd = 0.65
    current_height = data.qpos[2]
    yaw_error = 0.0

    obs = np.concatenate([
        g_body, body_ang_vel, body_lin_vel,
        qpos, qvel, prev_action,
        [height_cmd, current_height, yaw_error]
    ])

    # Get action from controller
    action_lqr = controller.compute_action(obs)
    log_lqr = run_open_loop_test(model, data, action_lqr, duration_s=2.0)
    print_test_summary("LQR/IK prior", log_lqr)

    print("\n" + "=" * 80)
    print("Diagnostic complete")
    print("=" * 80)

    # Summary interpretation
    print("\nInterpretation:")
    print("- If neutral standing is stable (roll < 5°), initial pose is OK")
    print("- If symmetric commands don't change roll much, hip_roll has weak authority")
    print("- If antisymmetric commands create large roll, sign convention is correct")
    print("- If LQR/IK prior fails quickly, check commanded hip_roll values")


if __name__ == "__main__":
    main()
