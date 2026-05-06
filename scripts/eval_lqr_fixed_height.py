"""Evaluate LQR/IK prior at fixed heights with proper initialization.

Tests whether the prior can maintain balance when initialized at the target height,
separating fixed-height balance capability from height-transition capability.
"""

import numpy as np
import mujoco
from pathlib import Path
import sys
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from wheeled_biped.utils.config import get_model_path
from wheeled_biped.controllers.lqr_ik_prior import create_lqr_ik_prior
from wheeled_biped.utils.math_utils import get_gravity_in_body_frame


@dataclass
class FixedHeightResult:
    """Results for a single fixed-height test."""
    height_cmd: float
    survival_time: float
    fell: bool
    pitch_rms_deg: float
    roll_rms_deg: float
    height_error_rmse: float
    wheel_speed_rms: float
    wheel_saturation_rate: float
    base_action_rms: float


def initialize_at_height(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller,
    target_height: float,
) -> None:
    """Initialize robot at target height using IK solution.

    Args:
        model: MuJoCo model
        data: MuJoCo data (will be modified)
        controller: LQR/IK controller
        target_height: Target height in meters
    """
    # Get IK solution for target height
    hip_pitch_des, knee_des = controller.height_ik(target_height)

    # Reset data
    mujoco.mj_resetData(model, data)

    # Set base position
    data.qpos[0:3] = [0, 0, target_height]
    data.qpos[3:7] = [1, 0, 0, 0]  # upright quaternion

    # Set joint positions (symmetric left/right)
    # Joint order: [l_hip_roll, l_hip_yaw, l_hip_pitch, l_knee, l_wheel,
    #               r_hip_roll, r_hip_yaw, r_hip_pitch, r_knee, r_wheel]
    data.qpos[7:17] = [
        0, 0, hip_pitch_des, knee_des, 0,  # left leg
        0, 0, hip_pitch_des, knee_des, 0,  # right leg
    ]

    # Forward kinematics
    mujoco.mj_forward(model, data)


def run_fixed_height_test(
    model: mujoco.MjModel,
    controller,
    height_cmd: float,
    duration_s: float = 10.0,
    dt: float = 0.01,
) -> FixedHeightResult:
    """Run fixed-height balance test.

    Args:
        model: MuJoCo model
        controller: LQR/IK controller
        height_cmd: Target height in meters
        duration_s: Test duration in seconds
        dt: Control timestep

    Returns:
        FixedHeightResult with metrics
    """
    data = mujoco.MjData(model)

    # Initialize at target height
    initialize_at_height(model, data, controller, height_cmd)

    # PID gains (from balance_env.py)
    kp = np.array([55.0, 40.0, 70.0, 70.0, 4.0, 55.0, 40.0, 70.0, 70.0, 4.0])
    kd = np.array([3.0, 2.0, 4.0, 4.0, 0.0, 3.0, 2.0, 4.0, 4.0, 0.0])
    wheel_mask = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

    # Joint limits for denormalization
    joint_mins = np.array([
        -0.7, -0.4, -0.5, -0.5, -20.0,
        -0.7, -0.4, -0.5, -0.5, -20.0,
    ])
    joint_maxs = np.array([
        0.7, 0.4, 1.8, 2.7, 20.0,
        0.7, 0.4, 1.8, 2.7, 20.0,
    ])

    # Logging arrays
    n_steps = int(duration_s / dt)
    pitch_history = []
    roll_history = []
    height_error_history = []
    wheel_speed_history = []
    wheel_saturation_history = []
    base_action_history = []

    # Initial observation
    quat = data.qpos[3:7]
    g_body = get_gravity_in_body_frame(quat)
    body_ang_vel = data.qvel[3:6]
    body_lin_vel = data.qvel[0:3]
    qpos = data.qpos[7:17]
    qvel = data.qvel[6:16]
    prev_action = np.zeros(10)
    current_height = data.qpos[2]
    yaw_error = 0.0

    obs = np.concatenate([
        g_body, body_ang_vel, body_lin_vel,
        qpos, qvel, prev_action,
        [height_cmd, current_height, yaw_error]
    ])

    fell = False
    survival_time = 0.0

    for step in range(n_steps):
        # Get action from controller
        action = controller.compute_action(obs)

        # Log metrics
        quat = data.qpos[3:7]
        g_body = get_gravity_in_body_frame(quat)
        pitch = -np.arcsin(np.clip(g_body[1], -1.0, 1.0))
        roll = np.arctan2(g_body[0], -g_body[2])

        pitch_history.append(np.degrees(pitch))
        roll_history.append(np.degrees(roll))
        height_error_history.append(data.qpos[2] - height_cmd)

        # Wheel speed (average of left and right)
        wheel_speed = (data.qvel[10] + data.qvel[15]) / 2.0  # qvel indices for wheels
        wheel_speed_history.append(abs(wheel_speed))

        # Wheel saturation (normalized action > 0.9)
        wheel_actions = np.abs([action[4], action[9]])
        wheel_saturation_history.append(np.mean(wheel_actions > 0.9))

        # Base action RMS
        base_action_history.append(np.linalg.norm(action))

        # Check termination
        if abs(pitch) > 0.8 or abs(roll) > 0.8 or data.qpos[2] < 0.3:
            fell = True
            survival_time = step * dt
            break

        # Denormalize action
        pos_target = joint_mins + (action + 1.0) * 0.5 * (joint_maxs - joint_mins)
        vel_target_whl = action * 20.0

        # Get current joint state
        joint_pos = data.qpos[7:17]
        joint_vel = data.qvel[6:16]

        # Compute PID control
        pos_err = pos_target - joint_pos
        error = (1.0 - wheel_mask) * pos_err + wheel_mask * (vel_target_whl - joint_vel)
        d_error = (1.0 - wheel_mask) * (-joint_vel)

        ctrl = kp * error + kd * d_error
        ctrl = np.clip(ctrl, -100.0, 100.0)

        # Apply control
        data.ctrl[:] = ctrl

        # Step simulation
        mujoco.mj_step(model, data)

        # Update observation
        quat = data.qpos[3:7]
        g_body = get_gravity_in_body_frame(quat)
        body_ang_vel = data.qvel[3:6]
        body_lin_vel = data.qvel[0:3]
        qpos = data.qpos[7:17]
        qvel = data.qvel[6:16]
        prev_action = action
        current_height = data.qpos[2]

        obs = np.concatenate([
            g_body, body_ang_vel, body_lin_vel,
            qpos, qvel, prev_action,
            [height_cmd, current_height, yaw_error]
        ])

    if not fell:
        survival_time = duration_s

    # Compute metrics
    pitch_rms = np.sqrt(np.mean(np.array(pitch_history)**2))
    roll_rms = np.sqrt(np.mean(np.array(roll_history)**2))
    height_error_rmse = np.sqrt(np.mean(np.array(height_error_history)**2))
    wheel_speed_rms = np.sqrt(np.mean(np.array(wheel_speed_history)**2))
    wheel_saturation_rate = np.mean(wheel_saturation_history)
    base_action_rms = np.mean(base_action_history)

    return FixedHeightResult(
        height_cmd=height_cmd,
        survival_time=survival_time,
        fell=fell,
        pitch_rms_deg=pitch_rms,
        roll_rms_deg=roll_rms,
        height_error_rmse=height_error_rmse,
        wheel_speed_rms=wheel_speed_rms,
        wheel_saturation_rate=wheel_saturation_rate,
        base_action_rms=base_action_rms,
    )


def main():
    print("=" * 80)
    print("LQR/IK Prior Fixed-Height Balance Evaluation")
    print("=" * 80)

    # Load model
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # Create controller
    config_path = Path(__file__).parent.parent / "configs" / "controllers" / "gain_scheduled_lqr.yaml"
    controller = create_lqr_ik_prior(config_path, model)

    # Test heights
    test_heights = [0.70, 0.65, 0.60, 0.55, 0.50]

    print("\nTesting fixed-height balance (10s episodes, 3 trials each):")
    print("-" * 80)

    results = []

    for height in test_heights:
        print(f"\nHeight {height:.2f}m:")
        trial_results = []

        for trial in range(3):
            result = run_fixed_height_test(model, controller, height, duration_s=10.0)
            trial_results.append(result)

            status = "SURVIVED" if not result.fell else f"FELL at {result.survival_time:.2f}s"
            print(f"  Trial {trial+1}: {status}, pitch={result.pitch_rms_deg:.2f}°, roll={result.roll_rms_deg:.2f}°")

        results.append((height, trial_results))

    # Summary table
    print("\n" + "=" * 80)
    print("Summary Table")
    print("=" * 80)
    print(f"{'Height (m)':<12} {'Survival':<12} {'Fall Rate':<12} {'Pitch RMS':<12} {'Roll RMS':<12} {'Height RMSE':<14}")
    print("-" * 80)

    for height, trial_results in results:
        survival_times = [r.survival_time for r in trial_results]
        fall_rate = sum(r.fell for r in trial_results) / len(trial_results)
        avg_survival = np.mean(survival_times)
        avg_pitch = np.mean([r.pitch_rms_deg for r in trial_results])
        avg_roll = np.mean([r.roll_rms_deg for r in trial_results])
        avg_height_err = np.mean([r.height_error_rmse for r in trial_results])

        print(f"{height:<12.2f} {avg_survival:<12.2f} {fall_rate:<12.1%} {avg_pitch:<12.2f} {avg_roll:<12.2f} {avg_height_err:<14.4f}")

    print("\n" + "=" * 80)
    print("Acceptance Criteria Check")
    print("=" * 80)

    # Check if any height meets strong prior criteria
    strong_prior = False
    for height, trial_results in results:
        avg_survival = np.mean([r.survival_time for r in trial_results])
        fall_rate = sum(r.fell for r in trial_results) / len(trial_results)
        avg_pitch = np.mean([r.pitch_rms_deg for r in trial_results])
        avg_roll = np.mean([r.roll_rms_deg for r in trial_results])

        if avg_survival >= 5.0 and fall_rate <= 0.2 and avg_pitch < 5.0 and avg_roll < 5.0:
            print(f"✓ Height {height:.2f}m meets strong prior criteria")
            strong_prior = True

    if strong_prior:
        print("\nResult: STRONG PRIOR - suitable for fixed-height nominal balance")
    else:
        print("\nResult: LIMITED PRIOR - needs verification of signs and bounded actions")
        print("        Height transitions likely require residual PPO")


if __name__ == "__main__":
    main()
