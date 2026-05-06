"""Diagnostic to verify joint position tracking.

Checks if actual joint positions match commanded positions from the controller.
"""

import numpy as np
import mujoco
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from wheeled_biped.utils.config import get_model_path
from wheeled_biped.controllers.lqr_ik_prior import create_lqr_ik_prior
from wheeled_biped.utils.math_utils import get_gravity_in_body_frame


def main():
    # Load model
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Create controller
    config_path = Path(__file__).parent.parent / "configs" / "controllers" / "gain_scheduled_lqr.yaml"
    controller = create_lqr_ik_prior(config_path, model)

    # Reset
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Build initial observation
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
    action = controller.compute_action(obs)

    # Joint limits for denormalization
    joint_mins = np.array([
        -0.7, -0.4, -0.5, -0.5, -20.0,
        -0.7, -0.4, -0.5, -0.5, -20.0,
    ])
    joint_maxs = np.array([
        0.7, 0.4, 1.8, 2.7, 20.0,
        0.7, 0.4, 1.8, 2.7, 20.0,
    ])

    # Denormalize action
    pos_target = joint_mins + (action + 1.0) * 0.5 * (joint_maxs - joint_mins)

    print("Joint tracking diagnostic:")
    print("=" * 80)
    print(f"\nCommanded height: {height_cmd} m")
    print(f"Initial torso height: {data.qpos[2]:.4f} m")
    print(f"\nNormalized actions:")
    print(f"  L_hip_roll:  {action[0]:7.4f}")
    print(f"  L_hip_yaw:   {action[1]:7.4f}")
    print(f"  L_hip_pitch: {action[2]:7.4f}")
    print(f"  L_knee:      {action[3]:7.4f}")
    print(f"  L_wheel:     {action[4]:7.4f}")

    print(f"\nDenormalized position targets (rad):")
    print(f"  L_hip_roll:  {pos_target[0]:7.4f}")
    print(f"  L_hip_yaw:   {pos_target[1]:7.4f}")
    print(f"  L_hip_pitch: {pos_target[2]:7.4f}")
    print(f"  L_knee:      {pos_target[3]:7.4f}")

    print(f"\nInitial joint positions (rad):")
    print(f"  L_hip_roll:  {qpos[0]:7.4f}")
    print(f"  L_hip_yaw:   {qpos[1]:7.4f}")
    print(f"  L_hip_pitch: {qpos[2]:7.4f}")
    print(f"  L_knee:      {qpos[3]:7.4f}")

    # Run simulation for 100 steps
    print(f"\nRunning simulation for 1.0s (100 steps)...")

    # Use actual environment PID gains (from balance_env.py default_kp/kd)
    kp = np.array([55.0, 40.0, 70.0, 70.0, 4.0, 55.0, 40.0, 70.0, 70.0, 4.0])
    kd = np.array([3.0, 2.0, 4.0, 4.0, 0.0, 3.0, 2.0, 4.0, 4.0, 0.0])
    wheel_mask = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

    # Track pitch and wheel velocity over time
    pitch_history = []
    wheel_vel_cmd_history = []

    for step in range(100):
        # Update observation to get fresh controller commands
        quat = data.qpos[3:7]
        g_body = get_gravity_in_body_frame(quat)
        body_ang_vel = data.qvel[3:6]
        body_lin_vel = data.qvel[0:3]
        qpos = data.qpos[7:17]
        qvel = data.qvel[6:16]
        current_height = data.qpos[2]

        obs = np.concatenate([
            g_body, body_ang_vel, body_lin_vel,
            qpos, qvel, action,  # use previous action
            [height_cmd, current_height, yaw_error]
        ])

        # Get fresh action from controller
        action = controller.compute_action(obs)
        pos_target = joint_mins + (action + 1.0) * 0.5 * (joint_maxs - joint_mins)

        # Track pitch and wheel commands
        pitch = -np.arcsin(np.clip(g_body[1], -1.0, 1.0))
        pitch_history.append(np.degrees(pitch))
        wheel_vel_cmd_history.append(action[4] * 20.0)  # L_WHEEL velocity command

        # Get current joint state
        joint_pos = data.qpos[7:17]
        joint_vel = data.qvel[6:16]

        # Compute PID control
        vel_target_whl = action * 20.0
        pos_err = pos_target - joint_pos
        error = (1.0 - wheel_mask) * pos_err + wheel_mask * (vel_target_whl - joint_vel)
        d_error = (1.0 - wheel_mask) * (-joint_vel)

        ctrl = kp * error + kd * d_error
        ctrl = np.clip(ctrl, -100.0, 100.0)

        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

    print(f"\nFinal joint positions after 1.0s (rad):")
    final_qpos = data.qpos[7:17]
    print(f"  L_hip_roll:  {final_qpos[0]:7.4f}  (target: {pos_target[0]:7.4f}, error: {final_qpos[0] - pos_target[0]:+7.4f})")
    print(f"  L_hip_yaw:   {final_qpos[1]:7.4f}  (target: {pos_target[1]:7.4f}, error: {final_qpos[1] - pos_target[1]:+7.4f})")
    print(f"  L_hip_pitch: {final_qpos[2]:7.4f}  (target: {pos_target[2]:7.4f}, error: {final_qpos[2] - pos_target[2]:+7.4f})")
    print(f"  L_knee:      {final_qpos[3]:7.4f}  (target: {pos_target[3]:7.4f}, error: {final_qpos[3] - pos_target[3]:+7.4f})")

    print(f"\nFinal torso height: {data.qpos[2]:.4f} m (target: {height_cmd} m, error: {data.qpos[2] - height_cmd:+.4f} m)")

    print(f"\nPitch and wheel velocity history:")
    print(f"  Initial pitch: {pitch_history[0]:+.2f}°")
    print(f"  Final pitch: {pitch_history[-1]:+.2f}°")
    print(f"  Pitch range: [{min(pitch_history):+.2f}°, {max(pitch_history):+.2f}°]")
    print(f"  Initial wheel vel cmd: {wheel_vel_cmd_history[0]:+.2f} rad/s")
    print(f"  Final wheel vel cmd: {wheel_vel_cmd_history[-1]:+.2f} rad/s")
    print(f"  Wheel vel cmd range: [{min(wheel_vel_cmd_history):+.2f}, {max(wheel_vel_cmd_history):+.2f}] rad/s")
    print("=" * 80)


if __name__ == "__main__":
    main()
