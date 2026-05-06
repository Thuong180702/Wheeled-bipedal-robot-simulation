"""Diagnostic script for sagittal LQR balance debugging.

Logs sagittal balance signals during fixed-height balance to identify sign errors
or gain issues in the LQR wheel velocity commands.
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
    print("=" * 80)
    print("Sagittal LQR Diagnostic - Fixed Height Balance")
    print("=" * 80)

    # Load model
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Create controller
    config_path = Path(__file__).parent.parent / "configs" / "controllers" / "gain_scheduled_lqr.yaml"
    controller = create_lqr_ik_prior(config_path, model)

    # Test at h=0.65m
    height_cmd = 0.65

    # Initialize at target height using IK
    hip_pitch_des, knee_des = controller.height_ik(height_cmd)
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0, 0, height_cmd]
    data.qpos[3:7] = [1, 0, 0, 0]  # upright quaternion
    data.qpos[7:17] = [
        0, 0, hip_pitch_des, knee_des, 0,  # left leg
        0, 0, hip_pitch_des, knee_des, 0,  # right leg
    ]
    mujoco.mj_forward(model, data)

    print(f"\nInitialized at h={height_cmd}m")
    print(f"  IK solution: hip_pitch={hip_pitch_des:.4f} rad, knee={knee_des:.4f} rad")
    print(f"  Initial torso height: {data.qpos[2]:.4f} m")
    print(f"  Initial torso pitch: {np.degrees(-np.arcsin(np.clip(get_gravity_in_body_frame(data.qpos[3:7])[1], -1.0, 1.0))):.2f}°")

    # PID gains
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

    # Build initial observation
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

    print("\n" + "=" * 80)
    print("Sagittal Balance Signals (first 100 steps = 1.0s)")
    print("=" * 80)
    print(f"{'Step':>5} | {'pitch°':>7} | {'pitch_rate':>11} | {'fwd_vel':>8} | {'fwd_pos':>8} | "
          f"{'wheel_cmd':>10} | {'wheel_vel':>10} | {'torso_z':>8} | {'hip_pitch':>10} | {'knee':>10}")
    print("-" * 120)

    for step in range(100):
        # Get action from controller
        action = controller.compute_action(obs)

        # Parse observation
        g_body = obs[0:3]
        body_ang_vel = obs[3:6]
        body_lin_vel = obs[6:9]
        qpos_obs = obs[9:19]
        qvel_obs = obs[19:29]

        # Compute pitch from gravity vector
        pitch_rad = -np.arcsin(np.clip(g_body[1], -1.0, 1.0))
        pitch_deg = np.degrees(pitch_rad)
        pitch_rate = body_ang_vel[1]  # pitch rate (rad/s)

        # Forward velocity and position
        fwd_vel = body_lin_vel[0]  # forward velocity (m/s)
        fwd_pos = data.qpos[0]  # forward position (m)

        # Wheel command (normalized action for left wheel)
        wheel_action_norm = action[4]
        wheel_cmd_rad_s = wheel_action_norm * 20.0  # denormalized wheel velocity command

        # Actual wheel velocity
        wheel_vel_actual = data.qvel[10]  # left wheel velocity (rad/s)

        # Torso height
        torso_z = data.qpos[2]

        # Joint positions
        hip_pitch_actual = data.qpos[9]  # left hip pitch
        knee_actual = data.qpos[10]  # left knee

        if step % 10 == 0:  # Print every 10 steps
            print(f"{step:5d} | {pitch_deg:7.2f} | {pitch_rate:11.4f} | {fwd_vel:8.4f} | {fwd_pos:8.4f} | "
                  f"{wheel_cmd_rad_s:10.2f} | {wheel_vel_actual:10.2f} | {torso_z:8.4f} | "
                  f"{hip_pitch_actual:10.4f} | {knee_actual:10.4f}")

        # Check for fall
        if abs(pitch_rad) > 0.8 or torso_z < 0.3:
            print(f"\n*** FELL at step {step} (t={step*0.01:.2f}s) ***")
            print(f"    Final pitch: {pitch_deg:.2f}°")
            print(f"    Final torso height: {torso_z:.4f} m")
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

    print("\n" + "=" * 80)
    print("Sign Convention Check")
    print("=" * 80)
    print("Expected behavior for stable balance:")
    print("  - Robot pitches forward (+pitch) -> wheels should move forward (+wheel_cmd) to catch fall")
    print("  - Robot pitches backward (-pitch) -> wheels should move backward (-wheel_cmd) to catch fall")
    print("\nExpected LQR response:")
    print("  - pitch > 0 -> wheel_cmd > 0 (forward)")
    print("  - pitch < 0 -> wheel_cmd < 0 (backward)")
    print("\nIf wheel_cmd has opposite sign from pitch, the LQR sign is WRONG.")
    print("=" * 80)


if __name__ == "__main__":
    main()
