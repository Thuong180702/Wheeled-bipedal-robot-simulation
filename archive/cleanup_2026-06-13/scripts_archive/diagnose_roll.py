"""Diagnostic script for roll stabilization debugging.

Logs roll signals over the first few timesteps to identify sign errors.
"""

import numpy as np
import mujoco
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from wheeled_biped.controllers.lqr_ik_prior import create_lqr_ik_prior
from wheeled_biped.utils.math_utils import get_gravity_in_body_frame


def main():
    # Load model using the same path as eval_balance.py
    from wheeled_biped.utils.config import get_model_path
    from wheeled_biped.utils.math_utils import get_gravity_in_body_frame

    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Create LQR/IK prior
    config_path = Path(__file__).parent.parent / "configs" / "controllers" / "gain_scheduled_lqr.yaml"
    controller = create_lqr_ik_prior(config_path, model)

    # Reset to initial state
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Build observation manually (42-dim)
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

    print("Timestep | g_body[0] | ang_vel[1] | roll_corr | hip_roll_act | hip_pitch_act | knee_act | torso_roll_deg | torso_z")
    print("-" * 120)

    # Joint limits for denormalization
    joint_mins = np.array([
        -0.7, -0.4, -0.5, -0.5, -20.0,  # left leg
        -0.7, -0.4, -0.5, -0.5, -20.0,  # right leg
    ])
    joint_maxs = np.array([
        0.7, 0.4, 1.8, 2.7, 20.0,  # left leg
        0.7, 0.4, 1.8, 2.7, 20.0,  # right leg
    ])

    # PID gains
    kp = 100.0
    kd = 10.0
    wheel_mask = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

    # Run for 50 steps (0.5s)
    for step in range(50):
        # Get action from controller
        action = controller.compute_action(obs)

        # Parse observation to get roll signals
        g_body = obs[0:3]
        body_ang_vel = obs[3:6]

        # Compute what the controller computed
        roll_error = g_body[0]
        roll_rate = body_ang_vel[1]
        roll_correction = (
            controller.config.roll_kp * roll_error +
            controller.config.roll_kd * roll_rate
        )
        roll_correction_clipped = np.clip(
            roll_correction,
            -controller.config.roll_max_correction,
            controller.config.roll_max_correction,
        )

        # Get action components (normalized)
        hip_roll_action = action[0]  # L_HIP_ROLL
        hip_pitch_action = action[2]  # L_HIP_PITCH
        knee_action = action[3]  # L_KNEE

        # Compute actual torso roll angle from gravity vector
        torso_roll_rad = np.arctan2(g_body[0], -g_body[2])
        torso_roll_deg = np.degrees(torso_roll_rad)

        # Get torso height
        torso_z = data.qpos[2]

        print(f"{step:8d} | {g_body[0]:9.4f} | {body_ang_vel[1]:10.4f} | "
              f"{roll_correction_clipped:9.4f} | {hip_roll_action:12.4f} | "
              f"{hip_pitch_action:13.4f} | {knee_action:8.4f} | "
              f"{torso_roll_deg:14.2f} | {torso_z:7.3f}")

        # Denormalize action to joint targets
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

        # Update observation for next iteration
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

        # Check for fall
        if torso_z < 0.2:
            print(f"\nRobot fell at step {step}")
            break


if __name__ == "__main__":
    main()
