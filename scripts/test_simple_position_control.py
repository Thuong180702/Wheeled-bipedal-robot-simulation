"""Test wheeled biped balance with Segway-style wheel velocity control.

Leg joints: position PD to maintain height/posture.
Wheel joints: velocity PD based on pitch error (Segway balance law).
"""

import mujoco
import numpy as np
from pathlib import Path


def get_pitch_from_quat(quat):
    """Extract pitch angle from quaternion [w, x, y, z]."""
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2)))


def main():
    print("=" * 80)
    print("Wheeled Biped Balance Test (Segway-style wheel control)")
    print("=" * 80)

    model_path = "assets/robot/wheeled_biped_real.xml"
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    # Ground-contact configuration: wheels at z=0.061m
    # CoM is 0.099m behind wheels → needs active Segway control
    torso_z = 0.71
    hip_pitch = 0.4   # rad (~23 deg)
    knee = 0.4        # rad (~23 deg)

    # Initialize robot at ground-contact configuration
    mj_data.qpos[:] = 0
    mj_data.qpos[2] = torso_z
    mj_data.qpos[3] = 1.0  # quaternion w (identity)
    # l_hip_pitch=qpos[9], l_knee=qpos[10], r_hip_pitch=qpos[14], r_knee=qpos[15]
    mj_data.qpos[9]  = hip_pitch
    mj_data.qpos[10] = knee
    mj_data.qpos[14] = hip_pitch
    mj_data.qpos[15] = knee
    mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)

    print(f"Initial config: torso_z={torso_z}m, hip_pitch={np.rad2deg(hip_pitch):.1f}deg, knee={np.rad2deg(knee):.1f}deg")

    # Target joint positions for leg joints (indices 0-9 in action space)
    # Order: l_hip_roll, l_hip_yaw, l_hip_pitch, l_knee, l_wheel,
    #        r_hip_roll, r_hip_yaw, r_hip_pitch, r_knee, r_wheel
    target_leg_pos = np.array([
        0.0, 0.0, hip_pitch, knee,  # left leg (no wheel target)
        0.0, 0.0, hip_pitch, knee,  # right leg (no wheel target)
    ])
    LEG_INDICES = [0, 1, 2, 3, 5, 6, 7, 8]  # exclude wheel indices 4, 9

    # PD gains for leg position control
    kp_leg = np.array([15, 10, 25, 25, 15, 10, 25, 25])
    kd_leg = np.array([1.5, 1.0, 2.5, 2.5, 1.5, 1.0, 2.5, 2.5])

    # Segway balance gains for wheel velocity control
    # target_pitch=0: stay upright, wheels compensate for any lean
    target_pitch = 0.0
    kp_pitch = 80.0   # pitch proportional gain
    kd_pitch = 15.0   # pitch derivative gain

    print(f"\nRunning simulation for 3000 steps (60 seconds)")
    print(f"{'Step':>6} {'h':>7} {'pitch':>8} {'pitch_rate':>11} {'wheel_cmd':>10} {'max_leg_tau':>12}")
    print("-" * 65)

    max_steps = 3000
    control_dt = 0.02
    physics_dt = mj_model.opt.timestep
    n_substeps = int(control_dt / physics_dt)

    prev_pitch = 0.0

    for step in range(max_steps):
        # Current state
        joint_pos = mj_data.qpos[7:17]   # all 10 joint positions
        joint_vel = mj_data.qvel[6:16]   # all 10 joint velocities
        quat = mj_data.qpos[3:7]         # [w, x, y, z]
        pitch = get_pitch_from_quat(quat)
        pitch_rate = mj_data.qvel[4]     # angular velocity around x-axis (pitch rate)

        # --- Leg position PD control ---
        tau = np.zeros(10)
        for i, idx in enumerate(LEG_INDICES):
            pos_err = target_leg_pos[i] - joint_pos[idx]
            tau[idx] = kp_leg[i] * pos_err - kd_leg[i] * joint_vel[idx]

        # --- Wheel velocity control (Segway balance law) ---
        # pitch_error > 0 (leaning forward) → drive wheels forward to catch
        # pitch_error < 0 (leaning backward) → drive wheels backward to catch
        pitch_error = pitch - target_pitch
        wheel_cmd = kp_pitch * pitch_error + kd_pitch * pitch_rate
        tau[4] = wheel_cmd   # l_wheel
        tau[9] = wheel_cmd   # r_wheel

        mj_data.ctrl[:] = tau

        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data)

        com_z = mj_data.qpos[2]
        roll = float(2 * np.arcsin(np.clip(2 * (quat[0] * quat[2] - quat[3] * quat[1]), -1, 1)))

        if step % 20 == 0:
            max_leg_tau = float(np.max(np.abs(tau[LEG_INDICES])))
            print(f"{step:6d} {com_z:7.3f} {np.rad2deg(pitch):8.2f} {np.rad2deg(pitch_rate):11.2f} {wheel_cmd:10.2f} {max_leg_tau:12.2f}")

        if com_z < 0.35:
            print(f"\n[TERMINATED] step {step}: height_too_low (h={com_z:.3f}m)")
            break
        if abs(pitch) > 0.785 or abs(roll) > 0.785:
            print(f"\n[TERMINATED] step {step}: orientation_fail (pitch={np.rad2deg(pitch):.1f}deg, roll={np.rad2deg(roll):.1f}deg)")
            break
    else:
        print(f"\n[SUCCESS] Completed {max_steps} steps!")

    print("=" * 80)


if __name__ == "__main__":
    main()
