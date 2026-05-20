"""Test hybrid controller: Segway wheel velocity + strong roll stabilization.

Combines:
- Segway-style wheel velocity control from LQR baseline (pitch balance)
- Strong roll stabilization from hierarchical WBC (lateral balance)
- Simple position PD for leg joints (height/posture)
"""

import mujoco
import numpy as np
from pathlib import Path


def get_pitch_roll_from_quat(quat):
    """Extract pitch and roll from quaternion [w, x, y, z]."""
    w, x, y, z = quat
    pitch = float(np.arctan2(2 * (w * y + x * z), 1 - 2 * (y**2 + z**2)))
    roll = float(2 * np.arcsin(np.clip(2 * (w * x - y * z), -1, 1)))
    return pitch, roll


def main():
    print("=" * 80)
    print("Hybrid Controller Test: Segway Wheels + Strong Roll Stabilization")
    print("=" * 80)

    model_path = "assets/robot/wheeled_biped_real.xml"
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    # Initialize from keyframe
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        print("[OK] Robot initialized from keyframe 0")

    mujoco.mj_forward(mj_model, mj_data)

    # Target leg configuration from keyframe (UPDATED: hip_pitch=0.4, knee=0.4)
    # qpos order: [root_pos(3) root_quat(4) | l_hip_roll l_hip_yaw l_hip_pitch l_knee l_wheel | r_hip_roll r_hip_yaw r_hip_pitch r_knee r_wheel]
    target_leg_pos = np.array([
        0.0,    # l_hip_roll
        0.0,    # l_hip_yaw
        0.4,    # l_hip_pitch (CORRECTED from 0.453)
        0.4,    # l_knee (CORRECTED from 2.765)
        0.0,    # r_hip_roll
        0.0,    # r_hip_yaw
        0.4,    # r_hip_pitch
        0.4,    # r_knee
    ])
    LEG_INDICES = [0, 1, 2, 3, 5, 6, 7, 8]  # exclude wheels 4, 9

    # PD gains for leg position control
    kp_leg = np.array([20, 15, 30, 30, 20, 15, 30, 30])  # Stronger hip_roll for lateral stability
    kd_leg = np.array([2.0, 1.5, 3.0, 3.0, 2.0, 1.5, 3.0, 3.0])

    # Segway balance gains (from LQR baseline)
    K_pitch = 87.7        # pitch proportional gain
    K_pitch_rate = 20.6   # pitch rate gain
    K_fwd_vel = 5.2       # forward velocity gain
    K_fwd_pos = 0.6       # forward position drift gain

    # Roll stabilization gains (VERY STRONG to prevent lateral tipping)
    kp_roll = 80.0   # Very strong roll stabilization
    kd_roll = 15.0   # Very strong roll damping

    # Yaw hold gains
    kp_yaw = 2.5
    kd_yaw = 0.2

    print(f"\nController gains:")
    print(f"  Segway: K_pitch={K_pitch:.1f}, K_pitch_rate={K_pitch_rate:.1f}")
    print(f"  Roll: kp={kp_roll:.1f}, kd={kd_roll:.1f}")
    print(f"  Leg position: kp_hip_roll={kp_leg[0]:.1f}, kp_hip_pitch={kp_leg[2]:.1f}")

    print(f"\nRunning simulation for 3000 steps (60 seconds)")
    print(f"{'Step':>6} {'h':>7} {'pitch':>8} {'roll':>8} {'roll_rate':>10} {'roll_corr':>10} {'l_hip_roll':>11} {'r_hip_roll':>11}")
    print("-" * 90)

    max_steps = 3000
    control_dt = 0.02
    physics_dt = mj_model.opt.timestep
    n_substeps = int(control_dt / physics_dt)

    # State for position drift integration
    fwd_pos_drift = 0.0

    for step in range(max_steps):
        # Current state
        joint_pos = mj_data.qpos[7:17]   # all 10 joint positions
        joint_vel = mj_data.qvel[6:16]   # all 10 joint velocities
        quat = mj_data.qpos[3:7]         # [w, x, y, z]

        pitch, roll = get_pitch_roll_from_quat(quat)
        pitch_rate = mj_data.qvel[4]     # angular velocity around x-axis
        roll_rate = mj_data.qvel[3]      # angular velocity around y-axis (roll)

        # Forward velocity (body frame y-axis, negative = forward in world -Y)
        fwd_vel = -mj_data.qvel[1]

        # Integrate position drift
        fwd_pos_drift += fwd_vel * control_dt

        # Yaw error and rate
        yaw_error = 0.0  # Assume no yaw drift for now
        yaw_rate = mj_data.qvel[5]

        # --- Leg position PD control ---
        tau = np.zeros(10)
        for i, idx in enumerate(LEG_INDICES):
            pos_err = target_leg_pos[i] - joint_pos[idx]
            tau[idx] = kp_leg[i] * pos_err - kd_leg[i] * joint_vel[idx]

        # --- Segway wheel velocity control (TWIP balance) ---
        # State: [pitch, pitch_rate, fwd_vel, fwd_pos_drift]
        # Control law: u = -(K @ x)
        wheel_vel_cmd = -(K_pitch * pitch + K_pitch_rate * pitch_rate +
                         K_fwd_vel * fwd_vel + K_fwd_pos * fwd_pos_drift)

        # Convert wheel velocity (rad/s) to torque via simple proportional gain
        # Wheels use velocity targets, but MuJoCo PID needs torque commands
        # Use a velocity feedback gain
        kp_wheel_vel = 2.0  # Nm/(rad/s) - velocity tracking gain
        l_wheel_vel = joint_vel[4]
        r_wheel_vel = joint_vel[9]

        tau[4] = kp_wheel_vel * (wheel_vel_cmd - l_wheel_vel)
        tau[9] = kp_wheel_vel * (wheel_vel_cmd - r_wheel_vel)

        # --- Roll stabilization (antisymmetric hip_roll) ---
        # Positive roll = leaning right → left hip_roll should push left leg out
        roll_correction = kp_roll * roll + kd_roll * roll_rate
        tau[0] += -roll_correction  # l_hip_roll (negative = push left leg out when leaning right)
        tau[5] += roll_correction   # r_hip_roll (positive = pull right leg in when leaning right)

        # --- Yaw hold (differential wheel speed) ---
        yaw_correction = kp_yaw * yaw_error + kd_yaw * yaw_rate
        tau[4] += -yaw_correction  # l_wheel
        tau[9] += yaw_correction   # r_wheel

        mj_data.ctrl[:] = tau

        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data)

        com_z = mj_data.qpos[2]

        if step % 1 == 0:  # Print every step for detailed debugging
            l_hip_roll_tau = tau[0]
            r_hip_roll_tau = tau[5]
            print(f"{step:6d} {com_z:7.3f} {np.rad2deg(pitch):8.2f} {np.rad2deg(roll):8.2f} {np.rad2deg(roll_rate):10.2f} {roll_correction:10.2f} {l_hip_roll_tau:11.2f} {r_hip_roll_tau:11.2f}")

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
